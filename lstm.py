import random as python_random
import json
import argparse
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.layers import Dense, Embedding, LSTM, BatchNormalization, Dropout, Bidirectional
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from keras.initializers import Constant
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Text Classification with Preprocessing and Machine Learning Models')

    # Define command-line arguments
    parser.add_argument('--file', type=str, required=True, help='Path to the reviews file')
    parser.add_argument('--models', type=str, nargs='+', default=["SVM", "n-grams", "LSTM", "DeBERTa"], help='List of models to run')    
    parser.add_argument('--dev', type=str, default='dev.tsv', help='Path to the dev file')
    parser.add_argument('--test_file', type=str, default='test.tsv', help='Path to the test file')
    parser.add_argument("-e", "--embeddings", default='glove_filtered.json', type=str, help="Embedding file we are using (default glove_filtered.json)")

    return parser.parse_args()

# Initialize NLTK Lemmatizer and stop words, removing certain negative words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
negative_words = ['no', 'not', "don't", "aren't", "couldn't", "didn't", "doesn't",
                  "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't",
                  "shouldn't", "wasn't", "weren't", "won't", "wouldn't", "shan't",
                  "none", "nothing", "neither", "nor", "cannot", "can't"]

for word in negative_words:
    stop_words.discard(word)


def expand_contractions(text):
    """
    Replace contractions in the text with their expanded forms.

    Args:
        text (str): Text with contractions.

    Returns:
        str: Text with contractions expanded.
    """
    contractions_dict = {
        "i 'm": "i'm",
        "you 're": "you're",
        "he 's": "he's",
        "she 's": "she's",
        "it 's": "it's",
        "we 're": "we're",
        "they 're": "they're",
        "that 's": "that's",
        "there 's": "there's",
        "who 's": "who's",
        "what 's": "what's",
        "where 's": "where's",
        "when 's": "when's",
        "why 's": "why's",
        "how 's": "how's",
        "i 've": "i've",
        "you 've": "you've",
        "we 've": "we've",
        "they 've": "they've",
        "i 'd": "i'd",
        "you 'd": "you'd",
        "he 'd": "he'd",
        "she 'd": "she'd",
        "it 'd": "it'd",
        "we 'd": "we'd",
        "they 'd": "they'd",
        "i 'll": "i'll",
        "you 'll": "you'll",
        "he 'll": "he'll",
        "she 'll": "she'll",
        "it 'll": "it'll",
        "we 'll": "we'll",
        "they 'll": "they'll",
        "do n't": "don't",
        "does n't": "doesn't",
        "did n't": "didn't",
        "is n't": "isn't",
        "are n't": "aren't",
        "was n't": "wasn't",
        "were n't": "weren't",
        "has n't": "hasn't",
        "have n't": "haven't",
        "had n't": "hadn't",
        "wo n't": "won't",
        "would n't": "wouldn't",
        "ca n't": "can't",
        "could n't": "couldn't",
        "should n't": "shouldn't",
        "might n't": "mightn't",
        "must n't": "mustn't"
    }
    for contraction, correct_form in contractions_dict.items():
        text = re.sub(contraction, correct_form, text)
    return text


def remove_extra_spaces(text):
    """
    Remove unnecessary spaces before apostrophes in the text.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text with no extra spaces before apostrophes.
    """
    return re.sub(r" '", "'", text)


def clean_review_text(text):
    """
    Perform text cleaning by lowercasing, expanding contractions, 
    removing URLs, special characters, and non-alphabetical characters.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    text = text.lower()  # Convert to lowercase
    text = remove_extra_spaces(text)  # Fix space issues with apostrophes
    text = expand_contractions(text)  # Expand custom contractions
    # text = contractions.fix(text)  # Use the contractions package to expand more
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    REPLACE_BY_SPACE_RE = re.compile('[/(){}—[]|@,;‘?|।!-॥–’-]')
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # Replace special characters with space
    text = re.sub("[^a-z]+", " ", text)  # Remove non-alphabet characters
    return text


def lemmatize_and_tokenize(text):
    """
    Tokenize and lemmatize the input text.

    Args:
        text (str): Input text for tokenization and lemmatization.

    Returns:
        str: Lemmatized text.
    """
    tokens = word_tokenize(text)  # Tokenize text
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(lemmatized_tokens)


def preprocess_text(text):
    """
    Preprocess the input text by cleaning, tokenizing, and lemmatizing.

    Args:
        text (str): Input text for preprocessing.

    Returns:
        str: Preprocessed text.
    """
    cleaned_text = clean_review_text(text)  # Clean the text
    lemmatized_text = lemmatize_and_tokenize(cleaned_text)  # Tokenize and lemmatize
    return lemmatized_text


def load_reviews_from_file(file_path):
    """
    Load reviews from the specified text file and preprocess the reviews.

    Args:
        file_path (str): Path to the text file containing the reviews.

    Returns:
        pd.DataFrame: DataFrame containing processed reviews.
    """
    reviews = []
    with open(file_path, encoding='utf-8') as file:
        for line in file.readlines():
            reviews.append(line.strip())

    content = []
    label = []
    
    # Splitting the reviews based on spaces
    for review in reviews:
        parts = review.rsplit('\t', 1) # Split the review and label
        content.append(parts[0].strip())  # Add the content
        label.append(parts[1].strip())    # Add the label
    
    data = {'content': content, 'label': label}
    df = pd.DataFrame(data, columns=['content', 'label'])
    
    # Apply text preprocessing
    df['processed_content'] = df['content'].apply(preprocess_text)
    
    return df


def prepare_features_and_labels(df, vectorizer_type='tfidf', classification_type='binary'):
    """
    Prepare the feature matrix and target labels for model training. Supports
    selection between TF-IDF or Bag of Words vectorization and binary or multiclass classification.

    Args:
        df (pd.DataFrame): DataFrame containing processed reviews.
        vectorizer_type (str): Choice of vectorizer ('tfidf' or 'bow'). Default is 'tfidf'.
        classification_type (str): Classification type ('binary' or 'multiclass'). Default is 'multiclass'.

    Returns:
        X (sparse matrix): Vectorized feature matrix.
        y (pd.Series): Target labels.
    """
    X = df['processed_content']  # Extract processed review texts

    # Choose vectorizer based on user input
    if vectorizer_type == 'bow':
        vectorizer = CountVectorizer(max_features=5000)  # Bag of Words
    else:
        vectorizer = TfidfVectorizer(max_features=5000)  # TF-IDF

    X_vectorized = vectorizer.fit_transform(X)  # Fit and transform the review texts

    # Initialize Label Encoder
    label_encoder = LabelEncoder()
    y = df['label']

    return X_vectorized, y


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_lstm_model(Y_train, emb_matrix):
    '''Create the Keras model to use'''
    # Define settings, you might want to create cmd line args for them
    learning_rate = 0.0001
    loss_function = 'categorical_crossentropy'
    optim = Adam(learning_rate=learning_rate)
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))
    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix),trainable=False))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16))

    # Ultimately, end with dense layer with softmax
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev):
    '''Train the model here. Note the different settings you can experiment with!'''
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    batch_size = 16
    epochs = 150
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size, validation_data=(X_dev, Y_dev))
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "train")
    return model


def test_set_predict(model, X_test, Y_test, ident):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    if Y_test.ndim > 1:
        Y_test = np.argmax(Y_test, axis=1)
    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))
    # Print the metrics
    compute_metrics(Y_test, Y_pred, f'LSTM Model Results on {ident} set')


# Evaluate model performance
def compute_metrics(y_true, y_pred, title):
    """
    Print the F1 score, precision, recall, and classification report for the model.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        title (str): Title for the evaluation report.
    """
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    error = np.mean(y_true != y_pred)

    print(f'\n{title}:')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Error Rate: {error:.4f}')
    print(f'Classification Report:')
    print(classification_report(y_true, y_pred))



def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    train_df = load_reviews_from_file(args.file)
    dev_df = load_reviews_from_file(args.dev)
    embeddings = read_embeddings(args.embeddings)

    X_train, Y_train = train_df['processed_content'], train_df['label']
    # prepare_features_and_labels(train_df, 'tfidf', 'binary')
    X_dev, Y_dev = dev_df['processed_content'], dev_df['label']
    # prepare_features_and_labels(test_df, 'tfidf', 'binary')


    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelEncoder()
    Y_train_enc = encoder.fit_transform(Y_train)
    Y_dev_enc = encoder.transform(Y_dev)

    # Convert to one-hot encoding
    Y_train_bin = tf.keras.utils.to_categorical(Y_train_enc, num_classes=2)
    Y_dev_bin = tf.keras.utils.to_categorical(Y_dev_enc, num_classes=2)

    # Create model
    model = create_lstm_model(Y_train, emb_matrix)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin)

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        test_df = load_reviews_from_file(args.test_file)
        X_test, Y_test = test_df['processed_content'], test_df['label']
        Y_test_bin = encoder.fit_transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        # Finally do the predictions
        test_set_predict(model, X_test_vect, Y_test_bin, "test")

if __name__ == '__main__':
    main()
