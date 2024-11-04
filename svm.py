import argparse
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

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


def create_svm_model():
    return SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f'Best hyperparameters for {model.__class__.__name__}: {grid_search.best_params_}')

    # Evaluation on the training set
    train_predictions = best_model.predict(X_train)
    compute_metrics(y_train, train_predictions, f'Training Results ({model.__class__.__name__})')

    # Evaluation on the test set
    best_model.fit(X_test, y_test)
    test_predictions = best_model.predict(X_test)
    compute_metrics(y_test, test_predictions, f'Test Results ({model.__class__.__name__})')


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
    """
    Main function to handle argument parsing, loading the dataset, and running 
    the specified models with chosen vectorization and classification settings.
    """
    parser = argparse.ArgumentParser(description='Text Classification with Preprocessing and Machine Learning Models')

    # Define command-line arguments
    parser.add_argument('--file', type=str, required=True, help='Path to the reviews file')
    parser.add_argument('--dev', type=str, default='dev.tsv', help='Path to the dev file')
    parser.add_argument('--test_file', type=str, default='test.tsv', help='Path to the test file')

    args = parser.parse_args()

    # Load and preprocess reviews
    train_df = load_reviews_from_file(args.file)
    test_df = load_reviews_from_file(args.test_file)

    # Run the specified models
    model = create_svm_model()

    # Prepare feature matrix and labels
    X_train, y_train = prepare_features_and_labels(train_df, 'tfidf', 'binary')
    X_test, y_test = prepare_features_and_labels(test_df, 'tfidf', 'binary')

    evaluate_model(model, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()