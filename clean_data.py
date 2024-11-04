import pandas as pd
import re

# Load the .tsv file without headers or column names
file_path = '/home/carmen/learningFromData/project/dev_noemoji.tsv'  # Replace with the actual path to your file
df = pd.read_csv(file_path, sep='\t', header=None)

# Assuming the tweet text is in the second column (index 1), replace hashtags with 'HASHTAG'
df[0] = df[0].apply(lambda text: re.sub(r'#\w+', 'HASHTAG', text))

# Display the first few rows to verify changes
print(df.head())

# Optionally, save the modified DataFrame to a new file without headers
df.to_csv('/home/carmen/learningFromData/project/dev_clean.tsv', sep='\t', index=False, header=False)
