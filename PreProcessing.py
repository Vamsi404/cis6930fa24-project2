import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
import argparse
import os
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')

def load_data(file_path, is_test=False):
    """Load data from the provided file path. For test data, `is_test` flag will skip split and name columns."""
    if is_test:
        # For test data, we only have 'id' and 'context'
        data = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='skip', names=['id', 'context'], header=0)
    else:
        # For training/validation data, we load all three columns
        data = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='skip', names=['split', 'name', 'context'], header=0)
    return data

def extract_features(row, is_test=False):
    """Extract features from the given row of data."""
    context = row['context']
    
    # For test data, we assume 'id' is used as the redacted length
    if is_test:
        redacted_blocks = re.findall(r'████+', context)  # Look for sequences of '████'
        redacted_length = sum(len(block) for block in redacted_blocks)  # Total length of all redacted blocks
    else:
        redacted_length = len(row['name'])  # Assuming 'name' is used for redacted length in train/val data
    
    words = word_tokenize(context)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment = sentiment_analyzer.polarity_scores(context)
    
    features = {
        'prev_word': words[max(0, len(words) - redacted_length - 1)] if len(words) > redacted_length else '',
        'next_word': words[min(len(words) - 1, redacted_length + 1)] if len(words) > redacted_length else '',
        'sentiment': sentiment['compound'],
        'redacted_length': redacted_length
    }
    return features

def preprocess_train_val_data(input_path, output_path):
    """Process training and validation data, extract features, and save to pickle."""
    data = load_data(input_path, is_test=False)
    
    # Split into training and validation sets
    training_data = data[data['split'] == 'training']
    validation_data = data[data['split'] == 'validation']
    
    # Extract features
    training_data['features'] = training_data.apply(extract_features, axis=1, is_test=False)
    validation_data['features'] = validation_data.apply(extract_features, axis=1, is_test=False)
    
    # Save the processed data
    output = {'training_features': training_data, 'validation_features': validation_data}
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "features.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"Preprocessed data saved to {output_path}")

def preprocess_test_data(input_path, output_path):
    """Process test data, extract features, and save to pickle."""
    data = load_data(input_path, is_test=True)
    
    # Extract features for the test data
    data['features'] = data.apply(extract_features, axis=1, is_test=True)
    
    # Save the processed test data
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "test_features.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Processed test data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess unredactor data.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input file (TSV format).")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output pickle file.")
    parser.add_argument('--data_type', type=str, required=True, choices=['train', 'test'], help="Type of data: 'train' or 'test'.")
    
    args = parser.parse_args()
    
    if args.data_type == 'train':
        preprocess_train_val_data(args.input, args.output)
    elif args.data_type == 'test':
        preprocess_test_data(args.input, args.output)

#pipenv run python PreProcessing.py --input Data/unredactor.tsv --output Features/ --data_type train
#pipenv run python PreProcessing.py --input Data/test.tsv --output Features/ --data_type test
