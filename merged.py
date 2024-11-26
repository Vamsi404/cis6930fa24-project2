import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pickle
import argparse

nltk.download('punkt')
nltk.download('vader_lexicon')

# ------------------- Preprocessing Section -------------------

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='skip', names=['split', 'name', 'context'], header=0)
    return data


def load_test_data(file_path):
    data = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='skip', names=['context'], header=0)
    return data


def extract_features(row):
    context = row['context']
    redacted_length = len(row.get('name', ''))  # Handle test data without a 'name' column

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


def preprocess_data(file_path, output_path, split=None, is_test=False):
    if is_test:
        data = load_test_data(file_path)
    else:
        data = load_data(file_path)
        if split:
            data = data[data['split'] == split]  # Filter based on 'split' column

    data['features'] = data.apply(extract_features, axis=1)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Preprocessed data saved to {output_path}")


# ------------------- Model Section -------------------

def train_model(features_path, model_path):
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(data['features'].tolist())
    y = data['name']

    model = RandomForestClassifier()
    model.fit(X, y)

    with open(model_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)

    print(f"Model saved to {model_path}")


def evaluate_model(features_path, model_path):
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)

    X = vectorizer.transform(data['features'].tolist())
    y = data['name']

    y_pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')

    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


# ------------------- Prediction Section -------------------

def predict_unredacted(features_path, model_path, output_path):
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)

    X = vectorizer.transform(data['features'].tolist())
    predictions = model.predict(X)

    data['predicted_name'] = predictions
    data.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# ------------------- Main Script -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Make predictions")
    parser.add_argument("--features", help="Path to preprocessed features file")
    parser.add_argument("--data", help="Path to raw data file")
    parser.add_argument("--model", help="Path to save/load the model")
    parser.add_argument("--output", help="Path to save predictions")
    parser.add_argument("--is_test", action="store_true", help="Flag for processing test data")

    args = parser.parse_args()

    if args.preprocess and args.data and args.features:
        preprocess_data(args.data, args.features, is_test=args.is_test)
    elif args.train and args.features and args.model:
        train_model(args.features, args.model)
    elif args.evaluate and args.features and args.model:
        evaluate_model(args.features, args.model)
    elif args.predict and args.features and args.model and args.output:
        predict_unredacted(args.features, args.model, args.output)
    else:
        parser.print_help()
