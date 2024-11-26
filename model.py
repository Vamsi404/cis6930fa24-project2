import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

def train_model(features_path, model_path):
    # Load features data
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract training data
    training_data = data['training_features']
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(training_data['features'].tolist())
    y_train = training_data['name']

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.pkl")

    # Save the trained model and vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)

    print(f"Model saved to {model_path}")

def evaluate_model(features_path, model_path):
    # Load features data
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract validation data
    validation_data = data['validation_features']
    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)

    # Transform validation data
    X_val = vectorizer.transform(validation_data['features'].tolist())
    y_val = validation_data['name']

    # Make predictions and evaluate
    y_pred = model.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the model.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--features", required=True, help="Path to preprocessed features file")
    parser.add_argument("--model", required=True, help="Path to save/load the model")

    args = parser.parse_args()

    if args.train:
        train_model(args.features, args.model)
    elif args.evaluate:
        evaluate_model(args.features, args.model)
#pipenv run python model.py --train --features Features/features.pkl --model ModelSaved
#pipenv run python model.py --evaluate --features Features/features.pkl --model ModelSaved/model.pkl
