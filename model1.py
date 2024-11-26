import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import argparse

def train_model(features_path, model_path, test_data_path):
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    # Convert features to matrix
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(data['features'].tolist())
    y = data['name']

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Data split: {X_train.shape[0]} train, {X_val.shape[0]} validation, {X_test.shape[0]} test")

    # Train the model on the training set
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='weighted')
    accuracy = accuracy_score(y_val, y_val_pred)

    print(f"Validation - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Save the model and vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)

    # Save the test data
    with open(test_data_path, 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)

    print(f"Model saved to {model_path}")
    print(f"Test data saved to {test_data_path}")

def evaluate_model(model_path, test_data_path):
    # Load the model and vectorizer
    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)

    # Load the test data
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    X_test = test_data['X_test']
    y_test = test_data['y_test']

    # Evaluate on test set
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--features", required="--train" in argparse._sys.argv, help="Path to preprocessed features file")
    parser.add_argument("--model", required=True, help="Path to save/load the model")
    parser.add_argument("--test_data", required="--evaluate" in argparse._sys.argv, help="Path to save/load the test data")

    args = parser.parse_args()

    if args.train:
        train_model(args.features, args.model, args.test_data)
    elif args.evaluate:
        evaluate_model(args.model, args.test_data)
