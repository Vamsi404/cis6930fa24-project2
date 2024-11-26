import pickle
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")

def predict_unredacted(features_path, model_path, output_path):
    # Load test data from the preprocessed pickle file
    with open(features_path, 'rb') as f:
        data = pickle.load(f) 
    
    X= data['features'].tolist()
    # Load the trained model and vectorizer
    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)

    # Transform text data to features
    X_transformed = vectorizer.transform(X)
    predictions = model.predict(X_transformed)

    # Create the output DataFrame
    output_data = pd.DataFrame({
        "id": data["id"],
        "predicted_name": predictions
    })

    # Save the predictions to a TSV file
    output_data.to_csv(output_path,sep='\t', index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict names from redacted data.")
    parser.add_argument("--features", required=True, help="Path to test features TSV file")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--output", required=True, help="Path to save the predictions CSV file")

    args = parser.parse_args()
    predict_unredacted(args.features, args.model, args.output)
#pipenv run python predict.py --features Data/test.tsv --ModelSaved/model.pkl --output output/predictions.csv
