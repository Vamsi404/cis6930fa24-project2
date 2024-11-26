# Unredactor Project

## Project Overview
The **Unredactor Project** is an end-to-end solution to process redacted text data, train a machine learning model to predict the redacted information, and make predictions on unseen test data. The pipeline includes three main steps: **preprocessing**, **model training & evaluation**, and **prediction**.

---

## Folder Structure

```
├── Data/
│   ├── test.tsv                 # Test dataset (TSV format with 'id' and 'context')
│   ├── unredactor.tsv           # Training and validation dataset (TSV format with 'split', 'name', 'context')
│
├── Features/
│   ├── features.pkl             # Extracted features for training/validation
│   ├── test_features.pkl        # Extracted features for test data
│
├── ModelSaved/
│   ├── model.pkl                # Trained Random Forest model and vectorizer
│
├── output/
│   ├── predictions.tsv          # Predictions generated for test data
│
├── PreProcessing.py             # Preprocessing script for feature extraction
├── model.py                     # Script for model training and evaluation
├── predict.py                   # Script for generating predictions
├── Pipfile                      # Pipenv environment file
├── Pipfile.lock                 # Pipenv lock file
```

---

## Explanation of the Code

### 1. **PreProcessing.py**
This script handles the preprocessing of raw datasets and feature extraction.  
**Key Functions**:
- **`load_data()`**: Loads TSV data into a Pandas DataFrame. Differentiates between training and test datasets.
- **`extract_features()`**: Extracts features such as:
  - **Previous and Next Words**: Words around redacted sections.
  - **Sentiment**: Sentiment score of the `context` using NLTK's VADER analyzer.
  - **Redacted Length**: Total length of redacted blocks (`████`).
- **`preprocess_train_val_data()`**: Extracts features for training and validation data and saves them as `features.pkl`.
- **`preprocess_test_data()`**: Extracts features for test data and saves them as `test_features.pkl`.

**Usage**:
```bash
pipenv run python PreProcessing.py --input Data/unredactor.tsv --output Features/ --data_type train
pipenv run python PreProcessing.py --input Data/test.tsv --output Features/ --data_type test
```

---

### 2. **model.py**
This script trains and evaluates a machine learning model using the preprocessed features.  
**Key Functions**:
- **`train_model()`**:
  - Loads preprocessed training features from `features.pkl`.
  - Trains a Random Forest model using `DictVectorizer` for feature encoding.
  - Saves the trained model and vectorizer as `model.pkl`.
- **`evaluate_model()`**:
  - Loads validation features and the saved model.
  - Evaluates the model using precision, recall, and F1-score metrics.

**Usage**:
- Train the model:
  ```bash
  pipenv run python model.py --train --features Features/features.pkl --model ModelSaved
  ```
- Evaluate the model:
  ```bash
  pipenv run python model.py --evaluate --features Features/features.pkl --model ModelSaved/model.pkl
  ```

---

### 3. **predict.py**
This script generates predictions on unseen test data using the trained model.  
**Key Functions**:
- **`predict_unredacted()`**:
  - Loads preprocessed test features from `test_features.pkl`.
  - Uses the trained model to predict redacted names.
  - Saves the predictions as a CSV file (`predictions.csv`) with columns `id` and `predicted_name`.

**Usage**:
```bash
pipenv run python predict.py --features Features/test_features.pkl --model ModelSaved/model.pkl --output output/predictions.tsv
```

---

## Pipeline Explanation
The pipeline is divided into the following steps:

1. **Data Preprocessing**:
   - Input: `Data/unredactor.tsv` and `Data/test.tsv`.
   - Output: `Features/features.pkl` (training/validation features) and `Features/test_features.pkl` (test features).
   - Features extracted include context sentiment, word positions, and redacted block length.

2. **Model Training and Evaluation**:
   - Input: `Features/features.pkl`.
   - Model: Random Forest Classifier trained on extracted features.
   - Output: `ModelSaved/model.pkl` (saved model and vectorizer).
   - Evaluation: Precision, Recall, and F1-Score metrics on validation data.

3. **Prediction**:
   - Input: `Features/test_features.pkl` and `ModelSaved/model.pkl`.
   - Output: `output/predictions.csv` with predicted names.

---

## How to Run

1. **Install Dependencies**:
   Ensure you have `pipenv` installed. Then, install dependencies using:
   ```bash
   pipenv install
   ```

2. **Preprocess Data**:
   - Preprocess training/validation data:
     ```bash
     pipenv run python PreProcessing.py --input Data/unredactor.tsv --output Features/ --data_type train
     ```
   - Preprocess test data:
     ```bash
     pipenv run python PreProcessing.py --input Data/test.tsv --output Features/ --data_type test
     ```

3. **Train the Model**:
   ```bash
   pipenv run python model.py --train --features Features/features.pkl --model ModelSaved
   ```

4. **Evaluate the Model**:
   ```bash
   pipenv run python model.py --evaluate --features Features/features.pkl --model ModelSaved/model.pkl
   ```

5. **Generate Predictions**:
   ```bash
   pipenv run python predict.py --features Features/test_features.pkl --model ModelSaved/model.pkl --output output/predictions.csv
   ```

---

## Additional Notes
- Ensure the datasets (`unredactor.tsv` and `test.tsv`) follow the required formats:
  - **Training/Validation**: `split`, `name`, `context`.
  - **Test**: `id`, `context`.
- Use the provided Pipfile to manage dependencies for reproducibility.
- Predictions will be saved in `output/predictions.csv`.


https://github.com/user-attachments/assets/7f8fb797-97f4-4fc7-bfcb-0ce1de3e7d4e




