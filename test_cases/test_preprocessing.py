import os
import pickle
import pandas as pd
from PreProcessing import load_data, extract_features, preprocess_train_val_data, preprocess_test_data

def test_load_data():
    """Test loading training and test data."""
    # Create a mock training file
    train_data = "split\tname\tcontext\ntraining\tJohn Doe\tThis is a test ████ data.\nvalidation\tJane Doe\tAnother █████ test."
    test_file = "test_train_data.tsv"
    with open(test_file, "w") as f:
        f.write(train_data)
    
    # Test loading training data
    df = load_data(test_file, is_test=False)
    assert not df.empty, "Training data should not be empty"
    assert list(df.columns) == ["split", "name", "context"], "Columns do not match for training data"
    
    os.remove(test_file)

    # Create a mock test file
    test_data = "id\tcontext\n1\tThis is a test ████ data.\n2\tAnother █████ test."
    test_file = "test_test_data.tsv"
    with open(test_file, "w") as f:
        f.write(test_data)
    
    # Test loading test data
    df = load_data(test_file, is_test=True)
    assert not df.empty, "Test data should not be empty"
    assert list(df.columns) == ["id", "context"], "Columns do not match for test data"
    
    os.remove(test_file)

def test_extract_features():
    """Test feature extraction."""
    row = {"context": "This is a test ████ data."}
    features = extract_features(row, is_test=True)
    
    assert "prev_word" in features, "Feature extraction failed: Missing 'prev_word'"
    assert "next_word" in features, "Feature extraction failed: Missing 'next_word'"
    assert "sentiment" in features, "Feature extraction failed: Missing 'sentiment'"
    assert "redacted_length" in features, "Feature extraction failed: Missing 'redacted_length'"

def test_preprocess_train_val_data():
    """Test preprocessing training and validation data."""
    train_data = "split\tname\tcontext\ntraining\tJohn Doe\tThis is a test ████ data.\nvalidation\tJane Doe\tAnother █████ test."
    input_file = "train_val_data.tsv"
    output_file = "train_val_features.pkl"
    
    with open(input_file, "w") as f:
        f.write(train_data)
    
    preprocess_train_val_data(input_file, output_file)
    assert os.path.exists(output_file), "Output file not created for training/validation data"
    
    with open(output_file, "rb") as f:
        data = pickle.load(f)
        assert "training_features" in data, "Missing 'training_features' in output"
        assert "validation_features" in data, "Missing 'validation_features' in output"
    
    os.remove(input_file)
    os.remove(output_file)

def test_preprocess_test_data():
    """Test preprocessing test data."""
    test_data = "id\tcontext\n1\tThis is a test ████ data.\n2\tAnother █████ test."
    input_file = "test_data.tsv"
    output_file = "test_features.pkl"
    
    with open(input_file, "w") as f:
        f.write(test_data)
    
    preprocess_test_data(input_file, output_file)
    assert os.path.exists(output_file), "Output file not created for test data"
    
    with open(output_file, "rb") as f:
        data = pickle.load(f)
        assert "id" in data.columns, "Missing 'id' column in test output"
        assert "features" in data.columns, "Missing 'features' column in test output"
    
    os.remove(input_file)
    os.remove(output_file)

if __name__ == "__main__":
    print("Running tests...")
    test_load_data()
    print("test_load_data: PASSED")
    
    test_extract_features()
    print("test_extract_features: PASSED")
    
    test_preprocess_train_val_data()
    print("test_preprocess_train_val_data: PASSED")
    
    test_preprocess_test_data()
    print("test_preprocess_test_data: PASSED")
