import os
import pickle
import pytest
from model import train_model, evaluate_model
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def prepare_test_data():
    """Fixture to prepare sample features data for testing."""
    # Create mock data for testing
    training_data = {
        'features': [{'prev_word': 'test', 'next_word': 'data', 'sentiment': 0.0, 'redacted_length': 4}],
        'name': ['John Doe']
    }
    validation_data = {
        'features': [{'prev_word': 'test', 'next_word': 'data', 'sentiment': 0.0, 'redacted_length': 4}],
        'name': ['Jane Doe']
    }
    
    # Prepare mock pickle file with sample data
    mock_data = {
        'training_features': training_data,
        'validation_features': validation_data
    }
    features_path = "test_features.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(mock_data, f)
    
    return features_path

def test_train_model(prepare_test_data):
    """Test the training functionality of the model."""
    features_path = prepare_test_data
    model_path = "test_model.pkl"
    
    # Train the model
    train_model(features_path, model_path)
    
    # Check if model and vectorizer are saved correctly
    assert os.path.exists(model_path), "Model file was not saved"
    
    # Load the saved model and check if it contains the right components
    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)
        
    assert isinstance(model, RandomForestClassifier), "The model is not of type RandomForestClassifier"
    assert hasattr(vectorizer, 'transform'), "The vectorizer does not have 'transform' method"
    
    os.remove(model_path)  # Clean up after test

def test_evaluate_model(prepare_test_data):
    """Test the evaluation functionality of the model."""
    features_path = prepare_test_data
    model_path = "test_model.pkl"
    
    # Train the model first
    train_model(features_path, model_path)
    
    # Now, evaluate the model
    evaluate_model(features_path, model_path)
    
    # Here, we are not checking the exact output since it's a print statement,
    # but we ensure that the process completes without exceptions.
    assert os.path.exists(model_path), "Model file not found for evaluation"
    
    os.remove(model_path)  # Clean up after test

def test_invalid_features_file():
    """Test invalid features file path during training."""
    invalid_features_path = "invalid_features.pkl"
    model_path = "test_model.pkl"
    
    try:
        train_model(invalid_features_path, model_path)
    except FileNotFoundError:
        pass  # Expected error
    
    assert not os.path.exists(model_path), "Model should not be created with invalid features file"

def test_invalid_model_file():
    """Test invalid model file path during evaluation."""
    features_path = "test_features.pkl"
    invalid_model_path = "invalid_model.pkl"
    
    try:
        evaluate_model(features_path, invalid_model_path)
    except FileNotFoundError:
        pass  # Expected error

def test_model_without_train():
    """Test evaluate functionality without training the model."""
    features_path = "test_features.pkl"
    model_path = "test_model.pkl"
    
    # Directly attempt evaluation without training first
    try:
        evaluate_model(features_path, model_path)
    except FileNotFoundError:
        pass  # Expected error

if __name__ == "__main__":
    pytest.main()
