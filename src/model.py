from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def create_model():
    """
    Creates a Random Forest Classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found.")
    return joblib.load(path)
