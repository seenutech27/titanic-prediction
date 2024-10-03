import joblib
import pandas as pd

class TitanicModel:
    def __init__(self, model_path, train_data_path=None):  # Added default value
        self.model = joblib.load(model_path)
        if train_data_path:
            self.train_data = pd.read_csv(train_data_path)  # Load the training data if provided

    def predict(self, features):
        # Ensure the features are in the correct format
        return self.model.predict(features)
