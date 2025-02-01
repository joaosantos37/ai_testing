import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # or whatever model type you're using

class TurnoverPredictor:
    def __init__(self):
        # Use relative paths
        self.model = joblib.load('random_forest_turnover_top10.joblib')
        self.scaler = joblib.load('scaler.joblib')
        
        # Define the top 10 features (replace with your actual top 10 features)
        self.features = ['satisfaction_level', 'number_project', 
                        'time_spend_company', 'average_montly_hours',
                        'last_evaluation', 'work_accident', 'salary_low',
                        'salary_high', 'department_sales', 'department_technical']
        
    def predict_turnover(self, data):
        try:
            # Convert data to numpy array if it's not already
            data = np.array(data).reshape(1, -1)
            
            # Scale the data
            scaled_data = self.scaler.transform(data)
            
            # Make prediction
            probability = self.model.predict_proba(scaled_data)[0, 1]
            prediction = 1 if probability > 0.5 else 0
            
            return probability, prediction
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise

    def get_feature_importance(self):
        """
        Get feature importance from the model
        """
        return pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    