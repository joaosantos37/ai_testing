import joblib
import pandas as pd
import numpy as np

class TurnoverPredictor:
    def __init__(self):
        # Load the model and scaler
        self.model = joblib.load('random_forest_turnover_top10.joblib')
        self.scaler = joblib.load('scaler.joblib')
        
        # Define the top 10 features (replace with your actual top 10 features)
        self.features = ['satisfaction_level', 'number_project', 
                        'time_spend_company', 'average_montly_hours',
                        'last_evaluation', 'work_accident', 'salary_low',
                        'salary_high', 'department_sales', 'department_technical']
        
    def predict_turnover(self, data):
        """
        Make turnover predictions using only top 10 features
        """
        # Ensure we only use top 10 features
        data = data[self.features]
        
        # Scale numerical features
        numerical_features = ['number_project', 'average_montly_hours', 'time_spend_company']
        scale_features = [f for f in numerical_features if f in self.features]
        if scale_features:
            data[scale_features] = self.scaler.transform(data[scale_features])
        
        # Make predictions
        probability = self.model.predict_proba(data)[:, 1]
        prediction = self.model.predict(data)
        
        return probability[0], prediction[0]

    def get_feature_importance(self):
        """
        Get feature importance from the model
        """
        return pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    