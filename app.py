import joblib
import pandas as pd
import numpy as np

class TurnoverPredictor:
    def __init__(self):
        # Load the model and scaler
        self.model = joblib.load('random_forest_turnover_top10.joblib')
        self.scaler = joblib.load('scaler.joblib')
        
        # Define the top 10 features
        self.features = ['satisfaction_level', 'number_project', 
                        'time_spend_company', 'average_montly_hours',
                        'last_evaluation', 'work_accident', 'salary_low',
                        'salary_high', 'department_sales', 'department_technical']
        
        # Define which features need scaling
        self.numerical_features = ['number_project', 'average_montly_hours', 'time_spend_company']
        
    def predict_turnover(self, data):
        """
        Make turnover predictions using only top 10 features
        """
        try:
            # Print debug information
            print("Input data shape before selection:", data.shape)
            print("Available columns:", data.columns)
            
            # Ensure we only use top 10 features
            data = data[self.features].copy()
            
            # Scale only the numerical features
            if self.numerical_features:
                # Reshape for single sample prediction
                numerical_data = data[self.numerical_features].values
                if len(numerical_data.shape) == 1:
                    numerical_data = numerical_data.reshape(1, -1)
                
                # Apply scaling
                scaled_numerical = self.scaler.transform(numerical_data)
                
                # Update the scaled values in the dataframe
                for idx, feature in enumerate(self.numerical_features):
                    data[feature] = scaled_numerical[:, idx]
            
            # Make predictions
            probability = self.model.predict_proba(data)[:, 1]
            prediction = self.model.predict(data)
            
            return probability[0], prediction[0]
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'no shape'}")
            raise

    def get_feature_importance(self):
        """
        Get feature importance from the model
        """
        return pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    