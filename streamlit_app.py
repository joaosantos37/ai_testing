import streamlit as st
import pandas as pd
import numpy as np
from app import TurnoverPredictor
import plotly.graph_objects as go

def create_gauge(probability):
    """Create a gauge chart for turnover probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Turnover Risk"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    return fig

def main():
    st.title('Employee Turnover Prediction')
    st.write('Enter employee information to predict turnover probability')
    
    # Initialize predictor
    predictor = TurnoverPredictor()
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            satisfaction_level = st.slider('Satisfaction Level', 0.0, 1.0, 0.5)
            number_project = st.number_input('Number of Projects', 2, 7, 4)
            average_montly_hours = st.number_input('Average Monthly Hours', 96, 310, 200)
            time_spend_company = st.number_input('Years at Company', 2, 10, 3)
            last_evaluation = st.slider('Last Evaluation Score', 0.0, 1.0, 0.5)
            
        with col2:
            work_accident = st.checkbox('Had Work Accident')
            salary = st.selectbox('Salary Level', ['low', 'medium', 'high'])
            department = st.selectbox('Department', ['sales', 'technical', 'other'])
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Prepare the data with only the top 10 features
            data = {
                'satisfaction_level': satisfaction_level,
                'number_project': number_project,
                'time_spend_company': time_spend_company,
                'average_montly_hours': average_montly_hours,
                'last_evaluation': last_evaluation,
                'work_accident': int(work_accident),
                'salary_low': 1 if salary == 'low' else 0,
                'salary_high': 1 if salary == 'high' else 0,
                'department_sales': 1 if department == 'sales' else 0,
                'department_technical': 1 if department == 'technical' else 0
            }
            
            # Create DataFrame
            input_data = pd.DataFrame([data])
            
            # Make prediction
            probability, prediction = predictor.predict_turnover(input_data)
            
            # Show results
            st.header('Prediction Results')
            
            # Display gauge chart
            fig = create_gauge(probability)
            st.plotly_chart(fig)
            
            if prediction == 1:
                st.error(f'⚠️ High risk of turnover! ({probability:.1%} probability)')
            else:
                st.success(f'✅ Low risk of turnover ({probability:.1%} probability)')
            
            # Show feature importance
            st.header('Feature Importance')
            feature_importance = predictor.get_feature_importance()
            if feature_importance is not None:
                fig = go.Figure(go.Bar(
                    x=feature_importance['importance'],
                    y=feature_importance['feature'],
                    orientation='h'
                ))
                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance',
                    yaxis_title='Feature',
                    height=400
                )
                st.plotly_chart(fig)
            
            # Show input summary
            st.header('Input Summary')
            st.dataframe(input_data)

if __name__ == '__main__':
    main()