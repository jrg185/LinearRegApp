import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import statsmodels.api as sm

# Suppress openpyxl warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file, engine='openpyxl')

def perform_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Add constant term to the features
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    # Fit the model
    model = sm.OLS(y_train, X_train_sm).fit()
    
    # Make predictions
    y_pred = model.predict(X_test_sm)
    
    return model, X_train_sm, X_test_sm, y_train, y_test, y_pred

def calculate_confidence_interval(model, X_pred, confidence=0.95):
    # Ensure X_pred is 2D
    if len(X_pred.shape) == 1:
        X_pred = X_pred.reshape(1, -1)
    
    # Check if X_pred has the correct number of features (excluding the constant)
    if X_pred.shape[1] != len(model.params) - 1:
        raise ValueError(f"Expected {len(model.params) - 1} features, but got {X_pred.shape[1]}")
    
    # Add constant term to X_pred
    X_pred_with_const = sm.add_constant(X_pred, has_constant='add')
    
    prediction = model.get_prediction(X_pred_with_const)
    frame = prediction.summary_frame(alpha=1-confidence)
    return frame['mean'].values[0], frame['obs_ci_lower'].values[0], frame['obs_ci_upper'].values[0]

def main():
    st.title("Multiple Linear Regression App")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        # Column selection
        st.subheader("Select Columns for Regression")
        x_cols = st.multiselect("Select the input (X) columns", df.columns)
        y_col = st.selectbox("Select the target (Y) column", df.columns)

        # Run regression
        if st.button("Run Regression"):
            if len(x_cols) == 0:
                st.error("Please select at least one input column.")
            else:
                X = df[x_cols]
                y = df[y_col]

                model, X_train, X_test, y_train, y_test, y_pred = perform_regression(X, y)

                st.session_state['model'] = model
                st.session_state['x_cols'] = x_cols

                st.subheader("Regression Results")
                
                # Display summary statistics with 95% CI
                st.write("Model Summary:")
                summary = model.summary()
                st.text(str(summary))
                
                # Display coefficient statistics in a more readable format
                st.write("Coefficient Statistics:")
                conf_int = model.conf_int(alpha=0.05)
                coef_df = pd.DataFrame({
                    'Coefficient': model.params,
                    'Std Error': model.bse,
                    't-statistic': model.tvalues,
                    'p-value': model.pvalues,
                    '95% CI Lower': conf_int[0],
                    '95% CI Upper': conf_int[1]
                })
                st.write(coef_df)

                st.write(f"R-squared: {model.rsquared:.4f}")
                st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
                st.write(f"F-statistic: {model.fvalue:.4f}")
                st.write(f"Prob (F-statistic): {model.f_pvalue:.4f}")
                
                if len(x_cols) == 1:
                    fig, ax = plt.subplots()
                    ax.scatter(X_test[x_cols[0]], y_test, color='blue', label='Actual')
                    ax.plot(X_test[x_cols[0]], y_pred, color='red', label='Predicted')
                    ax.set_xlabel(x_cols[0])
                    ax.set_ylabel(y_col)
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("Scatter plot is only available for single-variable regression.")

        # Point Estimate and Confidence Interval
        if 'model' in st.session_state:
            st.subheader("Point Estimate and Confidence Interval")
            
            point_estimate_inputs = []
            for col in st.session_state['x_cols']:
                value = st.number_input(f"Enter value for {col}", key=f"input_{col}")
                point_estimate_inputs.append(value)

            if st.button("Calculate Estimate"):
                try:
                    X_pred = np.array(point_estimate_inputs)
                    
                    point_estimate, ci_lower, ci_upper = calculate_confidence_interval(
                        st.session_state['model'], 
                        X_pred
                    )

                    st.write(f"Point Estimate: {point_estimate:.4f}")
                    st.write(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
                except ValueError as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please make sure you've entered values for all input features used in the model.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.info("Please check your inputs and try again.")

if __name__ == "__main__":
    main()
