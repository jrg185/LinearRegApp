import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats

def perform_regression(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred

def calculate_confidence_interval(model, X, y, X_pred, confidence=0.95):
    n = len(X)
    p = X.shape[1]
    dof = n - p - 1
    
    # Calculate MSE
    y_pred = model.predict(X)
    mse = np.sum((y - y_pred)**2) / dof
    
    # Calculate standard error
    X_mean_centered = X - np.mean(X, axis=0)
    cov_matrix = np.linalg.inv(X_mean_centered.T.dot(X_mean_centered))
    std_error = np.sqrt(mse * np.sum(X_pred.dot(cov_matrix) * X_pred, axis=1))
    
    # Calculate t-value
    t_value = stats.t.ppf((1 + confidence) / 2, dof)
    
    # Calculate prediction and confidence interval
    y_pred = model.predict(X_pred)
    ci_lower = y_pred - t_value * std_error
    ci_upper = y_pred + t_value * std_error
    
    return y_pred[0], ci_lower[0], ci_upper[0]

def main():
    st.title("Multiple Linear Regression App")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        st.write("Data Preview:")
        st.write(df.head())

        # Column selection
        st.subheader("Select Columns for Regression")
        x_cols = st.multiselect("Select the input (X) columns", df.columns)
        y_col = st.selectbox("Select the target (Y) column", df.columns)

        if st.button("Run Regression"):
            if len(x_cols) == 0:
                st.error("Please select at least one input column.")
                return

            # Prepare data
            X = df[x_cols].values
            y = df[y_col].values

            model, X_train, X_test, y_train, y_test, y_pred = perform_regression(X, y)

            # Display results
            st.subheader("Regression Results")
            for i, col in enumerate(x_cols):
                st.write(f"Coefficient for {col}: {model.coef_[i]:.4f}")
            st.write(f"Intercept: {model.intercept_:.4f}")
            st.write(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

            # Point estimate and confidence interval
            st.subheader("Point Estimate and Confidence Interval")
            point_estimate_inputs = []
            for col in x_cols:
                value = st.number_input(f"Enter value for {col}")
                point_estimate_inputs.append(value)

            X_pred = np.array(point_estimate_inputs).reshape(1, -1)
            point_estimate, ci_lower, ci_upper = calculate_confidence_interval(model, X_train, y_train, X_pred)

            st.write(f"Point Estimate: {point_estimate:.4f}")
            st.write(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

            # Plot results (for single input variable only)
            if len(x_cols) == 1:
                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='blue', label='Actual')
                ax.plot(X_test, y_pred, color='red', label='Predicted')
                ax.set_xlabel(x_cols[0])
                ax.set_ylabel(y_col)
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Scatter plot is only available for single-variable regression.")

if __name__ == "__main__":
    main()
