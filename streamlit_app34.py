import streamlit as st
import pickle
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

loaded_model = pickle.load(open("trained_model.sav", 'rb'))

def predict_sales_for_date(input_date):
    # Forecast sales for the input date
    sales_prediction = loaded_model.predict(pd.to_datetime(input_date))
    return sales_prediction

def main():
    st.title("Walmart Sales Forecasting App")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://img.freepik.com/premium-vector/calendar-with-checkmark-tick-approved-schedule-date-vector-stock-illustration_100456-6728.jpg?w=1060");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    input_date = st.date_input("Select a date", pd.to_datetime('today'))
    
    sales_prediction = ''
    
    if st.button("Predict Sales"):
        sales_prediction = abs(predict_sales_for_date(input_date))
        sales_prediction = round(sales_prediction * 100, 2)
        # Create a DataFrame to display the results in a table
        data = {'Date': [input_date], 'Forecasted Sales': [sales_prediction]}
        results_df = pd.DataFrame(data)
        
        # Display the results in a table
        st.table(results_df)

    
    
if __name__ == "__main__":
    main()
