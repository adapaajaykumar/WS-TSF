pip install statsmodels
import streamlit as st
import pickle
import pandas as pd
import statsmodels

loaded_model = pickle.load(open("trained_model.sav", 'rb'))

def predict_sales_for_date(input_date):
    # Forecast sales for the input date
    sales_prediction = loaded_model.predict(pd.to_datetime(input_date))
    return sales_prediction

def main():
    st.title("Walmart Sales Forecasting App")
    
    input_date = st.date_input("Select a date", pd.to_datetime('today'))
    
    sales_prediction = ''
    
    if st.button("Predict Sales"):
        sales_prediction = abs(predict_sales_for_date(input_date))
        sales_prediction = round(sales_prediction * 100, 2)
            
    st.success("Predicted sales for {}:   {}".format(input_date, sales_prediction))
    
if __name__ == "__main__":
    main()
