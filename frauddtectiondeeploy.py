#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 14:02:38 2023

@author: yassine
"""

import numpy as np 
import pickle 
import streamlit as st 

load_model = pickle.load(open("/home/yassine/trained_model.sav" , "rb"))

#creating a function for prediction 

def  fraud_detection(input_data) : 
    input_data_as_numpy_array  = np.asarray(input_data)
    input_data_reshape = input_data_as_numpy_array.reshape((1,-1))
    
    return load_model.predict(input_data_reshape)

def main():
    #giving a title to our user interface 
    st.title("fraud detection using machine learning web app ")
    #getting the input data 
    #"ttype", "amount", "oldbalanceOrg", "newbalanceOrig"
    # Create a mapping of categorical values to numerical values
    mapping = {"CASH_OUT": 1, "PAYMENT": 2,"CASH_IN": 3, "TRANSFER": 4,"DEBIT": 5}
    
    # Use the keys of the mapping as the options for the drop-down menu
    options = list(mapping.keys())
    
    # Create a radio button for the user to select a categorical value
    selected_category = st.radio("Select a category:", options)
    
    # Use the selected value to look up the corresponding numerical value
    ttype = mapping[selected_category]
    amount =  st.text_input("the amount transfer ")    
    oldbalanceOrg = st.text_input("the old balence before the transfer")
    newbalanceOrig = st.text_input("the new balance after the transfer ")
    #code for the prediction 
    pp = ""
    #creating a button for prediction 
    if st.button("prediction about the fraud detection") : 
        pp  = fraud_detection([ttype, amount, oldbalanceOrg, newbalanceOrig ])
    st.success(pp)
    

if __name__ == "__main__":
    main()
        