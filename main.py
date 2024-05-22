import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

API_URL = 'http://127.0.0.1:8000/predict'

st.set_page_config(page_icon="🫐")

#st.image('Banner.jpg')

features = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','AverageOfLowerTRange',
            'AverageRainingDays', 'fruitset', 'fruitmass','seeds']

def main():
    
    st.sidebar.markdown("<h2>About the app</h2>", unsafe_allow_html=True)
    st.sidebar.write("""
            This app will predict the crop yield of the wild berries given various factors, 
            including plant spatial arrangement, bee species compositions, weather conditions of the wild blueberry field.
             """)
    
    with st.form('prediction_form'):
        
        st.subheader('Enter the following details:')
        
        clonesize = st.selectbox('The average blueberry clone size (in m2) in the field', 
                                options=[37.5, 25.0, 12.5, 20.0, 10.0, 40.0] )
        honeybee = st.selectbox('Honeybee density (in bees/m2/min) in the field', 
                                options=[0.75, 0.25, 0.5, 0.0, 6.64, 18.43, 0.537] )
        bumbles = st.selectbox('Bumblebee density (in bees/m2/min) in the field', 
                               options=[0.25, 0.38, 0.117, 0.202, 0.0, 0.065, 0.042, 0.585, 0.293, 0.058] )
        andrena = st.selectbox('Andrena bee density (in bees/m2/min) in the field', 
                               options=[0.25, 0.38, 0.5, 0.63, 0.75, 0.409, 0.707, 0.0, 0.229, 0.147, 0.585, 0.234] )
        osmia = st.selectbox('Osmia bee density (in bees/m2/min) in the field', 
                             options=[0.25, 0.38, 0.5, 0.63, 0.75, 0.058, 0.101, 0.0, 0.033, 0.021, 0.585, 0.117] )
        AverageOfLowerTRange = st.selectbox('The average of the lower band daily air temperature (℃)', 
                                            options=[50.8, 55.9, 45.8, 41.2, 45.3] )
        AverageRainingDays = st.selectbox('The average of raining days of the entire bloom season', 
                                          options=[0.26, 0.1, 0.39, 0.56, 0.06] )
        fruitset = st.slider('Average fruitset of the wild berry',0.19, 0.65)
        fruitmass = st.slider('Fruit mass (kg) of the wild berry',0.31, 0.54)  
        seeds = st.slider('Seeds of wild berry produced',22.0, 46.6) 
           
        submit = st.form_submit_button('Predict')

    if submit:
        data = {
            "clonesize" : clonesize,
            "honeybee" : honeybee, 
            "bumbles" : bumbles,
            "andrena" : andrena,
            "osmia" : osmia,
            "AverageOfLowerTRange" : AverageOfLowerTRange,
            "AverageRainingDays" : AverageRainingDays,
            "fruitset" : fruitset,
            "fruitmass" : fruitmass,
            "seeds" : seeds
        }
        
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            prediction = response.json()
            st.write(f'The predicted wild berry yield is:  {prediction["prediction"]} 🫐'
        )
        else:
            st.write(f'Error: {response.status_code} - {response.text}')

if __name__ == '__main__':
    main()