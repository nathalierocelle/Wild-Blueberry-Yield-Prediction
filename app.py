import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from prediction import get_prediction, encode_value
from PIL import Image


model = joblib.load(r'final_model.joblib')

st.set_page_config(page_icon="🫐")

st.image('Banner.jpg')

features = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','AverageOfLowerTRange',
            'AverageRainingDays', 'fruitset', 'fruitmass','seeds']

def main():
    with st.form('prediction_form'):
        
        st.subheader("Enter the following details:")
    
        clonesize = st.slider("Days with below 20F temperature",10.0, 40.0)
        honeybee = st.slider("Days with below 10F temperature",0.0, 18.43)
        bumbles = st.slider("Days with above 100F temperature",0.0, 0.585)  
        andrena = st.slider("Days with below 20F temperature",0.0, 0.75)
        osmia = st.slider("Days with below 10F temperature",0.0, 0.75)
        AverageOfLowerTRange = st.slider("Days with above 100F temperature",41.2, 55.9)  
        AverageRainingDays = st.slider("Days with below 20F temperature",0.06, 0.56)
        fruitset = st.slider("Days with below 10F temperature",0.19, 0.65)
        fruitmass = st.slider("Days with above 100F temperature",0.31, 0.54)  
        seeds = st.slider("Days with above 100F temperature",22.0, 46.6) 
           
        submit = st.button("Predict")

    if submit:
        data = np.array([clonesize, honeybee,  ]).reshape(1,-1)
        #st.write(data)
        pred = get_prediction(data=data, model=model)
        st.write(f"The predicted wild berry yield is:  {pred[0]} 🫐")
           

if __name__ == '__main__':
    main()