import pandas as pd
import numpy as np
import pickle
import streamlit as st
import gdown

def predict_model():
    try:
        gdown.download('https://drive.google.com/drive/folders/1mLn2qIP3AsWDoe2sF_bsmBWW8d7elOPP?usp=sharing', 'model.pkl', quiet=False)
        f = open('model.pkl','rb')
        model = pickle.load(f)
        return model
    except Exception:
        st.write("Error loading predictive model")
        st.stop()
        
model = predict_model()

        
        
def main():
  st.title("Disease Prediction System Using Support Vector Machine")
  answer = []
  features = ['skin_rash', 'continuous_sneezing', 'shivering', 'fatigue',
       'irregular_sugar_level', 'high_fever', 'sweating', 'yellowish_skin',
       'dark_urine', 'nausea', 'pain_behind_the_eyes', 'back_pain',
       'constipation', 'diarrhoea', 'mild_fever', 'malaise',
       'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose',
       'congestion', 'obesity', 'loss_of_smell', 'toxic_look_(typhos)',
       'muscle_pain', 'red_spots_over_body', 'belly_pain',
       'watering_from_eyes', 'increased_appetite', 'polyuria',
       'family_history', 'coma', 'stomach_bleeding', 'blood_in_sputum']

  for i in features:
    x = st.radio("Do you suffer from "+i+" ?",('Yes','No'))
    if x == 'Yes':
        answer.append(1)
    else:
        answer.append(0)

  test = np.array(answer)
  st.write('You are diagnosed with : ',model.predict(test.reshape(1,-1))[0])

        
       
if __name__ == "__main__":
  main()