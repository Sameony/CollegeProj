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
  features = ['itching', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
       'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
       'vomiting', 'burning_micturition', 'fatigue', 'weight_loss', 'lethargy',
       'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
       'dehydration', 'indigestion', 'headache', 'yellowish_skin',
       'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
       'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
       'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
       'swelled_lymph_nodes', 'malaise', 'phlegm', 'throat_irritation',
       'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
       'chest_pain', 'fast_heart_rate', 'dizziness', 'muscle_weakness',
       'swelling_joints', 'movement_stiffness', 'loss_of_balance',
       'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
       'continuous_feel_of_urine', 'depression', 'irritability', 'muscle_pain',
       'red_spots_over_body', 'belly_pain', 'dischromic _patches',
       'watering_from_eyes', 'increased_appetite', 'rusty_sputum',
       'lack_of_concentration', 'visual_disturbances',
       'receiving_blood_transfusion', 'blood_in_sputum', 'painful_walking',
       'pus_filled_pimples', 'blackheads', 'scurring']

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