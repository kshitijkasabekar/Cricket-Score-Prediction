import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import load_model

model = load_model('model.h5')

df = pd.read_csv("ipl_data.csv") 

venue_options = df['venue'].unique()
venue_selection = st.selectbox('Select Venue:', venue_options)
batting_team_options = df['bat_team'].unique()
batting_team_selection = st.selectbox('Select Batting Team:', batting_team_options)
bowling_team_options = df['bowl_team'].unique()
bowling_team_selection = st.selectbox('Select Bowling Team:', bowling_team_options)
striker_options = df['batsman'].unique()
striker_selection = st.selectbox('Select Striker:', striker_options)
bowler_options = df['bowler'].unique()
bowler_selection = st.selectbox('Select Bowler:', bowler_options)
predict_button = st.button("Predict Score")
output = st.empty()

venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

venue_encoder.fit(venue_options)
batting_team_encoder.fit(batting_team_options)
bowling_team_encoder.fit(bowling_team_options)
striker_encoder.fit(striker_options)
bowler_encoder.fit(bowler_options)

def predict_score(venue, batting_team, bowling_team, striker, bowler):
    # Use the fitted encoders for transformation
    encoded_venue = venue_encoder.transform([venue])[0]
    encoded_batting_team = batting_team_encoder.transform([batting_team])[0]
    encoded_bowling_team = bowling_team_encoder.transform([bowling_team])[0]
    encoded_striker = striker_encoder.transform([striker])[0]
    encoded_bowler = bowler_encoder.transform([bowler])[0]

    input_data = np.array([encoded_venue, encoded_batting_team, encoded_bowling_team, encoded_striker, encoded_bowler])
    input_data = input_data.reshape(1, -1)

    # Make prediction using the loaded model
    predicted_score = model.predict(input_data)
    predicted_score = int(np.sum(predicted_score))  # Sum the predicted values

    return predicted_score

st.title('Cricket Score Prediction Dashboard')

if predict_button:
    # Predict and display the result
    predicted_score = predict_score(venue_selection, batting_team_selection, bowling_team_selection, striker_selection, bowler_selection)
    output.write(f"Predicted Score: {predicted_score}")
