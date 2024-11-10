# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 06:28:29 2024

@author: gauri
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import streamlit as st

# Initialize session state if not already done
if 'user_profile' not in st.session_state:
    st.session_state['user_profile'] = None
    
# Load datasets
df1 = pd.read_csv('C:/Downloads/train.csv')
df2 = pd.read_csv('C:/Downloads/test.csv')
df = pd.concat([df1, df2], ignore_index=True)
df.drop(columns=['Serial No.'], errors='ignore', inplace=True)

# Sample nutrient requirements
nutrient_requirements = {
    'Child': {'Protein_g': 20, 'Fat_g': 30, 'Energy_kcal': 1200},
    'Teen_Male': {'Protein_g': 60, 'Fat_g': 80, 'Energy_kcal': 2400},
    'Teen_Female': {'Protein_g': 50, 'Fat_g': 70, 'Energy_kcal': 2000},
    'Adult_Male': {'Protein_g': 56, 'Fat_g': 80, 'Energy_kcal': 2500},
    'Adult_Female': {'Protein_g': 46, 'Fat_g': 70, 'Energy_kcal': 2000},
    'Senior_Citizen': {'Protein_g': 50, 'Fat_g': 70, 'Energy_kcal': 2000}
}

# Authentication state
if 'user_database' not in st.session_state:
    st.session_state['user_database'] = {}
if 'logged_in_user' not in st.session_state:
    st.session_state['logged_in_user'] = None

# Registration function
def register_user():
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Create Account"):
        if username in st.session_state['user_database']:
            st.warning("Username already exists!")
        else:
            st.session_state['user_database'][username] = {'password': password, 'profile': None}
            st.success("Account created successfully. Please log in.")

# Login function
def login_user():
    st.title("Log In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state['user_database'] and st.session_state['user_database'][username]['password'] == password:
            st.session_state['logged_in_user'] = username
            st.success("Logged in successfully!")
        else:
            st.warning("Invalid username or password.")

# Profile creation function
def create_user_profile():
    st.title("Create or Update Your Profile")
    age_group = st.selectbox("Select your age group", ["Child", "Teen Male", "Teen Female", "Adult Male", "Adult Female", "Senior Citizen"])
    daily_calories = st.number_input("Enter your daily calorie intake (kcal)", min_value=0)
    protein = st.number_input("Protein intake (g)", min_value=0)
    fat = st.number_input("Fat intake (g)", min_value=0)
    carb = st.number_input("Carbohydrate intake (g)", min_value=0)
    
    dietary_preferences = {
        'Non-vegetarian': st.checkbox("Non-vegetarian"),
        'Vegetarian': st.checkbox("Vegetarian"),
        'Gluten-free': st.checkbox("Gluten-free"),
        'Vegan': st.checkbox("Vegan")
    }
    
    profile = {
        'age_group': age_group,
        'daily_calories': daily_calories,
        'protein': protein,
        'fat': fat,
        'carb': carb,
        'dietary_preferences': dietary_preferences
    }
    
    if st.button("Save Profile"):
        st.session_state['user_database'][st.session_state['logged_in_user']]['profile'] = profile
        st.success("Profile saved successfully!")

# Calculate nutrient deficiencies
def calculate_nutrient_gaps(user_profile):
    age_group = user_profile['age_group'].replace(" ", "_")
    reqs = nutrient_requirements.get(age_group, {})
    gaps = {}
    for nutrient, req in reqs.items():
        intake = user_profile.get(nutrient.lower(), 0)
        gap = req - intake
        if gap > 0:
            gaps[nutrient] = gap
    return gaps

# Filter food based on dietary preferences
def filter_food_by_dietary_preferences(dietary_preferences, df):
    if 'Food_Type' not in df.columns:
        df['Food_Type'] = 'Generic'  # Add a placeholder if Food_Type doesn't exist

    exclude_foods = []
    if dietary_preferences.get('Vegan'):
        exclude_foods += ['Meat', 'Dairy', 'Eggs']
    if dietary_preferences.get('Vegetarian'):
        exclude_foods.append('Meat')
    if dietary_preferences.get('Gluten-free'):
        exclude_foods.append('Wheat')
    
    filtered_df = df[~df['Food_Type'].isin(exclude_foods)]
    return filtered_df

# Food recommendation function
def generate_recommendations(user_profile, food_df, model, scaler, top_n=5):
    # Calculate nutrient gaps for the user
    user_nutrient_gaps = {}
    for nutrient in ['Energy_kcal', 'Protein_g', 'Fat_g', 'Carbohydrate_g']:
        user_nutrient_gaps[f"{nutrient}_gap"] = nutrient_needs_by_demographic[user_profile['age_group']].get(nutrient, 0) - user_profile['current_intake'].get(nutrient, 0)
    
    # Scale the input for prediction
    input_data = [[user_profile['current_intake']['Energy_kcal'], 
                   user_profile['current_intake']['Protein_g'], 
                   user_profile['current_intake']['Fat_g'], 
                   user_profile['current_intake']['Carbohydrate_g']]]
    input_scaled = scaler.transform(input_data)
    
    # Predict the nutrient gaps using the model
    predicted_gaps = model.predict(input_scaled)[0]
    print(f"Predicted Gaps: {predicted_gaps}")

    # Calculate a gap score for each food based on predicted gaps
    food_df['gap_score'] = (
        food_df['Protein_g'] * abs(predicted_gaps[1]) * 2 +  # Double the weight for protein
        food_df['Carbohydrate_g'] * abs(predicted_gaps[3]) +
        food_df['Fat_g'] * abs(predicted_gaps[2])  # Adjust fat as necessary
    )

    # Filter out non-meal categories
    food_df_filtered = food_df[~food_df['FoodGroup'].str.contains('Beverages|Sweets|Powder', case=False)]

    # Sort by gap score (higher scores are prioritized) and select the top N recommendations
    recommended_foods = food_df_filtered.sort_values('gap_score', ascending=False).head(top_n)
    
    return recommended_foods[['ID', 'FoodName', 'FoodGroup', 'Protein_g', 'Carbohydrate_g', 'Energy_kcal', 'gap_score']]

# App interface and logic
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sign Up", "Log In", "User Profile", "Recommendations"])

# Define the default page
if st.session_state.get('default_page') is None:
    st.session_state['default_page'] = "Home"

if page == "Home":
    st.session_state['default_page'] = "Home"
    st.title("Welcome to Your Personalized Nutritional Analysis App")
    st.subheader("Please Sign Up or Log In to Get Started")

elif page == "Sign Up":
    st.session_state['default_page'] = "Sign Up"
    register_user()

elif page == "Log In":
    st.session_state['default_page'] = "Log In"
    login_user()

elif page == "User Profile":
    if 'logged_in_user' in st.session_state and st.session_state['logged_in_user']:
        profile = st.session_state['user_database'][st.session_state['logged_in_user']].get('profile')
        if profile:
            st.write("Your profile already exists.")
            st.write(profile)
        else:
            create_user_profile()
    else:
        st.warning("Please log in to create a profile.")

elif page == "Recommendations":
    if 'logged_in_user' not in st.session_state or not st.session_state['logged_in_user']:
        st.warning("Please log in and create a profile to see recommendations.")
    else:
        profile = st.session_state['user_database'][st.session_state['logged_in_user']].get('profile')
        if profile:
            gaps = calculate_nutrient_gaps(profile)
            st.write("Identified Nutrient Deficiencies:", gaps)
            recommendations = generate_recommendations(profile, df)
            st.write("Recommended Foods to Overcome Deficiencies:")
            st.write(recommendations)
        else:
            st.warning("Please complete your profile first.")
