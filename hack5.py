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

# Load datasets
df1 = pd.read_csv('C:/Downloads/train.csv')
df2 = pd.read_csv('C:/Downloads/test.csv')
df = pd.concat([df1, df2], ignore_index=True)
df.drop(columns=['Serial No.'], errors='ignore', inplace=True)

# Handle missing data
def handle_missing_data(df):
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df

df = handle_missing_data(df)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')

# Define nutrient requirements for different demographics
nutrient_needs_by_demographic = {
    'Child': {'Protein_g': 20, 'Fat_g': 30, 'Energy_kcal': 1200},
    'Teen_Male': {'Protein_g': 60, 'Fat_g': 80, 'Energy_kcal': 2400},
    'Teen_Female': {'Protein_g': 50, 'Fat_g': 70, 'Energy_kcal': 2000},
    'Adult_Male': {'Protein_g': 56, 'Fat_g': 80, 'Energy_kcal': 2500},
    'Adult_Female': {'Protein_g': 46, 'Fat_g': 70, 'Energy_kcal': 2000},
    'Senior Citizen': {'Protein_g': 50, 'Fat_g': 70, 'Energy_kcal': 2000}
}

# Function to calculate nutrient gaps
def calculate_nutrient_gaps(user_profile, df):
    age_group = user_profile['age_group']
    required_nutrients = nutrient_needs_by_demographic.get(age_group, {})
    nutrient_columns = ['Energy_kcal', 'Protein_g', 'Fat_g', 'Carb_g']

    for nutrient in nutrient_columns:
        if nutrient in required_nutrients and nutrient in df.columns:
            gap_column = f"{nutrient}_gap"
            df[gap_column] = required_nutrients[nutrient] - df[nutrient]
            df[gap_column] = df[gap_column].apply(lambda x: max(x, 0))

    return df

# Function to create user profile
def create_user_profile():
    name = st.text_input("Enter your name")
    age_group = st.selectbox("Select your age group", ["Child", "Teenager", "Adult", "Senior Citizen"])
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    daily_calories = st.number_input("Enter your daily calorie intake (kcal)", min_value=0)
    dietary_preferences = {
        'non_vegetarian': st.checkbox("Non-vegetarian"),
        'vegetarian': st.checkbox("Vegetarian"),
        'gluten_free': st.checkbox("Gluten-free"),
        'vegan': st.checkbox("Vegan")
    }

    nutrient_intake = {
        'Protein_g': st.number_input("Protein intake (g)", min_value=0),
        'Fat_g': st.number_input("Fat intake (g)", min_value=0),
        'Carb_g': st.number_input("Carbohydrate intake (g)", min_value=0),
        'Energy_kcal': daily_calories
    }
    nutrient_priority = st.selectbox("Select Nutrient to Prioritize", ["None", "Protein_g", "Energy_kcal", "Fat_g", "Carb_g"])

    user_profile = {
        'name': name,
        'age_group': age_group.replace(" ", "_"),
        'gender': gender.lower(),
        'daily_calories': daily_calories,
        'dietary_preferences': dietary_preferences,
        'nutrient_intake': nutrient_intake,
        'nutrient_priority': nutrient_priority
    }
    return user_profile

# Streamlit app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "User Profile", "Dietary Preferences", "Recommendations"])

if page == "Home":
    st.title("Welcome to Your Personalized Nutritional Analysis!")

elif page == "User Profile":
    st.title("User Profile")
    user_profile = create_user_profile()

    if st.button("Save Profile"):
        st.session_state['user_profile'] = user_profile
        st.write("Profile saved successfully!")

elif page == "Recommendations":
    if 'user_profile' not in st.session_state:
        st.warning("Please complete the User Profile section first.")
    else:
        user_profile = st.session_state['user_profile']
        df = calculate_nutrient_gaps(user_profile, df.copy())

        # Model training and prediction
        features = ['Energy_kcal', 'Protein_g', 'Fat_g', 'Carb_g']
        target = ['Energy_kcal_gap', 'Protein_g_gap', 'Fat_g_gap', 'Carb_g_gap']

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Mean Absolute Error: {mae}")

        # Filter food based on dietary preferences
        def filter_food_by_dietary_preferences(dietary_preferences, food_df):
           # Initialize an empty list for food exclusions
            exclude_foods = []

            # If the user is Vegan, exclude any animal-based products
            if 'Vegan' in dietary_preferences:
                exclude_foods.append('Meat')
                exclude_foods.append('Dairy')
                exclude_foods.append('Eggs')

            # If the user is Vegetarian, exclude meat and fish
            if 'Vegetarian' in dietary_preferences:
                exclude_foods.append('Meat')

            # If the user is Gluten-Free, exclude gluten-based foods (you can customize this based on your dataset)
            if 'Gluten-Free' in dietary_preferences:
                exclude_foods.append('Wheat')

            # Filter the food dataframe to exclude any foods in the exclude_foods list
            filtered_df = food_df[~food_df['Food_Type'].isin(exclude_foods)]

            return filtered_df

        # Generate recommendations
        def generate_recommendations(user_profile, food_df, model, scaler, top_n=5):
            nutrient_gaps = [user_profile['nutrient_intake'][col] for col in features]
            nutrient_gaps_scaled = scaler.transform([nutrient_gaps])

            predicted_gaps = model.predict(nutrient_gaps_scaled)[0]
            food_df['gap_score'] = food_df.apply(lambda row: sum(abs(predicted_gaps[i] - row[features[i]]) for i in range(len(features))), axis=1)
            recommended_foods = food_df.sort_values(by='gap_score').head(top_n)

            return recommended_foods[['ID', 'FoodName', 'Protein_g', 'Fat_g', 'Energy_kcal']]

        recommendations = generate_recommendations(user_profile, filtered_df, model, scaler)
        st.write("Meal Recommendations:")
        st.dataframe(recommendations)