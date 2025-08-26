# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import pickle as pk
import datetime

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score

# Model
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# --- 1. Load and Clean Data ---
print("Step 1: Loading and Cleaning Data...")
try:
    df = pd.read_csv('CarDetails_105k.csv')
except FileNotFoundError:
    print("Error: 'CarDetails_105k.csv' not found.")
    exit()

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['mileage'] = df['mileage'].str.replace(' kmpl', '', regex=False).str.replace(' km/kg', '', regex=False).astype(float)
df['engine'] = df['engine'].str.replace(' CC', '', regex=False).astype(float)
df['max_power'] = df['max_power'].str.replace(' bhp', '', regex=False).astype(float)
print("Data Cleaning Complete.\n")

# --- 2. Feature Engineering ---
print("Step 2: Performing Feature Engineering...")
# Create 'age' feature
current_year = datetime.datetime.now().year
df['age'] = current_year - df['year']
# Create 'brand' feature
df['brand'] = df['name'].str.split().str[0]
print("Created 'age' and 'brand' features.\n")

# --- 3. Define Features, Preprocess, and Split Data ---
print("Step 3: Preparing Data for Modeling...")
# Define Features (X) and Target (y)
# We drop the original 'name', 'year', and 'torque' columns
X = df.drop(columns=['selling_price', 'name', 'year', 'torque'])
y = df['selling_price']

# Identify numerical and categorical features for our pipeline
numerical_features = ['km_driven', 'mileage', 'engine', 'max_power', 'seats', 'age']
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

# Create a professional preprocessing pipeline with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
print("Data Preparation Complete.\n")

# --- 4. Create and Train the Final Model Pipeline ---
print("Step 4: Training the Final XGBoost Model...")
# We bundle the preprocessor and the model into a single pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=7, n_jobs=-1, max_depth=8, n_estimators=200, learning_rate=0.1, subsample=0.9))
])

# Train the entire pipeline
final_pipeline.fit(X_train, y_train)

# --- 5. Evaluate the Final Model ---
print("Step 5: Evaluating the Final Model...")
score = r2_score(y_test, final_pipeline.predict(X_test))
print("Final Tuned Model R-squared Score: {:.4f}".format(score))

# --- 6. Save the Final Pipeline ---
print("Step 6: Saving the entire pipeline...")
pk.dump(final_pipeline, open('pipeline.pkl', 'wb'))
print("\nPipeline saved successfully to 'pipeline.pkl'.")