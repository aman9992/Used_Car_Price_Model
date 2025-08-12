import pandas as pd
import numpy as np
import pickle as pk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# 1. Load and Clean Data
print("Loading and cleaning data...")
df = pd.read_csv('CarDetails_105k.csv')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['mileage'] = df['mileage'].str.replace(' kmpl', '', regex=False).str.replace(' km/kg', '', regex=False).astype(float)
df['engine'] = df['engine'].str.replace(' CC', '', regex=False).astype(float)
df['max_power'] = df['max_power'].str.replace(' bhp', '', regex=False).astype(float)

# 2. Feature Engineering & Manual Label Encoding
print("Creating features and encoding data...")
df['brand'] = df['name'].str.split().str[0]
df['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
df['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
df['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
df['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)

# 3. Create and Save the Brand Mapping
unique_brands = df['brand'].unique()
brand_mapping = {}
for i, brand in enumerate(unique_brands):
    brand_mapping[brand] = i + 1
df['brand'].replace(brand_mapping, inplace=True)
pk.dump(brand_mapping, open('brand_mapping.pkl', 'wb'))
print("Saved 'brand_mapping.pkl'.")

# 4. Prepare Data for Modeling
X = df.drop(columns=['selling_price', 'name', 'torque'])
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# 5. Create, Fit, and Save the Preprocessor
preprocessor = StandardScaler()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
pk.dump(preprocessor, open('preprocessor.pkl', 'wb'))
print("Saved 'preprocessor.pkl'.")

# 6. Train and Save the Best Model
print("Training the best model (XGBoost)...")
xgb_model = XGBRegressor(random_state=7, n_jobs=-1, max_depth=6)
xgb_model.fit(X_train_scaled, y_train)
pk.dump(xgb_model, open('model.pkl', 'wb'))
print("Saved 'model.pkl'.")

print("\nAll necessary files have been created successfully!")