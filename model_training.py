import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

# 1. Load the cleaned data
df = pd.read_csv('cleaned_tourism_data.csv')

# --- FEATURE ENGINEERING [cite: 63, 65] ---

# Calculate historical popularity (average rating) for each attraction type
# This helps the Regression model understand which types are generally liked more
type_means = df.groupby('AttractionType')['Rating'].mean().to_dict()
df['Avg_Type_Rating'] = df['AttractionType'].map(type_means)

# Calculate User Travel Frequency
# This helps the Classification model identify veteran vs. new travelers 
user_freq = df['UserId'].value_counts().to_dict()
df['User_Travel_Freq'] = df['UserId'].map(user_freq)

# 2. Encoding Categorical Variables [cite: 64]
encoders = {}
categorical_cols = ['Continent', 'Country', 'VisitMode', 'AttractionType', 'Region']

for col in categorical_cols:
    le = LabelEncoder()
    # Fill any NaNs with 'Unknown' before encoding to prevent errors [cite: 58]
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders for the Streamlit app
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# --- TASK 1: CLASSIFICATION (Predicting Visit Mode) [cite: 24, 76] ---
# Features: Demographics, Attraction Type, Seasonality, and User Behavior [cite: 31-34]
X_cls = df[['Continent', 'Country', 'Region', 'AttractionType', 'VisitMonth', 'User_Travel_Freq']]
y_cls = df['VisitMode']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Use Random Forest Classifier [cite: 77]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_c, y_train_c)

# Evaluation [cite: 82]
cls_pred = clf.predict(X_test_c)
print(f"--- Classification Results ---")
print(f"Accuracy Score: {accuracy_score(y_test_c, cls_pred):.2f}")

# --- TASK 2: REGRESSION (Predicting Ratings) [cite: 11, 74] ---
# Features: Contextual visit details and Attraction popularity [cite: 18-21, 75]
X_reg = df[['Continent', 'Country', 'VisitMode', 'AttractionType', 'VisitMonth', 'Avg_Type_Rating']]
y_reg = df['Rating']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Use Random Forest Regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)

# Evaluation 
reg_pred = reg.predict(X_test_r)
print(f"\n--- Regression Results ---")
print(f"R2 Score: {r2_score(y_test_r, reg_pred):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test_r, reg_pred):.2f}")

# 3. Save the Models for Deployment [cite: 85]
with open('visit_mode_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('rating_model.pkl', 'wb') as f:
    pickle.dump(reg, f)

# Save the engineering dictionaries so Streamlit can map them
with open('eng_features.pkl', 'wb') as f:
    pickle.dump({'type_means': type_means, 'user_freq': user_freq}, f)

print("\nPhase 3 Complete: Models, Encoders, and Features saved successfully!")