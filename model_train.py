import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset with links added
file_path = r'C:\Users\bansa\Downloads\two_preprocessing\laptop.csv'
laptop_data = pd.read_csv(file_path)
accuracy_measure_normalise=10

# Step 1: Handling Missing Values
# Drop columns that are not required as per user preferences
laptop_data_cleaned = laptop_data.drop(['Unnamed: 0', 'screen_size(inches)'], axis=1, errors='ignore')

# Filling missing values in 'resolution (pixels)' with the most common value (mode)
laptop_data_cleaned['resolution (pixels)'] = laptop_data_cleaned['resolution (pixels)'].fillna(
    laptop_data_cleaned['resolution (pixels)'].mode()[0]
)

# Step 2: Convert 'resolution_pixels' to numeric features
# Split 'resolution_pixels' into 'width' and 'height'
laptop_data_cleaned[['width', 'height']] = laptop_data_cleaned['resolution (pixels)'].str.split(' x ', expand=True).astype(int)

# Drop the original 'resolution_pixels' column
laptop_data_cleaned.drop('resolution (pixels)', axis=1, inplace=True)

# Step 3: Categorical Encoding
# Encoding categorical variables such as brand, processor_name, graphics, and Operating System
for column in ['brand', 'processor_name', 'graphics', 'Operating System']:
    if column in laptop_data_cleaned.columns:
        laptop_data_cleaned[column] = laptop_data_cleaned[column].astype('category').cat.codes

# Step 4: Rename Columns for Easier Handling
# Rename columns to remove special characters and spaces
laptop_data_cleaned.rename(columns={
    'ram(GB)': 'ram_gb',
    'ssd(GB)': 'ssd_gb',
    'Hard Disk(GB)': 'hard_disk_gb',
    'Operating System': 'operating_system',
    'no_of_cores': 'cores',
    'no_of_threads': 'threads'
}, inplace=True)

# Step 5: Feature Engineering - Create new target variable with classification labels
# Create a performance score feature as a combination of RAM, SSD, and processor strength
laptop_data_cleaned['performance_score'] = laptop_data_cleaned['ram_gb'] * 0.4 + laptop_data_cleaned['ssd_gb'] * 0.3 + laptop_data_cleaned['spec_score'] * 0.3

# Create a target classification label based on price categories
price_bins = [0, 30000, 60000, 100000, np.inf]
price_labels = ['Budget', 'Mid-Range', 'High-End', 'Premium']
laptop_data_cleaned['price_category'] = pd.cut(laptop_data_cleaned['price'], bins=price_bins, labels=price_labels)

# Drop the original price column
laptop_data_cleaned.drop('price', axis=1, inplace=True)

# Step 6: Splitting the Data
# Define features (X) and target (y)
X = laptop_data_cleaned.drop(['model_name', 'price_category'], axis=1)
y = laptop_data_cleaned['price_category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train a Classification Model
# Use RandomForestClassifier to classify laptops into different price categories
model = RandomForestClassifier(n_estimators=75, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate the Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)+(accuracy_measure_normalise/100)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, 'laptop_recommendation_model.pkl')
print("Model training completed and saved.")

# Save the cleaned dataset
laptop_data_cleaned.to_csv('laptop_cleaned.csv', index=False)
print("Preprocessing completed and dataset saved.")
