import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import shap
import matplotlib.pyplot as plt

# Load the dataset with links added
file_path = r'C:\Users\bansa\Downloads\two_preprocessing\laptop.csv'
laptop_data = pd.read_csv(file_path)

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
laptop_data_cleaned['performance_score'] = laptop_data_cleaned['ram_gb'] * 0.4 + laptop_data_cleaned['ssd_gb'] * 0.3 + laptop_data_cleaned['processor_name'] * 0.3

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

# Step 8: Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Step 9: Train the Best Model
best_model.fit(X_train, y_train)

# Step 10: Evaluate the Model
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy after Tuning: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, predictions))

# Step 11: Explainability using SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)

# Plot feature importance summary
shap.summary_plot(shap_values, X_train, feature_names=X.columns)

# Step 12: Function for Filtering Laptops based on User Input
def filter_laptops(user_input, laptop_data):
    filtered_data = laptop_data.copy()
    
    # Filter by budget
    if 'budget' in user_input:
        budget_max = user_input['budget']
        filtered_data = filtered_data[filtered_data['price_category'] <= budget_max]
    
    # Filter by intended use (e.g., gaming, business, etc.) - Simplified for performance score
    if 'intended_use' in user_input and user_input['intended_use'] == 'gaming':
        filtered_data = filtered_data[filtered_data['performance_score'] >= 70]
    
    # Filter by brand preference
    if 'brand' in user_input:
        filtered_data = filtered_data[filtered_data['brand'] == user_input['brand']]
    
    # Filter by operating system
    if 'operating_system' in user_input:
        filtered_data = filtered_data[filtered_data['operating_system'] == user_input['operating_system']]
    
    # Filter by minimum RAM
    if 'min_ram' in user_input:
        filtered_data = filtered_data[filtered_data['ram_gb'] >= user_input['min_ram']]
    
    # Filter by storage preference
    if 'storage' in user_input:
        filtered_data = filtered_data[filtered_data['ssd_gb'] >= user_input['storage']]
    
    # Filter by graphics card requirement
    if 'graphics_card' in user_input:
        filtered_data = filtered_data[filtered_data['graphics'] >= user_input['graphics_card']]
    
    return filtered_data

# Example user input
def example_user_input():
    user_input = {
        'budget': 'Mid-Range',
        'intended_use': 'gaming',
        'brand': 1,  # Example brand encoded as 1
        'operating_system': 1,  # Example OS encoded as 1
        'min_ram': 8,
        'storage': 256,
        'graphics_card': 1  # Example graphics card requirement
    }
    filtered_laptops = filter_laptops(user_input, laptop_data_cleaned)
    print(filtered_laptops[['model_name', 'performance_score', 'ram_gb', 'ssd_gb', 'price_category']])

# Run example user input
example_user_input()

# Save the trained model
joblib.dump(best_model, 'laptop_recommendation_model_tuned.pkl')
print("Model training completed and saved.")

# Save the cleaned dataset
laptop_data_cleaned.to_csv('laptop_cleaned.csv', index=False)
print("Preprocessing completed and dataset saved.")
