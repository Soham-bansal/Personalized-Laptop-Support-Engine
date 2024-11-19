# Training the XGBoost model for price prediction
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

# Load the dataset again to ensure consistency
dataset_path = 'laptop_cleaned.csv'
preprocessed_data = pd.read_csv(dataset_path)

# Select features based on user inputs and target variable
features = ['brand', 'processor_name', 'ram_gb', 'ssd_gb', 'hard_disk_gb', 
            'operating_system', 'graphics', 'cores', 'threads', 'spec_score', 
            'width', 'height', 'performance_score']
target = 'price'

X = preprocessed_data[features]
y = preprocessed_data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Save the model to a file
model_save_path = 'xgboost_price_model_final.pkl'
joblib.dump(xgb_model, model_save_path)

# Output performance metrics and saved model path
mse, rmse, model_save_path
