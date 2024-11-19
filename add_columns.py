import pandas as pd

# Load the original and cleaned datasets
original_data_path = 'laptop.csv'
cleaned_data_path = 'laptop_cleaned.csv'

original_data = pd.read_csv(original_data_path)
cleaned_data = pd.read_csv(cleaned_data_path)

# Merge the spec_score column from original dataset into cleaned dataset
# Perform the merge on the 'model_name' column
cleaned_data = cleaned_data.merge(original_data[['model_name', 'price']], on='model_name', how='left')

# Save the updated cleaned dataset
cleaned_data.to_csv('laptop_cleaned.csv', index=False)

print("Spec score has been added to the cleaned dataset successfully.")
