import pandas as pd

# Load the cleaned dataset
cleaned_data_path = 'laptop_cleaned copy.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Drop the 'price' column if it exists
if 'price' in cleaned_data.columns:
    cleaned_data = cleaned_data.drop(columns=['price'])
    print("The 'price' column has been removed from the cleaned dataset.")
else:
    print("The 'price' column does not exist in the cleaned dataset.")

# Save the updated cleaned dataset
cleaned_data.to_csv(cleaned_data_path, index=False)
