
import joblib
import numpy as np

# Load the trained model
model_path = 'xgboost_price_model_final.pkl'
model = joblib.load(model_path)

def predict_price(features):
    
    # Ensure input is in the correct format (2D array for the model)
    features_array = np.array(features).reshape(1, -1)
    price_prediction = model.predict(features_array)
    return price_prediction[0]

# Example usage
if __name__ == "__main__":
    # Example features: [brand, processor_name, ram_gb, ssd_gb, hard_disk_gb, operating_system,
    #                    graphics, cores, threads, spec_score, width, height, performance_score]
    sample_features = [14, 10, 16, 512, 0, 4, 1, 4, 8, 70, 1920, 1080, 130.4]
    predicted_price = predict_price(sample_features)
    print(f"Predicted Price: INR {predicted_price:.2f}")
