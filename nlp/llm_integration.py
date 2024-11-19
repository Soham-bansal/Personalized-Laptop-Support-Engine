
import requests
import json
import numpy as np
from keras.models import load_model

# Integrate with LLM API
def predict_and_send(user_input, model_path, combined_features_path, llm_url, llm_key):
    model = load_model(model_path)
    combined_features = np.load(combined_features_path)

    # Fake prediction example
    prediction = model.predict(combined_features[:1])
    structured_output = {
        "RAM": f"{int(prediction[0][0])}GB",
        "SSD": f"{int(prediction[0][1])}GB",
        "GPU": "Yes" if prediction[0][2] > 0.5 else "No",
        "Processor": "Intel i7"  # Example hardcoded
    }

    # Send structured output to LLM API
    headers = {"Authorization": f"Bearer {llm_key}", "Content-Type": "application/json"}
    payload = {"user_input": user_input, "predicted_specs": structured_output}
    response = requests.post(llm_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        print("LLM Response:", response.json())
    else:
        print("Error:", response.status_code, response.text)

if __name__ == '__main__':
    user_input = "I am a video editor working on 4K projects."
    model_path = '/mnt/data/neural_network_model.h5'
    combined_features_path = '/mnt/data/combined_features.npy'
    llm_url = 'https://llm.api.example.com/v1/recommendations'  # Replace with actual API URL
    llm_key = 'YOUR_API_KEY'  # Replace with your API key
    predict_and_send(user_input, model_path, combined_features_path, llm_url, llm_key)
