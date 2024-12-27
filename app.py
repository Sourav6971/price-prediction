from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import json

app = Flask(__name__)

# Enable CORS with specific configuration to support credentials
CORS(app, resources={r"/*": {
    "origins": "http://localhost:5173",  # React frontend origin
    "allow_headers": "*",               # Allow all headers
    "supports_credentials": True        # Allow credentials (cookies, auth headers)
}})

# Load the saved model and scaler
with open("banglore_home_prices.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pickle", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load columns from JSON file - now using the data_columns key
with open("columns.json", "r") as f:
    columns_dict = json.load(f)
    columns = columns_dict['data_columns']  # Get the list from data_columns key

print("Debug: Model and data loaded")
print("Number of columns:", len(columns))
print("First few columns:", columns[:5])

@app.route('/', methods=['POST'])
def predict_price():
    try:
        # Parse JSON request body
        data = request.get_json()
        print("Debug: Received data:", data)
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        # Extract and validate required fields
        try:
            total_sqft = float(data.get("total_sqft"))
            bath = int(data.get("bath"))
            bhk = int(data.get("bhk"))
            location = str(data.get("location"))
            
            print("Debug: Extracted values:", {
                "total_sqft": total_sqft,
                "bath": bath,
                "bhk": bhk,
                "location": location
            })
            
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid data types. Ensure total_sqft is float, bath and bhk are integers"}), 400

        # Create input vector with correct length
        x = np.zeros(len(columns))
        print("Debug: Initial x shape:", x.shape)
        
        # Assign values
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk

        print("Debug: After assigning basic values:", x[:5])

        # Handle location one-hot encoding
        location_index = columns.index(location.lower()) if location.lower() in columns else -1
        if location_index != -1:
            x[location_index] = 1
            print(f"Debug: Location {location} found at index {location_index}")
        else:
            print(f"Debug: Location {location} not found in columns")

        print("Debug: Final x shape before reshape:", x.shape)
        x = x.reshape(1, -1)
        print("Debug: Shape after reshape:", x.shape)

        # Scale and predict
        x_scaled = scaler.transform(x)
        predicted_price = float(model.predict(x_scaled)[0])
        print("Debug: Prediction successful:", predicted_price)

        return jsonify({
            "predicted_price": predicted_price,
            "currency": "INR",
            "input_data": {
                "total_sqft": total_sqft,
                "bath": bath,
                "bhk": bhk,
                "location": location
            }
        })

    except Exception as e:
        print(f"Debug: Final error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
