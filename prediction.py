from flask import Flask, request, jsonify
import pickle
import pandas as pd
app = Flask(__name__)

# Load the saved model
with open('classification_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json

        # Convert the data to a DataFrame
        input_data = pd.DataFrame.from_dict(data)

        # Make predictions
        predictions = loaded_pipeline.predict(input_data)

        # Prepare the response
        response = {'predictions': predictions.tolist()}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)