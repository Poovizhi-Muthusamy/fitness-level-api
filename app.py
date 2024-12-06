from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model and label encoder
model = joblib.load('fitness_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_fitness_level():
    try:
        # Parse input JSON
        input_data = request.json
        features = [
            input_data['Age Group'],
            input_data['Physical Activity'],
            input_data['Activity Type'],
            input_data['Lifestyle'],
            input_data['Resting Heart Rate'],
            input_data['Fatigue'],
            input_data['Eating Habits'],
            input_data['Sitting Hours'],
            input_data['Sleep Quality'],
            input_data['Fitness Goal']
        ]

        # Convert input to NumPy array and reshape
        features = np.array(features).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(features)
        fitness_level = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'Fitness Level': fitness_level})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
