from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd
from functions import predict_pH
from Water_Quality_pH import predict_pH_mlp2
from Water_Quality_Temperature import predict_temperature_for_date2
from Water_Quality_Turbidity import predict_turbidity_for_date2
import json

app = Flask(__name__)

# Load the combined components
with open("combined_components.pkl", "rb") as file:
    model_components = pickle.load(file)

model = model_components["model"]
scaler = model_components["scaler"]
data = pd.read_csv('Edited Chemical Dataset.csv')


class DiseasePredictor:
    def __init__(self, model, scaler, label_encoder, conditions):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.conditions = conditions

    def check_conditions(self, pH, temperature):
        for disease, condition in self.conditions.items():
            pH_range = condition['pH_range']
            temperature_range = condition['temperature_range']

            if pH_range[0] <= pH <= pH_range[1] and temperature_range[0] <= temperature <= temperature_range[1]:
                return disease
        return 'Unknown'

    def predict_disease(self, date_time, temperature, ph, turbidity):
        date_time = pd.to_datetime(date_time, format="%m/%d/%Y %H:%M")
        date_time = date_time.timestamp()
        n_steps = 10
        n_features = 4
        condition_based_prediction = self.check_conditions(ph, temperature)
        if condition_based_prediction != 'Unknown':
            return condition_based_prediction

        input_data = np.array([[date_time, temperature, ph, turbidity]])
        input_data_scaled = self.scaler.transform(input_data)
        input_data_padded = np.pad(input_data_scaled, ((0, n_steps - 1), (0, 0)), mode='constant', constant_values=0)

        # Check if the padding operation created the correct shape
        assert input_data_padded.shape == (n_steps, n_features)

        input_data_reshaped = input_data_padded.reshape(1, n_steps, n_features)

        predictions = self.model.predict(input_data_reshaped)
        predicted_diseases = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_diseases)

        return predicted_labels[0]


# Load the DiseasePredictor instance from the pickle file
with open('disease_predictor.pickle', 'rb') as file:
    disease_predictor = pickle.load(file)

# Load the Keras model separately
keras_model = load_model('keras_model.h5')

# Assign the Keras model to the 'model' attribute of the DiseasePredictor instance
disease_predictor.model = keras_model


@app.route('/')
def index():
    return jsonify({"Status": "Server running"})


@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        data = request.get_json()

        if not data:
            return jsonify(error="Invalid JSON data"), 400

        date_time = data['date_time']
        temperature = data['temperature']
        ph = data['ph']
        turbidity = data['turbidity']

        # Predict the disease using the loaded model
        predicted_disease = disease_predictor.predict_disease(date_time, temperature, ph, turbidity)

        return jsonify({'predicted_disease': predicted_disease}), 200

    except Exception as e:
        return jsonify(error=str(e)), 500


# Load the trained model
model_disease = pickle.load(open('predict_diseases.pkl', 'rb'))


@app.route('/predict/disease', methods=['POST'])
def predict():
    try:
        data_json_disease = request.get_json()

        if not data_json_disease:
            return jsonify(error="Invalid JSON data"), 400

        Temperature = float(data_json_disease.get('Temperature'))
        pH = float(data_json_disease.get('pH'))
        turbidity = float(data_json_disease.get('Turbidity (NTU)'))
        Hour = int(data_json_disease.get('Hour'))
        Minute = int(data_json_disease.get('Minute'))
        Day = int(data_json_disease.get('Day'))
        Month = int(data_json_disease.get('Month'))
        Year = int(data_json_disease.get('Year'))

        # Make a prediction using the loaded model
        prediction = model_disease.predict([[Temperature, pH, turbidity, Hour, Minute, Day, Month, Year]])

        # Convert the prediction to an integer and return it as a JSON response
        disease = prediction.tolist()
        return jsonify(disease=disease), 200

    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/predict/pH', methods=['POST'])
def predictPH():
    try:
        data_json = request.json

        if not data_json:
            return jsonify(error="Invalid JSON data"), 400

        date_time_str = data_json['date_time']
        predicted_pH = predict_pH(date_time_str, model, scaler, data, 10)  # Pass the data DataFrame
        return jsonify({'predicted_pH': predicted_pH}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Load the trained model
model_chemical = pickle.load(open('predict_chemical.pkl', 'rb'))


@app.route('/predict/chemical', methods=['POST'])
def predictChemical():
    try:
        data_json_chemical = request.get_json()

        if not data_json_chemical or not all(key in data_json_chemical for key in ['pH', 'Day', 'Month', 'Year']):
            return jsonify(error="Invalid JSON data"), 400

        pH = float(data_json_chemical['pH'])
        Day = int(data_json_chemical['Day'])
        Month = int(data_json_chemical['Month'])
        Year = int(data_json_chemical['Year'])

        # Make a prediction using the loaded model
        prediction = model_chemical.predict([[pH, Day, Month, Year]])

        # Assuming the prediction is a single value, convert it to a JSON response
        chemical_prediction = prediction[0]
        return jsonify(chemical_prediction=chemical_prediction), 200

    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/predict/future/pH', methods=['POST'])
def predictDPH():
    try:
        data_json_future_chemical = request.get_json()
        print(data_json_future_chemical)

        if not data_json_future_chemical or not all(key in data_json_future_chemical for key in ['Day']):
            return jsonify(error="Invalid JSON data"), 400

        Day = data_json_future_chemical['Day']

        # Make a prediction using the model associated function
        prediction = predict_pH_mlp2(Day)

        # Example float32 value
        float32_value = np.float32(prediction)

        # Convert float32 to Python float
        python_float_value = float(float32_value)

        # Serialize the Python float to JSON
        json_data_future_pH = json.dumps(python_float_value)

        return jsonify(json_data_future_pH), 200

    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/predict/future/temperature', methods=['POST'])
def predictDTemperature():
    try:
        data_json_future_temperature = request.get_json()
        print(data_json_future_temperature)

        if not data_json_future_temperature or not all(key in data_json_future_temperature for key in ['Day']):
            return jsonify(error="Invalid JSON data"), 400

        Day = data_json_future_temperature['Day']

        # Make a prediction using the model associated function
        prediction = predict_temperature_for_date2(Day)

        # Example float32 value
        float32_value = np.float32(prediction)

        # Convert float32 to Python float
        python_float_value = float(float32_value)

        # Serialize the Python float to JSON
        json_data_future_temp = json.dumps(python_float_value)

        return jsonify(json_data_future_temp), 200

    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/predict/future/turbidity', methods=['POST'])
def predictDTurbidity():
    try:
        data_json_future_turbidity = request.get_json()
        print(data_json_future_turbidity)

        if not data_json_future_turbidity or not all(key in data_json_future_turbidity for key in ['Day']):
            return jsonify(error="Invalid JSON data"), 400

        Day = data_json_future_turbidity['Day']

        # Make a prediction using the model associated function
        prediction = predict_turbidity_for_date2(Day)

        # Example float32 value
        float32_value = np.float32(prediction)

        # Convert float32 to Python float
        python_float_value = float(float32_value)

        # Serialize the Python float to JSON
        json_data_future_turbidity = json.dumps(python_float_value)

        return jsonify(json_data_future_turbidity), 200

    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.run()
