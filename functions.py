import numpy as np
import random

import pandas as pd


def predict_pH(date_time_str, model, scaler, data, sequence_length):
    input_timestamp = pd.Timestamp(date_time_str).timestamp()
    data["Timestamp"] = pd.to_datetime(data["Date and Time"]).astype(np.int64)
    # Prepare the input features
    input_sequence = []
    for i in range(sequence_length):
        target_timestamp = input_timestamp - (sequence_length - i) * 3600
        nearest_index = np.argmin(np.abs(data['Timestamp'] - target_timestamp))
        input_sequence.append(data.iloc[nearest_index, 1:4].values)

    input_sequence = np.array(input_sequence)
    input_sequence_scaled = scaler.transform(input_sequence)

    predicted_pH = model.predict(input_sequence_scaled.reshape(1, -1))

    # pH prediction adjustment
    adjustment = random.uniform(-0.2, 0.2)
    predicted_pH_adjusted = predicted_pH[0] + adjustment
    predicted_pH_adjusted = np.clip(predicted_pH_adjusted, 0, 1)  # Clip to [0, 1]

    # Scale the adjusted pH value back to the original range [0, 14]
    predicted_pH_scaled = predicted_pH_adjusted * (scaler.data_max_[2] - scaler.data_min_[2]) + scaler.data_min_[2]

    return predicted_pH_scaled
