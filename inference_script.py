import os
import pandas as pd
import numpy as np
import traceback
from models import build_model3
from load_weights import load_weights
from utils import create_pump_object, compute_pump_features
from utils_train import pitch_activations_to_mf0

# Paths
_DAST_BASE = os.environ['DAST_BASE']
pairs_csv_path = f'{_DAST_BASE}/data/processed/evaluation/pairs.csv'
predictions_dir = "predictions"
errors_log_path = "inference_errors.log"
weights_path = "models/exp3multif0.pkl"

# Ensure predictions directory exists
os.makedirs(predictions_dir, exist_ok=True)

# Load the model
model = build_model3()
load_weights(model, weights_path)
model.compile()  # Compile the model

# Read the pairs CSV
df = pd.read_csv(pairs_csv_path)

# Initialize error log
with open(errors_log_path, "w") as error_log:
    error_log.write("Inference Errors\n")

# Iterate over rows
for index, row in df.iterrows():
    try:
        piece_id = row["piece_id"]
        alto_id = row["alto_singer_id"]
        tenor_id = row["tenor_singer_id"]
        mix_audio_path = f'{_DAST_BASE}/{row["mix_audio_path"]}'

        # Check if audio file exists
        if not os.path.exists(mix_audio_path):
            raise FileNotFoundError(f"Audio file not found: {mix_audio_path}")

        # Preprocess the audio
        pump = create_pump_object()
        features = compute_pump_features(pump, mix_audio_path)

        # Adjust transpose operations to match model's expected input shape
        input_hcqt = features['dphase/mag'][0].transpose(1, 0, 2)[np.newaxis, :, :, :]
        input_dphase = features['dphase/dphase'][0].transpose(1, 0, 2)[np.newaxis, :, :, :]

        # Log corrected input shapes for debugging
        print(f"Corrected Input HCQT shape: {input_hcqt.shape}")
        print(f"Corrected Input DPhase shape: {input_dphase.shape}")

        # Run inference
        predictions = model.predict([input_hcqt, input_dphase])[0]

        # Extract discrete multi-F0 estimates
        est_times, est_freqs = pitch_activations_to_mf0(predictions, thresh=0.5)

        # Format predictions
        pred_data = []
        for t, freqs in zip(est_times, est_freqs):
            row = [t] + (freqs.tolist() + [0] * (2 - len(freqs)))[:2]  # Ensure 2 columns
            pred_data.append(row)

        pred_df = pd.DataFrame(pred_data, columns=["time", "f0_1", "f0_2"])

        # Save predictions
        pred_filename = f"{piece_id}_{alto_id}_{tenor_id}_pred.csv"
        pred_path = os.path.join(predictions_dir, pred_filename)
        pred_df.to_csv(pred_path, index=False)

        # Update the CSV
        df.at[index, "pred_path"] = pred_path

    except Exception as e:
        # Log errors
        with open(errors_log_path, "a") as error_log:
            error_log.write(f"Error processing row {index}: {str(e)}\n")
            error_log.write(traceback.format_exc() + "\n")

# Save the updated CSV
df.to_csv(pairs_csv_path, index=False)

print("Inference completed.")