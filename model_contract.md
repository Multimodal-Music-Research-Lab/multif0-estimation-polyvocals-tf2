# Model Contract for `model3` (Late/Deep Model)

## 1. Expected Input Shape and Format
- **Input 1 (HCQT):**
  - Shape: `(None, 360, 5)`
  - Description: Harmonic Constant-Q Transform (HCQT) representation of the audio signal.
- **Input 2 (Phase Differentials):**
  - Shape: `(None, 360, 5)`
  - Description: Phase differentials computed from the audio signal.

## 2. Preprocessing Steps
- **HCQT and Phase Differentials Computation:**
  - The HCQT and phase differentials are computed using the `utils.create_pump_object()` and `utils.compute_pump_features()` functions.
  - The HCQT is accessed via `features['dphase/mag'][0]` and phase differentials via `features['dphase/dphase'][0]`.
- **Reshaping:**
  - Both inputs are transposed to the shape `(1, time, frequency, channels)` using:
    ```python
    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]
    input_dphase = input_dphase.transpose(1, 2, 0)[np.newaxis, :, :, :]
    ```

## 3. Output Shape and Format
- **Output:**
  - Shape: `(time_frames, frequency_bins)`
  - Description: A salience matrix where each value represents the likelihood of a specific frequency being active at a given time.
- **Time Resolution:**
  - Hop size: `512` samples
  - Sampling rate: `22050 Hz`
  - Time resolution: $\text{hop size} / \text{sampling rate} = 512 / 22050 \approx 0.0232$ seconds per frame.
- **Frequency Bins:**
  - Fixed at `360` bins, spanning the range of the HCQT.

## 4. Extracting Discrete F0 Values
- **Method:**
  - The salience matrix is processed using `utils_train.pitch_activations_to_mf0()`.
  - This function applies peak-picking and thresholding to extract discrete F0 values.
- **Thresholding:**
  - A threshold value is applied to the salience matrix to filter out low-likelihood values.
- **Peak-Picking:**
  - Peaks in the salience matrix are identified to determine the most likely F0 values at each time frame.

---

This document outlines the inference process for the `model3` (Late/Deep model) and provides details on the input, preprocessing, output, and postprocessing steps.