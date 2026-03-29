# Project purpose
This repo is a reference implementation of a polyphonic multi-F0 estimation model (model3 — 2-voice/duet).
It is used to benchmark model3's performance against a separate model in another repo.
The pipeline covers: audio mixing → feature extraction → training → inference → evaluation.

# Scope constraints — DO NOT change
- **Model architecture**: do not modify `models.py` (layer structure, filter sizes, activations)
- **Training hyperparameters**: do not modify learning rate, batch size, loss function, or metrics in `experiments/2_training.py` or `utils_train.py`
- **Feature extraction parameters**: do not modify HCQT parameters in `utils.py` (`get_hcqt_params()`) — these must stay consistent with the model's training assumptions
- **Model contract**: inputs, outputs, and preprocessing are fixed; see `model_contract.md`

# Environment
- **Required env variable**: `export DAST_BASE=/path/to/dAST` — all data paths derive from this
- **Conda env**: use `environment.yml` (Linux + CUDA 12.6); activate with `conda activate multif0_env`
- **System dependency**: `sox` must be installed (`apt-get install sox` or `conda install -c conda-forge sox`)
- The `ir/IR_greathall.wav` impulse response file must be present in `ir/` (not tracked in git; copy manually to server)


# Pipeline execution order
1. `python experiments/data_augmentation.py` — pitch-shift augmentation (one-time, if not done)
2. `python experiments/run_pipeline.py` — batch audio mixing + feature extraction (section by section to save disk)
3. `python experiments/2_training.py` — train model3
4. `python inference_script.py` — run inference on evaluation pairs
5. `python run_evaluation.py` — compute mir_eval metrics across all eval datasets

# Data paths
- Do NOT write into any `raw/` folders. All processed data must go into `processed/`
- Do not write unnecessary large files. Minimise storage utilisation.
- Audio mixtures: prefer FLAC over WAV (use `0_setup_duet.py`; delete mixes after feature extraction via `run_pipeline.py`)

## Execution
- **Never run scripts or commands yourself.** Always provide the command for the user to run manually in their remote SSH terminal.
- After providing a command to run, wait for the user to report back before proceeding.
 
## Logging
- Always wrap run commands with `tee` to save output to a timestamped log file:
  ```bash
  mkdir -p ~/logs
  python script.py 2>&1 | tee ~/logs/<script_name>_$(date +%Y%m%d_%H%M%S).log
  ```
- Always capture both stdout and stderr (`2>&1`).
 
## Debugging
- When debugging, always read the latest log file in `~/logs/` first before asking the user for error output.
- The logs directory is accessible locally at `~/remote-mount/logs/`.
