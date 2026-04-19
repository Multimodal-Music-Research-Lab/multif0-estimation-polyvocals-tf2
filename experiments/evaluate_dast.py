"""
evaluate_dast.py

Mirrors the run_evaluation() flow from experiments/2_training.py, adapted for
the dAST pipeline where pair annotations are stored as _output.npy salience maps
rather than annotation CSV files in a cluster-specific directory.

Steps (matching 2_training.py):
  1. utils_train.get_model_metrics   — Keras evaluate on train/val/test generators
  2. get_best_thresh_npy             — threshold sweep on validation set (npy-based)
  3. score_on_split_npy              — per-pair mir_eval scoring on requested splits

Usage (run from repo root with PYTHONPATH=.):
    PYTHONPATH=. uv run python experiments/evaluate_dast.py \\
        [--model-path /path/to/model.keras] \\
        [--save-key dast_model3_v1] \\
        [--data-splits-file data_splits.json] \\
        [--splits train,validate,test]
"""

import argparse
import json
import os

import mir_eval
import numpy as np
import pandas as pd

from experiments import config
import utils
import utils_train
from models import build_model3


# ---------------------------------------------------------------------------
# Keras 3 compatibility shim for get_model_metrics
# evaluate_generator() was removed in Keras 3; use evaluate() with steps arg.
# ---------------------------------------------------------------------------

def _get_model_metrics(data_object, model, model_scores_path):
    """Keras-3-compatible replacement for utils_train.get_model_metrics."""
    def _wrap(gen):
        """Wrap pescador generator to yield numpy arrays (required by Keras 3)."""
        for x, y in gen:
            yield (np.asarray(x[0]), np.asarray(x[1])), np.asarray(y)

    train_eval = model.evaluate(
        _wrap(data_object.get_train_generator()), steps=1000, verbose=0
    )
    valid_eval = model.evaluate(
        _wrap(data_object.get_validation_generator()), steps=1000, verbose=0
    )
    test_eval = model.evaluate(
        _wrap(data_object.get_test_generator()), steps=1000, verbose=0
    )

    df = pd.DataFrame(
        [train_eval, valid_eval, test_eval],
        index=['train', 'validation', 'test'],
    )
    print(df)
    df.to_csv(model_scores_path)


# ---------------------------------------------------------------------------
# Data class — verbatim from train_dast.py
# ---------------------------------------------------------------------------

class Data(object):
    """Loads data splits and provides train/validation/test generators."""

    def __init__(self, data_splits_path, data_path, input_patch_size,
                 batch_size, active_str, muxrate):

        self.data_splits_path = data_splits_path
        self.input_patch_size = input_patch_size
        self.data_path = data_path

        (self.train_set,
         self.validation_set,
         self.test_set) = self.load_data_splits()

        self.train_files = utils_train.get_file_paths(
            self.train_set, self.data_path)
        self.validation_files = utils_train.get_file_paths(
            self.validation_set, self.data_path)
        self.test_files = utils_train.get_file_paths(
            self.test_set, self.data_path)

        self.batch_size = batch_size
        self.active_str = active_str
        self.muxrate = muxrate

    def load_data_splits(self):
        with open(self.data_splits_path, 'r') as fh:
            data_splits = json.load(fh)
        return (
            data_splits['train'],
            data_splits['validate'],
            data_splits['test'],
        )

    def get_train_generator(self):
        return utils_train.keras_generator(
            self.train_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate,
        )

    def get_validation_generator(self):
        return utils_train.keras_generator(
            self.validation_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate,
        )

    def get_test_generator(self):
        return utils_train.keras_generator(
            self.test_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate,
        )


# ---------------------------------------------------------------------------
# npy-based reference helpers (replace cluster-path annotation CSV loading)
# ---------------------------------------------------------------------------

def _input_to_output_path(input_path):
    """Derive the _output.npy path from the corresponding _input.npy path."""
    return (input_path
            .replace('inputs/', 'outputs/')
            .replace('_input.npy', '_output.npy'))


def _salience_to_mf0(output_path, thresh=0.5):
    """Convert an _output.npy salience map [360, n_t] to (times, freqs).

    Uses a simple threshold (any bin >= thresh is active) rather than argrelmax,
    because the reference salience is already binary/near-binary.
    """
    salience = np.load(output_path, allow_pickle=True)  # shape (360, n_t)
    freq_grid = utils.get_freq_grid()
    time_grid = utils.get_time_grid(salience.shape[1])
    ref_freqs = [freq_grid[salience[:, t] >= thresh] for t in range(salience.shape[1])]
    return time_grid, ref_freqs


# ---------------------------------------------------------------------------
# Step 2: Best threshold — mirror of utils_train.get_best_thresh
# ---------------------------------------------------------------------------

def get_best_thresh_npy(dat, model):
    """Mirror of utils_train.get_best_thresh using _output.npy references.

    Sweeps threshold values [0.1, 0.9] on the validation set and returns the
    threshold that maximises mean mir_eval Accuracy.
    """
    thresh_vals = np.arange(0.1, 1.0, 0.1)
    thresh_scores = {t: [] for t in thresh_vals}

    for npy_file, _ in dat.validation_files:
        output_path = _input_to_output_path(npy_file)
        if not os.path.exists(output_path):
            print(f'  SKIP (missing output): {os.path.basename(npy_file)}')
            continue

        predicted_output, _, _ = utils_train.get_single_test_prediction(
            model=model, npy_file=npy_file
        )
        ref_times, ref_freqs = _salience_to_mf0(output_path, thresh=0.5)

        for thresh in thresh_vals:
            est_times, est_freqs = utils_train.pitch_activations_to_mf0(
                predicted_output, thresh
            )
            scores = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs)
            thresh_scores[thresh].append(scores['Accuracy'])

    avg_thresh = [np.mean(thresh_scores[t]) if thresh_scores[t] else 0.0
                  for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]
    print(f'Best threshold: {best_thresh:.1f}  '
          f'(val Accuracy: {np.max(avg_thresh):.4f})')
    print('Validation accuracy at 0.5: '
          f'{np.mean(thresh_scores[0.5]) if thresh_scores[0.5] else 0.0:.4f}')
    return best_thresh


# ---------------------------------------------------------------------------
# Step 3: Score on split — mirror of utils_train.score_on_test_set
# ---------------------------------------------------------------------------

def score_on_split_npy(model, file_list, save_path, split_name, thresh=0.5):
    """Mirror of utils_train.score_on_test_set using _output.npy references.

    Runs full-file inference on every pair in file_list, evaluates against
    the corresponding _output.npy reference, and writes per-pair CSV results.
    """
    all_scores = []

    for npy_file, _ in file_list:
        output_path = _input_to_output_path(npy_file)
        if not os.path.exists(output_path):
            print(f'  SKIP (missing output): {os.path.basename(npy_file)}')
            continue

        pair_id = os.path.basename(npy_file).replace('_input.npy', '')
        print(f'  Scoring: {pair_id}')

        predicted_output, _, _ = utils_train.get_single_test_prediction(
            model, npy_file
        )

        # Save raw prediction salience map
        np.save(
            os.path.join(save_path, f'{pair_id}_{split_name}_prediction.npy'),
            predicted_output.astype(np.float32),
        )

        est_times, est_freqs = utils_train.pitch_activations_to_mf0(
            predicted_output, thresh
        )
        ref_times, ref_freqs = _salience_to_mf0(output_path, thresh=0.5)

        scores = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs)
        scores['pair_id'] = pair_id
        all_scores.append(scores)

    if not all_scores:
        print(f'  No scores computed for split "{split_name}".')
        return

    df = pd.DataFrame(all_scores)
    p, r = df['Precision'], df['Recall']
    df['F1'] = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)

    scores_path   = os.path.join(save_path, f'{split_name}_scores.csv')
    summary_path  = os.path.join(save_path, f'{split_name}_score_summary.csv')
    df.to_csv(scores_path, index=False)
    df.describe().to_csv(summary_path)

    print(f'\n  {split_name} results ({len(df)} pairs):')
    print(df.describe().to_string())


# ---------------------------------------------------------------------------
# run_evaluation — mirrors 2_training.py's run_evaluation()
# ---------------------------------------------------------------------------

def run_evaluation(dat, model, save_key, splits):
    """Mirror of experiments/2_training.py run_evaluation(), adapted for dAST.

    Steps:
      1. Keras model metrics (loss/MSE/accuracy) on all generators
      2. Best threshold selection on validation set (npy-based)
      3. Full mir_eval scoring on each requested split (npy-based)
    """
    (save_path, _, _,
     model_scores_path, _, _) = utils_train.get_paths(config.exper_output, save_key)

    # Step 1: Keras model metrics
    print('Getting Keras model metrics (loss / MSE / soft_binary_accuracy)...')
    _get_model_metrics(dat, model, model_scores_path)

    # Step 2: Best threshold
    print('\nGetting best threshold on validation set...')
    thresh = get_best_thresh_npy(dat, model)

    # Step 3: Per-pair mir_eval scoring on requested splits
    split_file_map = {
        'train':    dat.train_files,
        'validate': dat.validation_files,
        'test':     dat.test_files,
    }
    for split in splits:
        files = split_file_map.get(split, [])
        print(f'\nScoring split "{split}" ({len(files)} pairs) with thresh={thresh:.1f}...')
        score_on_split_npy(model, files, save_path, split, thresh)

    print(f'\nEvaluation complete. Results saved to {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    data_splits_path = os.path.join(config.data_save_folder, args.data_splits_file)

    dat = Data(
        data_splits_path=data_splits_path,
        data_path=config.data_save_folder,
        input_patch_size=(360, 50),
        batch_size=32,
        active_str=100,
        muxrate=32,
    )

    print(f'Loading model from {args.model_path} ...')
    model = build_model3()
    model.load_weights(args.model_path)
    model.compile(
        loss=utils_train.bkld,
        metrics=['mse', utils_train.soft_binary_accuracy],
        optimizer='adam',
    )
    print('Model loaded.\n')

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    run_evaluation(dat, model, args.save_key, splits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained dAST model on train/val/test splits.'
    )
    parser.add_argument(
        '--model-path',
        default=os.path.join(config.models_save_folder, 'dast_model3_v1.keras'),
        help='Path to trained .keras model file',
    )
    parser.add_argument(
        '--save-key',
        dest='save_key',
        default='dast_model3_v1',
        help='Save key used by get_paths() for output directory naming',
    )
    parser.add_argument(
        '--data-splits-file',
        dest='data_splits_file',
        default='data_splits.json',
        help='Filename of the splits JSON inside data_save_folder',
    )
    parser.add_argument(
        '--splits',
        default='train,validate,test',
        help='Comma-separated split names to evaluate (train, validate, test)',
    )
    main(parser.parse_args())
