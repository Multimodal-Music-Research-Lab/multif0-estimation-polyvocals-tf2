"""
Batch evaluation script: run multi-F0 inference on 5 evaluation datasets
and compute mir_eval.multipitch metrics.
"""

import argparse
import os
import traceback
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import mir_eval

from experiments import config
from models import build_model3
from utils import create_pump_object, compute_pump_features
from utils_train import pitch_activations_to_mf0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DAST_BASE = os.environ['DAST_BASE']

DATASETS = {
    'pairs':              f'{DAST_BASE}/data/processed/evaluation/pairs.csv',
    'pairs_crepe':        f'{DAST_BASE}/data/processed/evaluation/pairs_crepe.csv',
    'pairs_pyin':         f'{DAST_BASE}/data/processed/evaluation/pairs_pyin.csv',
    'pairs_manual_crepe': f'{DAST_BASE}/data/processed/evaluation/pairs_manual_crepe.csv',
    'pairs_manual_pyin':  f'{DAST_BASE}/data/processed/evaluation/pairs_manual_pyin.csv',
}

PREDICTIONS_DIR = 'predictions'
MIXES_DIR = os.path.join(PREDICTIONS_DIR, 'mixes')
ERRORS_LOG = 'evaluation_errors.log'

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(MIXES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: pair key for caching
# ---------------------------------------------------------------------------
def pair_key(piece_id, alto_id, tenor_id):
    return f"{piece_id}_{alto_id}_{tenor_id}"


# ---------------------------------------------------------------------------
# Mixed audio creation
# ---------------------------------------------------------------------------
def create_mix(alto_path, tenor_path, out_path):
    if os.path.exists(out_path):
        return
    a, sr = librosa.load(alto_path, sr=22050, mono=True)
    t, _  = librosa.load(tenor_path, sr=22050, mono=True)
    min_len = min(len(a), len(t))
    mix = (a[:min_len] + t[:min_len]) / 2.0
    sf.write(out_path, mix, sr)
    print(f"  Created mix: {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(model, mix_path, pred_path):
    """Run model on mix_path, save predictions to pred_path."""
    pump = create_pump_object()
    features = compute_pump_features(pump, mix_path)

    input_hcqt   = features['dphase/mag'][0].transpose(1, 0, 2)[np.newaxis, :, :, :]
    input_dphase = features['dphase/dphase'][0].transpose(1, 0, 2)[np.newaxis, :, :, :]

    predictions = model.predict([input_hcqt, input_dphase])[0]
    est_times, est_freqs = pitch_activations_to_mf0(predictions, thresh=0.5)

    pred_data = []
    for t, freqs in zip(est_times, est_freqs):
        row_vals = [t] + (list(freqs) + [0, 0])[:2]
        pred_data.append(row_vals)

    pred_df = pd.DataFrame(pred_data, columns=['time', 'f0_1', 'f0_2'])
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved predictions: {os.path.basename(pred_path)}")


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------
CREPE_CONFIDENCE_THRESHOLD = 0.935
MIN_FREQUENCY_HZ = 130.0
MAX_FREQUENCY_HZ = 698.0


def _freq_mask(series):
    return (series >= MIN_FREQUENCY_HZ) & (series <= MAX_FREQUENCY_HZ)


def load_annotation_csv(path):
    """Return (times, freqs) arrays for voiced frames within the frequency range."""
    if 'CREPE' in path:
        df = pd.read_csv(path, header=None, names=['time', 'freq', 'conf'])
        voiced = df[_freq_mask(df['freq']) & (df['conf'] >= CREPE_CONFIDENCE_THRESHOLD)]
    elif 'PYIN' in path:
        df = pd.read_csv(path, header=None, names=['time', 'freq', 'conf'])
        voiced = df[_freq_mask(df['freq'])]
    else:
        df = pd.read_csv(path, header=None, names=['time', 'freq'])
        voiced = df[_freq_mask(df['freq'])]
    return voiced['time'].values, voiced['freq'].values


def combine_annotations(alto_path, tenor_path):
    """Combine two single-F0 annotations into multi-F0 ragged format."""
    alto_times, alto_freqs = load_annotation_csv(alto_path)
    tenor_times, tenor_freqs = load_annotation_csv(tenor_path)

    all_times = np.union1d(alto_times, tenor_times)
    alto_dict  = dict(zip(alto_times, alto_freqs))
    tenor_dict = dict(zip(tenor_times, tenor_freqs))

    ref_freqs = []
    for t in all_times:
        freqs = []
        if t in alto_dict:
            freqs.append(alto_dict[t])
        if t in tenor_dict:
            freqs.append(tenor_dict[t])
        ref_freqs.append(np.array(freqs))

    return all_times, ref_freqs


def combine_annotations_multi(annotation_pairs):
    """Merge multiple (alto_path, tenor_path) annotation source pairs into one ragged reference.

    Each annotation source contributes its own time grid and pitch values.
    All voiced pitches across all sources are unioned per time step.
    """
    from collections import defaultdict
    time_freqs = defaultdict(list)

    for alto_path, tenor_path in annotation_pairs:
        for path in [alto_path, tenor_path]:
            times, freqs = load_annotation_csv(path)
            for t, f in zip(times, freqs):
                time_freqs[round(float(t), 8)].append(f)

    if not time_freqs:
        return np.array([]), []

    all_times = np.array(sorted(time_freqs.keys()))
    ref_freqs = [np.array(time_freqs[t]) for t in all_times]
    return all_times, ref_freqs


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------
def load_prediction_csv(path):
    df = pd.read_csv(path)
    est_times = df['time'].values
    est_freqs = []
    for _, row in df.iterrows():
        freqs = [f for f in [row['f0_1'], row['f0_2']]
                 if MIN_FREQUENCY_HZ <= f <= MAX_FREQUENCY_HZ]
        est_freqs.append(np.array(freqs))
    return est_times, est_freqs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
SETTINGS = ['QuartetA', 'QuartetB', 'FullChoir']


def get_setting(pair_key_str):
    for s in SETTINGS:
        if s in pair_key_str:
            return s
    return 'Unknown'


def _print_summary(dataset_name, setting, scores_df, summary_rows):
    metric_cols = ['Precision', 'Recall', 'F1', 'Accuracy']
    metric_cols = [c for c in metric_cols if c in scores_df.columns]

    label = f"{dataset_name} / {setting}"
    print(f"\n  {label} ({len(scores_df)} pairs)")
    print(f"  {'Metric':<28} {'Mean':>8} {'Std':>8}")
    print('  ' + '-' * 46)
    for col in metric_cols:
        mean_val = scores_df[col].mean()
        std_val  = scores_df[col].std()
        print(f"  {col:<28} {mean_val:>8.4f} {std_val:>8.4f}")
        summary_rows.append({
            'dataset': dataset_name,
            'setting': setting,
            'metric': col,
            'mean': mean_val,
            'std': std_val,
            'n_pairs': len(scores_df),
        })


def _print_all_summaries(dataset_name, scores_df, summary_rows):
    """Print overall + per-setting breakdowns."""
    scores_df = scores_df.copy()
    scores_df['setting'] = scores_df['pair'].apply(get_setting)

    _print_summary(dataset_name, 'all', scores_df, summary_rows)

    for setting in SETTINGS:
        subset = scores_df[scores_df['setting'] == setting]
        if len(subset) > 0:
            _print_summary(dataset_name, setting, subset, summary_rows)

    qab = scores_df[scores_df['setting'].isin(['QuartetA', 'QuartetB'])]
    if len(qab) > 0:
        _print_summary(dataset_name, 'QuartetA+B', qab, summary_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    # Load model once
    print(f"Loading model from {args.model_path} ...")
    model = build_model3()
    model.load_weights(args.model_path)  # Keras 3 native — works with .keras files
    model.compile()
    print("Model loaded.\n")

    # Inference cache: pair_key -> pred_path (already computed)
    inference_cache = {}

    # Pre-seed cache with existing predictions
    for fname in os.listdir(PREDICTIONS_DIR):
        if fname.endswith('_pred.csv'):
            key = fname[:-len('_pred.csv')]
            inference_cache[key] = os.path.join(PREDICTIONS_DIR, fname)

    with open(ERRORS_LOG, 'w') as f:
        f.write("Evaluation Errors\n")

    summary_rows = []

    for dataset_name, csv_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        df = pd.read_csv(csv_path)
        all_scores = []

        for index, row in df.iterrows():
            piece_id  = row['piece_id']
            alto_id   = row['alto_singer_id']
            tenor_id  = row['tenor_singer_id']
            alto_audio  = os.path.join(DAST_BASE, row['alto_audio_path'])
            tenor_audio = os.path.join(DAST_BASE, row['tenor_audio_path'])
            alto_annot  = os.path.join(DAST_BASE, row['alto_annotation_path'])
            tenor_annot = os.path.join(DAST_BASE, row['tenor_annotation_path'])

            key = pair_key(piece_id, alto_id, tenor_id)
            pred_path = os.path.join(PREDICTIONS_DIR, f"{key}_pred.csv")

            try:
                # --- Step 1: Create mix if needed ---
                mix_path = os.path.join(MIXES_DIR, f"{key}_mix.wav")
                if not os.path.exists(pred_path):
                    # Only need the mix for inference
                    create_mix(alto_audio, tenor_audio, mix_path)

                # --- Step 2: Run inference if needed ---
                if key not in inference_cache:
                    print(f"  [{index}] Inference: {key}")
                    run_inference(model, mix_path, pred_path)
                    inference_cache[key] = pred_path
                else:
                    pred_path = inference_cache[key]

                # --- Step 3: Load reference annotations ---
                ref_times, ref_freqs = combine_annotations(alto_annot, tenor_annot)

                # --- Step 4: Load predictions ---
                est_times, est_freqs = load_prediction_csv(pred_path)

                # --- Step 5: Compute metrics ---
                scores = mir_eval.multipitch.evaluate(
                    ref_times, ref_freqs, est_times, est_freqs
                )
                scores['pair'] = key
                all_scores.append(scores)

            except Exception as e:
                msg = f"Error in {dataset_name} row {index} ({key}): {e}\n{traceback.format_exc()}\n"
                print(f"  ERROR: {e}")
                with open(ERRORS_LOG, 'a') as flog:
                    flog.write(msg)

        if not all_scores:
            print(f"  No scores computed for {dataset_name}")
            continue

        scores_df = pd.DataFrame(all_scores)

        # Compute F1 per pair
        p = scores_df['Precision']
        r = scores_df['Recall']
        scores_df['F1'] = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)

        # Save full results
        results_path = os.path.join(PREDICTIONS_DIR, f"evaluation_results_{dataset_name}.csv")
        scores_df.to_csv(results_path, index=False)

        _print_all_summaries(dataset_name, scores_df, summary_rows)

    # -----------------------------------------------------------------------
    # Combined dataset: all unique (pair, annotation-source) evaluations
    # across all 5 CSVs — 414 rows total (2 manual + 206 CREPE + 206 pYIN)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Dataset: combined (all annotation sources, 414 evaluations)")
    print(f"{'='*60}")

    # Collect unique (pair_key, alto_annot_path, tenor_annot_path) across all CSVs
    seen_eval = set()
    combined_eval_rows = []
    for csv_path in DATASETS.values():
        for _, row in pd.read_csv(csv_path).iterrows():
            k = pair_key(row['piece_id'], row['alto_singer_id'], row['tenor_singer_id'])
            dedup_key = (k, row['alto_annotation_path'], row['tenor_annotation_path'])
            if dedup_key not in seen_eval:
                seen_eval.add(dedup_key)
                combined_eval_rows.append({
                    'pair_key':    k,
                    'alto_annot':  os.path.join(DAST_BASE, row['alto_annotation_path']),
                    'tenor_annot': os.path.join(DAST_BASE, row['tenor_annotation_path']),
                })

    print(f"  Unique (pair, annotation-source) combinations: {len(combined_eval_rows)}")
    all_scores_combined = []

    for er in combined_eval_rows:
        key = er['pair_key']
        pred_path = inference_cache.get(key)
        if pred_path is None:
            continue
        try:
            ref_times, ref_freqs = combine_annotations(er['alto_annot'], er['tenor_annot'])
            est_times, est_freqs = load_prediction_csv(pred_path)

            scores = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs)
            scores['pair'] = key
            scores['alto_annot'] = er['alto_annot']
            all_scores_combined.append(scores)

        except Exception as e:
            msg = f"Error in combined ({key}): {e}\n{traceback.format_exc()}\n"
            print(f"  ERROR: {e}")
            with open(ERRORS_LOG, 'a') as flog:
                flog.write(msg)

    if all_scores_combined:
        scores_df = pd.DataFrame(all_scores_combined)
        p = scores_df['Precision']
        r = scores_df['Recall']
        scores_df['F1'] = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
        scores_df.to_csv(
            os.path.join(PREDICTIONS_DIR, 'evaluation_results_combined.csv'), index=False
        )
        _print_all_summaries('combined', scores_df, summary_rows)

    # Save overall summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(PREDICTIONS_DIR, 'evaluation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate multi-F0 model on dAST evaluation pair CSVs.'
    )
    parser.add_argument(
        '--model-path',
        default=os.path.join(config.models_save_folder, 'dast_model3_v1.keras'),
        help='Path to trained .keras model file (default: dast_model3_v1.keras in models_save_folder)',
    )
    main(parser.parse_args())
