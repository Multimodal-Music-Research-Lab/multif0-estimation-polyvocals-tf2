"""
prepare_dast_mixture_features.py

For each row in mixture_pairs.csv:
  - Loads and mixes audio from both tracks (mono, truncated to min length)
  - Combines per-singer annotations into a joint multi-pitch target
  - Extracts HCQT features from the mixture via pumpp
  - Saves input/output .npy files to features_targets/inputs/ and outputs/
  - Writes data_splits.json for use by train_dast.py

Temp WAV files written to audiomixtures/ are deleted after extraction.

Usage (run from repo root with PYTHONPATH=.):
    PYTHONPATH=. python experiments/prepare_dast_mixture_features.py [options]

Arguments:
    --n-jobs N               Parallel workers (default: 4; use 1 for debugging)
    --datasets DS [DS ...]   Restrict to these dataset values
    --splits-only            Skip extraction; only regenerate data_splits.json
                             from already-extracted npy files on disk
    --output-splits-file F   Filename for splits JSON in data_save_folder
                             (default: data_splits.json)
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

import utils
from experiments import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)

DAST_BASE = os.environ['DAST_BASE']
PROCESSED = os.path.join(DAST_BASE, 'data', 'processed')
PAIRS_PATH = os.path.join(DAST_BASE, 'data', 'processed', 'training', 'mixture_pairs.csv')
RAW = config.RAW_DATA_ROOT
SPLIT_MAP = {'train': 'train', 'val': 'validate', 'test': 'test'}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_paths(dataset, tid, csd_meta):
    """Return (audio_path, annot_path, fmt) for a single track."""
    if dataset == 'dagstuhl_manual':
        audio = os.path.join(
            RAW, 'DagstuhlChoirSet_V1.2.3', 'audio_wav_22050_mono',
            f'{tid}.wav'
        )
        # Manual annotations only exist for LRX microphone.
        # DYN/HSM variants of the same recording share the LRX annotation.
        if tid.endswith('_DYN') or tid.endswith('_HSM'):
            annot_tid = tid[:-4] + '_LRX'
        else:
            annot_tid = tid
        annot = os.path.join(
            RAW, 'DagstuhlChoirSet_V1.2.3', 'annotations_csv_F0_manual',
            f'{annot_tid}.csv'
        )
        fmt = 'csv_time_f0'

    elif dataset == 'dagstuhl_non_manual':
        audio = os.path.join(
            RAW, 'DagstuhlChoirSet_V1.2.3', 'audio_wav_22050_mono',
            f'{tid}.wav'
        )
        annot = os.path.join(
            RAW, 'DagstuhlChoirSet_V1.2.3', 'annotations_csv_F0_PYIN',
            f'{tid}.csv'
        )
        fmt = 'csv_time_f0'

    elif dataset == 'esmuc':
        audio = os.path.join(RAW, 'EsmucChoirDataset_v1.0.0', f'{tid}.wav')
        annot = os.path.join(RAW, 'EsmucChoirDataset_v1.0.0', f'{tid}.lab')
        fmt = 'lab_start_f0_dur'

    elif dataset == 'cantoria':
        audio = os.path.join(RAW, 'CantoriaDataset_v1.0.0', 'Audio', f'{tid}.wav')
        annot = os.path.join(RAW, 'CantoriaDataset_v1.0.0', 'F0_crepe', f'{tid}.csv')
        fmt = 'csv_time_f0'

    elif dataset == 'choral_singing_dataset':
        meta = csd_meta[tid]
        audio = os.path.join(RAW, 'ChoralSingingDataset', f'{tid}.wav')
        annot = os.path.join(
            RAW, 'ChoralSingingDataset',
            f"CSD_{meta['song']}_{meta['section']}_notes.lab"
        )
        fmt = 'csd_notes'

    else:
        raise ValueError(f'Unknown dataset: {dataset!r}')

    return audio, annot, fmt


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def load_annotations(annot_path, fmt):
    """Load annotation file; return (times, freqs) arrays with f0 > 0."""
    if fmt == 'csv_time_f0':
        df = pd.read_csv(annot_path, header=None)
        t = df.iloc[:, 0].to_numpy(dtype=float)
        f = df.iloc[:, 1].to_numpy(dtype=float)
        mask = f > 0
        return t[mask], f[mask]

    elif fmt == 'lab_start_f0_dur':
        # ESMUC: tab/space-separated [start_time, f0, duration]
        df = pd.read_csv(annot_path, sep=r'\s+', header=None)
        t = df.iloc[:, 0].to_numpy(dtype=float)
        f = df.iloc[:, 1].to_numpy(dtype=float)
        mask = f > 0
        return t[mask], f[mask]

    elif fmt == 'csd_notes':
        # CSD: space-separated [start_time, mean_f0, duration], note-level
        # Expand to frame-level at 5 ms intervals
        df = pd.read_csv(annot_path, sep=r'\s+', header=None)
        starts    = df.iloc[:, 0].to_numpy(dtype=float)
        f0s       = df.iloc[:, 1].to_numpy(dtype=float)
        durations = df.iloc[:, 2].to_numpy(dtype=float)
        ts, fs = [], []
        for start, f0, dur in zip(starts, f0s, durations):
            if f0 <= 0:
                continue
            frame_ts = np.arange(start, start + dur, 0.005)
            ts.append(frame_ts)
            fs.append(np.full(len(frame_ts), f0))
        if not ts:
            return np.array([]), np.array([])
        return np.concatenate(ts), np.concatenate(fs)

    else:
        raise ValueError(f'Unknown annotation format: {fmt!r}')


# ---------------------------------------------------------------------------
# CSD metadata loader
# ---------------------------------------------------------------------------

def load_csd_meta():
    """Return { track_id: {'song': str, 'section': str} }"""
    csv_path = os.path.join(PROCESSED, 'ChoralSingingDataset', 'csd_meta.csv')
    df = pd.read_csv(csv_path, index_col=0)
    return {
        tid: {'song': row['song'], 'section': row['section']}
        for tid, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Per-pair worker
# ---------------------------------------------------------------------------

def process_pair(row, save_dir, csd_meta):
    """
    Process one mixture pair row.
    Returns '{pair_id}.wav' on success, None on failure/skip.
    """
    pair_id = row['pair_id']
    dataset = row['dataset']
    track_a = row['track_a']
    track_b = row['track_b']

    inp = os.path.join(save_dir, 'inputs',  f'{pair_id}_input.npy')
    out = os.path.join(save_dir, 'outputs', f'{pair_id}_output.npy')
    if os.path.exists(inp) and os.path.exists(out):
        logger.info('SKIP %s — already extracted', pair_id)
        return f'{pair_id}.wav'

    try:
        audio_a, annot_a, fmt_a = resolve_paths(dataset, track_a, csd_meta)
        audio_b, annot_b, fmt_b = resolve_paths(dataset, track_b, csd_meta)
    except Exception as exc:
        logger.warning('SKIP %s — path resolution error: %s', pair_id, exc)
        return None

    for path, label in [
        (audio_a, f'{track_a} audio'), (audio_b, f'{track_b} audio'),
        (annot_a, f'{track_a} annot'), (annot_b, f'{track_b} annot'),
    ]:
        if not os.path.exists(path):
            logger.warning('SKIP %s — missing %s: %s', pair_id, label, path)
            return None

    try:
        times_a, freqs_a = load_annotations(annot_a, fmt_a)
        times_b, freqs_b = load_annotations(annot_b, fmt_b)
    except Exception as exc:
        logger.warning('SKIP %s — annotation load error: %s', pair_id, exc)
        return None

    try:
        sig_a, sr_a = sf.read(audio_a, always_2d=False)
        sig_b, sr_b = sf.read(audio_b, always_2d=False)
    except Exception as exc:
        logger.warning('SKIP %s — audio load error: %s', pair_id, exc)
        return None

    if sr_a != sr_b:
        logger.warning(
            'SKIP %s — sample rate mismatch: %d vs %d', pair_id, sr_a, sr_b
        )
        return None

    # Truncate to min length and mix
    min_len = min(len(sig_a), len(sig_b))
    sig_a = sig_a[:min_len].astype(np.float32)
    sig_b = sig_b[:min_len].astype(np.float32)
    mix = (sig_a + sig_b) / 2.0
    mix_duration = min_len / sr_a

    # Clip annotation times to mix duration and combine
    mask_a = times_a < mix_duration
    mask_b = times_b < mix_duration
    combined_times = np.concatenate([times_a[mask_a], times_b[mask_b]])
    combined_freqs = np.concatenate([freqs_a[mask_a], freqs_b[mask_b]])

    if len(combined_times) == 0:
        logger.warning('SKIP %s — empty combined annotation after clipping', pair_id)
        return None

    # Write temp WAV, extract HCQT, then delete
    mix_dir = config.audio_save_folder
    os.makedirs(mix_dir, exist_ok=True)
    temp_wav = os.path.join(mix_dir, f'{pair_id}.wav')
    try:
        sf.write(temp_wav, mix, sr_a)
        hcqt, annot_target, freq_grid, time_grid = utils.get_input_output_pairs_pump(
            temp_wav, combined_times, combined_freqs
        )
    except Exception as exc:
        logger.warning('SKIP %s — HCQT extraction error: %s', pair_id, exc)
        return None
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    utils.save_data(save_dir, inp, out, pair_id, hcqt, annot_target, freq_grid, time_grid)
    return f'{pair_id}.wav'


# ---------------------------------------------------------------------------
# data_splits.json writer
# ---------------------------------------------------------------------------

def write_data_splits(pairs, processed_ids, save_dir, filename):
    splits = {'train': [], 'validate': [], 'test': []}
    for _, row in pairs.iterrows():
        key = f"{row['pair_id']}.wav"
        if key not in processed_ids:
            continue
        dest = SPLIT_MAP.get(row['split'])
        if dest is None:
            logger.warning(
                'Unknown split value %r for pair %s', row['split'], row['pair_id']
            )
            continue
        splits[dest].append(key)

    out_path = os.path.join(save_dir, filename)
    with open(out_path, 'w') as fh:
        json.dump(splits, fh, indent=2)

    logger.info(
        'data_splits written to %s  (train=%d  validate=%d  test=%d)',
        out_path, len(splits['train']), len(splits['validate']), len(splits['test'])
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    save_dir = config.data_save_folder
    os.makedirs(os.path.join(save_dir, 'inputs'),  exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'outputs'), exist_ok=True)
    os.makedirs(config.models_save_folder, exist_ok=True)

    pairs = pd.read_csv(PAIRS_PATH)
    logger.info('Pairs loaded: %d total', len(pairs))

    if args.datasets:
        pairs = pairs[pairs['dataset'].isin(args.datasets)].reset_index(drop=True)
        logger.info('Filtered to datasets %s: %d pairs', args.datasets, len(pairs))

    csd_meta = load_csd_meta()
    rows = [row for _, row in pairs.iterrows()]

    if args.splits_only:
        logger.info('--splits-only: scanning existing npy files, skipping extraction')
        processed_ids = set()
        for row in rows:
            pid = row['pair_id']
            inp = os.path.join(save_dir, 'inputs',  f'{pid}_input.npy')
            out = os.path.join(save_dir, 'outputs', f'{pid}_output.npy')
            if os.path.exists(inp) and os.path.exists(out):
                processed_ids.add(f'{pid}.wav')
        logger.info('--splits-only: found %d already-extracted pairs', len(processed_ids))
    else:
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(process_pair)(row, save_dir, csd_meta) for row in rows
        )
        processed_ids = set(r for r in results if r is not None)
        logger.info(
            'Extraction complete: %d / %d pairs succeeded',
            len(processed_ids), len(rows)
        )

    write_data_splits(pairs, processed_ids, save_dir, args.output_splits_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare dAST mixture features and data_splits.json for training.'
    )
    parser.add_argument(
        '--n-jobs', type=int, default=4,
        help='Number of parallel joblib workers (default: 4; use 1 for debugging)'
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        metavar='DS',
        help='Restrict to these dataset values (space-separated)'
    )
    parser.add_argument(
        '--splits-only', action='store_true',
        help='Skip extraction; only regenerate splits JSON from existing npy files'
    )
    parser.add_argument(
        '--output-splits-file', default='data_splits.json',
        metavar='FILENAME',
        help='Output filename for splits JSON inside data_save_folder '
             '(default: data_splits.json)'
    )
    main(parser.parse_args())
