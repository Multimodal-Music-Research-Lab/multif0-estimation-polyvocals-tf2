"""
generate_mixture_pairs.py

Reads dAST split_manifest.csv and generates all valid 2-voice mixture pairs
for training model3 (Late/Deep). Each pair consists of two tracks from the
same dataset, same recording group (song/setting/take), and same split.

Pair ID format: {track_a}__{track_b}  (tracks in lexicographic order)

Output: $DAST_BASE/data/processed/training/mixture_pairs.csv
Columns: pair_id, dataset, split, track_a, track_b

Usage (run from repo root with PYTHONPATH=.):
    PYTHONPATH=. python experiments/generate_mixture_pairs.py
"""

import itertools
import logging
import os

import pandas as pd

from experiments import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)

DAST_BASE = os.environ['DAST_BASE']
PROCESSED = os.path.join(DAST_BASE, 'data', 'processed')
MANIFEST_PATH = os.path.join(PROCESSED, 'split_manifest.csv')
TRAINING_DIR = os.path.join(DAST_BASE, 'data', 'processed', 'training')
OUTPUT_PATH = os.path.join(TRAINING_DIR, 'mixture_pairs.csv')


# ---------------------------------------------------------------------------
# Dataset-specific configuration
# ---------------------------------------------------------------------------
#
# pair_types: list of (prefix1, prefix2) tuples; voice membership checked via
#             str.startswith() on voice_col.  None means all C(n,2) among
#             distinct-voice tracks within a group.
# voice_exclude: list of exact voice values to drop before pairing.

DATASET_CONFIGS = {
    'dagstuhl_manual': {
        'meta_path': os.path.join(PROCESSED, 'DagstuhlChoirSet_V1.2.3', 'dcs_meta_df.csv'),
        'meta_index': 'id',
        'group_by': ['song', 'setting', 'take', 'microphone'],
        'voice_col': 'voice',
        'pair_types': None,
    },
    'dagstuhl_non_manual': {
        'meta_path': os.path.join(PROCESSED, 'DagstuhlChoirSet_V1.2.3', 'dcs_meta_df.csv'),
        'meta_index': 'id',
        'group_by': ['song', 'setting', 'take', 'microphone'],
        'voice_col': 'voice',
        'pair_types': None,
    },
    'esmuc': {
        'meta_path': os.path.join(PROCESSED, 'EsmucChoirDataset_v1.0.0', 'esmuc_meta.csv'),
        'meta_index': 0,
        'group_by': ['song', 'setting', 'take'],
        'voice_col': 'voice',
        'pair_types': None,
    },
    'choral_singing_dataset': {
        'meta_path': os.path.join(PROCESSED, 'ChoralSingingDataset', 'csd_meta.csv'),
        'meta_index': 0,
        'group_by': ['song'],
        'voice_col': 'section',
        'pair_types': None,
    },
    'cantoria': {
        'meta_path': os.path.join(PROCESSED, 'CantoriaDataset_v1.0.0', 'cantoria_meta.csv'),
        'meta_index': 0,
        'group_by': ['song'],
        'voice_col': 'voice',
        'pair_types': None,
        'voice_exclude': ['Mix', 'MixOrgan'],
    },
}


# ---------------------------------------------------------------------------
# Pairing helpers
# ---------------------------------------------------------------------------

def make_pair_id(track_a, track_b):
    a, b = sorted([track_a, track_b])
    return f'{a}__{b}'


def pairs_all_combinations(group_df, voice_col):
    """Yield (track_a, track_b, split) for all same-split, different-voice combos."""
    rows = group_df[['track_id', voice_col, 'split']].to_dict('records')
    for r1, r2 in itertools.combinations(rows, 2):
        if r1['split'] == r2['split'] and r1[voice_col] != r2[voice_col]:
            yield r1['track_id'], r2['track_id'], r1['split']


def pairs_typed(group_df, voice_col, pair_types):
    """Yield (track_a, track_b, split) for typed (prefix1, prefix2) combos."""
    for p1, p2 in pair_types:
        voices1 = group_df[group_df[voice_col].str.startswith(p1)]
        voices2 = group_df[group_df[voice_col].str.startswith(p2)]
        for _, v1 in voices1.iterrows():
            for _, v2 in voices2.iterrows():
                if v1['split'] == v2['split']:
                    yield v1['track_id'], v2['track_id'], v1['split']


# ---------------------------------------------------------------------------
# Per-dataset generator
# ---------------------------------------------------------------------------

def generate_for_dataset(dataset, cfg, manifest):
    dataset_manifest = manifest[manifest['dataset'] == dataset].copy()
    if dataset_manifest.empty:
        logger.warning('No manifest rows for dataset %r', dataset)
        return []

    meta = pd.read_csv(cfg['meta_path'], index_col=cfg['meta_index'])
    meta.index.name = 'track_id'
    meta = meta.reset_index()

    merged = dataset_manifest.merge(meta, on='track_id', how='inner', suffixes=('', '_meta'))
    if merged.empty:
        logger.warning('Merge produced no rows for dataset %r — check track_id format', dataset)
        return []

    logger.info('%s: %d tracks joined with metadata', dataset, len(merged))

    voice_col = cfg['voice_col']
    voice_exclude = cfg.get('voice_exclude', [])
    if voice_exclude:
        merged = merged[~merged[voice_col].isin(voice_exclude)]

    pair_types = cfg.get('pair_types')
    group_by = cfg['group_by']

    seen = set()
    pairs = []
    for _, group in merged.groupby(group_by, dropna=False):
        if pair_types is None:
            gen = pairs_all_combinations(group, voice_col)
        else:
            gen = pairs_typed(group, voice_col, pair_types)

        for track_a, track_b, split in gen:
            pid = make_pair_id(track_a, track_b)
            if pid in seen:
                continue
            seen.add(pid)
            a, b = sorted([track_a, track_b])
            pairs.append({
                'pair_id': pid,
                'dataset': dataset,
                'split': split,
                'track_a': a,
                'track_b': b,
            })

    logger.info('%s: %d pairs generated', dataset, len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(TRAINING_DIR, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    logger.info('Manifest loaded: %d tracks', len(manifest))

    all_pairs = []
    for dataset, cfg in DATASET_CONFIGS.items():
        all_pairs.extend(generate_for_dataset(dataset, cfg, manifest))

    if not all_pairs:
        logger.error('No pairs generated — check dataset configs and manifest')
        return

    df = pd.DataFrame(all_pairs).drop_duplicates(subset='pair_id')
    df.to_csv(OUTPUT_PATH, index=False)

    for split in ['train', 'val', 'test']:
        n = (df['split'] == split).sum()
        logger.info('Split %-8s: %d pairs', split, n)
    logger.info('Total: %d pairs written to %s', len(df), OUTPUT_PATH)


if __name__ == '__main__':
    main()
