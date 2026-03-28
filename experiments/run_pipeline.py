"""
Batched section-by-section pipeline (Workaround 2).

Processes one dataset section at a time:
  1. Generate mix FLACs for the section  (0_setup_duet.py)
  2. Extract HCQT features               (1_prep.py)
  3. Delete the mix FLACs                (free disk space)
  4. Repeat for the next section

Peak disk usage = largest single section (~7 GB for ECS Seele Christi),
well within the 10 GB currently available.

Usage
-----
    cd /path/to/multif0-estimation-polyvocals-tf2
    PYTHONPATH=. python experiments/run_pipeline.py
    PYTHONPATH=. python experiments/run_pipeline.py --sections CSD ECS_DG
    PYTHONPATH=. python experiments/run_pipeline.py --dry-run

Strategy
--------
Phase 1 (metadata only, no audio):
    Run 0_setup_duet.py --no-audio-mix to build the complete
    mtracks_info.json covering all sections at once.

Phase 2 (section loop):
    For each section:
      a. Generate FLACs: 0_setup_duet.py --sections SECTION
      b. Extract features: 1_prep.py (skips already-done and missing files)
      c. Delete the section's FLAC files to reclaim disk space.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

# Path to the repo root (parent of this file's directory)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_SECTIONS = ['CSD', 'ECS_DG', 'ECS_DH', 'ECS_SC', 'DCS']


def _run(cmd, dry_run=False):
    print("  $ " + " ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd, cwd=_REPO_ROOT)
        if result.returncode != 0:
            print("ERROR: command failed with return code {}".format(result.returncode))
            sys.exit(result.returncode)


def _delete_section_flacs(mixes_path, metadata_path, section_keys, dry_run=False):
    """Delete FLAC mix files that belong to the current section."""
    deleted = 0
    for key in section_keys:
        fpath = os.path.join(mixes_path, key)
        if os.path.exists(fpath) and fpath.endswith('.flac'):
            if not dry_run:
                os.remove(fpath)
            deleted += 1
    print("  {} FLAC files {}deleted.".format(
        deleted, "(would be) " if dry_run else ""))


def _load_section_keys(metadata_path, mixes_path, section):
    """
    Return the set of mix filenames that belong to `section`.

    Since mtracks_info.json doesn't tag entries by section we identify them
    by which keys currently have a matching FLAC on disk (written in the
    immediately preceding 0_setup_duet.py call for this section only).
    """
    if not os.path.exists(metadata_path):
        return []
    with open(metadata_path, 'r') as fh:
        meta = json.load(fh)
    return [k for k in meta if os.path.exists(os.path.join(mixes_path, k))]


def main():
    parser = argparse.ArgumentParser(
        description="Batched section-by-section pipeline (Workaround 2).")
    parser.add_argument(
        '--sections', nargs='+', default=None,
        choices=ALL_SECTIONS,
        metavar='SECTION',
        help='Sections to process (default: all). Choices: ' + ', '.join(ALL_SECTIONS))
    parser.add_argument(
        '--metadata-file', default=None,
        help='Path to mtracks_info.json (default: derived from config).')
    parser.add_argument(
        '--audio-path', default=None,
        help='Path to audiomixtures folder (default: derived from config).')
    parser.add_argument(
        '--save-dir', default=None,
        help='Path to features_targets folder (default: derived from config).')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print commands without executing them.')
    args = parser.parse_args()

    # Resolve paths (use config defaults if not overridden)
    sys.path.insert(0, _REPO_ROOT)
    from experiments import config as cfg

    mixes_path = args.audio_path or cfg.audio_save_folder
    save_dir = args.save_dir or cfg.data_save_folder
    metadata_file = args.metadata_file or os.path.join(mixes_path, 'mtracks_info.json')
    sections = args.sections or ALL_SECTIONS

    python = sys.executable

    print("=" * 60)
    print("Phase 1: Build complete metadata (no audio)")
    print("=" * 60)
    _run([python, 'experiments/0_setup_duet.py', '--no-audio-mix'], dry_run=args.dry_run)

    for section in sections:
        print("\n" + "=" * 60)
        print("Phase 2 – Section: {}".format(section))
        print("=" * 60)

        # a) Generate FLACs for this section only
        print("  [a] Generating mix FLACs...")
        _run([python, 'experiments/0_setup_duet.py', '--sections', section],
             dry_run=args.dry_run)

        # Identify which keys were just written (have a FLAC on disk now)
        section_keys = _load_section_keys(metadata_file, mixes_path, section)
        print("  {} mix FLACs written for section {}.".format(len(section_keys), section))

        # b) Extract HCQT features (skips already-done and missing-audio entries)
        print("  [b] Extracting features...")
        _run([python, 'experiments/1_prep.py',
              '--metadata-path', metadata_file,
              '--audio-path', mixes_path,
              '--save-dir', save_dir],
             dry_run=args.dry_run)

        # c) Delete section FLACs to reclaim disk space
        print("  [c] Deleting section FLACs...")
        _delete_section_flacs(mixes_path, metadata_file, section_keys,
                              dry_run=args.dry_run)

    print("\nAll sections complete.")
    print("Features saved to: {}".format(save_dir))


if __name__ == '__main__':
    main()
