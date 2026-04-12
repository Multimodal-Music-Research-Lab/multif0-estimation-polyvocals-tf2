"""
train_dast.py

Trains model3 (Late/Deep) using dAST data splits produced by
prepare_dast_mixture_features.py.  Model hyperparameters and training settings are
identical to experiments/2_training.py (see CLAUDE.md constraints).

The only differences from 2_training.py:
  - model_save_path points to config.models_save_folder (not a hardcoded
    /scratch/hc2945 path)
  - run_evaluation() is omitted (depends on cluster-specific paths)
  - only model3 is available (this script is dAST-specific)

Usage (run from repo root with PYTHONPATH=.):
    PYTHONPATH=. python experiments/train_dast.py \\
        --save-key dast_model3_v1 \\
        [--data-splits-file data_splits.json]
"""

import argparse
import json
import os

import keras

from experiments import config
import utils_train
import models


# ---------------------------------------------------------------------------
# Data class (verbatim from experiments/2_training.py)
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
# Training (verbatim from experiments/2_training.py, model_save_path only)
# ---------------------------------------------------------------------------

def train(model, model_save_path, data_splits_file, batch_size, active_str, muxrate):

    data_path        = config.data_save_folder
    input_patch_size = (360, 50)
    data_splits_path = os.path.join(config.data_save_folder, data_splits_file)

    dat = Data(
        data_splits_path, data_path, input_patch_size,
        batch_size, active_str, muxrate,
    )

    train_generator      = dat.get_train_generator()
    validation_generator = dat.get_validation_generator()

    model.compile(
        loss=utils_train.bkld,
        metrics=['mse', utils_train.soft_binary_accuracy],
        optimizer='adam',
    )

    print(model.summary(line_length=80))

    history = model.fit_generator(
        train_generator,
        config.SAMPLES_PER_EPOCH,
        epochs=config.NB_EPOCHS,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=config.NB_VAL_SAMPLES,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                model_save_path, save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            keras.callbacks.EarlyStopping(patience=25, verbose=1),
        ],
    )

    model.load_weights(model_save_path)
    return model, history, dat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    batch_size = 32
    active_str = 100
    muxrate    = 32

    os.makedirs(config.models_save_folder, exist_ok=True)
    model_save_path = os.path.join(
        config.models_save_folder, f'{args.save_key}.pkl'
    )

    model = models.build_model3()

    model, history, dat = train(
        model, model_save_path, args.data_splits_file,
        batch_size, active_str, muxrate,
    )

    print(f'Training complete. Model saved to {model_save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train model3 (Late/Deep) on dAST data splits.'
    )
    parser.add_argument(
        '--save-key', dest='save_key', type=str, required=True,
        help='Identifier for this run; used as the model checkpoint filename.'
    )
    parser.add_argument(
        '--data-splits-file', dest='data_splits_file', type=str,
        default='data_splits.json',
        help='Filename of the splits JSON inside data_save_folder '
             '(default: data_splits.json).'
    )
    main(parser.parse_args())
