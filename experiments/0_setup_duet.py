'''
This script creates 2-voice (duet) audio mixtures for all working datasets.
All possible voice-pair combinations are generated per song, appropriate
for training a 2-voice F0 estimation model (model3).

Disk-space workarounds
----------------------
Workaround 1 - FLAC output (default):
    Output filenames use .flac instead of .wav (~40-60% smaller, lossless).
    pumpp reads FLAC natively; no changes needed downstream.

Workaround 2 - Batched section-by-section processing:
    Pass `sections` to `create_full_dataset_mixes()` (or use --sections on the
    CLI) to restrict which dataset sections are processed in a single run.
    Run 1_prep.py after each section, then delete the mix files before
    the next section.  See experiments/run_pipeline.py for the wrapper.
    Valid section names: CSD, ECS_DG, ECS_DH, ECS_SC, DCS

Workaround 3 - On-the-fly mixing (skip writing mix files entirely):
    Set `compute_audio_mix=False`.  The metadata JSON records `audio_folder`
    and `source_files` for each entry so that utils.compute_multif0_complete()
    can mix sources in-memory via librosa when the mix file is absent.
    Use --no-audio-mix on the CLI.
'''

import argparse
import itertools
import sox
import os

from experiments import config
import utils


# All available section identifiers (used by the --sections CLI flag)
ALL_SECTIONS = ['CSD', 'ECS_DG', 'ECS_DH', 'ECS_SC', 'DCS']


def combine_audio_files(params):

    cmb = sox.Combiner()
    cmb.convert(samplerate=22050)
    cmb.build(
        [os.path.join(params['audio_folder'], fn) for fn in params['filenames']],
        os.path.join(config.audio_save_folder, params['output_fname']), 'mix')


def create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder,
                      audio_folder=None, source_files=None):
    """Store metadata for one mix.

    audio_folder and source_files are optional extras used by Workaround 3
    (on-the-fly mixing): they let utils.compute_multif0_complete() recreate
    the mix from source WAVs without needing the pre-generated mix file on disk.
    """
    diction[audiofname] = dict()
    diction[audiofname]['audiopath'] = audiopath
    diction[audiofname]['annot_files'] = annot_files
    diction[audiofname]['annot_folder'] = annot_folder
    if audio_folder is not None:
        diction[audiofname]['audio_folder'] = audio_folder
    if source_files is not None:
        diction[audiofname]['source_files'] = source_files

    return diction


def create_full_dataset_mixes(dataset, mixes_wavpath, reverb=True,
                               compute_audio_mix=True, compute_metadata=True,
                               sections=None):
    """Generate mix audio files and/or metadata for all requested sections.

    Parameters
    ----------
    sections : list of str or None
        Subset of ALL_SECTIONS to process.  None means all sections.
        Example: ['CSD', 'ECS_DG'] processes only CSD and ECS Der Greis.
    """

    if sections is None:
        sections = ALL_SECTIONS

    mtracks = dict()

    # ------------ Process Choral Singing Dataset ------------ #

    if 'CSD' in sections:

        print("Processing Choral Singing Dataset...")

        csd_voices = ['soprano', 'alto', 'tenor', 'bass']
        csd_singer_ranges = [range(1, 5), range(1, 5), range(1, 5), range(1, 5)]

        for song in dataset['CSD']['songs']:
            for vi, vj in itertools.combinations(range(4), 2):
                for si in csd_singer_ranges[vi]:
                    for sj in csd_singer_ranges[vj]:

                        params = {}
                        params['audio_folder'] = config.csd_folder
                        params['annot_folder'] = config.csd_folder
                        params['sr'] = 44100
                        params['reverb'] = True

                        params['filenames'] = [
                            '{}_{}_{}.wav'.format(song, csd_voices[vi], si),
                            '{}_{}_{}.wav'.format(song, csd_voices[vj], sj),
                        ]

                        params['output_fname'] = '{}_{}{}_{}{}.flac'.format(
                            song, csd_voices[vi][0], si, csd_voices[vj][0], sj)

                        if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
                            combine_audio_files(params)

                        if compute_metadata:
                            annotation_files = [
                                '{}_{}_{}.jams'.format(song, csd_voices[vi], si),
                                '{}_{}_{}.jams'.format(song, csd_voices[vj], sj),
                            ]

                            mtracks = create_dict_entry(
                                mtracks, mixes_wavpath, params['output_fname'],
                                annotation_files, params['annot_folder'],
                                audio_folder=params['audio_folder'],
                                source_files=params['filenames'])

                            if reverb:
                                for idx, annot in enumerate(annotation_files):
                                    utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])

            print("Mixtures for {} have been created.".format(song))

    else:
        print("Skipping Choral Singing Dataset.")

    # ------------ Process ESMUC ChoralSet ------------ #

    if 'ECS_DG' in sections or 'ECS_DH' in sections or 'ECS_SC' in sections:
        print("Processing ESMUC Choral Dataset...")

    # Der Greis
    if 'ECS_DG' in sections:

        dg_voice_labels = ['S', 'A', 'T', 'B']
        dg_singer_ranges = [range(1, 5), range(1, 4), range(1, 4), range(1, 3)]

        for song in dataset['ECS']['DG_songs']:
            for vi, vj in itertools.combinations(range(4), 2):
                for si in dg_singer_ranges[vi]:
                    for sj in dg_singer_ranges[vj]:

                        params = {}
                        params['audio_folder'] = config.ecs_folder
                        params['annot_folder'] = config.ecs_folder
                        params['sr'] = 22050
                        params['reverb'] = True

                        params['filenames'] = [
                            '{}_{}{}.wav'.format(song, dg_voice_labels[vi], si),
                            '{}_{}{}.wav'.format(song, dg_voice_labels[vj], sj),
                        ]

                        params['output_fname'] = '{}_{}{}_{}{}.flac'.format(
                            song, dg_voice_labels[vi], si, dg_voice_labels[vj], sj)

                        if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
                            combine_audio_files(params)

                        if compute_metadata:
                            annotation_files = [
                                '{}_{}{}.jams'.format(song, dg_voice_labels[vi], si),
                                '{}_{}{}.jams'.format(song, dg_voice_labels[vj], sj),
                            ]

                            mtracks = create_dict_entry(
                                mtracks, mixes_wavpath, params['output_fname'],
                                annotation_files, params['annot_folder'],
                                audio_folder=params['audio_folder'],
                                source_files=params['filenames'])

                            if reverb:
                                for idx, annot in enumerate(annotation_files):
                                    utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])

            print('{} duets mixed and exported'.format(song))

    else:
        print("Skipping ECS Der Greis.")

    # Die Himmel
    if 'ECS_DH' in sections:

        dh_voice_offsets = [0, 5, 7, 10]
        dh_singer_counts = [5, 2, 3, 2]

        for song in dataset['ECS']['DH_songs']:
            for vi, vj in itertools.combinations(range(4), 2):
                for si in range(1, dh_singer_counts[vi] + 1):
                    for sj in range(1, dh_singer_counts[vj] + 1):

                        singer_i = dataset['ECS']['DH_singers'][dh_voice_offsets[vi] + si - 1]
                        singer_j = dataset['ECS']['DH_singers'][dh_voice_offsets[vj] + sj - 1]

                        params = {}
                        params['audio_folder'] = config.ecs_folder
                        params['annot_folder'] = config.ecs_folder
                        params['sr'] = 22050
                        params['reverb'] = True

                        params['filenames'] = [
                            '{}_{}.wav'.format(song, singer_i),
                            '{}_{}.wav'.format(song, singer_j),
                        ]

                        params['output_fname'] = '{}_{}_{}.flac'.format(song, singer_i, singer_j)

                        if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
                            combine_audio_files(params)

                        if compute_metadata:
                            annotation_files = [
                                '{}_{}.jams'.format(song, singer_i),
                                '{}_{}.jams'.format(song, singer_j),
                            ]

                            mtracks = create_dict_entry(
                                mtracks, mixes_wavpath, params['output_fname'],
                                annotation_files, params['annot_folder'],
                                audio_folder=params['audio_folder'],
                                source_files=params['filenames'])

                            if reverb:
                                for idx, annot in enumerate(annotation_files):
                                    utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])

            print('{} duets mixed and exported'.format(song))

    else:
        print("Skipping ECS Die Himmel.")

    # Seele Christi
    if 'ECS_SC' in sections:

        sc_voice_labels = ['S', 'A', 'T', 'B']
        sc_singer_ranges = [range(1, 6), range(1, 3), range(1, 4), range(1, 3)]

        for song in dataset['ECS']['SC_songs']:
            for vi, vj in itertools.combinations(range(4), 2):
                for si in sc_singer_ranges[vi]:
                    for sj in sc_singer_ranges[vj]:

                        params = {}
                        params['audio_folder'] = config.ecs_folder
                        params['annot_folder'] = config.ecs_folder
                        params['sr'] = 22050
                        params['reverb'] = True

                        params['filenames'] = [
                            '{}_{}{}.wav'.format(song, sc_voice_labels[vi], si),
                            '{}_{}{}.wav'.format(song, sc_voice_labels[vj], sj),
                        ]

                        params['output_fname'] = '{}_{}{}_{}{}.flac'.format(
                            song, sc_voice_labels[vi], si, sc_voice_labels[vj], sj)

                        if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
                            combine_audio_files(params)

                        if compute_metadata:
                            annotation_files = [
                                '{}_{}{}.jams'.format(song, sc_voice_labels[vi], si),
                                '{}_{}{}.jams'.format(song, sc_voice_labels[vj], sj),
                            ]

                            mtracks = create_dict_entry(
                                mtracks, mixes_wavpath, params['output_fname'],
                                annotation_files, params['annot_folder'],
                                audio_folder=params['audio_folder'],
                                source_files=params['filenames'])

                            if reverb:
                                for idx, annot in enumerate(annotation_files):
                                    utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])

            print('{} duets mixed and exported'.format(song))

    else:
        print("Skipping ECS Seele Christi.")

    # ------------ Process Dagstuhl ChoirSet ------------ #

    if 'DCS' in sections:

        print("Processing Dagstuhl ChoirSet...")

        # Full Choir setting
        for song in dataset['DCS']['FC_songs']:
            singers = dataset['DCS']['FC_singers']
            for i, j in itertools.combinations(range(4), 2):

                params = {}
                params['audio_folder'] = config.dcs_folder_audio
                params['annot_folder'] = config.dcs_folder_annot
                params['sr'] = 22050
                params['reverb'] = True

                params['filenames'] = [
                    '{}_{}.wav'.format(song, singers[i]),
                    '{}_{}.wav'.format(song, singers[j]),
                ]

                params['output_fname'] = '{}_{}_{}.flac'.format(song, singers[i], singers[j])

                if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
                    combine_audio_files(params)

                if compute_metadata:
                    annotation_files = [
                        '{}_{}.jams'.format(song, singers[i]),
                        '{}_{}.jams'.format(song, singers[j]),
                    ]

                    mtracks = create_dict_entry(
                        mtracks, mixes_wavpath, params['output_fname'],
                        annotation_files, params['annot_folder'],
                        audio_folder=params['audio_folder'],
                        source_files=params['filenames'])

                    if reverb:
                        for idx, annot in enumerate(annotation_files):
                            utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])

            print('{} duets mixed and exported'.format(song))

        # Quartet A setting
        for song in dataset['DCS']['QA_songs']:
            singers = dataset['DCS']['QA_singers']
            for i, j in itertools.combinations(range(4), 2):

                params = {}
                params['audio_folder'] = config.dcs_folder_audio
                params['annot_folder'] = config.dcs_folder_annot
                params['sr'] = 22050
                params['reverb'] = True

                params['filenames'] = [
                    '{}_{}.wav'.format(song, singers[i]),
                    '{}_{}.wav'.format(song, singers[j]),
                ]

                params['output_fname'] = '{}_{}_{}.flac'.format(song, singers[i], singers[j])

                if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
                    combine_audio_files(params)

                if compute_metadata:
                    annotation_files = [
                        '{}_{}.jams'.format(song, singers[i]),
                        '{}_{}.jams'.format(song, singers[j]),
                    ]

                    mtracks = create_dict_entry(
                        mtracks, mixes_wavpath, params['output_fname'],
                        annotation_files, params['annot_folder'],
                        audio_folder=params['audio_folder'],
                        source_files=params['filenames'])

                    if reverb:
                        for idx, annot in enumerate(annotation_files):
                            utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])

            print('{} duets mixed and exported'.format(song))

        # Quartet B setting
        for song in dataset['DCS']['QB_songs']:
            singers = dataset['DCS']['QB_singers']
            for i, j in itertools.combinations(range(4), 2):

                params = {}
                params['audio_folder'] = config.dcs_folder_audio
                params['annot_folder'] = config.dcs_folder_annot
                params['sr'] = 22050
                params['reverb'] = True

                params['filenames'] = [
                    '{}_{}.wav'.format(song, singers[i]),
                    '{}_{}.wav'.format(song, singers[j]),
                ]

                params['output_fname'] = '{}_{}_{}.flac'.format(song, singers[i], singers[j])

                if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
                    combine_audio_files(params)

                if compute_metadata:
                    annotation_files = [
                        '{}_{}.jams'.format(song, singers[i]),
                        '{}_{}.jams'.format(song, singers[j]),
                    ]

                    mtracks = create_dict_entry(
                        mtracks, mixes_wavpath, params['output_fname'],
                        annotation_files, params['annot_folder'],
                        audio_folder=params['audio_folder'],
                        source_files=params['filenames'])

                    if reverb:
                        for idx, annot in enumerate(annotation_files):
                            utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])

            print('{} duets mixed and exported'.format(song))

    else:
        print("Skipping Dagstuhl ChoirSet.")

    # Store the metadata file (always saved to the full path so batched runs
    # can accumulate entries across section calls using merge logic in the
    # orchestration script; see experiments/run_pipeline.py)
    if compute_metadata:
        utils.save_json_data(mtracks, os.path.join(mixes_wavpath, 'mtracks_info.json'))

    return mtracks


def main():

    parser = argparse.ArgumentParser(
        description="Create duet audio mixes and metadata for multif0 training.")
    parser.add_argument(
        '--sections', nargs='+', default=None,
        choices=ALL_SECTIONS,
        metavar='SECTION',
        help=('Dataset sections to process (default: all). '
              'Valid choices: ' + ', '.join(ALL_SECTIONS)))
    parser.add_argument(
        '--no-audio-mix', dest='compute_audio_mix', action='store_false',
        help='Skip writing mix audio files (Workaround 3: on-the-fly mixing).')
    parser.set_defaults(compute_audio_mix=True)
    args = parser.parse_args()

    # load the dataset info
    dataset = config.dataset

    print("Dataset info loaded.")

    # use the dataset information to create audio mixtures and annotations
    create_full_dataset_mixes(
        dataset, config.audio_save_folder,
        reverb=False,
        compute_audio_mix=args.compute_audio_mix,
        compute_metadata=True,
        sections=args.sections)


if __name__ == '__main__':

    main()
