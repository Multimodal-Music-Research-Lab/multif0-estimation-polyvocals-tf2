'''
Config file
Dataset structure is created here
'''

import itertools
import os
import numpy as np


'''Paths
'''
_DAST_BASE = os.environ['DAST_BASE']
audio_save_folder = f'{_DAST_BASE}/data/processed/training/audiomixtures/'
data_save_folder  = f'{_DAST_BASE}/data/processed/training/features_targets/'
exper_output      = f'{_DAST_BASE}/data/processed/training/experiment_output/'


# audio folders
csd_folder        = f'{_DAST_BASE}/data/raw/ChoralSingingDataset/'
ecs_folder        = f'{_DAST_BASE}/data/raw/EsmucChoirDataset_v1.0.0/'
dcs_folder_audio  = f'{_DAST_BASE}/data/raw/DagstuhlChoirSet_V1.2.3/audio_wav_22050_mono/'
dcs_folder_annot  = f'{_DAST_BASE}/data/raw/DagstuhlChoirSet_V1.2.3/annotations_csv_F0_PYIN/'


'''All variables and parameters related to the dataset creation
'''

dataset = dict()
dataset['CSD'] = dict()
dataset['DCS'] = dict()
dataset['ECS'] = dict()

augmentation_idx = ['0_', '1_', '2_', '3_', '4_']

'''Choral Singing Dataset
'''
csd_songs = ['CSD_ER', 'CSD_LI', 'CSD_ND']

singers_csd = [
        'soprano_1', 'soprano_2', 'soprano_3', 'soprano_4',
        'alto_1', 'alto_2', 'alto_3', 'alto_4',
        'tenor_1', 'tenor_2', 'tenor_3', 'tenor_4',
        'bass_1', 'bass_2', 'bass_3', 'bass_4']

dataset['CSD']['songs'] = []
for song in csd_songs:
    for idx in augmentation_idx:
        dataset['CSD']['songs'].append(idx + song)

dataset['CSD']['singers'] = singers_csd

singers_per_section = 4
x = np.arange(1, singers_per_section + 1).astype(np.int32)
combos = [p for p in itertools.product(x, repeat=4)]
dataset['CSD']['combos'] = combos



'''ESMUC ChoralSet (divided by songs for convenience)
'''

''' Der Greis
'''

ecs_dg = ['DG_FT_take1', 'DG_FT_take2', 'DG_FT_take3', 'DG_FT_take4']

singers_ecs_dg = [
    'S1', 'S2', 'S3', 'S4',
    'A1', 'A2', 'A3',
    'T1', 'T2', 'T3',
    'B1', 'B2']

dataset['ECS']['DG_singers'] = singers_ecs_dg

dataset['ECS']['DG_songs'] = []
for song in ecs_dg:
    for idx in augmentation_idx:
        dataset['ECS']['DG_songs'].append(idx + song)

sop = np.arange(1, 4 + 1)
alto = np.arange(1, 3 + 1)
ten = np.arange(1, 3 + 1)
bass = np.arange(1, 2 + 1)

combos = []
for s in sop:
    for a in alto:
        for t in ten:
            for b in bass:
                combos.append(np.array([s, a, t, b]))

combos = np.array(combos, dtype=np.int32)
dataset['ECS']['DG_combos'] = combos

''' Die Himmel
'''

ecs_dh = ['DH1_FT_take1', 'DH2_FT_take1']

singers_ecs_dh = [
    'S1', 'S2', 'S3', 'S4', 'S5',
    'A1', 'A2',
    'T1', 'T2', 'T3',
    'B1', 'B2'
]

dataset['ECS']['DH_singers'] = singers_ecs_dh

dataset['ECS']['DH_songs'] = []
for song in ecs_dh:
    for idx in augmentation_idx:
        dataset['ECS']['DH_songs'].append(idx + song)

sop = np.arange(1, 5 + 1)
alto = np.arange(1, 2 + 1)
ten = np.arange(1, 3 + 1)
bass = np.arange(1, 2 + 1)

combos = []
for s in sop:
    for a in alto:
        for t in ten:
            for b in bass:
                combos.append(np.array([s, a, t, b]))

combos = np.array(combos, dtype=np.int32)
dataset['ECS']['DH_combos'] = combos


''' Seele Christi
'''

ecs_sc = ['SC1_FT_take1', 'SC1_FT_take2', 'SC1_FT_take3',
          'SC2_FT_take1', 'SC2_FT_take2', 'SC2_FT_take3',
          'SC3_FT_take1', 'SC3_FT_take2']


singers_ecs_sc = [
    'S1', 'S2', 'S3', 'S4', 'S5',
    'A1', 'A2',
    'T1', 'T2', 'T3',
    'B1', 'B2'
]

dataset['ECS']['SC_singers'] = singers_ecs_sc

dataset['ECS']['SC_songs'] = []
for song in ecs_sc:
    for idx in augmentation_idx:
        dataset['ECS']['SC_songs'].append(idx + song)

sop = np.arange(1, 5 + 1)
alto = np.arange(1, 2 + 1)
ten = np.arange(1, 3 + 1)
bass = np.arange(1, 2 + 1)

combos = []
for s in sop:
    for a in alto:
        for t in ten:
            for b in bass:
                combos.append(np.array([s, a, t, b]))

combos = np.array(combos, dtype=np.int32)
dataset['ECS']['SC_combos'] = combos


'''Dagstuhl ChoirSet
'''

dcs_settings = ['All', 'QuartetA', 'QuartetB']

singers_QB = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']
singers_QA = ['S2_DYN', 'A1_DYN', 'T1_DYN', 'B1_DYN']
singers_all_dyn = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']

# no combos because these are quartets (inside the full choir)
dcs_songs_fc = ['DCS_LI_FullChoir_Take01', 'DCS_LI_FullChoir_Take02', 'DCS_LI_FullChoir_Take03']
dcs_singers_fc = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']

dataset['DCS']['FC_singers'] = dcs_singers_fc
dataset['DCS']['FC_songs'] = []
for song in dcs_songs_fc:
    for idx in augmentation_idx:
        dataset['DCS']['FC_songs'].append(idx + song)



dcs_songs_qa = ['DCS_LI_QuartetA_Take01', 'DCS_LI_QuartetA_Take02', 'DCS_LI_QuartetA_Take03',
                              'DCS_LI_QuartetA_Take04', 'DCS_LI_QuartetA_Take05', 'DCS_LI_QuartetA_Take06']
dcs_singers_qa = ['S2_DYN', 'A1_DYN', 'T1_DYN', 'B1_DYN']

dataset['DCS']['QA_singers'] = dcs_singers_qa
dataset['DCS']['QA_songs'] = []

for song in dcs_songs_qa:
    for idx in augmentation_idx:
        dataset['DCS']['QA_songs'].append(idx + song)


dcs_songs_qb = ['DCS_LI_QuartetB_Take01', 'DCS_LI_QuartetB_Take02', 'DCS_LI_QuartetB_Take03',
               'DCS_LI_QuartetB_Take04', 'DCS_LI_QuartetB_Take05']
dcs_singers_qb = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']

dataset['DCS']['QB_singers'] = dcs_singers_qb
dataset['DCS']['QB_songs'] = []
for song in dcs_songs_qb:
    for idx in augmentation_idx:
        dataset['DCS']['QB_songs'].append(idx + song)


'''Training parameters
'''
SAMPLES_PER_EPOCH = 3072
NB_EPOCHS = 100
NB_VAL_SAMPLES = 256
