import py4DSTEM
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from copy import deepcopy
import pickle
from pathlib import Path
from matplotlib.colors import hsv_to_rgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import joblib
import sys
import os 
from sklearn.metrics import log_loss
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import PointGroup, SpaceGroup
from mp_api.client import MPRester
from RF_Functions import *

print('starting')

with open('train_df.pkl', 'rb') as f:
    train_df = pickle.load(f) 
    
with open('test_df.pkl', 'rb') as f:
    test_df = pickle.load(f) 
    
print('datasets loaded')
    
new_input_train = []
for inp in train_df['radial_200_ang_Colin_basis'].to_numpy():
    new_input_train.append(np.asarray(inp))
new_input_train = np.asarray(new_input_train)

input_train_temp = []
for unit in new_input_train:
    input_train_temp.append(flatten(unit))

input_train = []
for row in input_train_temp:
    temp = []
    abs_i = np.abs(row)
    angle_i = np.angle(row)
    for i in range(0, len(abs_i)):
        temp.append(abs_i[i])
        temp.append(angle_i[i])
        # print(temp)
    input_train.append(temp)
    
ids_train = []
for inp_id in train_df['mat_id'].to_numpy():
    ids_train.append(inp_id)

print('training set')


new_input_test = []
for test in test_df['radial_200_ang_Colin_basis'].to_numpy():
    new_input_test.append(np.asarray(test))
new_input_test = np.asarray(new_input_test)

input_test_temp = []
for unit in new_input_test:
    input_test_temp.append(flatten(unit))

input_test = []
for row in input_test_temp:
    temp = []
    abs_i = np.abs(row)
    angle_i = np.angle(row)
    for i in range(0, len(abs_i)):
        temp.append(abs_i[i])
        temp.append(angle_i[i])
        # print(temp)
    input_test.append(temp)
    
ids_test = []
for test_id in test_df['mat_id'].to_numpy():
    ids_test.append(test_id)

print('test set')

radial_inputs_0 = np.asarray(input_train)
radial_inputs_0_5 = np.asarray(ids_train)
radial_inputs_1 = np.asarray(input_test)
radial_inputs_1_5 = np.asarray(ids_test)
radial_inputs_2 = train_df['crystal system']
radial_inputs_3 = test_df['crystal system']

print('dataset conversions')

joblib.dump(radial_inputs_0, 'radial_inputs_0.joblib')
joblib.dump(radial_inputs_0_5, 'radial_inputs_0_5.joblib')
joblib.dump(radial_inputs_1, 'radial_inputs_1.joblib')
joblib.dump(radial_inputs_1_5, 'radial_inputs_1_5.joblib')
radial_inputs_2.to_pickle('radial_inputs_2.pkl')
radial_inputs_3.to_pickle('radial_inputs_3.pkl')

print('datasets saved')

rf_output_radial = rf_diffraction(radial_inputs_2, radial_inputs_3, radial_inputs_0, radial_inputs_1, num_trees = 80, max_depth = 40,
                                      index_to_use = 9, max_features = 'auto')
joblib.dump(rf_output_radial, 'reconstructed_test_set_full_output_200_ang_full_dataset_80_trees_max_depth_40.joblib')
