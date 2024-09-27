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
from RF_Diffraction_utils import *


print('starting')

with open('Final_0_05_spacing_radial_dataframe_100423.pkl', 'rb') as f:
    radial_dataframe = pickle.load(f)
    
radial_dataframe_no_triclinic = radial_dataframe.drop(radial_dataframe.loc[radial_dataframe['crystal system'] == 'triclinic'].index)
radial_dataframe = radial_dataframe_no_triclinic
radial_dataframe.reset_index(inplace = True)

try:
    with open('Model_data/Crystal_sys_inputs/train_df.pkl', 'rb') as f:
        train_df = pickle.load(f) 
    
    with open('Model_data/Crystal_sys_inputs/test_df.pkl', 'rb') as f:
        test_df = pickle.load(f) 
    
except:
    crystal_sys_alph = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal']

    full_ids_train = []
    full_ids_test = []
    for cry in crystal_sys_alph:
        if os.path.exists('Model_data/Crystal_sys_dataframes/full_data_'+cry+'.pkl'):
            print(cry + ' full data df exists')
        else:
            subdf = radial_dataframe.loc[radial_dataframe['crystal system'] == cry]
            subdf.reset_index(inplace = True)
            subdf = subdf.drop(['index'], axis = 1)
            subdf.to_pickle('Model_data/Crystal_sys_dataframes/full_data_'+cry+'.pkl')

    for system in crystal_sys_alph:
        with open('Model_data/Crystal_sys_dataframes/full_data_'+system+'.pkl', 'rb') as f: 
                sys_df = pickle.load(f)
        train_test_ids = lattice_prepare_training_and_test(sys_df, split_by_material = True, use_scaled_cols = False, return_ids = True)
        for mat1 in train_test_ids[0]:
            full_ids_train.append(mat1)
        for mat2 in train_test_ids[1]:
            full_ids_test.append(mat2)
        
    test_df = radial_dataframe.loc[radial_dataframe['mat_id'].isin(full_ids_test)]
    test_df.drop('index', axis = 1, inplace = True)
    test_df.reset_index(inplace = True)
    test_df.to_pickle('Model_data/Crystal_sys_inputs/test_df.pkl')
    
    train_df = radial_dataframe.loc[radial_dataframe['mat_id'].isin(full_ids_train)]
    train_df.drop('index', axis = 1, inplace = True)
    train_df.reset_index(inplace = True)
    train_df.to_pickle('Model_data/Crystal_sys_inputs/train_df.pkl')
    
    print('train test split generated')
    
try:
    radial_inputs_0 = joblib.load('Model_data/Crystal_sys_inputs/radial_inputs_0.joblib')
    radial_inputs_1 = joblib.load('Model_data/Crystal_sys_inputs/radial_inputs_1.joblib')

    with open('Model_data/Crystal_sys_inputs/radial_inputs_2.pkl', 'rb') as f:
        radial_inputs_2 = pickle.load(f)
    with open('Model_data/Crystal_sys_inputs/radial_inputs_3.pkl', 'rb') as f:
        radial_inputs_3 = pickle.load(f)
        
    radial_train_ids = joblib.load('Model_data/Crystal_sys_inputs/radial_inputs_0_5.joblib')
    radial_test_ids = joblib.load('Model_data/Crystal_sys_inputs/radial_inputs_1_5.joblib')

except: 
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

    joblib.dump(radial_inputs_0, 'Model_data/Crystal_sys_inputs/radial_inputs_0.joblib')
    joblib.dump(radial_inputs_0_5, 'Model_data/Crystal_sys_inputs/radial_inputs_0_5.joblib')
    joblib.dump(radial_inputs_1, 'Model_data/Crystal_sys_inputs/radial_inputs_1.joblib')
    joblib.dump(radial_inputs_1_5, 'Model_data/Crystal_sys_inputs/radial_inputs_1_5.joblib')
    radial_inputs_2.to_pickle('Model_data/Crystal_sys_inputs/radial_inputs_2.pkl')
    radial_inputs_3.to_pickle('Model_data/Crystal_sys_inputs/radial_inputs_3.pkl')
    
    radial_train_ids = radial_inputs_0_5
    radial_test_ids = radial_inputs_1_5

    print('datasets saved')



if os.path.exists('Model_data/Crystal_sys_outputs/crystal_system_model.joblib'):
    print('loading crystal system model')
    rf_output_radial = joblib.load('Model_data/Crystal_sys_outputs/crystal_system_model.joblib')
    print('crystal system model loaded')

else:
    rf_output_radial = rf_diffraction(radial_inputs_2, radial_inputs_3, radial_inputs_0, radial_inputs_1, num_trees = 80, max_depth = 40,
                                      index_to_use = 9, max_features = 'auto')
    joblib.dump(rf_output_radial, 'Model_data/Crystal_sys_outputs/crystal_system_model.joblib', compress = 3)

if os.path.exists('Model_data/Crystal_sys_outputs/output_df_radial.joblib'):
    print('loading crystal system output df')
    output_df_radial = joblib.load('Model_data/Crystal_sys_outputs/output_df_radial.joblib')
    print('loading crystal system output df')

else:
    test_indicies = radial_dataframe.loc[radial_dataframe['mat_id'].isin(radial_test_ids)].index
    output_df_radial = show_cm_and_uncertianty(rf_output_radial[0], rf_output_radial[1], rf_output_radial[2], radial_dataframe, 
                                    radial_inputs_3, test_indicies, type_to_show = 'crystal system', 
                                           point_group_df = None,  predicted_quant ='crystal system')
    joblib.dump(output_df_radial, 'Model_data/Crystal_sys_outputs/output_df_radial.joblib')

# visualize_predictions_by_material(output_df_radial, random_inds = 100, show_individual = False)



crystal_sys_alph = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal']

for cry in crystal_sys_alph:
    if os.path.exists('Model_data/Crystal_sys_dataframes/full_data_'+cry+'.pkl'):
        print(cry + ' full data df exists')
    else:
        subdf = radial_dataframe.loc[radial_dataframe['crystal system'] == cry]
        subdf.reset_index(inplace = True)
        subdf = subdf.drop(['index'], axis = 1)
        subdf.to_pickle('Model_data/Crystal_sys_dataframes/full_data_'+cry+'.pkl')
    
for system in crystal_sys_alph:
    print(system)
    with open('Model_data/Crystal_sys_dataframes/full_data_'+system+'.pkl', 'rb') as f:
        sys_df = pickle.load(f)

    if os.path.exists('Model_data/Lattice_inputs_and_outputs/radial_'+system+'_df_w_lattice.joblib'):
        print(system + ' lattice full data df exists')

    else:
        print(len(sys_df.mat_id.unique()))
        sys_df_w_lattice = update_df_lattice(sys_df)
        joblib.dump(sys_df_w_lattice, 'Model_data/Lattice_inputs_and_outputs/radial_'+system+'_df_w_lattice.joblib')
    
if os.path.exists('Model_data/Lattice_inputs_and_outputs/SCALED_radial_orthorhombic_df_w_lattice.joblib'):
    print(system + ' SCALED lattice full data df exists')

else:
    ortho_df = joblib.load('Model_data/Lattice_inputs_and_outputs/radial_orthorhombic_df_w_lattice.joblib')
    ortho_df_scaled = scale_df_lattice(ortho_df)
    joblib.dump(ortho_df_scaled, 'Model_data/Lattice_inputs_and_outputs/SCALED_radial_orthorhombic_df_w_lattice.joblib')

for system in crystal_sys_alph:
    if os.path.exists('Model_data/Lattice_inputs_and_outputs/'+system+'_lattice_model.joblib'):
        print(system+'_lattice_model.joblib exists')
        pass
    else:
        if system != 'orthorhombic':
            lattice_df = joblib.load('Model_data/Lattice_inputs_and_outputs/radial_'+system+'_df_w_lattice.joblib')
            lattice_radial_inputs = lattice_prepare_training_and_test(lattice_df, split_by_material = True, use_scaled_cols = False)
        else:
            lattice_df = joblib.load('Model_data/Lattice_inputs_and_outputs/SCALED_radial_'+system+'_df_w_lattice.joblib')
            lattice_radial_inputs = lattice_prepare_training_and_test(lattice_df, split_by_material = True, use_scaled_cols = True)

        lattice_rf_output = lattice_rf_diffraction(lattice_radial_inputs[2], lattice_radial_inputs[3], lattice_radial_inputs[0], lattice_radial_inputs[1], 
                                                   num_trees = 80, max_depth = 100, max_features = 'sqrt')


        new_input_train = []
        for inp in lattice_radial_inputs[0]:
            new_input_train.append(np.asarray(inp))
        new_input_train = np.asarray(new_input_train)
        joblib.dump(new_input_train, 'Model_data/Lattice_inputs_and_outputs/'+system+'_lattice_input_train_0.joblib', compress = 0)

        new_input_test = []
        for test in lattice_radial_inputs[1]:
            new_input_test.append(np.asarray(test))
        new_input_test = np.asarray(new_input_test)
        joblib.dump(new_input_test,  'Model_data/Lattice_inputs_and_outputs/'+system+'_lattice_input_test_1.joblib', compress = 0)

        lattice_radial_inputs[2].to_pickle('Model_data/Lattice_inputs_and_outputs/'+system+'_lattice_inputs_labels_train_2.pkl')
        lattice_radial_inputs[3].to_pickle('Model_data/Lattice_inputs_and_outputs/'+system+'_lattice_labels_test_3.pkl')

        joblib.dump(lattice_rf_output, 'Model_data/Lattice_inputs_and_outputs/'+system+'_lattice_model.joblib')
    
for system in crystal_sys_alph:
    space_group_df = joblib.load('Model_data/Lattice_inputs_and_outputs/radial_'+system+'_df_w_lattice.joblib')
    space_group_radial_inputs = prepare_training_and_test(space_group_df, split_by_material = True, quantity = 'space_group_first', column_to_use = '200_ang')
    
    radial_inputs_0 = space_group_radial_inputs[0]
    radial_inputs_1 = space_group_radial_inputs[1] 
    radial_inputs_2 = space_group_radial_inputs[2] 
    radial_inputs_3 = space_group_radial_inputs[3] 
    
    rf_output_radial = rf_diffraction(radial_inputs_2, radial_inputs_3, radial_inputs_0, radial_inputs_1, num_trees = 128, max_depth = 30,
                                  index_to_use = 9, max_features = 'auto')
    
    new_input_train = []
    for inp in space_group_radial_inputs[0]:
        new_input_train.append(np.asarray(inp))
    new_input_train = np.asarray(new_input_train)
    joblib.dump(new_input_train, 'Model_data/Space_group_inputs_and_outputs/'+system+'_space_groups_input_train_0.joblib', compress = 0)

    new_input_test = []
    for test in space_group_radial_inputs[1]:
        new_input_test.append(np.asarray(test))
    new_input_test = np.asarray(new_input_test)
    joblib.dump(new_input_test, 'Model_data/Space_group_inputs_and_outputs/'+system+'_space_groups_input_test_1.joblib', compress = 0)

    space_group_radial_inputs[2].to_pickle('Model_data/Space_group_inputs_and_outputs/'+system+'_space_groups_inputs_labels_train_2.pkl')
    space_group_radial_inputs[3].to_pickle('Model_data/Space_group_inputs_and_outputs/'+system+'_space_groups_labels_test_3.pkl')
    
    joblib.dump(rf_output_radial, 'Model_data/Space_group_inputs_and_outputs/'+system+'_space_group_model.joblib')
