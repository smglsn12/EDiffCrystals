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
from sklearn.metrics import log_loss
from matplotlib.ticker import PercentFormatter
import warnings
from matplotlib.cm import ScalarMappable
import matplotlib
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.metrics import r2_score
from RF_Functions import *
import math
from RF_Diffraction_OOP import *
import matplotlib as mpl
mpl.style.use('classic')
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
plt.rcdefaults()
plt.rcParams['pdf.fonttype'] = 'truetype'

def add_lattice_results_to_output_df(df):
    new_rows = []
    for i in range(0, len(df)):
        new_rows.append(None)
    df['pred_cry_sys_full_predictions_a'] = new_rows
    df['pred_cry_sys_full_predictions_b'] = new_rows
    df['pred_cry_sys_full_predictions_c'] = new_rows

    df['true_cry_sys_full_predictions_a'] = new_rows
    df['true_cry_sys_full_predictions_b'] = new_rows
    df['true_cry_sys_full_predictions_c'] = new_rows
    
    print('df ready')
    
    for cry_sys in ['trigonal', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'cubic']:        
        print(cry_sys)
        rf_model = joblib.load('Model_data/Lattice_inputs_and_outputs/' +cry_sys+'_lattice_model.joblib')[2]
        print('model loaded')
        trees = rf_model.estimators_
        # for k in range(0, len(df)):
        for k in range(0, len(df)):
            row = df.iloc[k]
            if row['Predictions Crystal System'] == cry_sys:
                cry_sys_vals = np.asarray(row.radial_200_ang_Colin_basis) 
                input_train_temp = flatten(cry_sys_vals)

                
                abs_i = np.abs(input_train_temp)
                angle_i = np.angle(input_train_temp)
                        # print(temp)
                input_train = []
                for i in range(0, len(abs_i)):
                    input_train.append(abs_i[i])
                    input_train.append(angle_i[i])

                cry_sys_use = np.asarray(input_train).reshape(1, -1)
                # print(cry_sys_use)
                predictions_full = []
                for tree in trees:
                    predictions_full.append(tree.predict(cry_sys_use))

                predictions_ordered = np.asarray(predictions_full).T
                # print(predictions_ordered[0])
                df.at[k, 'pred_cry_sys_full_predictions_a'] = predictions_ordered[0][0]
                df.at[k, 'pred_cry_sys_full_predictions_b'] = predictions_ordered[1][0]
                df.at[k, 'pred_cry_sys_full_predictions_c'] = predictions_ordered[2][0]
                
            if row['True Values Crystal System'] == cry_sys:
                cry_sys_vals = np.asarray(row.radial_200_ang_Colin_basis) 
                input_train_temp = flatten(cry_sys_vals)

                
                abs_i = np.abs(input_train_temp)
                angle_i = np.angle(input_train_temp)
                        # print(temp)
                input_train = []
                for i in range(0, len(abs_i)):
                    input_train.append(abs_i[i])
                    input_train.append(angle_i[i])

                cry_sys_use = np.asarray(input_train).reshape(1, -1)
                # print(cry_sys_use)
                predictions_full = []
                for tree in trees:
                    predictions_full.append(tree.predict(cry_sys_use))

                predictions_ordered = np.asarray(predictions_full).T
                # print(predictions_ordered[0])
                df.at[k, 'true_cry_sys_full_predictions_a'] = predictions_ordered[0][0]
                df.at[k, 'true_cry_sys_full_predictions_b'] = predictions_ordered[1][0]
                df.at[k, 'true_cry_sys_full_predictions_c'] = predictions_ordered[2][0]


    return df

rf_diff_obj = RF_Diffraction_model('Final_0_05_spacing_radial_dataframe_100423.pkl', 
                                   ['Model_data/Lattice_inputs_and_outputs/radial_cubic_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/radial_monoclinic_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_hexagonal_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/SCALED_radial_orthorhombic_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_tetragonal_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/radial_trigonal_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_triclinic_df_w_lattice.joblib'])

output_df_radial = joblib.load('Model_data/Crystal_sys_outputs/output_df_radial.joblib')
print('output_df_loaded')
output_df_temp = add_lattice_results_to_output_df(output_df_radial)
joblib.dump(output_df_temp, 'Model_data/Crystal_sys_outputs/Ortho_fixed_MONO_FIXED_LATTICE_UPDATED_output_df_radial.joblib')

