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
%load_ext autoreload
%autoreload 2
from RF_Diffraction_OOP import *
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
plt.rcdefaults()

rf_diff_obj = RF_Diffraction_model('Final_0_05_spacing_radial_dataframe_100423.pkl', 
                                   ['Model_data/Lattice_inputs_and_outputs/radial_cubic_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/radial_monoclinic_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_hexagonal_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/SCALED_radial_orthorhombic_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_tetragonal_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/radial_trigonal_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_triclinic_df_w_lattice.joblib'])

def r2(true, pred):
    print('starting r2')
    errors = []
    squares_true = []
    mean_true = np.mean(true)
    for i in range(0, len(true)):
        errors.append(np.square(true[i]-pred[i]))
        squares_true.append(np.square(true[i]-mean_true))
    sum_er = sum(errors)
    sum_squares = sum(squares_true)
    
    corr_coeff = sum_er/sum_squares 
    r2 = 1-corr_coeff 
    return r2


rf_diff_obj.loaded_submodels = [False, '']
# crystal_sys_alph = ['cubic', 'hexagonal', 'monoclinic', 'tetragonal', 'trigonal', 'orthorhombic']
crystal_sys_alph = ['monoclinic', 'orthorhombic']
maes = {}
medians = {}
for cry_sys in crystal_sys_alph:
    
    mae_list = []
    median_list = []
    if cry_sys == 'orthorhombic':
        params = ['a_sorted', 'b_sorted', 'c_sorted']
        rf_diff_obj.visualize_lattice_results_mat_id(cry_sys, material_id = 'all', show_plots = False, use_scaled = True)
    else:
        params = ['a', 'b', 'c']
        rf_diff_obj.visualize_lattice_results_mat_id(cry_sys, material_id = 'all', show_plots = False, use_scaled = False)
    for param in params:
        print(param + ' R2')
        fig = plt.figure(figsize=(8, 7))
        all_true = []
        all_pred = []
        
        for i in range(0, len(rf_diff_obj.full_out_df)):
            true_param = rf_diff_obj.full_out_df.iloc[i][param+'_true']
            pred_param = rf_diff_obj.full_out_df.iloc[i][param+'_mode']
            all_true.append(true_param)
            all_pred.append(pred_param)
                
        all_r2 = r2(all_true, all_pred)
        print(all_r2)
        if param == 'c' or param=='c_sorted':
            plt.xlabel('True c (Angstroms)', fontsize=32)
        if cry_sys == 'cubic':
            plt.ylabel('Predicted ' + param + ' (Angstroms)', fontsize=32)
        plt.scatter(all_true, all_pred, alpha = 1)
        plt.plot(np.linspace(min(all_true), max(all_true), 100), np.linspace(min(all_true), max(all_true), 100), linestyle = '--', color = 'k')
        plt.yticks(fontsize=32)
        plt.xticks(fontsize = 32)        
        plt.savefig('Figures/Figure S6/R2_plot_' + cry_sys + '_' + param + '.pdf', bbox_inches="tight", transparent=True)
        plt.show()
