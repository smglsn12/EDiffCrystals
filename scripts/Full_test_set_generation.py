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
from RF_Diffraction_OOP import *
from matplotlib.ticker import PercentFormatter

print('starting 090923')

rf_diff_obj = RF_Diffraction_model('symmetrized_full_data_only_radial_200_ang_pkls_0_36000.pkl', 
                                   ['Model_data/Lattice_inputs_and_outputs/radial_cubic_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/radial_monoclinic_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_hexagonal_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/SCALED_radial_orthorhombic_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_tetragonal_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/radial_trigonal_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_triclinic_df_w_lattice.joblib'])
rf_diff_obj.load_full_df() 
rf_diff_obj.condensed_output_df = joblib.load('Model_data/Crystal_sys_outputs/condensed_output_df.joblib')

for cry_sys in ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal']:
    true_cry_sys = rf_diff_obj.condensed_output_df.loc[rf_diff_obj.condensed_output_df['True Values Crystal System']== cry_sys]
    true_cry_sys_ids = true_cry_sys.mat_id.to_numpy()
    test_cry_sys_correct_predictions = rf_diff_obj.condensed_output_df.loc[(rf_diff_obj.condensed_output_df['Aggregate Predictions Crystal System'] == cry_sys) & 
                                                                           (rf_diff_obj.condensed_output_df['True Values Crystal System'] == cry_sys)]
    correct_cry_sys_ids = test_cry_sys_correct_predictions.mat_id.to_numpy()
    test_set_cry_sys_mispredictions = rf_diff_obj.condensed_output_df.loc[(rf_diff_obj.condensed_output_df['Aggregate Predictions Crystal System'] == cry_sys) & 
                                                                        (rf_diff_obj.condensed_output_df['True Values Crystal System'] != cry_sys)]
    wrong_cry_sys_prediction_ids = test_set_cry_sys_mispredictions.mat_id.to_numpy()
    
    if cry_sys == 'orthorhombic': 
        rf_diff_obj.visualize_lattice_results_mat_id(cry_sys, material_id = 'all', show_plots = False, use_scaled = True)
        
    else: 
        rf_diff_obj.visualize_lattice_results_mat_id(cry_sys, material_id = 'all', show_plots = False, use_scaled = False)

    lattice_test_cry_sys_ids = list(rf_diff_obj.full_out_df['material_id'])

    universal_cry_sys_test_ids = []
    for mat_id in correct_cry_sys_ids:
        if mat_id in lattice_test_cry_sys_ids:
            universal_cry_sys_test_ids.append(mat_id)
    print(len(lattice_test_cry_sys_ids))
    print(len(universal_cry_sys_test_ids))

    # for i in integers_to_use:
        # universal_cry_sys_test_ids.append(wrong_cry_sys_prediction_ids[i])

    for mat_id in wrong_cry_sys_prediction_ids:
        universal_cry_sys_test_ids.append(mat_id)

    cry_sys_test_df = pd.DataFrame(columns = rf_diff_obj.full_df.columns)
    for mat_id in universal_cry_sys_test_ids:
        # print(mat_id)
        subdf = rf_diff_obj.full_df.loc[rf_diff_obj.full_df.mat_id == mat_id]
        cry_sys_test_df = pd.concat([cry_sys_test_df, subdf])

    try: 
        with open('Model_data/Lattice_inputs_and_outputs/lattice_'+cry_sys+'_cry_sys_test_df.pkl', 'rb') as f:
            cry_sys_test_df = pickle.load(f)
    except:
        cry_sys_test_df = update_df_lattice(cry_sys_test_df)
        cry_sys_test_df.to_pickle('Model_data/Lattice_inputs_and_outputs/lattice_'+cry_sys+'_cry_sys_test_df.pkl')
        if cry_sys == 'orthorhombic':
            cry_sys_test_df = scale_df_lattice(cry_sys_test_df)
            cry_sys_test_df.drop('x', axis = 1, inplace = True)
            cry_sys_test_df.drop('y', axis = 1, inplace = True)
            cry_sys_test_df.drop('z', axis = 1, inplace = True)
            cry_sys_test_df = cry_sys_test_df.rename(columns={'a_sorted': 'x', 'b_sorted': 'y', 'c_sorted': 'z'})
            cry_sys_test_df.to_pickle('Model_data/Lattice_inputs_and_outputs/lattice_'+cry_sys+'_cry_sys_test_df.pkl')

    cry_sys_vals = np.asarray(cry_sys_test_df.radial_200_ang_Colin_basis) 

    input_train_temp = []
    for unit in cry_sys_vals:
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
        input_train.append(np.asarray(temp))

    cry_sys_use = input_train

    pred_test = rf_diff_obj.lattice_rf_output[2].predict(cry_sys_use)

    a_pred = []
    b_pred = []
    c_pred = []
    alpha_pred =[]
    beta_pred = []
    gamma_pred = []
    for prediction in pred_test:
        a_pred.append(prediction[0])
        b_pred.append(prediction[1])
        c_pred.append(prediction[2])
        alpha_pred.append(prediction[3])
        beta_pred.append(prediction[4])
        gamma_pred.append(prediction[5])

    cry_sys_test_df['a_pred'] = a_pred
    cry_sys_test_df['b_pred'] = b_pred
    cry_sys_test_df['c_pred'] = c_pred
    cry_sys_test_df['alpha_pred'] = alpha_pred
    cry_sys_test_df['beta_pred'] = beta_pred
    cry_sys_test_df['gamma_pred'] = gamma_pred

    FINAL_cry_sys_test_df = pd.DataFrame( columns=['a_full_predictions', 'a_mode', 'a_true',
                                         'b_full_predictions', 'b_mode', 'b_true',
                                         'c_full_predictions', 'c_mode', 'c_true',
                                         'alpha_full_predictions', 'alpha_mode', 'alpha_true',
                                         'beta_full_predictions', 'beta_mode', 'beta_true',
                                         'gamma_full_predictions', 'gamma_mode', 'gamma_true',
                                          'material_id', 'true_crystal_sys'])
    for mat_id in cry_sys_test_df.mat_id.unique():
        # print(mat_id)
        subdf = cry_sys_test_df.loc[cry_sys_test_df.mat_id == mat_id]
        out_df = lattice_visualize_predictions_by_material(subdf, show_plots = False)
        FINAL_cry_sys_test_df = pd.concat([FINAL_cry_sys_test_df, out_df])


    for char in ['a', 'b', 'c']:
        
        print('Accurate MAE')
        errors_acc = np.abs(np.asarray(FINAL_cry_sys_test_df.loc[FINAL_cry_sys_test_df['true_crystal_sys'] == cry_sys][char+'_true']) - 
                            np.asarray(FINAL_cry_sys_test_df.loc[FINAL_cry_sys_test_df['true_crystal_sys'] == cry_sys][char+'_mode']))
        MAE_acc = np.mean(errors_acc)
        print(MAE_acc)
        
        print('Innacuracy MAE')
        errors_innac = np.abs(np.asarray(FINAL_cry_sys_test_df.loc[FINAL_cry_sys_test_df['true_crystal_sys'] != cry_sys][char+'_true']) - 
                              np.asarray(FINAL_cry_sys_test_df.loc[FINAL_cry_sys_test_df['true_crystal_sys'] != cry_sys][char+'_mode']))
        MAE_inacc = np.mean(errors_innac)
        print(MAE_inacc)
        
        print('All MAE')
        errors_all = np.abs(np.asarray(FINAL_cry_sys_test_df[char+'_true']) -  np.asarray(FINAL_cry_sys_test_df[char+'_mode']))
        MAE_all = np.mean(errors_all)
        print(MAE_all)
        
        
        plt.figure(figsize=(8, 7))
        hist_acc = plt.hist(errors_acc, bins = 20, weights=np.ones(len(errors_acc)) / len(errors_all))
        hist_innac = plt.hist(errors_innac, bins = 20, weights=np.ones(len(errors_innac)) / len(errors_all))

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.title('Error Histogram', fontsize=18)
        plt.vlines(MAE_all, max(hist_acc[0]), min(hist_acc[0]), color='limegreen', linewidth=5, label='RMSE')
        plt.text(MAE_all + 0.25, max(hist_acc[0]) - 0.1 * max(hist_acc[0]), 'RMSE all = ' + str(round(MAE_all, 2)),
                 horizontalalignment='center', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Absolute Error', fontsize=16)
        plt.ylabel('Percent', fontsize=16)
        plt.show()

    plt.show()