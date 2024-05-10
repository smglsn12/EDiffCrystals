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
print('starting')

# output_df_radial = joblib.load('Model_data/Crystal_sys_outputs/output_df_radial.joblib')
# visualize_predictions_by_material(output_df_radial, random_inds = 100, show_individual = False, savefigure = True)

rf_diff_obj = RF_Diffraction_model('Final_0_05_spacing_radial_dataframe_100423.pkl', 
                                   ['Model_data/Lattice_inputs_and_outputs/radial_cubic_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/radial_monoclinic_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_hexagonal_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/SCALED_radial_orthorhombic_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_tetragonal_df_w_lattice.joblib', 
                                    'Model_data/Lattice_inputs_and_outputs/radial_trigonal_df_w_lattice.joblib',
                                    'Model_data/Lattice_inputs_and_outputs/radial_triclinic_df_w_lattice.joblib'])
rf_diff_obj.load_full_df() # load data from path provided 
rf_diff_obj.load_output_df_radial(path = 'Model_data/Crystal_sys_outputs/output_df_radial.joblib')
rf_diff_obj.condense_crystal_system_output()
joblib.dump(rf_diff_obj.condensed_output_df, 'Model_data/Crystal_sys_outputs/condensed_output_df.joblib')