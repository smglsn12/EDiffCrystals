import pickle
import pandas as pd
import numpy as np
# from pathlib import Path
import joblib
# import itertools as it
import py4DSTEM
# import glob
import sys
import os
from random import sample
# from time import time
# import collections
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from typing import Union, Optional
from scipy.signal import medfilt
# from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
# from EDiffCrystals import diffraction_analysis_utils as dau
import EDiffCrystals.diffraction_analysis_utils as dau

class Diffraction_Analysis():
    def __init__(self, pl_filepath, model_path_dict):
        self.pl_filepath = pl_filepath
        self.model_path_dict = model_path_dict
        self.pls = None
        self.sum_scaling = None
        self.num_spots_needed = None
        self.remove_central_beam = None
        self.scaled_patterns = None
        self.pl_indicies = None
        self.input_vector = None
        self.df_predictions = None
        self.lattice_cmap = None

    def load_pls(self):
        print('loading point lists stored at ' + self.pl_filepath)
        self.pls = py4DSTEM.read(self.pl_filepath)

    def prep_pls_for_prediction(self, sum_scaling=0.3, num_spots_needed=5, remove_central_beam=True):
            self.sum_scaling = sum_scaling
            self.num_spots_needed = num_spots_needed
            self.remove_central_beam = remove_central_beam
            self.scaled_patterns, self.pl_indicies = dau.clean_and_scale_pls(
                                                                            pls=self.pls,
                                                                            sum_scaling=self.sum_scaling,
                                                                            num_spots_needed=self.num_spots_needed,
                                                                            remove_central_beam=self.remove_central_beam
                                                                            )

    def predict_set_of_point_lists(self, lattice_pred_type = 'all',
                                       save_df=True, save_path=None):

        print('loading crystal system model')
        cry_sys_model = joblib.load(self.model_path_dict['cry_sys'])
        print('crystal system model loaded!')

        self.lattice_pred_type = lattice_pred_type

        print('building input vector')
        self.input_vector = dau.build_input_vector(self.scaled_patterns)
        print('finished input vector')

        print('predicting input vector')
        pred_df = dau.predict_exp_patterns(cry_sys_model,
                                       self.input_vector,
                                       list_of_labels_for_patterns=None,
                                       pl_indicies_reference=self.pl_indicies,
                                       lattice_model_dict=self.model_path_dict,
                                       cry_sys_lattice=self.lattice_pred_type )

        print('finished predictions, adding confidence')

        self.df_predictions = dau.add_confidences_to_pred_df(pred_df,
                                                     num_trees=80)

        if save_df:
            print('saving prediction dataframe')
            if save_path == None:
                sum_scaling_string = str(self.sum_scaling)
                sum_scaling_string = sum_scaling_string.replace('.', '_')
                pl_filepath = self.pl_filepath.replace('.', '_')
                self.df_predictions.to_pickle(
                    str(self.num_spots_needed) + '_spots_required_sum_scaling_' + sum_scaling_string + '_' + pl_filepath + '_lattice_unaugmented_model.pkl')
            else:
                self.df_predictions.to_pickle(save_path)

    def generate_lattice_cmap(self, insert_length = 15, resample_length = 75):
        self.lattice_cmap = dau.generate_lattice_cmap(insert_length, resample_length)


    def visualize_real_space_predictions(self,
                                         im_shape,
                                         c_vals = (),
                                         cry_sys = ('cubic', 'hexagonal', 'tetragonal', 'trigonal', 'orthorhombic', 'monoclinic'),
                                         plot_result = True,
                                         mask_threshold = None,
                                         threshold_image = True,
                                         legend_type = 'Diff Con Percents',
                                         mask_range = (0.005, 0.05),
                                         medfilt_shape = (5,5),
                                         save_figure = False,
                                         figure_path = '',
                                         lattice_param = 'a',
                                        ):

        if len(c_vals) == 0:
            c_vals = np.array([
                [1, 20 / 255, 20 / 255],
                [1, 191 / 255, 0],
                [42 / 255, 1, 48 / 255],
                [1, 242 / 255, 0],
                [190 / 255, 106 / 255, 1],  # cyan
                [0, 174 / 255, 239 / 255],  # violet
            ]),

        self.pred_image, self.stack_con = dau.confidence_image(
                             self.df_predictions,
                             im_shape=im_shape,
                             # c_vals=c_vals,
                             cry_sys=cry_sys,
                             plot_result=plot_result,
                             mask_threshold=mask_threshold,
                             threshold_image=threshold_image,
                             mask_range=mask_range,
                             medfilt_shape=medfilt_shape,
                             legend_type=legend_type,
                             save_figure=save_figure,
                             figure_path=figure_path,
                             lattice_color_map=self.lattice_cmap,
                            lattice_param = lattice_param,

                             )

    def visualize_lattice_distributions(self, im_shape,
                                        crystal_systems = ('cubic', 'hexagonal', 'tetragonal', 'trigonal',
                                                           'orthorhombic', 'monoclinic'),
                                        lattice_params = ('a', 'b', 'c'),
                                        bin_width = 0.1):
        lattice_mask = np.zeros((im_shape[0], im_shape[1]))
        for a1 in range(3):
            temp = self.stack_con[a1, :, :] > 0
            lattice_mask += temp.astype(int)

        inds = np.argwhere(lattice_mask > 0)
        list_inds = []
        for ind in inds:
            list_inds.append(list(ind))

        # t = list(self.df_predictions[self.df_predictions['pl_indicies'].isin(list_inds)].pl_indicies)

        subdf = self.df_predictions[self.df_predictions['pl_indicies'].isin(list_inds)]

        # for ind in list_inds:
            # if (ind in t) == False:
                # print(ind)

        if len(crystal_systems) == 1:
            for param in lattice_params:
                param_label = param + '_median'
                bot = round(min(subdf[param_label])-bin_width, 1)
                top = round(max(subdf[param_label])-bin_width, 1)
                num_vals = round((top-bot)/bin_width)+1

                # print(bot, top, num_vals)
                bins = np.linspace(bot, top, num_vals)
                # print(bins)
                hist = plt.hist(
                    subdf[param_label],
                    edgecolor='k', linewidth=1,
                    bins=bins, stacked=True, density=True)
                median = np.median(subdf[param_label])
                mean = round(np.mean(subdf[param_label]), 2)


                print('median ' + str(median))
                print('mean ' + str(mean))
                plt.vlines(median, min(hist[0]), max(hist[0]), color = 'red', zorder = 10, linewidth = 3)
                plt.title(param + ' Axis', fontsize=14)
                plt.show()

        else:
            print(len(crystal_systems))
            subdf_cubic = subdf.loc[subdf.prediction == 'cubic']
            subdf_hexagonal = subdf.loc[subdf.prediction == 'hexagonal']
            subdf_tetragonal = subdf.loc[subdf.prediction == 'tetragonal']
            subdf_trigonal = subdf.loc[subdf.prediction == 'trigonal']
            subdf_orthorhombic = subdf.loc[subdf.prediction == 'orthorhombic']
            subdf_monoclinic = subdf.loc[subdf.prediction == 'monoclinic']

            subdf_dict = {'cubic':subdf_cubic,
                          'hexagonal':subdf_hexagonal,
                          'tetragonal':subdf_tetragonal,
                          'trigonal':subdf_trigonal,
                          'orthorhombic':subdf_orthorhombic,
                          'monoclinic':subdf_monoclinic}

            for param in lattice_params:
                param_label = param + '_median'
                bot = round(min(subdf[param_label])-bin_width, 1)
                top = round(max(subdf[param_label])-bin_width, 1)
                num_vals = round((top-bot)/bin_width)+1

                # print(bot, top, num_vals)
                bins = np.linspace(bot, top, num_vals)
                # print(bins)

                lattice_param_vals = []
                cry_sys_vals = []
                medians = []
                legend_vals = []

                for cry_sys in crystal_systems:
                    lattice_param_vals.append(subdf_dict[cry_sys][param_label])
                    cry_sys_vals.append(cry_sys)
                    med = np.median(subdf_dict[cry_sys][param_label])
                    medians.append(med)
                    legend_vals.append(cry_sys + ' ' + str(round(med, 2)))

                hist = plt.hist(
                    lattice_param_vals,
                    edgecolor='k', linewidth=1,
                    bins=bins, stacked=True, density=True)

                plt.title(param + ' Axis', fontsize = 14)
                plt.legend(legend_vals)
                plt.show()