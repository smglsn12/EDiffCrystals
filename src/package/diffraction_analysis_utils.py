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
from tqdm import tqdm

def flatten(list1):
    return [item for sublist in list1 for item in sublist]


def test_radial_representation(df, indicies, thicknesses, df_only_radial=None):
    df_copy = df.copy()
    if type(df_only_radial) == type(None):
        output_list = []
        for thickness in thicknesses:
            for index in indicies:
                print([index, df_copy.iloc[index]])
                pattern_scaled_use = np.zeros((41, 13), dtype=np.complex128)
                basis, mask = calc_basis_scaled_df(df.iloc[index]['thickness_' + thickness], 2, 0.05, 12)
                pattern_scaled = basis * df.iloc[index]['thickness_' + thickness].data['intensity'][mask]

                for i in range(0, len(pattern_scaled)):
                    for j in range(0, len(pattern_scaled[0])):
                        pattern_scaled_use[i][j] = sum(pattern_scaled[i][j])

                test_passed = True
                for i in range(0, len(df_copy.iloc[index]['radial_' + thickness + '_Colin_basis'])):
                    for j in range(0, len(df_copy.iloc[index]['radial_' + thickness + '_Colin_basis'][0])):
                        assert (df_copy.iloc[index]['radial_' + thickness + '_Colin_basis'][i][j] ==
                                pattern_scaled_use[i][j])
                        if not df_copy.iloc[index]['radial_' + thickness + '_Colin_basis'][i][j] == \
                               pattern_scaled_use[i][j]:
                            test_passed = False

                if test_passed == False:
                    output_list.append([test_passed, thickness, index])

                print(thickness, index, test_passed)

        if len(output_list) == 0:
            return 'All tests passed'
        else:
            return output_list

    else:
        output_list = []
        for thickness in thicknesses:
            for index in indicies:
                test_passed = True
                pattern_scaled_use = df_only_radial.iloc[index]['radial_' + thickness + '_Colin_basis']

                for i in range(0, len(df_copy.iloc[index]['radial_' + thickness + '_Colin_basis'])):
                    for j in range(0, len(df_copy.iloc[index]['radial_' + thickness + '_Colin_basis'][0])):
                        assert (df_copy.iloc[index]['radial_' + thickness + '_Colin_basis'][i][j] ==
                                pattern_scaled_use[i][j])
                        if not df_copy.iloc[index]['radial_' + thickness + '_Colin_basis'][i][j] == \
                               pattern_scaled_use[i][j]:
                            test_passed = False

                if test_passed == False:
                    output_list.append([test_passed, thickness, index])

                print(thickness, index, test_passed)

        if len(output_list) == 0:
            return 'All tests passed'
        else:
            return output_list


def construct_basis(kx, ky, k_max, dk, order_max, sine_basis=False):
    """


    Placeholder

    """
    # k bin boundaries starts at zero extends to kmax
    k_bins = np.arange(0, k_max + dk, dk)

    # no elements, size of the basis, so
    # radial_bins,
    if kx.ndim == 1:
        basis_size = (k_bins.shape[0], order_max + 1, kx.shape[0])
    elif kx.ndim > 1:
        basis_size = (k_bins.shape[0], order_max + 1, *kx.shape)
    else:
        print('error')

    basis = np.zeros(basis_size, dtype=np.complex128)
    # print(basis.shape)
    kr = np.hypot(kx, ky)
    # ensure ky, kx (alex check why)
    phi = np.arctan2(ky, kx)

    # loop over the bins
    for ind, k in enumerate(k_bins):
        # calculate the basis functions
        # create the mask to select
        sub = np.logical_and(kr > k - dk, kr < k + dk)

        b_radial = 1 - np.abs(kr[sub] - k) / dk
        if sine_basis:
            b_radial = np.sin(b_radial * (np.pi / 2)) ** 2

        for ind_order, order in enumerate(range(order_max + 1)):
            #             b_annular =  np.cos(order * phi[sub]) + 1j * np.sin(order * phi[sub])

            b_annular = np.exp((1j * order) * phi[sub])

            basis[ind, ind_order][sub] = b_radial * b_annular

    return basis


def calc_basis_scaled_df(bragg_list, k_max, dk, order_max, sine_basis=False, remove_central_beam=True):
    new_bragg_list = deepcopy(bragg_list)  # create a copy # probably not needed

    if remove_central_beam:
        # print(new_pl.data.shape)
        # print(new_bragg_list[0])
        # print(new_bragg_list.data)
        # print(new_bragg_list.data['intensity'])
        mask = np.ones_like(new_bragg_list.data['intensity'], dtype=bool)
        index = np.where(new_bragg_list.data['intensity'] == np.max(new_bragg_list.data['intensity']))[0][0]
        mask[index] = False
        # new_bragg_list.data = new_bragg_list.data[mask] return to this when going back to 13.3
        new_bragg_list._data = new_bragg_list.data[mask]

        # print(new_pl.data.shape)

    basis = construct_basis(new_bragg_list.data['qx'], new_bragg_list.data['qy'], k_max, dk, order_max, sine_basis)

    if remove_central_beam:
        return [basis, mask]
    else:
        return basis


def visualize_specific_pattern(point_lists, index, intensity_scaling=0.3):
    pl = point_lists.cal[index[0], index[1]]  # indvidual pointlist qx,qy,ints
    pl.data['intensity'] = pl.data['intensity'] / (sum(pl.data['intensity']) / intensity_scaling)
    fig, ax = plot_diffraction_pattern(pl, returnfig=True)
    return fig


def clean_and_scale_pls(pls, sum_scaling=0.3, num_spots_needed=5, remove_central_beam=True):
    print('scaling patterns to sum intensity = ' + str(sum_scaling) + ' and filtering out patterns with fewer than '
          + str(num_spots_needed) + ' diffraction spots')

    pl_indicies = []
    scaled_patterns = []
    for i in tqdm(range(pls.shape[0])):
        for j in range(pls.shape[1]):
            pattern = pls.cal[i, j]
            if len(pattern.data['intensity']) >= num_spots_needed:
                pattern.data['intensity'] = pattern.data['intensity'] / (sum(pattern.data['intensity']) / sum_scaling)
                basis, mask = calc_basis_scaled_df(pattern, 2, 0.05, 12, remove_central_beam=remove_central_beam)
                pattern_scaled_use = np.zeros((41, 13), dtype=np.complex128)
                pattern_scaled = basis * pattern.data['intensity'][mask]
                # pattern_scaled = basis*pattern.data['intensity']
                for x in range(0, len(pattern_scaled)):
                    for y in range(0, len(pattern_scaled[0])):
                        pattern_scaled_use[x][y] = sum(pattern_scaled[x][y])
                scaled_patterns.append(pattern_scaled_use)
                pl_indicies.append([i, j])
            else:
                pass
            # except:
            # pass

    print('finished scaling and filtering!')
    return scaled_patterns, pl_indicies


def add_confidences_to_pred_df(df_with_predictions, num_trees=80):
    new_df_with_predictions = df_with_predictions.copy()

    predictions_diff_confidence = []
    predictions_std = []
    for i in tqdm(range(0, len(df_with_predictions))):
        # print(i/len(df_with_predictions))
        row = df_with_predictions.iloc[i]
        full_pred = row.full_predictions
        # print(full_pred)
        pred_df_temp = pd.DataFrame(full_pred, columns=['Full_predictions'])
        # print(pred_df_temp)
        # print(pred_df_temp.Full_predictions)
        # print(pred_df_temp.Full_predictions.value_counts())
        vals = np.asarray(pred_df_temp.Full_predictions.value_counts())
        predictions_std.append(vals[0] / num_trees)
        if len(vals) > 1:
            predictions_diff_confidence.append(vals[0] / num_trees - vals[1] / num_trees)
        else:
            predictions_diff_confidence.append(vals[0] / num_trees)

    new_df_with_predictions['Difference_Confidence'] = predictions_diff_confidence
    new_df_with_predictions['Confidence'] = predictions_std

    return new_df_with_predictions

def plot_diffraction_pattern(
    bragg_peaks,
    bragg_peaks_compare = None,
    scale_markers: float = 500,
    scale_markers_compare: Optional[float] = None,
    power_markers: float = 1,
    plot_range_kx_ky: Optional[Union[list, tuple, np.ndarray]] = None,
    add_labels: bool = True,
    shift_labels: float = 0.08,
    shift_marker: float = 0.005,
    min_marker_size: float = 1e-6,
    max_marker_size: float = 1000,
    figsize: Union[list, tuple, np.ndarray] = (12, 6),
    returnfig: bool = False,
    input_fig_handle=None,
):
    """
    2D scatter plot of the Bragg peaks

    Args:
        bragg_peaks (PointList):        numpy array containing ('qx', 'qy', 'intensity', 'h', 'k', 'l')
        bragg_peaks_compare(PointList): numpy array containing ('qx', 'qy', 'intensity')
        scale_markers (float):          size scaling for markers
        scale_markers_compare (float):  size scaling for markers of comparison
        power_markers (float):          power law scaling for marks (default is 1, i.e. amplitude)
        plot_range_kx_ky (float):       2 element numpy vector giving the plot range
        add_labels (bool):              flag to add hkl labels to peaks
        min_marker_size (float):        minimum marker size for the comparison peaks
        max_marker_size (float):        maximum marker size for the comparison peaks
        figsize (2 element float):      size scaling of figure axes
        returnfig (bool):               set to True to return figure and axes handles
        input_fig_handle (fig,ax)       Tuple containing a figure / axes handle for the plot.
    """

    # 2D plotting
    if input_fig_handle is None:
        # fig = plt.figure(figsize=figsize)
        # ax = fig.add_subplot()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = input_fig_handle[0]
        ax_parent = input_fig_handle[1]
        ax = ax_parent[0]

    if power_markers == 2:
        marker_size = scale_markers * bragg_peaks.data["intensity"]
    else:
        marker_size = scale_markers * (
            bragg_peaks.data["intensity"] ** (power_markers / 2)
        )

    # Apply marker size limits to primary plot
    marker_size = np.clip(marker_size, min_marker_size, max_marker_size)

    if bragg_peaks_compare is None:
        ax.scatter(
            bragg_peaks.data["qy"], bragg_peaks.data["qx"], s=marker_size, facecolor="k"
        )
    else:
        if scale_markers_compare is None:
            scale_markers_compare = scale_markers

        if power_markers == 2:
            marker_size_compare = np.clip(
                scale_markers_compare * bragg_peaks_compare.data["intensity"],
                min_marker_size,
                max_marker_size,
            )
        else:
            marker_size_compare = np.clip(
                scale_markers_compare
                * (bragg_peaks_compare.data["intensity"] ** (power_markers / 2)),
                min_marker_size,
                max_marker_size,
            )

        ax.scatter(
            bragg_peaks_compare.data["qy"],
            bragg_peaks_compare.data["qx"],
            s=marker_size_compare,
            marker="o",
            facecolor=[0.0, 0.7, 1.0],
        )
        ax.scatter(
            bragg_peaks.data["qy"],
            bragg_peaks.data["qx"],
            s=marker_size,
            marker="+",
            facecolor="k",
        )

    ax.set_xlabel("$q_y$ [Å$^{-1}$]")
    ax.set_ylabel("$q_x$ [Å$^{-1}$]")

    if plot_range_kx_ky is not None:
        plot_range_kx_ky = np.array(plot_range_kx_ky)
        if plot_range_kx_ky.ndim == 0:
            plot_range_kx_ky = np.array((plot_range_kx_ky, plot_range_kx_ky))
        ax.set_xlim((-plot_range_kx_ky[0], plot_range_kx_ky[0]))
        ax.set_ylim((-plot_range_kx_ky[1], plot_range_kx_ky[1]))
    else:
        k_range = 1.05 * np.sqrt(
            np.max(bragg_peaks.data["qx"] ** 2 + bragg_peaks.data["qy"] ** 2)
        )
        ax.set_xlim((-k_range, k_range))
        ax.set_ylim((-k_range, k_range))

    ax.invert_yaxis()
    ax.set_box_aspect(1)
    ax.xaxis.tick_top()

    # Labels for all peaks
    if add_labels is True:
        text_params = {
            "ha": "center",
            "va": "center",
            "family": "sans-serif",
            "fontweight": "normal",
            "color": "r",
            "size": 10,
        }

        def overline(x):
            return str(x) if x >= 0 else (r"\overline{" + str(np.abs(x)) + "}")

        # for a0 in range(bragg_peaks.data.shape[0]):
        #     h = bragg_peaks.data["h"][a0]
        #     k = bragg_peaks.data["k"][a0]
        #     l = bragg_peaks.data["l"][a0]

        #     ax.text(
        #         bragg_peaks.data["qy"][a0],
        #         bragg_peaks.data["qx"][a0]
        #         - shift_labels
        #         - shift_marker * np.sqrt(marker_size[a0]),
        #         "$" + overline(h) + overline(k) + overline(l) + "$",
        #         **text_params,
        #     )

    # Force plot to have 1:1 aspect ratio
    ax.set_aspect("equal")

    if input_fig_handle is None:
        plt.show()

    if returnfig:
        return fig, ax


def load_output_data(save_path):
    with open(save_path, 'rb') as f:
        pred_df = pickle.load(f)

    return pred_df

def build_input_vector(patterns_vec):
    # run an array of vectorized patterns through here - will calculate angle/abs and turn into 1d array

    new_input_test = []
    for test in patterns_vec:
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
    return input_test


def predict_exp_patterns(cry_sys_model, vectorized_array_of_patterns, list_of_labels_for_patterns,
                         pl_indicies_reference, lattice_model_dict,
                         cry_sys_lattice):
    print('starting')
    if type(list_of_labels_for_patterns) == type(None):
        list_of_labels_for_patterns = []
        for i in range(0, len(vectorized_array_of_patterns)):
            list_of_labels_for_patterns.append('NA')
    out_list = []
    predictions = cry_sys_model.predict(vectorized_array_of_patterns)
    predictions_full = []
    trees = cry_sys_model.estimators_
    count = 0
    for tree_ind in tqdm(range(len(trees))):
        tree = trees[tree_ind]
        # print(count)
        predictions_full.append(tree.predict(np.asarray(vectorized_array_of_patterns)))
        count += 1
        # print(tree.predict(np.asarray(updated_spectra_test)))
    predictions_ordered = np.asarray(predictions_full).T

    for i in tqdm(range(0, len(vectorized_array_of_patterns))):
        # print(i)
        new_full_pred = []
        for pred in predictions_ordered[i]:
            new_full_pred.append(cry_sys_model.classes_[int(pred)])
        out_list.append([list_of_labels_for_patterns[i], predictions[i], new_full_pred, vectorized_array_of_patterns[i],
                         pl_indicies_reference[i]])

    out_df = pd.DataFrame(out_list, columns=['label', 'prediction', 'full_predictions', 'input_vector', 'pl_indicies'])

    new_rows = []
    for i in range(0, len(out_df)):
        new_rows.append(None)
    out_df['lattice_full_predictions_a'] = new_rows
    out_df['lattice_full_predictions_b'] = new_rows
    out_df['lattice_full_predictions_c'] = new_rows

    out_df['a_median'] = new_rows
    out_df['b_median'] = new_rows
    out_df['c_median'] = new_rows

    if cry_sys_lattice != 'all':
        lattice_model_path = lattice_model_dict[cry_sys_lattice]
        print('loading ' + cry_sys_lattice)
        lattice_model = joblib.load(lattice_model_path)
        print(cry_sys_lattice + ' loaded!')

        count = 0
        lattice_predictions_full = []
        lattice_trees = lattice_model.estimators_
        for lattice_tree_ind in tqdm(range(len(lattice_trees))):
            lattice_tree = lattice_trees[lattice_tree_ind]
            # print(count)
            lattice_predictions_full.append(lattice_tree.predict(np.asarray(vectorized_array_of_patterns)))
            # count += 1
            # print(tree.predict(np.asarray(updated_spectra_test)))
        lattice_predictions_ordered = np.asarray(lattice_predictions_full).T

        for i in range(0, len(out_df)):
            out_df.at[i, 'lattice_full_predictions_a'] = lattice_predictions_ordered[0][i]
            out_df.at[i, 'a_median'] = np.median(lattice_predictions_ordered[0][i])
            out_df.at[i, 'lattice_full_predictions_b'] = lattice_predictions_ordered[1][i]
            out_df.at[i, 'b_median'] = np.median(lattice_predictions_ordered[1][i])
            out_df.at[i, 'lattice_full_predictions_c'] = lattice_predictions_ordered[2][i]
            out_df.at[i, 'c_median'] = np.median(lattice_predictions_ordered[2][i])

    if cry_sys_lattice == 'all':
        for cry_sys in out_df.prediction.unique():
            subdf = out_df.loc[out_df.prediction == cry_sys]
            subdf.reset_index(inplace=True)

            print('loading ' + cry_sys)
            lattice_model_path = lattice_model_dict[cry_sys]
            lattice_model = joblib.load(lattice_model_path)
            print(cry_sys + ' loaded!')

            for k in tqdm(range(0, len(subdf))):
                row = subdf.iloc[k]
                ind = row['index']
                cry_sys_input = row['input_vector']
                lattice_trees = lattice_model.estimators_

                # print(cry_sys_use)
                predictions_full = []
                for lattice_tree in lattice_trees:
                    predictions_full.append(lattice_tree.predict([cry_sys_input]))

                # print(predictions_full)

                predictions_ordered = np.asarray(predictions_full).T

                # return predictions_ordered

                # print(predictions_ordered)
                # print(predictions_ordered.shape)

                out_df.at[ind, 'lattice_full_predictions_a'] = predictions_ordered[0]
                out_df.at[ind, 'a_median'] = np.median(predictions_ordered[0])
                out_df.at[ind, 'lattice_full_predictions_b'] = predictions_ordered[1]
                out_df.at[ind, 'b_median'] = np.median(predictions_ordered[1])
                out_df.at[ind, 'lattice_full_predictions_c'] = predictions_ordered[2]
                out_df.at[ind, 'c_median'] = np.median(predictions_ordered[2])

    return out_df

def generate_lattice_cmap(insert_length = 15, resample_length = 75):
    turbos = mpl.colormaps['turbo'].resampled(resample_length)
    newcolors = turbos(np.linspace(0, 1, resample_length))
    k = [0, 0, 0, 1]
    newcolorlist = list(newcolors)
    # for i in range(0, 15): # for c axis
    for i in range(0, insert_length):  # for a axis
        newcolorlist.insert(0, k)
    newcolors = np.asarray(newcolorlist)
    newcmp = ListedColormap(newcolors)
    return newcmp


def predict_set_of_point_lists(pls, cry_sys_model, pl_filepath, lattice_model_dict,
                               cry_sys_lattice, sum_scaling=0.3,
                               num_spots_needed=5, remove_central_beam=True,
                               save_df=True, save_path=None):
    scaled_patterns, pl_indicies = clean_and_scale_pls(pls,
                                                       sum_scaling=sum_scaling,
                                                       num_spots_needed=num_spots_needed,
                                                       remove_central_beam=remove_central_beam)

    print('building input vector')
    input_vector = build_input_vector(scaled_patterns)
    print('finished input vector')

    print('predicting input vector')
    pred_df = predict_exp_patterns(cry_sys_model,
                                   input_vector,
                                   list_of_labels_for_patterns=None,
                                   pl_indicies_reference=pl_indicies,
                                   lattice_model_dict=lattice_model_dict,
                                   cry_sys_lattice=cry_sys_lattice)

    print('finished predictions, adding confidence')

    pred_df_updated = add_confidences_to_pred_df(pred_df,
                                                 num_trees=80)

    if save_df:
        print('saving prediction dataframe')
        if save_path == None:
            sum_scaling_string = str(sum_scaling)
            sum_scaling_string = sum_scaling_string.replace('.', '_')
            pl_filepath = pl_filepath.replace('.', '_')
            pred_df_updated.to_pickle(
                str(num_spots_needed) + '_spots_required_sum_scaling_' + sum_scaling_string + '_' + pl_filepath + '_lattice_unaugmented_model.pkl')
        else:
            pred_df_updated.to_pickle(save_path)

    return pred_df_updated


def visualize_output(df, title, col_to_plot):
    plt.title(title + ' model value counts', fontsize=12)
    df.prediction.value_counts(normalize=True).plot(kind='bar')
    print(df.prediction.value_counts(normalize=True))
    plt.show()

    plt.title(title + ' model sum ' + col_to_plot, fontsize=12)
    cry_sys_xs = []
    cry_sys_ys = []
    for cry_sys in df.prediction.value_counts().index:
        cry_sys_xs.append(cry_sys)
        cry_sys_ys.append(sum(np.asarray(df.loc[df.prediction == cry_sys][col_to_plot])))
    plt.bar(cry_sys_xs, cry_sys_ys)
    plt.show()

    plt.title(title + ' model ' + col_to_plot, fontsize=12)
    for cry_sys in df.prediction.value_counts().index[0:3]:
        plt.hist(df.loc[df.prediction == cry_sys][col_to_plot], bins=np.linspace(0, 1, 21), label=cry_sys)
    plt.legend()
    plt.show()


def confidence_image(
        df_with_predictions,
        cry_sys=(
            'cubic',
            'hexagonal',
            'tetragonal',
            'trigonal',
            'orthorhombic',
            'monoclinic',
        ),
        c_vals=np.array([
            [1, 20 / 255, 20 / 255],
            [1, 191 / 255, 0],
            [42 / 255, 1, 48 / 255],
            [1, 242 / 255, 0],
            [190 / 255, 106 / 255, 1],  # cyan
            [0, 174 / 255, 239 / 255],  # violet
        ]),
        mask_threshold=(),
        mask_range=(0.2, 0.3),
        im_shape=None,
        medfilt_shape=None,
        plot_result=False,
        threshold_image=False,
        save_figure=False,
        figure_path=None,
        legend_type='Percent Counts',
        show_lattice=True,
        lattice_color_map=None,
        lattice_param = 'a',
):
    df_copy = df_with_predictions.copy()
    # init output
    stack_con = np.zeros((len(cry_sys), im_shape[0], im_shape[1]))
    lattice_vals = np.zeros((im_shape[0], im_shape[1]))
    # print(stack_con.shape)

    coords = np.asarray(df_copy.pl_indicies).tolist()
    # print(coords)
    coords = np.array(coords).astype('int')

    lattice_param_label = lattice_param+'_median'

    lattice_vals_raw = np.asarray(df_copy[lattice_param_label]).tolist()

    inds = np.ravel_multi_index((coords[:, 0], coords[:, 1]), im_shape)
    lattice_vals.ravel()[inds] = lattice_vals_raw

    # plt.imshow(lattice_vals, cmap=lattice_color_map)
    # plt.colorbar()

    for a0 in range(len(cry_sys)):

        coords = np.asarray(df_copy.loc[df_copy.prediction == cry_sys[a0]].pl_indicies).tolist()
        # print(coords)
        if len(coords) > 0:
            coords = np.array(coords).astype('int')
            diff_con = np.asarray(df_copy.loc[df_copy.prediction == cry_sys[a0]].Difference_Confidence).tolist()
            # diff_con = np.asarray(df_copy.loc[df_copy.prediction == cry_sys[a0]].Confidence).tolist()
            # print(diff_con)
            inds = np.ravel_multi_index((coords[:, 0], coords[:, 1]), im_shape)
            stack_con[a0].ravel()[inds] = diff_con

    if medfilt_shape is not None:
        stack_con = medfilt(stack_con, (1, medfilt_shape[0], medfilt_shape[1]))

    # find most probable phase and display difference confidence
    # stack_sort = np.sort(stack_con, axis = 0)
    # im_diff_con = stack_sort[-1] - stack_sort[-2]
    im_index = np.argmax(stack_con, axis=0)
    im_diff_con = np.max(stack_con, axis=0)
    # print(im_diff_con)
    # generate color image
    im_rgb_phases = np.zeros((im_shape[0], im_shape[1], 3))
    for a0 in range(len(cry_sys)):
        for a1 in range(3):
            im_rgb_phases[:, :, a1][im_index == a0] = c_vals[a0, a1]

    # masked color image
    mask = np.clip(
        (im_diff_con - mask_range[0]) / (mask_range[1] - mask_range[0]),
        0, 1,
    )
    im_rgb = im_rgb_phases * mask[:, :, None]

    percents = []
    percents_raw = []
    for a1 in range(len(cry_sys)):
        count = 0
        count_raw = 0
        for b0 in range(len(stack_con[a1, :, :])):
            for b1 in range(len(stack_con[a1, :, :][0])):
                if stack_con[a1, :, :][b0, b1] > mask_range[0]:
                    # if legend_type == 'Percent Counts':
                    if legend_type in ['Diff Con', 'Diff Con Percents']:
                        count += stack_con[a1, :, :][b0, b1]
                        count_raw += 1

        percents.append(count)
        percents_raw.append(count_raw)

    if legend_type in ['Percent Counts', 'Diff Con Percents']:
        percents = np.asarray(percents) / sum(percents)
        percents_raw = np.asarray(percents_raw) / sum(percents_raw)

    # plot result
    if plot_result:
        percents_plot = []
        for percent in percents:
            if percent > 10 ** -10:
                percents_plot.append(percent)

        plt.figure(figsize=(6, 5))
        print(percents_plot)
        bar_list = plt.bar(cry_sys[0:len(percents_plot)], percents_plot)
        for i in range(len(percents_plot)):
            bar_list[i].set_color(c_vals[i])

        percents_raw_plot = []
        for percent_raw in percents_raw:
            if percent_raw > 10 ** -10:
                percents_raw_plot.append(percent_raw)
        bar_list = plt.bar(cry_sys[0:len(percents_raw_plot)], percents_raw_plot, fill=False, linewidth=5)
        for i in range(len(percents_plot)):
            bar_list[i].set_color([0, 0, 0])

        if save_figure:
            plt.rcParams['pdf.fonttype'] = 'truetype'
            plt.savefig(figure_path + '_prediction_histogram' + '.pdf', bbox_inches="tight")

        fig, ax = plt.subplots()
        ax.imshow(
            im_rgb,
            cmap='turbo', vmax=2, vmin=0)

        custom_lines = [Line2D([0], [0], color=c_vals[0], lw=4),
                        Line2D([0], [0], color=c_vals[1], lw=4),
                        Line2D([0], [0], color=c_vals[2], lw=4),
                        Line2D([0], [0], color=c_vals[3], lw=4),
                        Line2D([0], [0], color=c_vals[4], lw=4),
                        Line2D([0], [0], color=c_vals[5], lw=4)]

        cry_sys_label = cry_sys.copy()
        for c0 in range(len(percents)):
            if legend_type in ['Percent Counts', 'Diff Con Percents']:
                percents[c0] = percents[c0] * 100
                cry_sys_label[c0] += ' ' + str(round(percents[c0], 2)) + '%'
            else:
                cry_sys_label[c0] += ' ' + str(round(percents[c0], 2))
        ax.legend(custom_lines, cry_sys_label, loc='upper right', bbox_to_anchor=(1.5, 1))
        if save_figure:
            plt.rcParams['pdf.fonttype'] = 'truetype'
            plt.savefig(figure_path + '_prediction_image' + '.pdf', bbox_inches="tight")
        plt.show()

    if show_lattice:
        lattice_mask = np.zeros((im_shape[0], im_shape[1]))
        for a1 in range(len(cry_sys)):
            temp = stack_con[a1, :, :] > 0
            lattice_mask += temp.astype(int)

        # plt.imshow(lattice_mask)

        lattice_vals_masked = lattice_vals * lattice_mask
        plt.imshow(lattice_vals_masked, cmap=lattice_color_map)
        plt.colorbar()
        if save_figure:
            plt.rcParams['pdf.fonttype'] = 'truetype'
            plt.savefig(figure_path + '_lattice_prediction_image' + '.pdf', bbox_inches="tight")
        plt.show()

        # lattice_vals_count = []
        # for l1 in range(len(lattice_vals_masked)):
        #     for l2 in range(len(lattice_vals_masked[0])):
        #         if lattice_vals_masked[l1, l2] > 0:
        #             lattice_vals_count.append(lattice_vals_masked[l1, l2])
        # hist = plt.hist(lattice_vals_count, edgecolor='black', linewidth=1, density=True,
        #                 bins=np.linspace(2.7, 14.7, 121))
        # print(np.median(lattice_vals_count))
        # print(np.mean(lattice_vals_count))
        # plt.vlines(np.median(lattice_vals_count), 0, max(hist[0]), zorder=5, linewidth=3, color='r',
        #            label=str(round(np.median(lattice_vals_count), 2)))
        # plt.legend(fontsize=12)
        # if save_figure:
        #     plt.rcParams['pdf.fonttype'] = 'truetype'
        #     plt.savefig(figure_path + '_lattice_histogram' + '.pdf', bbox_inches="tight")
        # plt.show()

    if threshold_image:
        thresholded_image = im_rgb.copy()
        for col_ind in range(3):
            thresholded_image[:, :, col_ind] = thresholded_image[:, :, col_ind] * mask_threshold

        stack_con_thresholded = np.zeros((len(cry_sys), im_shape[0], im_shape[1]))
        for a1 in range(len(cry_sys)):
            stack_con_thresholded[a1, :, :] = stack_con[a1, :, :] * mask_threshold

        percents = []
        for a1 in range(len(cry_sys)):
            count = 0
            count_raw = 0
            for b0 in range(len(stack_con_thresholded[a1, :, :])):
                for b1 in range(len(stack_con_thresholded[a1, :, :][0])):
                    if stack_con_thresholded[a1, :, :][b0, b1] > mask_range[0]:
                        if legend_type in ['Percent Counts']:
                            count += 1
                        if legend_type in ['Diff Con', 'Diff Con Percents']:
                            count += stack_con_thresholded[a1, :, :][b0, b1]

            percents.append(count)
        if legend_type in ['Percent Counts', 'Diff Con Percents']:
            percents = np.asarray(percents) / sum(percents)
        # print(percents)

        if plot_result:
            percents_plot = []
            for percent in percents:
                if percent > 10 ** -10:
                    percents_plot.append(percent)

            plt.figure(figsize=(6, 5))
            bar_list = plt.bar(cry_sys[0:len(percents_plot)], percents_plot)
            for i in range(len(percents_plot)):
                bar_list[i].set_color(c_vals[i])

            if save_figure:
                plt.rcParams['pdf.fonttype'] = 'truetype'
                plt.savefig(figure_path + '_prediction_histogram_post_thresholding' + '.pdf', bbox_inches="tight")

            fig, ax = plt.subplots()
            ax.imshow(
                thresholded_image,
                cmap='turbo', vmax=2, vmin=0)

            custom_lines = [Line2D([0], [0], color=c_vals[0], lw=4),
                            Line2D([0], [0], color=c_vals[1], lw=4),
                            Line2D([0], [0], color=c_vals[2], lw=4),
                            Line2D([0], [0], color=c_vals[3], lw=4),
                            Line2D([0], [0], color=c_vals[4], lw=4),
                            Line2D([0], [0], color=c_vals[5], lw=4)]

            print(cry_sys)
            cry_sys_label = cry_sys
            print(cry_sys_label)
            for c0 in range(len(percents)):
                if legend_type in ['Percent Counts', 'Diff Con Percents']:
                    percents[c0] = percents[c0] * 100
                    cry_sys_label[c0] += ' ' + str(round(percents[c0], 2)) + '%'
                else:
                    cry_sys_label[c0] += ' ' + str(round(percents[c0], 2))

            ax.legend(custom_lines, cry_sys_label, loc='upper right', bbox_to_anchor=(1.5, 1))

            #             try:
            #                 for col_ind in range(3):
            #                 #     for i in range(0, len(mask_threshold)):
            #                 #         for j in range(0, len(mask_threshold)):
            #                 #             # print(len(thresholded_image[:,:,col_ind]))
            #                 #             # print(range(len(thresholded_image[:,:,col_ind])))
            #                 #             # print(i,j)
            #                 #             if mask_threshold[i][j] == 1:
            #                 #                 thresholded_image[:,:,col_ind][i+1][j] = thresholded_image[:,:,col_ind][i][j]
            #                 #                 thresholded_image[:,:,col_ind][i-1][j] = thresholded_image[:,:,col_ind][i][j]
            #                 #                 thresholded_image[:,:,col_ind][i+1][j+1] = thresholded_image[:,:,col_ind][i][j]
            #                 #                 thresholded_image[:,:,col_ind][i-1][j+1] = thresholded_image[:,:,col_ind][i][j]
            #                 #                 thresholded_image[:,:,col_ind][i+1][j-1] = thresholded_image[:,:,col_ind][i][j]
            #                 #                 thresholded_image[:,:,col_ind][i-1][j-1] = thresholded_image[:,:,col_ind][i][j]
            #                 #                 thresholded_image[:,:,col_ind][i][j+1] = thresholded_image[:,:,col_ind][i][j]
            #                 #                 thresholded_image[:,:,col_ind][i][j-1] = thresholded_image[:,:,col_ind][i][j]
            #                     thresholded_image[:,:,col_ind][binary_dilation(mask_threshold,np.ones((3,3),dtype='bool'))] =

            #             except IndexError:
            #                 print(i,j)
            # thresholded_image = binary_dilation(
            #     thresholded_image.astype('bool'),
            #     structure = np.ones((3,3,1),dtype='bool'),
            # )
            # k = np.array([
            #     [0.4,0.8,0.4],
            #     [0.8,1.0,0.8],
            #     [0.4,0.8,0.4],
            # ])
            k = np.array([
                [0.0, 0.2, 0.5, 0.2, 0.0],
                [0.2, 0.8, 1.0, 0.8, 0.2],
                [0.5, 1.0, 1.0, 1.0, 0.5],
                [0.2, 0.8, 1.0, 0.8, 0.2],
                [0.0, 0.2, 0.5, 0.2, 0.0],
            ])
            for a0 in range(3):
                thresholded_image[:, :, a0] = convolve2d(
                    thresholded_image[:, :, a0],
                    k,
                    mode='same',
                )

            thresholded_image = np.clip(thresholded_image, 0, 1)

            fig, ax = plt.subplots()
            ax.imshow(
                thresholded_image,
                # cmap = 'turbo',
                # vmax = 2,
                # vmin = 0,
                interpolation='bilinear',
            )
            if save_figure:
                plt.rcParams['pdf.fonttype'] = 'truetype'
                plt.savefig(figure_path + '_prediction_image_post_thresholding' + '.pdf', bbox_inches="tight")
    return im_rgb, stack_con


def threshold_particles(
        df_image,
        threshold=0.2,
        plot_result=True,
        sigma=0.0,
):
    im = df_image.copy().astype('float')

    if sigma > 0:
        im = gaussian_filter(
            im,
            sigma=sigma,
            mode='nearest',
        )

    keep = np.logical_and.reduce((
        im > np.roll(im, (-1, -1), axis=(0, 1)),
        im > np.roll(im, (0, -1), axis=(0, 1)),
        im > np.roll(im, (1, -1), axis=(0, 1)),
        im > np.roll(im, (-1, 0), axis=(0, 1)),
        im > np.roll(im, (1, 0), axis=(0, 1)),
        im > np.roll(im, (-1, 1), axis=(0, 1)),
        im > np.roll(im, (0, 1), axis=(0, 1)),
        im > np.roll(im, (1, 1), axis=(0, 1)),
        im > threshold,
    ))
    keep[:, 0] = False
    keep[:, -1] = False
    keep[0, :] = False
    keep[-1, :] = False

    xy_keep = np.argwhere(keep)

    mask_keep = np.zeros(im.shape, dtype='bool')
    mask_keep.ravel()[np.ravel_multi_index(
        (xy_keep[:, 0], xy_keep[:, 1]),
        im.shape)] = True

    if plot_result:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            mask_keep,
            cmap='gray',
        )
        # ax.scatter(
        #     xy_keep[:,1],
        #     xy_keep[:,0],
        #     marker = '+',
        #     s = 10,
        #     color = 'r',
        # )

    return mask_keep

def threshold_scaled_bf_image(bf_image, threshold = 0.4):
    indicies = []
    plt.imshow(bf_image, cmap = 'gist_gray')
    plt.colorbar()
    plt.show()
    for i in range(0, len(bf_image)):
        for j in range(0, len(bf_image[1])):
            if bf_image[i][j] < threshold:
                bf_image[i][j] = 0
            else:
                indicies.append([i,j])

    plt.imshow(bf_image, cmap = 'gist_gray')
    plt.colorbar()
    plt.show()
    return [bf_image, indicies]