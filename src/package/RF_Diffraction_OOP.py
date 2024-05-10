# import cudf
# import cuml
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
mpl.style.use('default')
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
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
# from dask.distributed import Client, progress, wait
# import dask
# import dask_cudf
# import cudf
# from cuml.dask.ensemble import RandomForestClassifier as cumlDaskRF
# from cuml.dask.common import utils as dask_utils
# from dask_cuda import LocalCUDACluster, initialize
# from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.metrics import r2_score
import matplotlib
import math
import scipy
from matplotlib.colors import hsv_to_rgb
import warnings
warnings.filterwarnings('ignore')


def generate_diff_aggregate_space_group(cry_sys, subdf, show_plots = True, include_legend = [False], include_difference_aggregation = True):
    # print(test_indicies)
    material_id = subdf.iloc[0].mat_id
    sg_true = subdf.iloc[0]['space_group_first']
    true_cry_sys = subdf.iloc[0]['crystal system']
    
    sg_full_pred = []
    sg_modes = []
    
    output_list = []
    
    all_predictions = []
    for i in range(0, len(subdf)):
        row = subdf.iloc[i]
        x_predictions = row['SG_full_pred']
        for x in x_predictions:
            all_predictions.append(x)
    
    # print(len(all_predictions))
    all_pred_df = pd.DataFrame(all_predictions, columns = ['all_predictions'])

    
    weighted_ag = {}
    sgs = all_pred_df['all_predictions'].unique()
    for sg in sgs:
        weighted_ag[sg] = []
    
    for i in range(0, len(subdf)):
        row = subdf.iloc[i]
        x_predictions = row['SG_full_pred']
        # print(x_predictions)


        mode = scipy.stats.mode(x_predictions)[0][0]

        sg_full_pred.append(x_predictions)
        sg_modes.append(mode)


        x_pred_df = pd.DataFrame(x_predictions, columns = ['x_predictions'])


        vals = pd.DataFrame(x_pred_df['x_predictions']).value_counts()
        # if mat == 'mp-10020':
        #     print(weighted_ag)
        vales_percent = vals/sum(vals)
        
        if len(vales_percent) == 1:
            diff = np.asarray(vales_percent)[0]
        else:
            diff = np.asarray(vales_percent)[0] - np.asarray(vales_percent)[1]
        # print(row)
        # if mat == 'mp-10020':
            # print(weighted_ag.keys())
        weighted_ag[row['SG_pred']].append(diff)
        
    confidence_df = pd.DataFrame([weighted_ag])
    weighted_sum = {}
    for key in confidence_df.columns:
        weighted_sum[str(key)] = sum(np.asarray(confidence_df[key])[0])


    max_sg = max(weighted_sum, key=weighted_sum.get)

    max_sg_val = weighted_sum[max_sg]
    confidence = max_sg_val/sum(weighted_sum.values())

    
    output_list = [sg_full_pred, sg_modes, sg_true, material_id, true_cry_sys, cry_sys, max_sg, weighted_sum, weighted_ag, confidence]
    
    # print(output_list)

    # print(len(np.asarray(output_list, dtype = 'object')))
    output_df = pd.DataFrame([np.asarray(output_list, dtype = 'object')], columns=['space_group_full_predictions', 'space_group_modes', 'space_group_true',
                                      'material_id', 'true_crystal_sys', 'predicted_crystal_sys', 'difference_aggregate_prediction', 
                                     'averaged_weights_space_group', 
                                  'full_weights_space_group',  'prediction_confidence'])
    # print(output_df)
    return output_df


class RF_Diffraction_model():
    def __init__(self, full_df_filepath, subdf_filepaths):
        self.full_df_filepath = full_df_filepath
        self.full_df = None
        self.subdf_filepaths = subdf_filepaths
        self.rf_model = None
        self.output_df = None
        self.loaded_submodels = [False, '']
        self.loaded_submodels_space_group = [False, '']
        self.lattice_rf_output = None
        self.radial_df_w_lattice = None
        self.lattice_radial_inputs = None
        self.rf_output_radial = None
    def load_full_df(self):
        with open(self.full_df_filepath, 'rb') as f:
            self.full_df = pickle.load(f)

    def flatten(self, list1):
        return [item for sublist in list1 for item in sublist]

    def map_predictions(self, prediction, rf_model=None, list_of_classes=None):
        # print(prediction)
        if rf_model != None:
            print(rf_model.classes_)
            classes = list(rf_model.classes_)
            index = classes.index(prediction)
            return index
        if list_of_classes != None:
            index = list_of_classes.index(prediction)
            return index
    
    def show_distributions(self, figure_path = 'Figures/Figure S1/Figure_S1.pdf', savefigure = False):
        self.full_df['crystal system'].value_counts().plot(kind='bar')
        plt.title('Number of Patterns Per Crystal System', fontsize = 20)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel('Crystal System', fontsize = 18)
        plt.ylabel('Number of Patterns', fontsize = 18)
        if savefigure:
            plt.savefig(figure_path, bbox_inches="tight")
        plt.show()
        
    def load_cry_sys_ouptut(self):
        print('loading crystal system model')
        self.rf_output_radial = joblib.load('Model_data/Crystal_sys_outputs/crystal_system_model.joblib')
        self.radial_train_ids = joblib.load('Model_data/Crystal_sys_inputs/radial_inputs_0_5.joblib')
        self.radial_test_ids = joblib.load('Model_data/Crystal_sys_inputs/radial_inputs_1_5.joblib')
        print('crystal system model loaded')
        
        
    def show_cm_and_uncertianty_individual_prediction(self, show_cm = False, show_uncertianty = False, savefigure = False, reset_output_df = False,
                                                     figure_path = []):
        
        if reset_output_df:
            self.rf_output_radial = None
        
        if type(self.rf_output_radial) == type(None): 
            self.load_cry_sys_ouptut()
        
        predictions_ordered = self.rf_output_radial[0]
        pred_crystal_system = self.rf_output_radial[1]
        rf_model = self.rf_output_radial[2]
        
        test_indicies = self.full_df.loc[self.full_df['mat_id'].isin(self.radial_test_ids)].index 
                
        
        labels_test_cry_sys = np.asarray(self.full_df.iloc[test_indicies]['crystal system'])
        mp_ids = np.asarray(self.full_df.iloc[test_indicies]['mat_id'])

        cm = confusion_matrix(labels_test_cry_sys, pred_crystal_system, labels = ['cubic', 'hexagonal', 'trigonal', 'tetragonal', 'monoclinic', 'orthorhombic'])
        trues = 0
        for i in range(0, len(cm)):
            trues += cm[i][i]
        
        accuracy = trues/len(pred_crystal_system)
        print('crystal system ' + str(accuracy))
        
        cm = cm/len(pred_crystal_system)
        for i in range(0, len(cm)):
            for j in range(0, len(cm[0])):
                cm[i][j] = round(cm[i][j]*100, 1)
            

        
        accuracy = trues/len(pred_crystal_system)
        print('point group ' + str(accuracy))
        # crystal_sys_alph = ['C', 'H', 'M', 'O', 'Te', 'Tr']
        crystal_sys_alph = ['C', 'H', 'Tr', 'Te', 'M', 'O']


        df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)
        
        # df_cm.to_pickle('df_cm_percent_unrounded.pkl')
        
        # print(df_cm)

        plt.figure(figsize=(15, 20))
        # sn.set(font_scale=1.4) # for label size 
        # cmap=matplotlib.cm.get_cmap('plasma')
        # ax = sn.heatmap(df_cm, annot=True, vmin = 0, vmax = 0.22, cmap = sn.color_palette("rocket_r", as_cmap=True))


        ax = sn.heatmap(df_cm, annot=True, cmap = 'Blues', vmin = 0, vmax = 23, cbar_kws={"ticks":[0.0,5,10,15,20], "location":'bottom', 
                                                                                         "fraction":0.2, 'pad':0.1, 'label':'Percent Test Set'})
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(42)
            
        ax.tick_params(rotation=0)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(42)
        # ax.set_title('Confusion Matrix with labels\n', fontsize = 36);
        ax.set_xlabel('Predicted Values', fontsize = 46)
        ax.set_ylabel('Actual Values ', fontsize = 46)
        if savefigure:
            plt.savefig(figure_path[0], bbox_inches="tight")
        plt.show()

        
        # crystal_sys_alph = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal']
        crystal_sys_alph = ['cubic', 'hexagonal', 'trigonal', 'tetragonal', 'monoclinic', 'orthorhombic']           
        
        # if os.path.exists('Model_data/Crystal_sys_outputs/prediction_matrix_confidence.npy') == False:
        predictions_matrix = []

        for j in range(0, len(crystal_sys_alph)):
            row = []
            for k in range(0, len(crystal_sys_alph)):
                row.append([])
            predictions_matrix.append(row)

        predictions_confidence = []

        predictions_ordered_cry_sys_full = []

        for i in range(0, len(predictions_ordered)):
            prediction_ordered_crystal_system = []
            for j in predictions_ordered[i]:
                prediction_ordered_crystal_system.append(rf_model.classes_[int(j)])

            prediction_ordered_cry_sys = prediction_ordered_crystal_system


            prediction_df_crystal_sys = pd.DataFrame(prediction_ordered_cry_sys, columns=['Predictions'])
            val_counts = prediction_df_crystal_sys['Predictions'].value_counts()

            confidence = max(val_counts) / sum(val_counts)
            predictions_confidence.append(confidence)

            prediction_mapped = self.map_predictions(pred_crystal_system[i], list_of_classes = crystal_sys_alph)
            true_mapped = self.map_predictions(labels_test_cry_sys[i], list_of_classes = crystal_sys_alph)
            predictions_matrix[true_mapped][prediction_mapped].append(confidence)

            predictions_ordered_cry_sys_full.append(prediction_ordered_cry_sys)


        for k in range(0, len(predictions_matrix)):
            for l in range(0, len(predictions_matrix)):
                if len(predictions_matrix[k][l]) == 0:
                    predictions_matrix[k][l] = 0
                else:
                    predictions_matrix[k][l] = np.mean(predictions_matrix[k][l])

            # np.save('Model_data/Crystal_sys_outputs/prediction_matrix_confidence.npy', predictions_matrix)  
        
        # else:
            # predictions_matrix = np.load('Model_data/Crystal_sys_outputs/prediction_matrix_confidence.npy')
        # crystal_sys_alph = ['C', 'H', 'M', 'O', 'Te', 'Tr']
        crystal_sys_alph = ['C', 'H', 'Tr', 'Te', 'M', 'O']

        df_cm = pd.DataFrame(predictions_matrix, crystal_sys_alph, crystal_sys_alph)
        plt.figure(figsize=(15, 20))
        # sn.set(font_scale=1.4) # for label size
        # ax = sn.heatmap(df_cm, annot=True, vmin=0.27, vmax=1, cmap = sn.color_palette("rocket", as_cmap=True))
        Reds = mpl.colormaps['Reds'].resampled(75)
        newcolors = Reds(np.linspace(0, 1, 75))
        white = Reds(range(75))[0]
        newcolorlist = list(newcolors)
        for i in range(0, 25):
            newcolorlist.insert(0, white)
        newcolors = np.asarray(newcolorlist)
        newcmp = ListedColormap(newcolors)
        
        cm = np.asarray(df_cm)
        for i in range(0, len(cm)):
            for j in range(0, len(cm[0])):
                # print(cm[i][j])
                cm[i][j] = np.mean(cm[i][j]).astype('float64')
        
        for i in range(0, len(cm)):
            for j in range(0, len(cm[0])):
                # print(cm[i][j])
                cm[i][j] = round(cm[i][j]*100, 0)
        
        df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)
        print(df_cm)
        
        ax = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=100, cmap = newcmp, cbar_kws={"location":'bottom', 
                                                                                         "fraction":0.2, 'pad':0.1, 'label':"Percent Trees"})

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(42)
            
        ax.tick_params(rotation=0)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(42)
        # ax.set_title('Confusion Matrix with labels\n', fontsize = 36);
        ax.set_xlabel('Predicted Values', fontsize = 46)
        ax.set_ylabel('Actual Values ', fontsize = 46)
        if savefigure:
            plt.savefig(figure_path[1], bbox_inches="tight")
        plt.show()
        
        
        
        output_list = []
        for i in range(0, len(labels_test_cry_sys)):
            output_list.append([pred_crystal_system[i], labels_test_cry_sys[i], 
                                predictions_confidence[i], test_indicies[i], 
                                 predictions_ordered_cry_sys_full[i], mp_ids[i]])



        output_df = pd.DataFrame(np.asarray(output_list),
                                 columns = ['Predictions Crystal System', 'True Values Crystal System', 
                                            'Confidence Crystal System', 'Full DF Indicies',
                                            'Full Predictions Crystal System', 'mat_id']
                                )

        patterns = []
        for i in np.asarray(output_df['Full DF Indicies']):
            patterns.append(self.full_df.iloc[i]['radial_200_ang_Colin_basis'])
        output_df['radial_200_ang_Colin_basis'] = patterns
        
        self.output_df = output_df
        
    def add_lattice_data_to_condensed_df(self):
        new_rows = []
        for i in range(0, len(self.condensed_output_df)):
            new_rows.append(None)
        self.condensed_output_df['pred_cry_sys_full_predictions_a'] = new_rows
        self.condensed_output_df['pred_cry_sys_full_predictions_b'] = new_rows
        self.condensed_output_df['pred_cry_sys_full_predictions_c'] = new_rows

        self.condensed_output_df['true_cry_sys_full_predictions_a'] = new_rows
        self.condensed_output_df['true_cry_sys_full_predictions_b'] = new_rows
        self.condensed_output_df['true_cry_sys_full_predictions_c'] = new_rows

        print('df ready')

        for cry_sys in ['trigonal', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'cubic']:    
        # for cry_sys in ['cubic']:
            print(cry_sys)
            rf_model = joblib.load('Model_data/Lattice_inputs_and_outputs/' +cry_sys+'_lattice_model.joblib')[2]
            print('model loaded')
            trees = rf_model.estimators_
            # for k in range(0, len(df)):
            count = 0
            for k in range(0, len(self.condensed_output_df)):
                row = self.condensed_output_df.iloc[k]
                if row['Aggregate Predictions Crystal System'] == cry_sys:
                    full_df_vals = row['Full Df Indicies']
                    subdf = self.output_df.loc[self.output_df['Full DF Indicies'].isin(full_df_vals)]                

                    cry_sys_vals = np.asarray(subdf.radial_200_ang_Colin_basis) 
                    input_train_temp = []
                    for unit in cry_sys_vals:
                        input_train_temp.append(flatten(unit))

                    input_train = []
                    for line in input_train_temp:
                        temp = []
                        abs_i = np.abs(line)
                        angle_i = np.angle(line)
                        for i in range(0, len(abs_i)):
                            temp.append(abs_i[i])
                            temp.append(angle_i[i])
                            # print(temp)
                        input_train.append(np.asarray(temp))

                    cry_sys_use = input_train

                    # print(cry_sys_use)
                    predictions_full = []
                    for tree in trees:
                        predictions_full.append(tree.predict(cry_sys_use))

                    predictions_ordered = np.asarray(predictions_full).T

                    # return predictions_ordered
                    # print(predictions_ordered[0])
                    self.condensed_output_df.at[k, 'pred_cry_sys_full_predictions_a'] = predictions_ordered[0]
                    self.condensed_output_df.at[k, 'pred_cry_sys_full_predictions_b'] = predictions_ordered[1]
                    self.condensed_output_df.at[k, 'pred_cry_sys_full_predictions_c'] = predictions_ordered[2]

                if row['True Values Crystal System'] == cry_sys:
                    full_df_vals = row['Full Df Indicies']
                    subdf = self.output_df.loc[self.output_df['Full DF Indicies'].isin(full_df_vals)]
                    cry_sys_vals = np.asarray(subdf.radial_200_ang_Colin_basis) 
                    input_train_temp = []
                    for unit in cry_sys_vals:
                        input_train_temp.append(flatten(unit))

                    input_train = []
                    for line in input_train_temp:
                        temp = []
                        abs_i = np.abs(line)
                        angle_i = np.angle(line)
                        for i in range(0, len(abs_i)):
                            temp.append(abs_i[i])
                            temp.append(angle_i[i])
                            # print(temp)
                        input_train.append(np.asarray(temp))

                    cry_sys_use = input_train
                    # print(cry_sys_use)
                    predictions_full = []
                    for tree in trees:
                        predictions_full.append(tree.predict(cry_sys_use))

                    predictions_ordered = np.asarray(predictions_full).T
                    # print(predictions_ordered[0])
                    self.condensed_output_df.at[k, 'true_cry_sys_full_predictions_a'] = predictions_ordered[0]
                    self.condensed_output_df.at[k, 'true_cry_sys_full_predictions_b'] = predictions_ordered[1]
                    self.condensed_output_df.at[k, 'true_cry_sys_full_predictions_c'] = predictions_ordered[2]
        
        
    def add_lattice_results_to_output_df(self, save_lattice_updated_df = True):
        new_rows = []
        for i in range(0, len(self.output_df)):
            new_rows.append(None)
        self.output_df['pred_cry_sys_full_predictions_a'] = new_rows
        self.output_df['pred_cry_sys_full_predictions_b'] = new_rows
        self.output_df['pred_cry_sys_full_predictions_c'] = new_rows

        self.output_df['true_cry_sys_full_predictions_a'] = new_rows
        self.output_df['true_cry_sys_full_predictions_b'] = new_rows
        self.output_df['true_cry_sys_full_predictions_c'] = new_rows

        print('df ready')

        # for cry_sys in ['trigonal', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'cubic']:  
        for cry_sys in ['hexagonal']:
            print(cry_sys)
            rf_model = joblib.load('Model_data/Lattice_inputs_and_outputs/' +cry_sys+'_lattice_model.joblib')[2]
            print('model loaded')
            trees = rf_model.estimators_
            # for k in range(0, len(df)):
            for k in range(0, len(self.output_df)):
                row = self.output_df.iloc[k]
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
                    self.output_df.at[k, 'pred_cry_sys_full_predictions_a'] = predictions_ordered[0][0]
                    self.output_df.at[k, 'pred_cry_sys_full_predictions_b'] = predictions_ordered[1][0]
                    self.output_df.at[k, 'pred_cry_sys_full_predictions_c'] = predictions_ordered[2][0]

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
                    self.output_df.at[k, 'true_cry_sys_full_predictions_a'] = predictions_ordered[0][0]
                    self.output_df.at[k, 'true_cry_sys_full_predictions_b'] = predictions_ordered[1][0]
                    self.output_df.at[k, 'true_cry_sys_full_predictions_c'] = predictions_ordered[2][0]

    def visualize_lattice_results_individual(self, cry_sys, cry_sys_type = 'Predictions Crystal System', prediction_type = 'Median',
                                            show_hists = False, show_r2 = False, update_output_df_with_lattice = False, 
                                             load_lattice_output_df_from_path = False,
                                            output_df_path = None, savefig = False, savenp=False, xlim = [0,4], use_xlim = True,
                                            ylims=[0,9]):
        mpl.style.use('default')

        if update_output_df_with_lattice:
            self.add_lattice_results_to_output_df()
        
        if load_lattice_output_df_from_path:
            self.output_df = joblib.load(output_df_path)
            
        # print(self.output_df)
        
        if cry_sys_type == 'True Values Crystal System':
            col = 'true_cry_sys_full_predictions'
        if cry_sys_type == 'Predictions Crystal System':
            col = 'pred_cry_sys_full_predictions'

        medians = []

        for char in ['a', 'b', 'c']:
            print(char)
            # print('Accurate MAE')
            if cry_sys == 'orthorhombic':
                # try:
                    # ortho_subdf = joblib.load(cry_sys_type[0:4]+'_ortho_lattice_subdf.joblib')

                # except FileNotFoundError:
                ortho_subdf = self.output_df.loc[self.output_df[cry_sys_type] == cry_sys]
                ortho_subdf = scale_df_lattice(ortho_subdf, 'mat_id')

                joblib.dump(ortho_subdf, cry_sys_type[0:4]+'_ortho_lattice_subdf.joblib')

                mean_per_pattern = []
                for i in np.asarray(self.output_df.loc[self.output_df[cry_sys_type] == cry_sys][col+'_'+char]):
                    if prediction_type == 'Mean':
                        mean_per_pattern.append(np.mean(i))
                    if prediction_type == 'Median':
                        mean_per_pattern.append(np.median(i))
                char_ortho = char + '_sorted'
                errors_all = np.abs(np.asarray(ortho_subdf.loc[ortho_subdf[cry_sys_type] == cry_sys][char_ortho]) - mean_per_pattern)
                cry_sys_subdf = ortho_subdf

                MAE_all = np.mean(errors_all)

                median_all = np.median(errors_all)

                medians.append(median_all)

                # print('MAE All ' + str(MAE_all))
                print('Median All ' + str(median_all))
                medians.append(median_all)
                mean_per_pattern_acc = []
                for i in np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys][col+'_'+char]):
                    if prediction_type == 'Mean':
                        mean_per_pattern_acc.append(np.mean(i))
                    if prediction_type == 'Median':
                        mean_per_pattern_acc.append(np.median(i))

                errors_acc = np.abs(np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys][char_ortho]) - mean_per_pattern_acc)
                MAE_acc = np.mean(errors_acc)

                mean_per_pattern_inacc = []
                for i in np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys][col+'_'+char]):
                    if prediction_type == 'Mean':
                        mean_per_pattern_inacc.append(np.mean(i))
                    if prediction_type == 'Median':
                        mean_per_pattern_inacc.append(np.median(i))

                errors_inacc = np.abs(np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys][char_ortho]) - mean_per_pattern_inacc)
                MAE_inacc = np.mean(errors_inacc)
                # print(MAE_acc)

            else:

                cry_sys_subdf = self.output_df.loc[self.output_df[cry_sys_type] == cry_sys]
                mean_per_pattern = []
                for i in np.asarray(self.output_df.loc[self.output_df[cry_sys_type] == cry_sys][col+'_'+char]):
                    if prediction_type == 'Mean':
                        mean_per_pattern.append(np.mean(i))
                    if prediction_type == 'Median':
                        mean_per_pattern.append(np.median(i))

                errors_all = np.abs(np.asarray(self.output_df.loc[self.output_df[cry_sys_type] == cry_sys][char]) - mean_per_pattern)

                MAE_all = np.mean(errors_all)

                median_all = np.median(errors_all)
                # print('MAE All ' + str(MAE_all))
                print('Median All ' + str(median_all))
                mean_per_pattern_acc = []
                for i in np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys][col+'_'+char]):
                    if prediction_type == 'Mean':
                        mean_per_pattern_acc.append(np.mean(i))
                    if prediction_type == 'Median':
                        mean_per_pattern_acc.append(np.median(i))

                errors_acc = np.abs(np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys][char]) - mean_per_pattern_acc)
                MAE_acc = np.mean(errors_acc)

                mean_per_pattern_inacc = []
                for i in np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys][col+'_'+char]):
                    if prediction_type == 'Mean':
                        mean_per_pattern_inacc.append(np.mean(i))
                    if prediction_type == 'Median':
                        mean_per_pattern_inacc.append(np.median(i))

                errors_inacc = np.abs(np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys][char]) - mean_per_pattern_inacc)
                MAE_inacc = np.mean(errors_inacc)
            # print(MAE_inacc)
                # print(MAE_acc)

            medians.append(median_all)

            median_acc = np.median(errors_acc)
            # print('MAE Accurate ' + str(MAE_acc))
            # print('Median Accurate ' + str(median_acc))        



            median_inacc = np.median(errors_inacc)
            # print('MAE Inaccurate ' + str(MAE_inacc))
            # print('Median Inaccurate ' + str(median_inacc))  
            # print(' ')

            if show_r2:
                plt.figure(figsize=(8, 7))
                true_vals = np.asarray(self.output_df.loc[self.output_df[cry_sys_type] == cry_sys][char])
                # plt.scatter(true_vals, mean_per_pattern, alpha = 1)
                r2 = r2_score(true_vals, mean_per_pattern)
                print(r2)
                true_val_inacc = np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys][char])
                plt.scatter(true_val_inacc, mean_per_pattern_inacc, alpha = 0.1, label = "Innacurate Crystal System", 
                           color = '#ff7f0e')

                true_val_acc = np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys][char])
                plt.scatter(true_val_acc, mean_per_pattern_acc, alpha = 0.1, label = 'True Crystal System',
                           color = '#1f77b4')


                plt.plot(np.linspace(min(true_vals), max(true_vals), 100), np.linspace(min(true_vals), max(true_vals), 100), linestyle = '--', color = 'k')
                plt.yticks(fontsize=20)
                plt.xticks(fontsize = 20)   
                plt.xlabel('True', fontsize = 30)
                plt.ylabel('Prediction', fontsize = 30)
                plt.legend(fontsize = 18)
                if savefig:
                    plt.savefig('Figures/Figure SI R2 individual/R2_plot_' + cry_sys + '_' + char + '.pdf', bbox_inches="tight", transparent=True)
                if savenp:
                    np.save('individual_pattern_'+cry_sys+'_'+char+'.npy', np.asarray([true_val_acc, mean_per_pattern_acc, true_val_inacc,
                                                                                       mean_per_pattern_inacc]))
                plt.show()        

            if show_hists:
                plt.figure(figsize=(8, 7))
                # hist_all = plt.hist(errors_all, bins = 20, weights=np.ones(len(errors_all)) / len(errors_all))
                # hist_acc = plt.hist(errors_acc, bins = 20, weights=np.ones(len(errors_acc)) / len(errors_all), label='Accurate Crystal System')
                # hist_innac = plt.hist(errors_inacc, bins = 20, weights=np.ones(len(errors_inacc)) / len(errors_all), label = 'Innacurate Crystal System')
                # print(np.arange(0.0, round(max(errors_all), 1)+0.1, 0.1))
                hist = plt.hist([errors_acc, errors_inacc], 
                                bins=np.arange(0.0, round(max(errors_all), 1)+0.1, 0.1), stacked=True, 
                                label=['Accuracte Prediction', 'Inaccurate Prediction'], density = True)
                
                # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
                # plt.title('Error Histogram', fontsize=18)
                plt.title(prediction_type+' Trees (Per Pattern Case)', fontsize = 32)
                # plt.vlines(median_all, max(hist_acc[0]), min(hist_acc[0]), color='k', linewidth=5, label='Median')
                # plt.vlines(MAE_all, max(hist_acc[0]), min(hist_acc[0]), color='red', linewidth=5, label='MAE')

                # plt.text(MAE_all + 0.5*max(errors_all), max(hist_acc[0]) - 0.1 * max(hist_acc[0]), 'MAE = ' + str(round(MAE_all, 2)),
                #          horizontalalignment='center', fontsize=32)
                plt.text(xlim[1]/2, max(hist[0][1])-0.2* max(hist[0][1]), 'Median Error = ' + str(round(median_all, 2)),
                         horizontalalignment='center', fontsize=32)
                if use_xlim:
                    plt.xlim(xlim)
                if type(ylims) == list:
                    plt.ylim(ylims)
                plt.yticks(fontsize=24)
                plt.xticks(fontsize = 24)



                # plt.xlim([0,4])
                plt.legend(fontsize = 17, loc='lower right')
                if char == 'c' or char=='c_sorted':
                    plt.xlabel('Absolute Error (Angstroms)', fontsize=32)
                if cry_sys == 'cubic':
                    plt.ylabel('Frequency', fontsize=42)
                if savefig:
                    plt.rcParams['pdf.fonttype'] = 'truetype'
                    plt.savefig('Figures/Figure SI Error Hist Individual/MAE_plot_' + cry_sys + '_' + char + '.pdf', bbox_inches="tight", transparent=True)
                plt.show()

        return medians
    
    
    def visualize_space_group_results_aggregate(self, cry_sys):
        
        warnings.filterwarnings('ignore')
        
        true_cry_sys = self.condensed_output_df.loc[self.condensed_output_df['True Values Crystal System']== cry_sys]
        true_cry_sys_ids = true_cry_sys.mat_id.to_numpy()
        test_cry_sys_correct_predictions = self.condensed_output_df.loc[(self.condensed_output_df['Aggregate Predictions Crystal System'] == cry_sys) & 
                                                                               (self.condensed_output_df['True Values Crystal System'] == cry_sys)]
        correct_cry_sys_ids = test_cry_sys_correct_predictions.mat_id.to_numpy()
        test_set_cry_sys_mispredictions = self.condensed_output_df.loc[(self.condensed_output_df['Aggregate Predictions Crystal System'] == cry_sys) & 
                                                                            (self.condensed_output_df['True Values Crystal System'] != cry_sys)]
        wrong_cry_sys_prediction_ids = test_set_cry_sys_mispredictions.mat_id.to_numpy()

        if cry_sys == 'orthorhombic': 
            self.visualize_space_group_results_mat_id(cry_sys, material_id = 'all', show_plots = False, use_scaled = True)

        else: 
            self.visualize_space_group_results_mat_id(cry_sys, material_id = 'all', show_plots = False, use_scaled = False)    



        universal_cry_sys_test_ids = []
        for mat_id in correct_cry_sys_ids:
            universal_cry_sys_test_ids.append(mat_id)
        print(len(universal_cry_sys_test_ids))

        # for i in integers_to_use:
            # universal_cry_sys_test_ids.append(wrong_cry_sys_prediction_ids[i])

        # try: 
        #     print('trying')
        #     with open('Model_data/Space_group_inputs_and_outputs/SG_'+cry_sys+'_cry_sys_test_df.pkl', 'rb') as f:
        #         cry_sys_test_df = pickle.load(f)
        #     print('loaded successfully')        

        # except:
        for mat_id in wrong_cry_sys_prediction_ids:
            universal_cry_sys_test_ids.append(mat_id)

        cry_sys_test_df = pd.DataFrame(columns = self.full_df.columns)
        for mat_id in universal_cry_sys_test_ids:
            # print(mat_id)
            subdf = self.full_df.loc[self.full_df.mat_id == mat_id]
            cry_sys_test_df = pd.concat([cry_sys_test_df, subdf])
        # cry_sys_test_df.to_pickle('Model_data/Space_group_inputs_and_outputs/SG_'+cry_sys+'_cry_sys_test_df.pkl')



        cry_sys_vals = np.asarray(cry_sys_test_df.radial_200_ang_Colin_basis) 

        print('starting input processing')

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

        pred_test = self.space_group_rf_output[2].predict(cry_sys_use)

        SG_pred = []

        for prediction in pred_test:
            SG_pred.append(prediction)

        cry_sys_test_df['SG_pred'] = SG_pred

        trees = self.space_group_rf_output[2].estimators_
        predictions_full = []
        for tree in trees:
            predictions_full.append(tree.predict(np.asarray(cry_sys_use)))
        predictions_ordered = np.asarray(predictions_full).T

        sg_full_pred = []
        for i in range(0, len(predictions_ordered)):
            prediction_ordered_sg = []
            for j in predictions_ordered[i]:
                prediction_ordered_sg.append(self.space_group_rf_output[2].classes_[int(j)])
            sg_full_pred.append(prediction_ordered_sg)

        cry_sys_test_df['SG_full_pred'] = sg_full_pred


        FINAL_cry_sys_test_df = pd.DataFrame( columns=['space_group_full_predictions', 'space_group_modes', 'space_group_true',
                                          'material_id', 'true_crystal_sys', 'predicted_crystal_sys', 'difference_aggregate_prediction', 
                                         'averaged_weights_space_group', 
                                      'full_weights_space_group',  'prediction_confidence'])
        print('starting aggregation')

        for mat_id in cry_sys_test_df.mat_id.unique():
            # print(mat_id)
            subdf = cry_sys_test_df.loc[cry_sys_test_df.mat_id == mat_id]
            out_df = generate_diff_aggregate_space_group(cry_sys, subdf, show_plots = False)
            FINAL_cry_sys_test_df = pd.concat([FINAL_cry_sys_test_df, out_df])




        print('Accurate Accuracy')
        acc_df = FINAL_cry_sys_test_df.loc[FINAL_cry_sys_test_df['true_crystal_sys'] == cry_sys]
        acc_df.reset_index(inplace = True)
        acc_df.drop('index', axis = 1)


        true_count_acc = 0
        for i in range(0, len(acc_df)):
            row = acc_df.iloc[i]
            if int(row['difference_aggregate_prediction']) == int(row['space_group_true']):
                true_count_acc += 1
        print(true_count_acc/len(acc_df))

        print('Inaccurate Accuracy')
        inacc_df = FINAL_cry_sys_test_df.loc[FINAL_cry_sys_test_df['true_crystal_sys'] != cry_sys]
        inacc_df.reset_index(inplace = True)
        inacc_df.drop('index', axis = 1)


        true_count_inacc = 0
        for i in range(0, len(inacc_df)):
            row = inacc_df.iloc[i]
            if int(row['difference_aggregate_prediction']) == int(row['space_group_true']):
                true_count_inacc += 1
        print(true_count_inacc/len(inacc_df))

        print('All Accuracy')
        print((true_count_inacc + true_count_acc)/len(FINAL_cry_sys_test_df))

    
        return FINAL_cry_sys_test_df
    
    
    def visualize_lattice_results_aggregate(self, cry_sys, cry_sys_type = 'True Values Crystal System', prediction_type = 'Median',
                                            show_hists = False, show_r2 = False, savefig = False, savenp=False, 
                                           use_xlim = True, xlim = [-0.3,5], ylims = None):
        mpl.style.use('default')
        plt.rcParams['pdf.fonttype'] = 'truetype'
        if cry_sys_type == 'True Values Crystal System':
            col = 'true_median'
        if cry_sys_type == 'Aggregate Predictions Crystal System':
            col = 'pred_median'

        medians = []

        for char in ['a', 'b', 'c']:
            print(char)
            # print('Accurate MAE')
            if cry_sys == 'orthorhombic':
                # try:
                    # ortho_subdf = joblib.load(cry_sys_type[0:4]+'_ortho_lattice_subdf.joblib')

                # except FileNotFoundError:
                ortho_subdf = self.condensed_output_df.loc[self.condensed_output_df[cry_sys_type] == cry_sys]
                ortho_subdf = ortho_subdf.rename(columns={'true_a': 'a', 'true_b': 'b', 'true_c': 'c'})
                ortho_subdf = scale_df_lattice(ortho_subdf, 'mat_id')
                    # joblib.dump(ortho_subdf, cry_sys_type[0:4]+'_ortho_lattice_subdf.joblib')

                mean_per_pattern = []
                for i in np.asarray(ortho_subdf.loc[ortho_subdf[cry_sys_type] == cry_sys][col+'_'+char]):
                    if prediction_type == 'Mean':
                        mean_per_pattern.append(np.mean(i))
                    if prediction_type == 'Median':
                        mean_per_pattern.append(np.median(i))
                char_ortho = char + '_sorted'
                errors_all = np.abs(np.asarray(ortho_subdf[ortho_subdf[cry_sys_type] == cry_sys][char_ortho]) - mean_per_pattern)
                cry_sys_subdf = ortho_subdf

                MAE_all = np.mean(errors_all)

                median_all = np.median(errors_all)

                medians.append(median_all)

                # print('MAE All ' + str(MAE_all))
                print('Median All ' + str(median_all))
                medians.append(median_all)
                mean_per_pattern_acc = []
                for i in np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys][col+'_'+char]):
                    if prediction_type == 'Mean':
                        mean_per_pattern_acc.append(np.mean(i))
                    if prediction_type == 'Median':
                        mean_per_pattern_acc.append(np.median(i))

                errors_acc = np.abs(np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys][char_ortho]) - mean_per_pattern_acc)
                MAE_acc = np.mean(errors_acc)

                mean_per_pattern_inacc = []
                for i in np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys][col+'_'+char]):
                    if prediction_type == 'Mean':
                        mean_per_pattern_inacc.append(np.mean(i))
                    if prediction_type == 'Median':
                        mean_per_pattern_inacc.append(np.median(i))

                errors_inacc = np.abs(np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys][char_ortho]) - mean_per_pattern_inacc)
                MAE_inacc = np.mean(errors_inacc)
                cry_sys_subdf = cry_sys_subdf.rename(columns={'a_sorted': 'true_a', 'b_sorted': 'true_b', 'c_sorted': 'true_c'})
                # print(MAE_acc)

            else:

                cry_sys_subdf = self.condensed_output_df.loc[self.condensed_output_df[cry_sys_type] == cry_sys]
                # mean_per_pattern = []
                # for i in np.asarray(output_df.loc[output_df[cry_sys_type] == cry_sys][col+'_'+char]):
                    # if prediction_type == 'Mode':
                        # todo finish

                if prediction_type == 'Median':
                    mean_per_pattern = np.asarray(self.condensed_output_df.loc[self.condensed_output_df[cry_sys_type] == cry_sys][col+'_'+char])

                errors_all = np.abs(np.asarray(self.condensed_output_df.loc[self.condensed_output_df[cry_sys_type] == cry_sys]['true_'+char]) - mean_per_pattern)

                MAE_all = np.mean(errors_all)

                median_all = np.median(errors_all)
                # print('MAE All ' + str(MAE_all))
                print('Median All ' + str(median_all))
                mean_per_pattern_acc = np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys][col+'_'+char])
                # mean_per_pattern_acc = []
                # for i in np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys][col+'_'+char]):
                    # if prediction_type == 'Mean':
                    #     mean_per_pattern_acc.append(np.mean(i))
                    # if prediction_type == 'Median':
                    #     mean_per_pattern_acc.append(np.median(i))

                errors_acc = np.abs(np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys]['true_'+char]) 
                                    - mean_per_pattern_acc)
                MAE_acc = np.mean(errors_acc)

                mean_per_pattern_inacc = np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys][col+'_'+char])
                # mean_per_pattern_inacc = []
                # for i in np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys][col+'_'+char]):
                #     if prediction_type == 'Mean':
                #         mean_per_pattern_inacc.append(np.mean(i))
                #     if prediction_type == 'Median':
                #         mean_per_pattern_inacc.append(np.median(i))

                errors_inacc = np.abs(np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys]['true_'+char]) 
                                      - mean_per_pattern_inacc)
                MAE_inacc = np.mean(errors_inacc)
            # print(MAE_inacc)
                # print(MAE_acc)

            medians.append(median_all)

            median_acc = np.median(errors_acc)
            # print('MAE Accurate ' + str(MAE_acc))
            # print('Median Accurate ' + str(median_acc))        



            median_inacc = np.median(errors_inacc)
            # print('MAE Inaccurate ' + str(MAE_inacc))
            # print('Median Inaccurate ' + str(median_inacc))  
            # print(' ')
            
            if show_r2:
                plt.figure(figsize=(8, 7))
                true_vals = np.asarray(cry_sys_subdf.loc[cry_sys_subdf[cry_sys_type] == cry_sys]['true_'+char])
                # plt.scatter(true_vals, mean_per_pattern, alpha = 1)
                r2 = r2_score(true_vals, mean_per_pattern)
                print(r2)
                true_val_inacc = np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] != cry_sys]['true_'+char])
                plt.scatter(true_val_inacc, mean_per_pattern_inacc, alpha = 0.33, label = "Innacurate Crystal System", 
                           color = '#ff7f0e', s=125)

                true_val_acc = np.asarray(cry_sys_subdf.loc[cry_sys_subdf['True Values Crystal System'] == cry_sys]['true_'+char])
                plt.scatter(true_val_acc, mean_per_pattern_acc, alpha = 0.33, label = 'True Crystal System',
                           color = '#1f77b4', s=125)


                plt.plot(np.linspace(min(true_vals), max(true_vals), 100), np.linspace(min(true_vals), max(true_vals), 100), linestyle = '--', color = 'k')
                plt.yticks(fontsize=32)
                plt.xticks(fontsize = 32)   
                plt.xlabel('True', fontsize = 40)
                plt.ylabel('Prediction', fontsize = 40)
                plt.legend(fontsize = 18)
                if savenp:
                    np.save('10_aggregate_'+cry_sys+'_'+char+'.npy', np.asarray([true_val_acc, mean_per_pattern_acc, true_val_inacc,
                                                                                       mean_per_pattern_inacc]))
                if savefig:
                    plt.savefig('Figures/Figure SI R2 Aggregate/R2_plot_' + cry_sys + '_' + char + '.pdf', bbox_inches="tight", transparent=True)
                plt.show()        

            if show_hists:
                plt.figure(figsize=(8, 7))
                # hist_all = plt.hist(errors_all, bins = 20, weights=np.ones(len(errors_all)) / len(errors_all))
                # hist_acc = plt.hist(errors_acc, bins = 20, weights=np.ones(len(errors_acc)) / len(errors_all), label='Accurate Crystal System')
                # hist_innac = plt.hist(errors_inacc, bins = 20, weights=np.ones(len(errors_inacc)) / len(errors_all), label = 'Innacurate Crystal System')
                hist = plt.hist([errors_acc, errors_inacc], 
                                bins=np.arange(0.0, round(max(errors_all), 1)+0.1, 0.1), stacked=True, 
                                label=['Accuracte Prediction', 'Inaccurate Prediction'], density = True)
                
                # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
                # plt.title('Error Histogram', fontsize=18)
                plt.title(prediction_type+' '+ cry_sys +' ' + char + ' Patterns', fontsize = 32)
                # plt.vlines(median_all, max(hist_acc[0]), min(hist_acc[0]), color='k', linewidth=5, label='Median')
                # plt.vlines(MAE_all, max(hist_acc[0]), min(hist_acc[0]), color='red', linewidth=5, label='MAE')

                # plt.text(MAE_all + 0.5*max(errors_all), max(hist_acc[0]) - 0.1 * max(hist_acc[0]), 'MAE = ' + str(round(MAE_all, 2)),
                #          horizontalalignment='center', fontsize=32)
                plt.text(xlim[1]/2, max(hist[0][1])-0.2* max(hist[0][1]), 'Median Error = ' + str(round(median_all, 2)),
                         horizontalalignment='center', fontsize=32)
                if use_xlim:
                    plt.xlim(xlim)
                if type(ylims) == list:
                    plt.ylim(ylims)

                plt.yticks(fontsize=24)
                plt.xticks(fontsize = 24)



                # plt.xlim([0,4])
                plt.legend(fontsize = 17, loc='lower right')
                if char == 'c' or char=='c_sorted':
                    plt.xlabel('Absolute Error (Angstroms)', fontsize=32)
                if cry_sys == 'cubic':
                    plt.ylabel('Frequency', fontsize=42)
                if savefig:
                    plt.savefig('Figures/Figure SI Error Hist Aggregate/MAE_plot_' + cry_sys + '_' + char + '.pdf', bbox_inches="tight", transparent=True)
                plt.show()

        return cry_sys_subdf
    
    
    
    
    def show_individual_sg_predictions(self, cry_sys, use_only_correct_cry_sys=False):
        self.loaded_submodels = [False, '']
        warnings.filterwarnings('ignore')
        print(cry_sys)
        if use_only_correct_cry_sys == False:
            true_cry_sys = self.output_df.loc[self.output_df['True Values Crystal System']== cry_sys]
            true_cry_sys_indicies = true_cry_sys.index
            test_cry_sys_correct_predictions = self.output_df.loc[(self.output_df['Predictions Crystal System'] == cry_sys) & 
                                                                                   (self.output_df['True Values Crystal System'] == cry_sys)]
            correct_cry_sys_indicies = test_cry_sys_correct_predictions.index
            test_set_cry_sys_mispredictions = self.output_df.loc[(self.output_df['Predictions Crystal System'] == cry_sys) & 
                                                                                (self.output_df['True Values Crystal System'] != cry_sys)]
            wrong_cry_sys_prediction_indicies= test_set_cry_sys_mispredictions.index


            universal_cry_sys_test_indicies = []
            for index in correct_cry_sys_indicies:
                universal_cry_sys_test_indicies.append(index)
            # print(len(universal_cry_sys_test_indicies))

            # for i in integers_to_use:
                # universal_cry_sys_test_ids.append(wrong_cry_sys_prediction_ids[i])

            # try: 
                # print('trying')
                # with open('Model_data/Space_group_inputs_and_outputs/individual_SG_'+cry_sys+'_cry_sys_test_df.pkl', 'rb') as f:
                #     cry_sys_test_df = pickle.load(f)
                # print('loaded successfully')        

            # except:
            for index in wrong_cry_sys_prediction_indicies:
                universal_cry_sys_test_indicies.append(index)

            cry_sys_test_df = pd.DataFrame(columns = self.output_df.columns)
            cry_sys_test_df = self.output_df.iloc[universal_cry_sys_test_indicies]

        # cry_sys_test_df.to_pickle('Model_data/Space_group_inputs_and_outputs/individual_SG_'+cry_sys+'_cry_sys_test_df.pkl')
        
        else:
            true_cry_sys = self.output_df.loc[self.output_df['True Values Crystal System']== cry_sys]
            true_cry_sys_indicies = true_cry_sys.index
            cry_sys_test_df = pd.DataFrame(columns = self.output_df.columns)
            cry_sys_test_df = self.output_df.iloc[true_cry_sys_indicies]

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
        rf_model = joblib.load('Model_data/Space_group_inputs_and_outputs/' +cry_sys+'_space_group_model.joblib')[2]
        pred_test = rf_model.predict(cry_sys_use)
        trees = rf_model.estimators_
        predictions_full = []
        for tree in trees:
            predictions_full.append(tree.predict(np.asarray(cry_sys_use)))
        predictions_ordered = np.asarray(predictions_full).T

        SG_pred = []

        for prediction in pred_test:
            SG_pred.append(prediction)

        cry_sys_test_df['SG_pred'] = SG_pred
        sg_full_pred = []
        for i in range(0, len(predictions_ordered)):
            prediction_ordered_sg = []
            for j in predictions_ordered[i]:
                prediction_ordered_sg.append(rf_model.classes_[int(j)])
            sg_full_pred.append(prediction_ordered_sg)

        cry_sys_test_df['SG_full_pred'] = sg_full_pred


        print('Accurate Accuracy')
        acc_df = cry_sys_test_df.loc[cry_sys_test_df['True Values Crystal System'] == cry_sys]
        acc_df.reset_index(inplace = True)
        acc_df.drop('index', axis = 1)


        true_count_acc = 0
        for i in range(0, len(acc_df)):
            row = acc_df.iloc[i]
            if row['SG_pred'] == row['True Value Space Group']:
                true_count_acc += 1
        print(true_count_acc/len(acc_df))

        print('Inaccurate Accuracy')
        inacc_df = cry_sys_test_df.loc[cry_sys_test_df['True Values Crystal System'] != cry_sys]
        inacc_df.reset_index(inplace = True)
        inacc_df.drop('index', axis = 1)


        true_count_inacc = 0
        for i in range(0, len(inacc_df)):
            row = inacc_df.iloc[i]
            if row['SG_pred'] == row['True Value Space Group']:
                true_count_inacc += 1
        print(true_count_inacc/len(inacc_df))

        print('All Accuracy')
        print((true_count_inacc + true_count_acc)/len(cry_sys_test_df))

        return cry_sys_test_df

    def show_specific_pattern(self, materials_id, zone, thickness_col = 'thickness_200_ang', include_y_label=True, include_x_label=True):
        # subdf = self.full_df.loc[self.full_df['mat_id'] == materials_id]
        # if len(zone) == 0:
        #     zone_to_use = subdf_id.iloc[0]['zone']
        # else:
        #     zone_to_use = zone
            
            
        subdf = self.full_df.loc[self.full_df['mat_id'] == materials_id]
        points = subdf.loc[subdf['zone'] == zone]['thickness_200_ang'].to_numpy()[0]
        # print(points)
        fig, ax = plot_diffraction_pattern(
            points,
            include_y_label = include_y_label,
            include_x_label = include_x_label,
            scale_markers=10000,
            scale_markers_compare = 5000,
            shift_labels=0.01,
            shift_marker=0.004,
            add_labels = False,
            min_marker_size=0,
            figsize=(8,8),
            returnfig = True
            )
        
        return fig, ax


    

    def show_feature_importances(self):
        plt.figure(figsize=(12, 10))
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.ylabel('Normalized Variance Reduction', fontsize=30)
        plt.xlabel('Radial Index', fontsize=30)
        plt.title('Feature Importances', fontsize=30)
        plt.plot(np.arange(0, 546, 1),
                 self.rf_model.feature_importances_, linewidth=3, color='#ff7f0e')
        loc = 0
        for i in range(0, 21):
            plt.vlines(loc, 0, max(self.rf_model.feature_importances_), color='k', linestyle='--')
            loc += 26
        plt.show()

    def lattice_visualize_predictions(self, predictions_ordered, predictions, rf_model, labels_test, inputs_test,
                                      test_indicies, predict_mode = True, return_df = False):
        print('really starting')
        count = 0
        predictions_full = []
        trees = rf_model.estimators_
        # print(len(trees))
        # for tree in trees:
        # predictions_full.append(tree.predict(np.asarray(inputs_test)))
        # predictions_ordered = np.asarray(predictions_full).T
        uncertianties = []
        for param in ['x', 'y', 'z', 'alpha', 'beta', 'gamma']:
            # print(predictions_ordered[0])

            # print(predictions_std)
            plt.figure(figsize=(8, 7))
            x_labels = labels_test[param].to_numpy()
            x_predictions = predictions.T[count]
            # print(x_labels)
            # print(x_predictions)
            # errors = np.abs(x_labels - x_predictions)
            # errors = np.asarray(errors)
            # predictions_std = []
            # for prediction in predictions_ordered[count]:
                # predictions_std.append(np.std(prediction))
            # uncertianties.append(predictions_std)
            # print(len(errors))
            # print(len(predictions_std))
            # print(errors)
            # print(predictions_std)
            # plt.scatter(errors, predictions_std)
            # plt.title('Errors vs Prediction Std', fontsize=18)
            # plt.xticks(fontsize=16)
            # plt.yticks(fontsize=16)
            # plt.xlabel('Error in ' + param + ' Prediction', fontsize=16)
            # plt.ylabel('Prediction Std', fontsize=16)
            # plt.show()

            # MSE = np.square(errors).mean()
            # RMSE = math.sqrt(MSE)
            # print('RMSE ' + str(RMSE))

            # plt.figure(figsize=(8, 7))
            # plt.title('Error Histogram', fontsize=18)
            # hist = plt.hist(errors, bins=50)
            # plt.vlines(RMSE, max(hist[0]), min(hist[0]), color='limegreen', linewidth=5, label='RMSE')
            # plt.text(RMSE + 0.25, max(hist[0]) - 0.1 * max(hist[0]), 'RMSE = ' + str(round(RMSE, 3)),
            #          horizontalalignment='center', fontsize=16)
            # plt.xticks(fontsize=16)
            # plt.yticks(fontsize=16)
            # plt.xlabel('Error', fontsize=16)
            # plt.ylabel('Frequency', fontsize=16)
            # plt.show()

            print(str(r2_score(x_labels, x_predictions)))

            plt.figure(figsize=(8, 7))
            plt.scatter(x_labels, x_predictions, s = 50)
            # plt.plot([2,12.5], [2,12.5], color = 'k')
            # plt.xlim([1.9, 3.1])
            # plt.ylim([1.9, 3.1])
            plt.title('Predicted vs True a', fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel('a Prediction', fontsize=22)
            plt.ylabel('True a', fontsize=22)
            plt.show()
            count += 1
            
            
        if return_df:
            output_list = []
            print('starting loop')
            for i in range(0, len(labels_test)):
                output_list.append([predictions[i], predictions[i][0], predictions[i][1], predictions[i][2],
                                    predictions[i][3], predictions[i][4], predictions[i][5],
                                    np.asarray(labels_test)[i], np.asarray(labels_test)[i][0],
                                    np.asarray(labels_test)[i][1],
                                    np.asarray(labels_test)[i][2], np.asarray(labels_test)[i][3],
                                    np.asarray(labels_test)[i][4],
                                    np.asarray(labels_test)[i][5],
                                    test_indicies[i], uncertianties[0][i], uncertianties[1][i], uncertianties[2][i],
                                    uncertianties[3][i],
                                    uncertianties[4][i], uncertianties[5][i]])

            output_df = pd.DataFrame(np.asarray(output_list),
                                     columns=['All Predictions', 'Predictions a', 'Predictions b', 'Predictions c',
                                              'Predictions alpha', 'Predictions beta', 'Predictions gamma',
                                              'Full Labels Test',
                                              'True a', 'True b', 'True c', 'True alpha', 'True beta', 'True gamma',
                                              'Full DF Index', 'std a', 'std b', 'std c', 'std alpha', 'std beta',
                                              'std gamma']

                                     )

            return output_df

    
    
    
    def lattice_visualize_predictions_by_material(self, material_id, show_plots, param_list = None, index_list = None,
                                                 savefigure=False, include_legend = [False], filename = None, xticks = None):
        
        row = self.condensed_output_df.loc[self.condensed_output_df['mat_id'] == material_id]
        
        output_list = []
        temp_list = []        
        
        if index_list == None:
            index_list = [0,1,2,3,4,5]
        if param_list == None:
            if scaled:
                param_list = ['a_sorted', 'b_sorted', 'c_sorted', 'alpha', 'beta', 'gamma']
            else:
                param_list = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        
        count = 0
        columns_for_df = []
        full_predictions = []
        for param in param_list:

            x_labels = row['true_'+param]
            x_predictions = row['full_true_median_'+param]
            x_ag_prediction = row['true_median_'+param]
            param_predictions = row['true_cry_sys_full_predictions_'+param]

            plt.figure(figsize=(8, 7))
            x_predictions = x_predictions.to_numpy()[0]
            self.x_predictions_kde = scipy.stats.gaussian_kde(x_predictions)
            # hist_pred = plt.hist(x_predictions, bins = 12, edgecolor = "black", linewidth = 3)
            vals = np.arange(min(x_predictions)-0.1*min(x_predictions), max(x_predictions)+0.1*max(x_predictions), 0.01)
            calc_pdf =  self.x_predictions_kde.pdf(vals)
            plt.plot(vals, calc_pdf, color = '#1f77b4', linewidth = 5)
            ax = plt.gca()
            if type(xticks) == type(None):
                plt.xticks(fontsize = 34)
            else:
                plt.xticks(xticks[count], fontsize = 34)

            plt.yticks(fontsize = 34)

            plt.ylabel('Percent Patterns', fontsize = 34)
            
            plt.vlines(x_labels, max(calc_pdf), 0, label = 'True ' + param, color = 'magenta', linewidth = 10)
            plt.vlines(x_ag_prediction, max(calc_pdf), 0, label = 'Prediction ' + param, color = 'gold', linewidth = 10)
            plt.xlabel('Predicted ' + param, fontsize = 34)

            if include_legend[count]:
                plt.legend(fontsize = 28)

            if savefigure:
                plt.savefig(filename+param+'.pdf', bbox_inches="tight")
            plt.show()
            count +=1         
        
        output_df = row
        return output_df
        

    
    def lattice_prepare_training_and_test(self, df, df_type='radial', test_fraction=0.25, column_to_use='200_ang',
                                          split_by_material=True, use_scaled_cols = False):

        if df_type == 'radial':
            col_name = 'radial_' + column_to_use + '_Colin_basis'
        if df_type == 'zernike':
            col_name = 'zernike_' + column_to_use

        input_col = np.asarray(df[col_name])
        labels = df[['x', 'y', 'z', 'alpha', 'beta', 'gamma']]

        if split_by_material:
            mat_ids = df['mat_id'].unique()
            dummy_output = np.ones((len(mat_ids)))
            mat_ids_train, mat_ids_test, dummy_train, dummy_test = train_test_split(mat_ids, dummy_output,
                                                                                    test_size=test_fraction,
                                                                                    random_state=32)
            print(len(mat_ids_train))
            print(len(mat_ids_test))

            input_train_first = df.loc[df['mat_id'].isin(mat_ids_train)][col_name]
            input_test_first = df.loc[df['mat_id'].isin(mat_ids_test)][col_name]

            if use_scaled_cols:
                print('using scaled cols')
                labels_train = df.loc[df['mat_id'].isin(mat_ids_train)][['a_sorted', 'b_sorted', 'c_sorted', 'alpha', 'beta', 'gamma', 'mat_id']]
                labels_test = df.loc[df['mat_id'].isin(mat_ids_test)][['a_sorted', 'b_sorted', 'c_sorted', 'alpha', 'beta', 'gamma', 'mat_id']]

            else:
                print('using unscaled cols')
                labels_train = df.loc[df['mat_id'].isin(mat_ids_train)][['x', 'y', 'z', 'alpha', 'beta', 'gamma', 'mat_id']]
                labels_test = df.loc[df['mat_id'].isin(mat_ids_test)][['x', 'y', 'z', 'alpha', 'beta', 'gamma', 'mat_id']]

        else:
            input_train_first, input_test_first, labels_train, labels_test = train_test_split(input_col, labels,
                                                                                              test_size=test_fraction,
                                                                                              random_state=32)
        if len(input_train_first[0].shape) < 2:
            input_train_temp = input_train_first
            input_test_temp = input_test_first

        else:
            input_train_temp = []
            for unit in input_train_first:
                input_train_temp.append(self.flatten(unit))

            input_test_temp = []
            for unit in input_test_first:
                input_test_temp.append(self.flatten(unit))

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

        input_test = []
        for row in input_test_temp:
            temp = []
            abs_i = np.abs(row)
            angle_i = np.angle(row)
            for i in range(0, len(abs_i)):
                temp.append(abs_i[i])
                temp.append(angle_i[i])
            input_test.append(temp)

        return [input_train, input_test, labels_train, labels_test, mat_ids_train, mat_ids_test]

    def visualize_lattice_results(self, crystal_system = 'orthorhombic'):
        
        print(crystal_system)

        # crystal_system_caps = crystal_system.upper()
        crystal_system_caps = crystal_system
        
        lattice_rf_output = joblib.load('Model_data/Lattice_inputs_and_outputs/'+crystal_system_caps+'_lattice_model.joblib')
        radial_df_w_lattice = joblib.load('Model_data/Lattice_inputs_and_outputs/radial_radial_' + crystal_system + '_df_w_lattice.joblib')
        lattice_radial_inputs = self.lattice_prepare_training_and_test(radial_df_w_lattice,
                                                                  split_by_material=True)
        print('starting')
        full_out_df = self.lattice_visualize_predictions(lattice_rf_output[0], lattice_rf_output[1], lattice_rf_output[2],
                                                    lattice_radial_inputs[3], lattice_radial_inputs[1],
                                                    list(lattice_radial_inputs[3].index))
        
    def load_submodels(self, crystal_system, crystal_system_caps, use_scaled):
        path_front = 'Model_data/'
        if use_scaled:
            print('using scaled')
            self.lattice_rf_output = joblib.load(path_front+'Lattice_inputs_and_outputs/'+crystal_system+'_lattice_model.joblib')
            self.radial_df_w_lattice = joblib.load(path_front+'Lattice_inputs_and_outputs/SCALED_radial_' + crystal_system + '_df_w_lattice.joblib')
        else:
            print('not using scaled')
            self.lattice_rf_output = joblib.load(path_front+'Lattice_inputs_and_outputs/'+crystal_system+'_lattice_model.joblib')
            self.radial_df_w_lattice = joblib.load(path_front+'Lattice_inputs_and_outputs/radial_' + crystal_system + '_df_w_lattice.joblib')
        
        try:
            lattice_radial_inputs_0 = joblib.load(path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_input_train_0.joblib')
            lattice_radial_inputs_1 = joblib.load(path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_input_test_1.joblib')
            with open(path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_inputs_labels_train_2.pkl', 'rb') as f:
                lattice_radial_inputs_2 = pickle.load(f)
                
            if 'x' in lattice_radial_inputs_2.columns:
                lattice_radial_inputs_2 = lattice_radial_inputs_2.rename(columns={'x': 'a', 'y': 'b', 'z': 'c'})
                lattice_radial_inputs_2.to_pickle(path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_inputs_labels_train_2.pkl')
            
            with open(path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_labels_test_3.pkl', 'rb') as f:
                lattice_radial_inputs_3 = pickle.load(f)
                
            if 'x' in lattice_radial_inputs_3.columns:
                lattice_radial_inputs_3 = lattice_radial_inputs_3.rename(columns={'x': 'a', 'y': 'b', 'z': 'c'})
                lattice_radial_inputs_3.to_pickle(path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_labels_test_3.pkl')

            lattice_radial_inputs_4 = joblib.load(path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_train_ids.joblib')
            lattice_radial_inputs_5 = joblib.load(path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_test_ids.joblib')


            self.lattice_radial_inputs = [lattice_radial_inputs_0, lattice_radial_inputs_1, lattice_radial_inputs_2, lattice_radial_inputs_3, 
                                         lattice_radial_inputs_4, lattice_radial_inputs_5]
            self.lattice_radial_inputs[3] = self.lattice_radial_inputs[3].reset_index()
            self.lattice_radial_inputs[3].drop(columns = ['index'], axis = 1, inplace = True)
            
            self.lattice_radial_inputs[3]['mat_id'] = self.full_df.loc[self.full_df['mat_id'].isin(self.lattice_radial_inputs[5])].mat_id.to_numpy()
            
            
        
        except:
            self.lattice_radial_inputs = self.lattice_prepare_training_and_test(self.radial_df_w_lattice,
                                                                      split_by_material=True, use_scaled_cols = use_scaled)
            self.lattice_radial_inputs[3] = self.lattice_radial_inputs[3].reset_index()
            self.lattice_radial_inputs[3].drop(columns = ['index'], axis = 1, inplace = True)
            
            joblib.dump(self.lattice_radial_inputs[4], path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_train_ids.joblib')
            joblib.dump(self.lattice_radial_inputs[5], path_front+'Lattice_inputs_and_outputs/'+crystal_system + '_lattice_test_ids.joblib')


        
        self.loaded_submodels = [True, crystal_system]
        

    
        
    def load_output_df_radial(self, path = None):
        if path == None:
            output_df_radial =  joblib.load('output_df_radial_cry_sys_split_by_mat_id.joblib')
        else:
            output_df_radial =  joblib.load(path)
        
        self.output_df = output_df_radial
        
    def add_full_pred_median_to_condensed_df(self):
        for col in ['pred_cry_sys_full_predictions_a',
                    'pred_cry_sys_full_predictions_b',	
                    'pred_cry_sys_full_predictions_c',	
                    'true_cry_sys_full_predictions_a',	
                    'true_cry_sys_full_predictions_b',	
                    'true_cry_sys_full_predictions_c']:
            print(col)
            full_pred_median = []
            ag_medians = []
            for i in range(0, len(self.condensed_output_df)):
                row = self.condensed_output_df.iloc[i]
                full_medians = []
                for full_pred in row[col]:
                    full_medians.append(np.median(full_pred))
                full_pred_median.append(full_medians)

            for medians in full_pred_median:
                ag_medians.append(np.median(medians))
            self.condensed_output_df[col[0:4]+'_median_'+col[len(col)-1]] = ag_medians
            self.condensed_output_df['full_'+col[0:4]+'_median_'+col[len(col)-1]] = full_pred_median
            
        ids = []
        a_trues = []
        b_trues = []
        c_trues = []
        for i in range(0, len(self.output_df)):
            row = self.output_df.iloc[i]
            mat_id = row['mat_id']
            if mat_id not in ids:
                print(i)
                ids.append(row['mat_id'])
                a_trues.append(row['a'])
                b_trues.append(row['b'])
                c_trues.append(row['c'])

        self.condensed_output_df['reference_id'] = ids
        self.condensed_output_df['true_a'] = a_trues
        self.condensed_output_df['true_b'] = b_trues
        self.condensed_output_df['true_c'] = c_trues

    
    def condense_crystal_system_output(self, method = 'naive', load_from_path = False, path = None, num = 100):
        if load_from_path == False:
            ids = self.output_df.mat_id.unique()
            final_full_out_df = pd.DataFrame(columns = ['mat_id', 'True Values Crystal System', 'Full Predictions Crystal System', 
                                                        'Aggregate Predictions Crystal System', 'Prediction Confidence', 'Full Df Indicies'])

        if method == 'naive':
            for mat in ids:    
                print(mat)
                subdf = self.output_df.loc[self.output_df['mat_id'] == mat]
                labels_test_crystal_sys = subdf.iloc[0]['True Values Crystal System']

                predictions_across_zones_cry_sys = subdf['Predictions Crystal System']

                old_df_indicies = list(subdf['Full DF Indicies'])
                xs_crystal = np.asarray(predictions_across_zones_cry_sys.value_counts().index)
                ys_crystal = np.asarray(predictions_across_zones_cry_sys.value_counts())
                predictions_across_zones_cry_sys = np.asarray(predictions_across_zones_cry_sys)

                full_out_df = pd.DataFrame(np.asarray([[mat], [labels_test_crystal_sys], [predictions_across_zones_cry_sys], [xs_crystal[0]], 
                                                       [ys_crystal[0]/len(subdf)], [old_df_indicies]], dtype = object).T, 
                                      columns=['mat_id', 'True Values Crystal System', 'Full Predictions Crystal System', 
                                               'Aggregate Predictions Crystal System', 'Prediction Confidence', 'Full Df Indicies'])
                final_full_out_df = pd.concat([final_full_out_df, full_out_df])
        
        if method == 'Full Histogram':
            for mat in ids:    
                print(mat)
                subdf = self.output_df.loc[self.output_df['mat_id'] == mat]
                labels_test_crystal_sys = subdf.iloc[0]['True Values Crystal System']
                predictions_across_zones_cry_sys = []
                for i in range(0, len(subdf)):
                    for pred in subdf.iloc[i]['Full Predictions Crystal System']:
                        predictions_across_zones_cry_sys.append(pred)
                old_df_indicies = list(subdf['Full DF Indicies'])
                predictions_across_zones_cry_sys_df = pd.DataFrame(predictions_across_zones_cry_sys, columns = ['Full_predictions_across_zones'])
                xs_crystal = np.asarray(predictions_across_zones_cry_sys_df.value_counts().index)
                ys_crystal = np.asarray(predictions_across_zones_cry_sys_df.value_counts())
                full_out_df = pd.DataFrame(np.asarray([[mat], [labels_test_crystal_sys], [predictions_across_zones_cry_sys], xs_crystal[0], 
                                                       [ys_crystal[0]/(80*len(subdf))], [old_df_indicies]], dtype = object).T, 
                                      columns=['mat_id', 'True Values Crystal System', 'Full Predictions Crystal System',
                                               'Aggregate Predictions Crystal System', 
                                               'Prediction Confidence', 'Full Df Indicies'])
                final_full_out_df = pd.concat([final_full_out_df, full_out_df])
            
            
        if method == 'Weighted Aggregation':
            for mat in ids:    
                print(mat)
                subdf = self.output_df.loc[self.output_df['mat_id'] == mat]
                
                weighted_ag = {'cubic':[], 'hexagonal':[], 'tetragonal':[], 'trigonal':[], 'monoclinic':[], 'orthorhombic':[]}
                
                for i in range(0, len(subdf)):
                    row = subdf.iloc[i]
                    weighted_ag[row['Predictions Crystal System']].append(row['Confidence Crystal System'])
                    
                confidence_df = pd.DataFrame([weighted_ag])
                
                weighted_sum = {'cubic':sum(np.asarray(confidence_df['cubic'])[0]),
                                'hexagonal':sum(np.asarray(confidence_df['hexagonal'])[0]), 
                                'tetragonal':sum(np.asarray(confidence_df['tetragonal'])[0]), 
                                'trigonal':sum(np.asarray(confidence_df['trigonal'])[0]), 
                                'monoclinic':sum(np.asarray(confidence_df['monoclinic'])[0]), 
                                'orthorhombic':sum(np.asarray(confidence_df['orthorhombic'])[0])}
                
                max_cry_sys = max(weighted_sum, key=weighted_sum.get)
                
                max_cry_sys_val = weighted_sum[max_cry_sys]
                confidence = max_cry_sys_val/sum(weighted_sum.values())
                
                labels_test_crystal_sys = subdf.iloc[0]['True Values Crystal System']
                full_pred_crystal_system = list(subdf['Full Predictions Crystal System'])


                old_df_indicies = list(subdf['Full DF Indicies'])


                full_out_df = pd.DataFrame(np.asarray([[mat], [labels_test_crystal_sys], [full_pred_crystal_system], [max_cry_sys], [weighted_ag], [weighted_sum],
                                                       [confidence], [old_df_indicies]], dtype = object).T, 
                                      columns=['mat_id', 'True Values Crystal System',  'Full Predictions Crystal System', 'Aggregate Predictions Crystal System', 
                                               'Averaged Weights Crystal System', 
                                              'Full Weights Crystal System',  'Prediction Confidence', 'Full Df Indicies'])
                

                final_full_out_df = pd.concat([final_full_out_df, full_out_df])
                
                
        if method == 'Difference Aggregation':
            if load_from_path:
                self.condensed_output_df =  joblib.load(path)
                if 'true_a' not in self.condensed_output_df.columns:
                    self.add_full_pred_median_to_condensed_df()
                    joblib.dump(self.condensed_output_df, path)
            
            else:
                for mat in ids:    
                    print(mat)
                    subdf = self.output_df.loc[self.output_df['mat_id'] == mat]
                    subdf.reset_index(inplace = True)
                    subdf.drop('index', axis=1, inplace = True)
                    inds = np.random.randint(0,100,num)
                    subdf = subdf.iloc[inds]
                    subdf.reset_index(inplace = True)
                    subdf.drop('index', axis=1, inplace = True)
                               
                    weighted_ag = {'cubic':[], 'hexagonal':[], 'tetragonal':[], 'trigonal':[], 'monoclinic':[], 'orthorhombic':[]}

                    for i in range(0, len(subdf)):
                        # print(subdf)
                        row = subdf.iloc[i]
                        vals = pd.DataFrame(row['Full Predictions Crystal System']).value_counts()
                        # if mat == 'mp-10020':
                        #     print(weighted_ag)
                        vales_percent = vals/sum(vals)
                        if len(vales_percent) == 1:
                            diff = vales_percent[0]
                        else:
                            diff = vales_percent[0]-vales_percent[1]
                        # print(row)
                        # if mat == 'mp-10020':
                            # print(weighted_ag.keys())
                        weighted_ag[row['Predictions Crystal System']].append(diff)


                    confidence_df = pd.DataFrame([weighted_ag])

                    weighted_sum = {'cubic':sum(np.asarray(confidence_df['cubic'])[0]),
                                    'hexagonal':sum(np.asarray(confidence_df['hexagonal'])[0]), 
                                    'tetragonal':sum(np.asarray(confidence_df['tetragonal'])[0]), 
                                    'trigonal':sum(np.asarray(confidence_df['trigonal'])[0]), 
                                    'monoclinic':sum(np.asarray(confidence_df['monoclinic'])[0]), 
                                    'orthorhombic':sum(np.asarray(confidence_df['orthorhombic'])[0])}
                    max_cry_sys = max(weighted_sum, key=weighted_sum.get)

                    max_cry_sys_val = weighted_sum[max_cry_sys]
                    confidence = max_cry_sys_val/sum(weighted_sum.values())

                    labels_test_crystal_sys = subdf.iloc[0]['True Values Crystal System']


                    old_df_indicies = list(subdf['Full DF Indicies'])
                    full_pred_crystal_system = list(subdf['Full Predictions Crystal System'])


                    full_out_df = pd.DataFrame(np.asarray([[mat], [labels_test_crystal_sys], [full_pred_crystal_system],
                                                           [max_cry_sys], [weighted_ag], [weighted_sum],
                                                           [confidence], [old_df_indicies]], dtype = object).T, 
                                          columns=['mat_id', 'True Values Crystal System',  'Full Predictions Crystal System',
                                                   'Aggregate Predictions Crystal System', 
                                                   'Averaged Weights Crystal System', 
                                                  'Full Weights Crystal System',  'Prediction Confidence', 'Full Df Indicies'])


                    final_full_out_df = pd.concat([final_full_out_df, full_out_df])

                final_full_out_df.reset_index(inplace = True)            
                self.condensed_output_df = final_full_out_df
            

    def show_aggregate_confusion_matrix(self, sample_mat_id = 'mp-1005760', random_inds = None, show_all_materials = True, show_individual = False,
                                     show_triclinic = False, savefigure = True, filenames=None):
    
        if show_triclinic:
            crystal_sys_alph = ['Cubic', 'Hexagonal', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Triclinic', 'Trigonal']
        else:
            crystal_sys_alph = ['Cubic', 'Hexagonal', 'Trigonal', 'Tetragonal', 'Monoclinic', 'Orthorhombic']


        if show_individual:
            subdf = self.output_df.loc[self.output_df['mat_id'] == sample_mat_id]

            if random_inds == None:
                random_inds = len(subdf)

            subdf = subdf.sample(random_inds, random_state=42)
            row = subdf.iloc[0]

            true_val_crys_sys = row['True Values Crystal System']


            predictions_across_zones_cry_sys = subdf['Majority Crystal System']  
            xs = np.asarray(predictions_across_zones_cry_sys.value_counts().index)
            ys = np.asarray(predictions_across_zones_cry_sys.value_counts())


            # prediction_point_group = row['Predictions Point Group']
            # predictions_point_group = row['Full Predictions Point Group']  

            # prediction_space_group = row['Predictions Space Group']
            # predictions_space_group = row['Full Predictions Space Group']  


            # ys_percentage = 100*ys*(1/len(subdf))

            plt.figure(figsize=(10, 8))
            plt.title('Prediction Histogram ' + sample_mat_id, fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('Prediction', fontsize=24)
            plt.ylabel('Count', fontsize=24)
            plt.bar(xs, ys, edgecolor='k', facecolor='grey', fill=True, linewidth=3)
            # plt.xticks(rotation = 270)
            height = max(ys)
            height_index = list(ys).index(height)

            plt.bar(xs[height_index], ys[height_index], edgecolor = 'r', facecolor='r', 
                    fill=False, hatch='/', label = 'Prediction')
            true_ind = list(xs).index(true_val_crys_sys)
            plt.bar(xs[true_ind], ys[true_ind], edgecolor = 'b', facecolor='b', 
                    fill=False, hatch='..', label = 'True')
            # plt.vlines(labels_test[0], 0, height, color='blue', label='True Space Group', linewidth=5)
            # plt.vlines(predictions[0], 0, height, color='red', label='Predicted Space Group',linewidth=5,linestyle=':')
            plt.legend(fontsize=16)
            plt.show()

        if show_all_materials == True:

            # try: 
                # with open('Model_data/Crystal_sys_outputs/cm_ag_percent.pkl', 'rb') as f:
                    # df_cm = pickle.load(f)
            # except:
                # print('file not found')
            pred_crystal_system = []
            labels_test_cry_sys = []



            for mat in self.condensed_output_df['mat_id'].unique():
                # print(mat)
                subdf = self.condensed_output_df.loc[self.condensed_output_df['mat_id'] == mat]
                labels_test_cry_sys.append(subdf.iloc[0]['True Values Crystal System'])

                # subdf = subdf.sample(random_inds, random_state=42)

                predictions_across_zones_cry_sys = subdf['Aggregate Predictions Crystal System']  

                # xs_crystal = np.asarray(predictions_across_zones_cry_sys.value_counts().index)
                # ys_crystal = np.asarray(predictions_across_zones_cry_sys.value_counts())
                pred_crystal_system.append(predictions_across_zones_cry_sys)


            cm = confusion_matrix(labels_test_cry_sys, pred_crystal_system, 
                                  labels = ['cubic', 'hexagonal', 'trigonal','tetragonal', 'monoclinic', 'orthorhombic'])

            trues = 0
            for i in range(0, len(cm)):
                trues += cm[i][i]

            cm = cm/len(pred_crystal_system)
            for i in range(0, len(cm)):
                for j in range(0, len(cm[0])):
                    cm[i][j] = round(cm[i][j]*100, 1)
            accuracy = trues/len(self.condensed_output_df['mat_id'].unique())
            print('crystal system ' + str(accuracy))
            crystal_sys_alph = ['cubic', 'hexagonal','trigonal','tetragonal', 'monoclinic', 'orthorhombic']

            
            df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)

            df_cm.to_pickle('Model_data/Crystal_sys_outputs/cm_ag_percent.pkl')

            # cm_point = confusion_matrix(labels_test_point_group, pred_point_group)
            # trues = 0
            # for i in range(0, len(cm_point)):
                # trues += cm_point[i][i]

            # accuracy = trues/len(predictions)
            # print('point group ' + str(accuracy))


            plt.figure(figsize=(15, 20))
            # sn.set(font_scale=1.4) # for label size
            # ax = sn.heatmap(df_cm, annot=True, cmap = sn.color_palette("rocket_r", as_cmap=True))
            cm = np.asarray(df_cm)
            for i in range(0, len(cm)):
                for j in range(0, len(cm[0])):
                    cm[i][j] = round(cm[i][j], 1)

            crystal_sys_alph = ['C', 'H', 'Tr','Te', 'M', 'O']
            df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)


            ax = sn.heatmap(df_cm, annot=True, cmap = 'Blues', vmin = 0.0, vmax = 23, cbar_kws={"ticks":[0.0,5,10,15,20], "location":'bottom', 
                                                                                             "fraction":0.2, 'pad':0.1, 'label':'Percent Test Set'})
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(42)

            ax.tick_params(rotation=0)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(42)
            # ax.set_title('Confusion Matrix with labels\n', fontsize = 36);
            ax.set_xlabel('Predicted Values', fontsize = 46)
            ax.set_ylabel('Actual Values ', fontsize = 46)
            if savefigure:
                plt.savefig(filenames[0]+'.pdf', bbox_inches="tight")
            plt.show()

            crystal_sys_alph = ['cubic', 'hexagonal', 'trigonal', 'tetragonal', 'monoclinic', 'orthorhombic']
            # try: 
                # with open('Model_data/Crystal_sys_outputs/confidence_cm_ag.pkl', 'rb') as f:
                    # df_cm = pickle.load(f)
            # except:
            predictions_matrix = []

            for j in range(0, len(crystal_sys_alph)):
                row = []
                for k in range(0, len(crystal_sys_alph)):
                    row.append([])
                predictions_matrix.append(row)

            predictions_confidence = []

                # predictions_ordered_cry_sys_full = []
                # predictions_ordered_space_group_full = []
                # predictions_ordered_point_group_full = []

            prediction_majority_crystal_sys = []
                    # prediction_majority_point_group = []

            for mat in self.condensed_output_df['mat_id'].unique():
                # print(mat)
                subdf = self.condensed_output_df.loc[self.condensed_output_df['mat_id'] == mat]
                # labels_test_cry_sys.append(subdf.iloc[0]['True Values Crystal System'])
                true_val = subdf.iloc[0]['True Values Crystal System']

                predictions_across_zones_cry_sys = np.asarray(subdf['Aggregate Predictions Crystal System'])[0] 
                # print(predictions_across_zones_cry_sys)

                prediction_majority_crystal_sys.append(predictions_across_zones_cry_sys)

                    # prediction_ordered_space_group = []
                    # for j in predictions_ordered[i]:
                        # prediction_ordered_space_group.append(rf_model.classes_[int(j)])

                   #  predictions_ordered_space_group_full.append(prediction_ordered_space_group)

                    # prediction_ordered_cry_sys = np.asarray(predictions_across_zones_cry_sys)

                    # prediction_ordered_point_group = point_group_from_space_group(prediction_ordered_space_group, point_group_df)

                    # prediction_df_crystal_sys = pd.DataFrame(prediction_ordered_cry_sys, columns=['Predictions'])
                    # val_counts = prediction_df_crystal_sys['Predictions'].value_counts()
                    # prediction_majority_crystal_sys.append(val_counts.index[0])

                    # prediction_df_point_group = pd.DataFrame(prediction_ordered_point_group, columns=['Predictions'])
                    # val_counts = prediction_df_point_group['Predictions'].value_counts()
                    # prediction_majority_point_group.append(val_counts.index[0])
                confidence = subdf['Prediction Confidence']  
                predictions_confidence.append(confidence)

                prediction_mapped = self.map_predictions(predictions_across_zones_cry_sys, list_of_classes = crystal_sys_alph)
                true_mapped = self.map_predictions(true_val, list_of_classes = crystal_sys_alph)
                predictions_matrix[true_mapped][prediction_mapped].append(confidence)

                    # predictions_ordered_cry_sys_full.append(prediction_ordered_cry_sys)
                    # predictions_ordered_point_group_full.append(prediction_ordered_point_group)

            for k in range(0, len(predictions_matrix)):
                for l in range(0, len(predictions_matrix)):
                    if len(predictions_matrix[k][l]) == 0:
                        predictions_matrix[k][l] = 0
                    else:
                        predictions_matrix[k][l] = np.mean(predictions_matrix[k][l])
            crystal_sys_alph = ['C', 'H', 'Tr', 'Te', 'M', 'O']
            df_cm = pd.DataFrame(predictions_matrix, crystal_sys_alph, crystal_sys_alph)
            # df_cm.to_pickle('Model_data/Crystal_sys_outputs/confidence_cm_ag.pkl')


            plt.figure(figsize=(15, 20))
            # sn.set(font_scale=1.4) # for label size
            # ax = sn.heatmap(df_cm, annot=True, vmin=0.27, vmax=1, cmap = sn.color_palette("rocket", as_cmap=True))
            Reds = mpl.colormaps['Reds'].resampled(75)
            newcolors = Reds(np.linspace(0, 1, 75))
            white = Reds(range(75))[0]
            newcolorlist = list(newcolors)
            for i in range(0, 25):
                newcolorlist.insert(0, white)
            newcolors = np.asarray(newcolorlist)
            newcmp = ListedColormap(newcolors)

            cm = np.asarray(df_cm)
            for i in range(0, len(cm)):
                for j in range(0, len(cm[0])):
                    cm[i][j] = round(cm[i][j]*100, 0)

            crystal_sys_alph = ['C', 'H', 'Tr', 'Te', 'M', 'O']
            df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)


            ax = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=100, cmap = newcmp, cbar_kws={"location":'bottom', 
                                                                                             "fraction":0.2, 'pad':0.1, 'label':'Percent Trees/Patterns'})
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(42)

            ax.tick_params(rotation=0)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(42)
            # ax.set_title('Confusion Matrix with labels\n', fontsize = 36);
            ax.set_xlabel('Predicted Values', fontsize = 46)
            ax.set_ylabel('Actual Values ', fontsize = 46)
            # ax.set_title('Confusion Matrix with labels\n', fontsize = 36);
            if savefigure:
                plt.savefig(filenames[1]+'.pdf', bbox_inches="tight")
            plt.show()

            # cm_point = confusion_matrix(labels_test_point_group, pred_point_group)
            # trues = 0
            # for i in range(0, len(cm_point)):
                # trues += cm_point[i][i]

            # accuracy = trues/len(predictions)
            # print('point group ' + str(accuracy))

        return None

    def visualize_space_group_results_mat_id(self, crystal_system = 'orthorhombic', material_id = None, show_plots = False, use_scaled = True,
                                            savefigure=False, filename=None):
        
        print(crystal_system)
        crystal_system_caps = crystal_system.upper()
        if self.loaded_submodels_space_group[0] == False:
            print('loading dfs')
            self.load_submodels_space_group(crystal_system)
        
        if self.loaded_submodels_space_group[0]:
            if self.loaded_submodels_space_group[1] != crystal_system:
                print('loading dfs')
                self.load_submodels_space_group(crystal_system)
        
        
        # self.subdf = self.full_df.loc[self.full_df['crystal system'] == crystal_system]
        if material_id == 'all':
            count = -1
            for mat_id in self.space_group_radial_inputs[3].mat_id.unique():
                count += 1 
                try:
                    self.space_group_mat_id_rows = self.space_group_radial_inputs[3].loc[self.space_group_radial_inputs[3].mat_id == mat_id]
                except:
                    print('material id not in subdf, maybe not a member of this crystal system?')

                indicies = self.space_group_mat_id_rows.index
                # print(indicies)




                # print('showing results for material ' + mat_id)
                full_out_df = self.space_group_visualize_predictions_by_material(self.space_group_rf_output[0], self.space_group_rf_output[1],
                                                                                 self.space_group_rf_output[2],
                                                            self.space_group_radial_inputs[3], self.space_group_radial_inputs[1],
                                                            list(self.space_group_radial_inputs[3].index), indicies, mat_id, show_plots =
                                                                                 show_plots, scaled = use_scaled, filename=filename)
                if count == 0:
                    self.space_group_full_out_df = full_out_df
                else:
                    # print(full_out_df)
                    self.space_group_full_out_df = pd.concat([self.space_group_full_out_df, full_out_df])
                    # print(self.full_out_df)

        else:
            print('starting for material ' + material_id)
            try:
                self.space_group_mat_id_rows = self.space_group_radial_inputs[3].loc[self.space_group_radial_inputs[3].mat_id == material_id]
            except:
                print('material id not in subdf, maybe not a member of this crystal system?')

            indicies = self.space_group_mat_id_rows.index
            # print(indicies)




            print('showing results for material ' + material_id)
            full_out_df = self.space_group_visualize_predictions_by_material(self.space_group_rf_output[0], self.space_group_rf_output[1],
                                                                             self.space_group_rf_output[2],
                                                        self.space_group_radial_inputs[3], self.space_group_radial_inputs[1],
                                                        list(self.space_group_radial_inputs[3].index), indicies, material_id, show_plots =
                                                                             show_plots, scaled = use_scaled, savefigure=savefigure, filename=filename)
            self.space_group_full_out_df = full_out_df        
    
    
    def space_group_visualize_predictions_by_material(self, predictions_ordered_input, predictions_input, rf_model, labels_test_input,
                                                      inputs_test_input, test_indicies, material_indicies, material_id, show_plots, scaled,
                                                     savefigure=False, filename=None):
        
        count = 0
        predictions_full = []
        trees = rf_model.estimators_
        # print(len(trees))
        # for tree in trees:
        # predictions_full.append(tree.predict(np.asarray(inputs_test)))
        # predictions_ordered = np.asarray(predictions_full).T
        uncertianties = []
        bot = material_indicies[0] + 1 
        top = material_indicies[len(material_indicies)-1] + 1 
        # print(bot, top)

        predictions = predictions_input[bot:top]
        labels_test = labels_test_input.iloc[bot:top]
        # print(labels_test)
        # print(list(labels_test.index))
        # print(labels_test.iloc[bot])
        # print(labels_test)
        inputs_test = inputs_test_input[bot:top]
        # print(inputs_test)
        test_indicies = test_indicies[bot:top]
        # print(test_indicies)
        output_list = []
        temp_list = []

        param_list = ['space_group_first']
        for param in param_list:
            # print(predictions)
            # print(param)
            # print(predictions_ordered[0])

            # print(predictions_std)
            x_labels = labels_test[param].to_numpy()
            x_predictions = predictions
            # print(x_labels)
            # print(x_predictions)

            # print(len(errors))
            # print(len(predictions_std))
            # print(errors)
            # print(predictions_std)
            if show_plots:
                sg_df = pd.DataFrame(x_predictions, columns = ['Predictions Space Group'])
                # print(sg_df)
                # print(sg_df['Predictions Space Group'].value_counts())
                xs_temp = np.asarray(sg_df['Predictions Space Group'].value_counts().index)
                ys = np.asarray(sg_df['Predictions Space Group'].value_counts())




                xs = []
                for x in xs_temp:
                    # print(x)
                    xs.append(x)
                # ys = ys/100

                # print(xs)
                # print(ys)
                
                
                plt.figure(figsize=(8, 7))

                plt.bar(xs[0:1], ys[0:1], edgecolor='k', facecolor='red', fill=True, linewidth=3, label = 'Prediction')
                plt.bar(xs[1:3], ys[1:3], edgecolor='k', facecolor='grey', fill=True, linewidth=3)

                height = max(ys)
                height_index = list(ys).index(height)

                # plt.bar(xs[height_index], ys[height_index], edgecolor='r', facecolor='r',
                #         fill=False, hatch='/', label='Prediction')
                true_ind = list(xs).index(x_labels[0])
                # plt.bar(xs[true_ind], ys[true_ind], edgecolor='b', facecolor='b',
                #         fill=False, hatch='..', label='True')                     
                
                # plt.title('Predictions Space Group', fontsize = 34)

                plt.xticks(xs[0:3], fontsize = 34)
                plt.yticks(fontsize = 34)
                plt.xlabel("Prediction", fontsize = 34)
                plt.ylabel('Percent Patterns', fontsize = 34)

                plt.legend(fontsize = 34)
                if savefigure:
                    plt.savefig(filename+'.pdf', bbox_inches="tight")
                plt.show()
                
            else:
                hist_pred = np.histogram(x_predictions, bins = 12)
                bins_list = list(hist_pred[1])
                n_list = list(hist_pred[0])
                mode_index = n_list.index(max(n_list))
                # print(bins_list)
                mode = (bins_list[mode_index] + bins_list[mode_index+1])/2
            count += 1
            # print(x_predictions)
            temp_list.append(x_predictions)
            temp_list.append(scipy.stats.mode(x_predictions, keepdims=True)[0][0])
            temp_list.append(x_labels[0])
        temp_list.append(material_id)
        output_list.append(temp_list)
        # print(len(np.asarray(output_list, dtype = 'object')))
        output_df = pd.DataFrame(np.asarray(output_list, dtype = 'object'),
                                 columns=['space_group_predictions', 'space_group_mode', 'true_space_group',
                                          'material_id'])
        return output_df
        

        
    def load_submodels_space_group(self, crystal_system):
        print('starting')
        self.space_group_rf_output = joblib.load('Model_data/Space_group_inputs_and_outputs/'+crystal_system+'_space_group_model.joblib')
        with open('Model_data/Crystal_sys_dataframes/full_data_'+crystal_system+'.pkl', 'rb') as f:
            self.space_group_radial_df = pickle.load(f)
        print('loaded dfs')
        self.space_group_radial_inputs = self.space_group_prepare_training_and_test(self.space_group_radial_df,
                                                                  split_by_material=True)
        self.space_group_radial_inputs[3] = self.space_group_radial_inputs[3].reset_index()
        self.space_group_radial_inputs[3].drop(columns = ['index'], axis = 1, inplace = True)

        self.loaded_submodels_space_group = [True, crystal_system]


    def space_group_prepare_training_and_test(self, df, df_type='radial', test_fraction=0.25, column_to_use='200_ang',
                                          split_by_material=True):

        if df_type == 'radial':
            col_name = 'radial_' + column_to_use + '_Colin_basis'
        if df_type == 'zernike':
            col_name = 'zernike_' + column_to_use

        input_col = np.asarray(df[col_name])
        labels = df['space_group_first']

        if split_by_material:
            mat_ids = df['mat_id'].unique()
            dummy_output = np.ones((len(mat_ids)))
            mat_ids_train, mat_ids_test, dummy_train, dummy_test = train_test_split(mat_ids, dummy_output,
                                                                                    test_size=test_fraction,
                                                                                    random_state=32)
            print(len(mat_ids_train))
            print(len(mat_ids_test))

            input_train_first = df.loc[df['mat_id'].isin(mat_ids_train)][col_name]
            input_test_first = df.loc[df['mat_id'].isin(mat_ids_test)][col_name]


            labels_train = df.loc[df['mat_id'].isin(mat_ids_train)][['space_group_first', 'mat_id']]
            labels_test = df.loc[df['mat_id'].isin(mat_ids_test)][['space_group_first', 'mat_id']]

        else:
            input_train_first, input_test_first, labels_train, labels_test = train_test_split(input_col, labels,
                                                                                              test_size=test_fraction,
                                                                                              random_state=32)
        if len(input_train_first[0].shape) < 2:
            input_train_temp = input_train_first
            input_test_temp = input_test_first

        else:
            input_train_temp = []
            for unit in input_train_first:
                input_train_temp.append(self.flatten(unit))

            input_test_temp = []
            for unit in input_test_first:
                input_test_temp.append(self.flatten(unit))

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

        input_test = []
        for row in input_test_temp:
            temp = []
            abs_i = np.abs(row)
            angle_i = np.angle(row)
            for i in range(0, len(abs_i)):
                temp.append(abs_i[i])
                temp.append(angle_i[i])
            input_test.append(temp)

        return [input_train, input_test, labels_train, labels_test, mat_ids_train, mat_ids_test]
    

def calc_basis_scaled_df(bragg_list, k_max, dk, order_max, sine_basis=False, remove_central_beam = True):
    
    if remove_central_beam:
        new_bragg_list = deepcopy(bragg_list) # create a copy # probably not needed
        # print(new_pl.data.shape)
        mask = np.ones_like(new_bragg_list.data['intensity'], dtype=bool)
        index = np.where(new_bragg_list.data['intensity'] == np.max(new_bragg_list.data['intensity']))[0][0]
        mask[index] = False
        new_bragg_list.data = new_bragg_list.data[mask]
        # print(new_pl.data.shape)
    
    basis = construct_basis(new_bragg_list.data['qx'], new_bragg_list.data['qy'], k_max, dk, order_max, sine_basis)
    
    if remove_central_beam:
        return [basis, mask] 
    else:
        return basis
    
def construct_basis(kx, ky, k_max, dk, order_max, sine_basis=False):
    """
    
    
    Placeholder
    
    """
    # k bin boundaries starts at zero extends to kmax
    k_bins = np.arange(0, k_max+dk, dk)
    
    # no elements, size of the basis, so 
    # radial_bins, 
    if kx.ndim == 1:
        basis_size = (k_bins.shape[0], order_max+1, kx.shape[0])
    elif kx.ndim > 1:
        basis_size = (k_bins.shape[0], order_max+1, *kx.shape)
    else:
        print('error')
        
    basis = np.zeros(basis_size, dtype=np.complex128)
    # print(basis.shape)
    kr = np.hypot(kx, ky)
    # ensure ky, kx (alex check why)
    phi = np.arctan2(ky,kx)
    
    # loop over the bins
    for ind, k in enumerate(k_bins):
        # calculate the basis functions
        # create the mask to select
        sub = np.logical_and(kr > k - dk,  kr < k + dk)
        
        b_radial = 1 - np.abs(kr[sub] - k) / dk
        if sine_basis:
            b_radial = np.sin(b_radial * (np.pi / 2) ) ** 2
        
        for ind_order, order in enumerate(range(order_max + 1)): 
#             b_annular =  np.cos(order * phi[sub]) + 1j * np.sin(order * phi[sub])
            
            b_annular = np.exp((1j * order) * phi[sub])
            
            basis[ind, ind_order][sub] = b_radial * b_annular
            
            
            
    return basis

def Complex2RGB_transparent(complex_array, vmin=None, vmax=None, hue_start=90, transparent_thresh = 0.1):
    """
    Function to turn a complex array into rgb for plotting
    Args:
        complex_array (2D array)          : complex array
        vmin (float, optional)            : minimum absolute value
        vmax (float, optional)            : maximum absolute value
            if None, vmin/vmax are set to fractions of the distribution of pixel values in the array, 
            e.g. vmin=0.02 will set the minumum display value to saturate the lower 2% of pixels
        hue_start(float, optional)        : phase offset (degrees)
        transprent_threh (float, optional): magntidue of complex array below which the array will be set to transparent 
    Returns: 
        rgb array for plotting
    """
    amp = np.abs(complex_array)

    if np.max(amp) == np.min(amp):
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = np.max(amp)
    else:
        if vmin is None:
            vmin = 0.02
        if vmax is None:
            vmax = 0.98
        vals = np.sort(amp[~np.isnan(amp)])
        ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
        ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
        ind_vmin = np.max([0, ind_vmin])
        ind_vmax = np.min([len(vals) - 1, ind_vmax])
        vmin = vals[ind_vmin]
        vmax = vals[ind_vmax]

    amp = np.where(amp < vmin, vmin, amp)
    amp = np.where(amp > vmax, vmax, amp)

    ph = np.angle(complex_array, deg=1) + hue_start

    h = (ph % 360) / 360
    s = 0.85 * np.ones_like(h)
    v = (amp - vmin) / (vmax - vmin)

    hsv = np.dstack((h, s, v))
    if hsv.shape[-1] != 3:
        hsv = hsv.reshape(hsv.shape[0], hsv.shape[1], 3, hsv.shape[2] // 3)
        hsv = hsv.swapaxes(-1, -2)

    rgb =  hsv_to_rgb(hsv)
    alpha = ~np.all(rgb < transparent_thresh, axis=2) * 255
    rgba = np.dstack((rgb * 255, alpha)).astype(np.uint8)
    
    return rgba

def visualize_radial_components(model_object, zone, radial_params = [[12, 6]], include_y_label = True, include_x_label = True,):
        
    k = np.linspace(-2.4, 2.4, 201)
    kya, kxa = np.meshgrid(k,k)
    k_max=2
    dk = 0.1
    order_max = 12
    # k_max spaced by dk = num radials 
    # order max (0 to order max) = number of oscillations
    basis = construct_basis(kxa, kya, k_max, dk, order_max, False)
    fig, ax = model_object.show_specific_pattern('mp-1001786', zone = zone, 
                                                include_y_label = include_y_label, include_x_label = include_x_label)
    if include_y_label:
        ax.set_yticks((-2, -1, 0, 1, 2))
        ax.tick_params(labelsize = 30)
    if include_x_label:
        ax.set_xticks((-2, -1, 0, 1, 2))
        ax.tick_params(labelsize = 30)
    
    for radial_param in radial_params: 
        test = Complex2RGB_transparent(basis[radial_param[0], radial_param[1]], transparent_thresh=0.1)
        test2 = show_complex(basis[radial_param[0], radial_param[1]])
        ax.imshow(test, extent=[-2,2, -2, 2])

    # ax.imshow(cmap = "viridis")


    
    # ax.set_yticks(fontsize = 16)

    return fig, test2

def plot_diffraction_pattern(
    bragg_peaks = None,
    bragg_peaks_compare = None,
    scale_markers = 500,
    scale_markers_compare = None,
    power_markers = 1,
    plot_range_kx_ky = None,
    add_labels = True,
    shift_labels = 0.08,
    shift_marker= 0.005,
    min_marker_size = 1e-6,
    max_marker_size = 500,
    figsize = (12, 6),
    returnfig = False,
    input_fig_handle =None,
    include_y_label = True,
    include_x_label = True,
    
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

    if plot_range_kx_ky is not None:
        plot_range_kx_ky = np.array(plot_range_kx_ky)
        if plot_range_kx_ky.ndim == 0:
            plot_range_kx_ky = np.array((plot_range_kx_ky,plot_range_kx_ky))
        ax.set_xlim((-plot_range_kx_ky[0], plot_range_kx_ky[0]))
        ax.set_ylim((-plot_range_kx_ky[1], plot_range_kx_ky[1]))
    else:
        k_range = 1.05 * np.sqrt(
            np.max(bragg_peaks.data["qx"] ** 2 + bragg_peaks.data["qy"] ** 2)
        )
        ax.set_xlim((-k_range, k_range))
        ax.set_ylim((-k_range, k_range))
        
    if include_x_label:
        ax.set_xticks((-2, -1, 0, 1, 2))
        ax.tick_params(labelsize = 30)
        ax.set_xlabel("$q_y$ [Å$^{-1}$]", fontsize = 32)
        ax.xaxis.set_label_position('top') 
        ax.xaxis.tick_top()
       
    else:
        ax.xaxis.set_ticks([])

    if include_y_label:
        ax.set_yticks((-2, -1, 0, 1, 2))
        ax.tick_params(labelsize = 30)
        ax.set_ylabel("$q_x$ [Å$^{-1}$]", fontsize = 32)
        ax.invert_yaxis()
        
    else:
        ax.yaxis.set_ticks([])
        ax.invert_yaxis()

    ax.set_box_aspect(1)

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

        for a0 in range(bragg_peaks.data.shape[0]):
            h = bragg_peaks.data["h"][a0]
            k = bragg_peaks.data["k"][a0]
            l = bragg_peaks.data["l"][a0]

            ax.text(
                bragg_peaks.data["qy"][a0],
                bragg_peaks.data["qx"][a0]
                - shift_labels
                - shift_marker * np.sqrt(marker_size[a0]),
                "$" + overline(h) + overline(k) + overline(l) + "$",
                **text_params,
            )

    # Force plot to have 1:1 aspect ratio
    ax.set_aspect("equal")

    if input_fig_handle is None:
        plt.show()
    
    if returnfig:
        return fig, ax
    
def flatten(list1):
    return [item for sublist in list1 for item in sublist]

def scale_df_lattice(df, ids_column):
    a_s = []
    b_s = []
    c_s = []
    count = -1
    for mat_id in df[ids_column].unique():
        count += 1 
        # print(100*count/len(df.ids.unique()))
        subdf = df.loc[df[ids_column] == mat_id]

        a = np.asarray(subdf[['a', 'b', 'c']]).T[0][0]
        b = np.asarray(subdf[['a', 'b', 'c']]).T[1][0]
        c = np.asarray(subdf[['a', 'b', 'c']]).T[2][0]

        lattice_params = [a, b, c]
        lattice_params.sort()

        reverse_indicies=[2,1,0]

        for i in reverse_indicies:
            if a == lattice_params[i]:
                for a_entry in np.asarray(subdf[['a', 'b', 'c']]).T[0]:
                    if i == 2:
                        a_s.append(a_entry)
                    if i == 1:
                        b_s.append(a_entry)
                    if i == 0:
                        c_s.append(a_entry)

            elif b == lattice_params[i]:
                for b_entry in np.asarray(subdf[['a', 'b', 'c']]).T[1]:
                    if i == 2:
                        a_s.append(b_entry)
                    if i == 1:
                        b_s.append(b_entry)
                    if i == 0:
                        c_s.append(b_entry)    

            elif c == lattice_params[i]:
                for c_entry in np.asarray(subdf[['a', 'b', 'c']]).T[2]:
                    if i == 2:
                        a_s.append(c_entry)
                    if i == 1:
                        b_s.append(c_entry)
                    if i == 0:
                        c_s.append(c_entry)

    df['c_sorted'] = a_s
    df['b_sorted'] = b_s
    df['a_sorted'] = c_s
    
    return df    

"""
def lattice_visualize_predictions_by_material(subdf, show_plots = True, include_legend = [False]):
    # print(test_indicies)
    material_id = subdf.iloc[0].mat_id
    true_crystal_sys = subdf.iloc[0]['crystal system']
    output_list = []
    temp_list = []

    param_list = ['a_pred', 'b_pred', 'c_pred', 'alpha_pred', 'beta_pred', 'gamma_pred']
    count = 0
    for param in param_list:
        # print(param)
        # print(predictions_ordered[0])

        # print(predictions_std)
        if param == 'a_pred':
            x_labels = subdf['x'].to_numpy()
        elif param == 'b_pred':
            x_labels = subdf['y'].to_numpy()
        elif param == 'c_pred':
            x_labels = subdf['z'].to_numpy()       
        else:  
            x_labels = subdf[param[0:len(param)-5]].to_numpy()

        x_predictions = subdf[param].to_numpy()
        # print(x_labels)
        # print(x_predictions)

        # print(len(errors))
        # print(len(predictions_std))
        # print(errors)
        # print(predictions_std)
        if show_plots:
            errors = np.abs(x_labels - x_predictions)
            errors = np.asarray(errors)

            MSE = np.square(errors).mean()
            RMSE = math.sqrt(MSE)
            print('RMSE ' + str(RMSE))

            plt.figure(figsize=(8, 7))
            plt.title('Error Histogram', fontsize=18)
            hist = plt.hist(errors, bins=50)
            plt.vlines(RMSE, max(hist[0]), min(hist[0]), color='limegreen', linewidth=5, label='RMSE')
            plt.text(RMSE + 0.25, max(hist[0]) - 0.1 * max(hist[0]), 'RMSE = ' + str(round(RMSE, 3)),
                     horizontalalignment='center', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Error', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.show()

            plt.figure(figsize=(8, 7))
            hist_pred = plt.hist(x_predictions, bins = 12)
            if param == 'x':
                plt.title('Prediction Histogram a', fontsize = 24)
            elif param == 'y':
                plt.title('Prediction Histogram b', fontsize = 24)
            elif param == 'z':
                plt.title('Prediction Histogram c', fontsize = 24)
            else:
                plt.title('Prediction Histogram ' + param, fontsize = 24)

            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.ylabel('Num Patterns', fontsize = 22)
            if param == 'x':
                plt.vlines(x_labels[0], max(hist_pred[0]), min(hist_pred[0]), label = 'True a', color = 'r', linewidth = 3)
                plt.xlabel('Predicted a', fontsize = 22)

            elif param == 'y':
                plt.vlines(x_labels[0], max(hist_pred[0]), min(hist_pred[0]), label = 'True b', color = 'r', linewidth = 3)
                plt.xlabel('Predicted b', fontsize = 22)

            elif param == 'z':
                plt.vlines(x_labels[0], max(hist_pred[0]), min(hist_pred[0]), label = 'True c', color = 'r', linewidth = 3)
                plt.xlabel('Predicted c', fontsize = 22)

            else:
                plt.vlines(x_labels[0], max(hist_pred[0]), min(hist_pred[0]), label = 'True ' + param, color = 'r', linewidth = 3)
                plt.xlabel('Predicted ' + param, fontsize = 22)

            bins_list = list(hist_pred[1])
            n_list = list(hist_pred[0])
            mode_index = n_list.index(max(n_list))
            print(bins_list)
            mode = (bins_list[mode_index] + bins_list[mode_index+1])/2
            plt.vlines(mode, max(hist_pred[0]), min(hist_pred[0]), label = 'Aggregate Prediction', color = 'limegreen', linewidth = 3)
            if include_legend[count]:
                plt.legend(fontsize = 20)
            count += 1
            plt.show()

        else:
            hist_pred = np.histogram(x_predictions, bins = 12)
            bins_list = list(hist_pred[1])
            n_list = list(hist_pred[0])
            mode_index = n_list.index(max(n_list))
            # print(bins_list)
            mode = (bins_list[mode_index] + bins_list[mode_index+1])/2
        temp_list.append(x_predictions)
        temp_list.append(mode)
        temp_list.append(x_labels[0])
    temp_list.append(material_id)
    temp_list.append(true_crystal_sys)
    output_list.append(temp_list)
    # print(len(np.asarray(output_list, dtype = 'object')))
    output_df = pd.DataFrame(np.asarray(output_list, dtype = 'object'),
                             columns=['a_full_predictions', 'a_mode', 'a_true',
                                     'b_full_predictions', 'b_mode', 'b_true',
                                     'c_full_predictions', 'c_mode', 'c_true',
                                     'alpha_full_predictions', 'alpha_mode', 'alpha_true',
                                     'beta_full_predictions', 'beta_mode', 'beta_true',
                                     'gamma_full_predictions', 'gamma_mode', 'gamma_true',
                                      'material_id', 'true_crystal_sys'])
    return output_df
"""
def show_complex(
    ar_complex,
    vmin=None,
    vmax=None,
    cbar=True,
    scalebar=False,
    pixelunits="pixels",
    pixelsize=1,
    returnfig=True,
    hue_start = 0,
    invert=False,
    **kwargs):
    """
    Function to plot complex arrays

    Args:
        ar_complex (2D array)       : complex array to be plotted. If ar_complex is list of complex arrarys
            such as [array1, array2], then arrays are horizonally plotted in one figure
        vmin (float, optional)      : minimum absolute value
        vmax (float, optional)      : maximum absolute value
            if None, vmin/vmax are set to fractions of the distribution of pixel values in the array, 
            e.g. vmin=0.02 will set the minumum display value to saturate the lower 2% of pixels
        cbar (bool, optional)       : if True, include color wheel
        scalebar (bool, optional)   : if True, adds scale bar
        pixelunits (str, optional)  : units for scalebar
        pixelsize (float, optional) : size of one pixel in pixelunits for scalebar
        returnfig (bool, optional)  : if True, the function returns the tuple (figure,axis)
        hue_start (float, optional) : rotational offset for colormap (degrees)
        inverse (bool)              : if True, uses light color scheme
    
    Returns:
        if returnfig==False (default), the figure is plotted and nothing is returned.
        if returnfig==True, return the figure and the axis.
    """
    # convert to complex colors
    ar_complex = ar_complex[0] if (isinstance(ar_complex,list) and len(ar_complex) == 1) else ar_complex
    if isinstance(ar_complex, list):
        if isinstance(ar_complex[0], list):
            rgb = [Complex2RGB(ar, vmin, vmax, hue_start = hue_start, invert=invert) for sublist in ar_complex for ar in sublist]
            H = len(ar_complex)
            W = len(ar_complex[0])

        else:
            rgb = [Complex2RGB(ar, vmin, vmax, hue_start=hue_start, invert=invert) for ar in ar_complex]
            if len(rgb[0].shape) == 4:
                H = len(ar_complex)
                W = rgb[0].shape[0]
            else:
                H = 1
                W = len(ar_complex)
        is_grid = True
    else:
        rgb = Complex2RGB(ar_complex, vmin, vmax, hue_start=hue_start, invert=invert)
        if len(rgb.shape) == 4:
            is_grid = True
            H = 1
            W = rgb.shape[0]
        elif len(rgb.shape) == 5:
            is_grid = True
            H = rgb.shape[0]
            W = rgb.shape[1]
            rgb = rgb.reshape((-1,)+rgb.shape[-3:])
        else:
            is_grid = False
    # plot
    if is_grid:
        from py4DSTEM.visualize import show_image_grid

        fig, ax = show_image_grid(
            get_ar=lambda i: rgb[i],
            H=H,
            W=W,
            vmin=0,
            vmax=1,
            intensity_range="absolute",
            returnfig=True,
            **kwargs,
        )
        if scalebar is True:
            scalebar = {
                "Nx": ar_complex[0].shape[0],
                "Ny": ar_complex[0].shape[1],
                "pixelsize": pixelsize,
                "pixelunits": pixelunits,
            }

            add_scalebar(ax[0, 0], scalebar)
    else:
        fig, ax = py4DSTEM.visualize.show(
            rgb, vmin=0, vmax=1, intensity_range="absolute", returnfig=True, **kwargs
        )

        if scalebar is True:
            scalebar = {
                "Nx": ar_complex.shape[0],
                "Ny": ar_complex.shape[1],
                "pixelsize": pixelsize,
                "pixelunits": pixelunits,
            }

            add_scalebar(ax, scalebar)

    # add color bar
    if cbar == True:
        ax0 = fig.add_axes([1, 0.35, 0.3, 0.3])

        # create wheel
        AA = 1000
        kx = np.fft.fftshift(np.fft.fftfreq(AA))
        ky = np.fft.fftshift(np.fft.fftfreq(AA))
        kya, kxa = np.meshgrid(ky, kx)
        kra = (kya**2 + kxa**2) ** 0.5
        ktheta = np.arctan2(-kxa, kya)
        ktheta = kra * np.exp(1j * ktheta)

        # convert to hsv
        rgb = Complex2RGB(ktheta, 0, 0.4, hue_start = hue_start, invert=invert)
        ind = kra > 0.4
        rgb[ind] = [1, 1, 1]

        # plot
        ax0.imshow(rgb)

        # add axes
        ax0.axhline(AA / 2, 0, AA, color="k")
        ax0.axvline(AA / 2, 0, AA, color="k")
        ax0.axis("off")

        label_size = 16

        ax0.text(AA, AA / 2, 1, fontsize=label_size)
        ax0.text(AA / 2, 0, "i", fontsize=label_size)
        ax0.text(AA / 2, AA, "-i", fontsize=label_size)
        ax0.text(0, AA / 2, -1, fontsize=label_size)

    if returnfig == True:
        return fig, ax
    
def Complex2RGB(complex_data, vmin=None, vmax = None, hue_start = 0, invert=False):
    """
    complex_data (array): complex array to plot
    vmin (float)        : minimum absolute value 
    vmax (float)        : maximum absolute value 
    hue_start (float)   : rotational offset for colormap (degrees)
    inverse (bool)      : if True, uses light color scheme
    """
    amp = np.abs(complex_data)
    if np.isclose(np.max(amp),np.min(amp)):
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = np.max(amp)
    else:
        if vmin is None:
            vmin = 0.02
        if vmax is None:
            vmax = 0.98
        vals = np.sort(amp[~np.isnan(amp)])
        ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
        ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
        ind_vmin = np.max([0, ind_vmin])
        ind_vmax = np.min([len(vals) - 1, ind_vmax])
        vmin = vals[ind_vmin]
        vmax = vals[ind_vmax]

    amp = np.where(amp < vmin, vmin, amp)
    amp = np.where(amp > vmax, vmax, amp)

    phase = np.angle(complex_data) + np.deg2rad(hue_start)
    amp /= np.max(amp)
    rgb = np.zeros(phase.shape +(3,))
    rgb[...,0] = 0.5*(np.sin(phase)+1)*amp
    rgb[...,1] = 0.5*(np.sin(phase+np.pi/2)+1)*amp
    rgb[...,2] = 0.5*(-np.sin(phase)+1)*amp
    
    return 1-rgb if invert else rgb




"""
        
    def visualize_lattice_results_mat_id(self, crystal_system = 'orthorhombic', material_id = None, show_plots = False, use_scaled = True, 
                                         param_list = None, index_list = None,
                                        savefigure=False, include_legend = [False], filename=None, xticks = None):
        self.full_out_df = []
        print(crystal_system)
        crystal_system_caps = crystal_system.upper()
        if self.loaded_submodels[0] == False:
            print('loading dfs')
            self.load_submodels(crystal_system, crystal_system_caps,use_scaled)
        
        if self.loaded_submodels[0]:
            if self.loaded_submodels[1] != crystal_system:
                print('loading dfs')
                self.load_submodels(crystal_system, crystal_system_caps,use_scaled)
        
        
        # self.subdf = self.full_df.loc[self.full_df['crystal system'] == crystal_system]
        if material_id == 'all':
            count = -1
            for mat_id in self.lattice_radial_inputs[5]:
                count += 1 
                try:
                    self.mat_id_rows = self.lattice_radial_inputs[3].loc[self.lattice_radial_inputs[3].mat_id == mat_id]
                except:
                    print('material id not in subdf, maybe not a member of this crystal system?')

                indicies = self.mat_id_rows.index
                # print(indicies)




                # print('showing results for material ' + mat_id)
                full_out_df = self.lattice_visualize_predictions_by_material(self.lattice_rf_output[0], self.lattice_rf_output[1],
                                                                             self.lattice_rf_output[2],
                                                            self.lattice_radial_inputs[3], self.lattice_radial_inputs[1],
                                                            list(self.lattice_radial_inputs[3].index), indicies, mat_id, show_plots = show_plots,
                                                                             scaled = use_scaled, param_list = param_list, index_list = index_list,
                                                                            include_legend=include_legend, filename=filename, xticks=xticks)
                if len(self.full_out_df) == 0:
                    self.full_out_df = full_out_df
                else:
                    # print(full_out_df)
                    self.full_out_df = pd.concat([self.full_out_df, full_out_df])
                    # print(self.full_out_df)

        else:
            if type(self.condensed_df) == None:
                self.condensed_df = joblib.load('Model_data/Crystal_sys_outputs/lattice_added_difference_aggregation_condensed_df.joblib')

            print('showing results for material ' + material_id)
            
            full_out_df = self.lattice_visualize_predictions_by_material(material_id, show_plots = show_plots,
                                                                         param_list = param_list, index_list = index_list, 
                                                                        savefigure=savefigure, include_legend=include_legend, filename=filename)
            self.full_out_df = full_out_df
            
def visualize_predictions_by_material(self, output_df_path = 'NO_MISMATCH_NO_TRICLINIC_output_df_radial.joblib', 
                                          sample_mat_id='mp-1005760', random_inds=None,
                                          show_all_materials=True, show_individual=False, show_space_group =True, 
                                         zone = (), savefigure=False, filenames = None, individual_pattern_num = '#1', 
                                         filename = None):

        # print('loading output df')
                                 
        
        full_out = self.output_df

            
        
        # print(self.output_df)
        # print(show_individual)
        if show_individual:
            # print('starting')
            
            subdf = self.output_df.loc[self.output_df['mat_id'] == sample_mat_id]
            # print(subdf)
            # print(zone)
            if random_inds == None:
                # print(len(zone))
                if len(zone)==0:
                    random_inds = len(subdf)
                    subdf = subdf.sample(random_inds, random_state=42)
                    row = subdf.iloc[0]
                
                else:
                    # print('really starting')
                    for i in np.asarray(subdf['Full DF Indicies']):
                        # print(i)
                        # print(subdf)
                        # print(subdf.index)
                        # print(self.full_df.iloc[i])
                        if self.full_df.iloc[i]['zone'] == zone: 
                            # print(self.full_df.iloc[i]['zone'])
                            # print(zone)
                            subdf = subdf.loc[subdf['Full DF Indicies'] == i]
                    row = subdf
            
            
            
            # print(subdf)
            # index = row['Full DF Indicies']
            # print(index)
            # print(row)
            # print(self.full_df.iloc[index])
            
            if show_space_group:
                true_val_crys_sys = row['True Values Space Group']
                predictions_across_zones_cry_sys = subdf['Predictions Space Group']
                # print(predictions_across_zones_cry_sys)
                xs_temp = np.asarray(predictions_across_zones_cry_sys['Predictions Space Group'].value_counts().index)
                ys = np.asarray(predictions_across_zones_cry_sys['Predictions Space Group'].value_counts())


            
            else:
                if len(zone)==0:
                    true_val_crys_sys = row['True Values Crystal System']
                    predictions_across_zones = np.asarray(subdf['Predictions Crystal System'])
                    predictions_across_zones_cry_sys = pd.DataFrame(predictions_across_zones, columns = ['Predictions Crystal System'])


                    # print(predictions_across_zones_cry_sys.columns)
                    xs_temp = np.asarray(predictions_across_zones_cry_sys['Predictions Crystal System'].value_counts().index)
                    ys = np.asarray(predictions_across_zones_cry_sys['Predictions Crystal System'].value_counts())
                else:
                    true_val_crys_sys = np.asarray(row['True Values Crystal System'])[0].capitalize()
                    predictions_across_zones = np.asarray(subdf['Full Predictions Crystal System'])[0]
                    predictions_across_zones_cry_sys = pd.DataFrame(predictions_across_zones, columns = ['Full Predictions Crystal System'])


                    xs_temp = np.asarray(predictions_across_zones_cry_sys['Full Predictions Crystal System'].value_counts().index)
                    ys = np.asarray(predictions_across_zones_cry_sys['Full Predictions Crystal System'].value_counts())

            xs = []
            for x in xs_temp:
                # print(x)
                xs.append(x.capitalize())
            
            # if len(zone) == 0:
            #     ys = ys/len(subdf)
       
            
            # print(xs)
            # print(ys)
            # prediction_point_group = row['Predictions Point Group']
            # predictions_point_group = row['Full Predictions Point Group']

            # prediction_space_group = row['Predictions Space Group']
            # predictions_space_group = row['Full Predictions Space Group']

            # ys_percentage = 100*ys*(1/len(subdf))
            plt.rcdefaults()
            plt.figure(figsize=(10, 8))
            # plt.title('Prediction Histogram (' + str(round(zone[0], 2)) + ' ' + str(round(zone[1], 2)) + ' ' + str(round(zone[2], 2)) + ')', fontsize=30)
            if len(zone) != 0:
                plt.title('Individual Pattern ' + individual_pattern_num, fontsize = 48)
            else:
                plt.title('Aggregate Prediction', fontsize = 48)

            plt.xticks(fontsize=44)
            plt.yticks(fontsize=44)
            
            plt.xlabel('Prediction', fontsize=50)
            if len(zone) == 0:
                plt.ylabel('Percent Patterns', fontsize=50)
            else:
                plt.ylabel('Num Trees', fontsize = 50)
            # print(xs)
            # print(ys)
            

            # plt.xticks(rotation = 270)
            if len(zone) != 0:
                plt.bar(xs[0:1], ys[0:1], edgecolor='orange', facecolor='white', fill=True, linewidth=15, label='Prediction')
                plt.bar(xs[1:3], ys[1:3], edgecolor='forestgreen', facecolor='white', fill=True, linewidth=15)
                plt.ylim([0,80])
                plt.yticks([0,20,40,60,80])
                plt.legend(fontsize=44)
            else:
                plt.bar(xs[0:1], ys[0:1], edgecolor='k', facecolor='orange', fill=True, linewidth=2, label='Prediction')
                plt.bar(xs[1:3], ys[1:3], edgecolor='k', facecolor='forestgreen', fill=True, linewidth=2)
                plt.legend(fontsize=40)
                plt.yticks([0,20,40,60])
            height = max(ys)
            height_index = list(ys).index(height)

            # plt.bar(xs[height_index], ys[height_index], edgecolor='r', facecolor='r',
            #         fill=False, hatch='/', label='Prediction')
            true_ind = list(xs).index(true_val_crys_sys.capitalize())
            # plt.bar(xs[true_ind], ys[true_ind], edgecolor='b', facecolor='b',
            #         fill=False, hatch='..', label='True')
            # plt.vlines(labels_test[0], 0, height, color='blue', label='True Space Group', linewidth=5)
            # plt.vlines(predictions[0], 0, height, color='red', label='Predicted Space Group',linewidth=5,linestyle=':')
            if savefigure:
                plt.rcParams['pdf.fonttype'] = 'truetype'
                plt.savefig(filename+'.pdf', bbox_inches="tight")
            plt.show()

        if show_all_materials == True:

            pred_crystal_system = []
            labels_test_cry_sys = []

            for mat in self.output_df['mat_id'].unique():
                subdf = self.output_df.loc[self.output_df['mat_id'] == mat]
                if show_space_group:
                    labels_test_cry_sys.append(subdf.iloc[0]['True Values Space Group'])

                    subdf = subdf.sample(random_inds, random_state=42)

                    predictions_across_zones_cry_sys = subdf['Predictions Space Group']

                
                else:
                    labels_test_cry_sys.append(subdf.iloc[0]['True Values Crystal System'])

                    subdf = subdf.sample(random_inds, random_state=42)

                    predictions_across_zones_cry_sys = subdf['Predictions Crystal System']

                xs_crystal = np.asarray(predictions_across_zones_cry_sys.value_counts().index)
                ys_crystal = np.asarray(predictions_across_zones_cry_sys.value_counts())
                pred_crystal_system.append(xs_crystal[0])

            cm = confusion_matrix(labels_test_cry_sys, pred_crystal_system)
            trues = 0
            for i in range(0, len(cm)):
                trues += cm[i][i]

            accuracy = trues / len(self.output_df['mat_id'].unique())
            print('crystal system ' + str(accuracy))

            # cm_point = confusion_matrix(labels_test_point_group, pred_point_group)
            # trues = 0
            # for i in range(0, len(cm_point)):
            # trues += cm_point[i][i]

            # accuracy = trues/len(predictions)
            # print('point group ' + str(accuracy))
            crystal_sys_alph = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal',
                                'trigonal']
            df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)
            plt.figure(figsize=(10, 8))
            # sn.set(font_scale=2.5) # for label size
            ax = sn.heatmap(df_cm, annot=True, cmap='Blues')
            ax.set_title('Confusion Matrix with labels\n');
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ');
            plt.show()

            predictions_matrix = []

            for j in range(0, len(crystal_sys_alph)):
                row = []
                for k in range(0, len(crystal_sys_alph)):
                    row.append([])
                predictions_matrix.append(row)

            predictions_confidence = []

            # predictions_ordered_cry_sys_full = []
            # predictions_ordered_space_group_full = []
            # predictions_ordered_point_group_full = []

            prediction_majority_crystal_sys = []
            # prediction_majority_point_group = []

            for mat in self.output_df['mat_id'].unique():
                subdf = self.output_df.loc[self.output_df['mat_id'] == mat]
                # labels_test_cry_sys.append(subdf.iloc[0]['True Values Crystal System'])
                true_val = subdf.iloc[0]['True Values Crystal System']
                subdf = subdf.sample(random_inds, random_state=42)

                predictions_across_zones_cry_sys = subdf['Predictions Crystal System']
                xs = np.asarray(predictions_across_zones_cry_sys.value_counts().index)
                ys = np.asarray(predictions_across_zones_cry_sys.value_counts())
                prediction_majority_crystal_sys.append(xs[0])

                # prediction_ordered_space_group = []
                # for j in predictions_ordered[i]:
                # prediction_ordered_space_group.append(rf_model.classes_[int(j)])

                #  predictions_ordered_space_group_full.append(prediction_ordered_space_group)

                # prediction_ordered_cry_sys = np.asarray(predictions_across_zones_cry_sys)

                # prediction_ordered_point_group = point_group_from_space_group(prediction_ordered_space_group, point_group_df)

                # prediction_df_crystal_sys = pd.DataFrame(prediction_ordered_cry_sys, columns=['Predictions'])
                # val_counts = prediction_df_crystal_sys['Predictions'].value_counts()
                # prediction_majority_crystal_sys.append(val_counts.index[0])

                # prediction_df_point_group = pd.DataFrame(prediction_ordered_point_group, columns=['Predictions'])
                # val_counts = prediction_df_point_group['Predictions'].value_counts()
                # prediction_majority_point_group.append(val_counts.index[0])

                confidence = max(ys) / sum(ys)
                predictions_confidence.append(confidence)

                prediction_mapped = self.map_predictions(xs[0], list_of_classes=crystal_sys_alph)
                true_mapped = self.map_predictions(true_val, list_of_classes=crystal_sys_alph)
                predictions_matrix[true_mapped][prediction_mapped].append(confidence)

                # predictions_ordered_cry_sys_full.append(prediction_ordered_cry_sys)
                # predictions_ordered_point_group_full.append(prediction_ordered_point_group)

            for k in range(0, len(predictions_matrix)):
                for l in range(0, len(predictions_matrix)):
                    if len(predictions_matrix[k][l]) == 0:
                        predictions_matrix[k][l] = 0
                    else:
                        predictions_matrix[k][l] = np.mean(predictions_matrix[k][l])

            df_cm = pd.DataFrame(predictions_matrix, crystal_sys_alph, crystal_sys_alph)
            plt.figure(figsize=(10, 8))
            # sn.set(font_scale=1.4) # for label size
            ax = sn.heatmap(df_cm, annot=True, cmap='Blues', vmin=0.35)
            ax.set_title('Average Confidence in Prediction\n');
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ');
            plt.show()

            # cm_point = confusion_matrix(labels_test_point_group, pred_point_group)
            # trues = 0
            # for i in range(0, len(cm_point)):
            # trues += cm_point[i][i]

            # accuracy = trues/len(predictions)
            # print('point group ' + str(accuracy))

        return None
"""