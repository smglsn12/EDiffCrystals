import py4DSTEM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
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
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import PointGroup, SpaceGroup
from mp_api.client import MPRester
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import scipy
from scipy.ndimage import gaussian_filter
plt.rcParams['pdf.fonttype'] = 'truetype'
sn.set(font_scale=3)

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

def plot_hist_2D(
    x0,
    y0,
    x1,
    y1,
    dr = 0.05,
    r_max = 15,
    r_sigma = 0.1,
    scale_power = 1.0,
    scale_saturation = (50.0, 50.0),
    figsize = (3,3),
    returnfig = True,
    returnim = False,
    title=None
    # cmap = 'turbo',#'Blues',
    ):
    """
    Generate a 2D histogram plot using KDE for 2 overlapping datasets.
    """
    
    # ensure numpy arrays
    scale_saturation = np.array(scale_saturation)
    
    # coordinates
    r = np.arange(0.0,r_max,dr)
    im_size = (r.shape[0],r.shape[0])
    
    # histogram - image 0
    x0_inds = np.round((x0 - r[0]) / dr).astype('int')
    y0_inds = np.round((y0 - r[0]) / dr).astype('int')
    keep0 = np.logical_and(
        x0_inds < im_size[0],
        y0_inds < im_size[1],
    )
    inds0 = np.ravel_multi_index(
        (y0_inds[keep0],x0_inds[keep0]),
        im_size,
    )
    im0_sig = np.reshape(
            np.bincount(
            inds0,
            minlength = np.prod(im_size),
        ),
        im_size,
    ).astype('float')
    
    # histogram - image 1
    x1_inds = np.round((x1 - r[0]) / dr).astype('int')
    y1_inds = np.round((y1 - r[0]) / dr).astype('int')
    keep1 = np.logical_and(
        x1_inds < im_size[0],
        y1_inds < im_size[1],
    )
    inds1 = np.ravel_multi_index(
        (y1_inds[keep1],x1_inds[keep1]),
        im_size,
    )
    im1_sig = np.reshape(
            np.bincount(
            inds1,
            minlength = np.prod(im_size),
        ),
        im_size,
    ).astype('float')
    
    # smoothing
    sigma = r_sigma / dr
    im0_sm = gaussian_filter(
        im0_sig,
        sigma,
    )
    im1_sm = gaussian_filter(
        im1_sig,
        sigma,
    )
    
    # normalize images
    scale = np.maximum(
        np.max(im0_sm),
        np.max(im1_sm),
    )
    im0_sm /= scale
    im1_sm /= scale
    
    # scale images
    im0_sm = im0_sm**scale_power * scale_saturation[0]
    im1_sm = im1_sm**scale_power * scale_saturation[1]
    
    # combined image from two colormaps
    # cmap0 = get_cmap('Blues')
    # cmap1 = get_cmap('Oranges')
    # cmap = matplotlib.colormaps['Blues'](np.linspace(0.0,0.5,256))
    # im0_rgb = cmap[np.clip(np.round(im0_sm*255).astype('int'),0,255)]
    #     im0_rgb = matplotlib.colormaps['Blues'](im0_sm)
    #     im1_rgb = matplotlib.colormaps['Reds'](im1_sm)

    #     im_rgb = np.minimum(
    #         im0_rgb,
    #         im1_rgb,
    #     )
    # print(cmap0.shape)

    # manual combined image 
    im_rgb = np.ones((
        im_size[0],
        im_size[1],
        3,
    ))
    im_rgb[:,:,0] -= im0_sm
    im_rgb[:,:,1] -= 0.25*np.clip(im0_sm,0,1)

    im_rgb[:,:,1] -= 0.75*np.clip(im1_sm,0,1)
    im_rgb[:,:,2] -= im1_sm
    im_rgb = np.clip(im_rgb,0,1)
    
    # plotting    
    fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(
        im_rgb,
        extent = [0,r_max,0,r_max],
        origin = 'lower',
    )
    # labels
    ax.set_xlabel('True Value')
    ax.set_ylabel('Prediction')
    ax.set_title(title, fontsize = 28)
    # appearance
    if r_max > 50:
        t = np.arange(0,r_max+20,20)
    else:
        t = np.arange(0,r_max+10,10)
    ax.set_xticks(t)
    ax.set_yticks(t)
    ax.set_xlim([0, r_max])
    ax.set_ylim([0, r_max])
    
    plt.plot([2, r_max-2],[2, r_max-2], linestyle = '--', zorder=0, color='k', linewidth =0.5)
    
    plt.yticks(fontsize=28)
    plt.xticks(fontsize = 28)
    plt.xlabel('True', fontsize = 32)
    plt.ylabel('Prediction', fontsize = 32)
    # ax.text(2, 14, 'your legend', bbox={'facecolor': 'white', 'pad': 10})
    # plt.legend(['True Crystal System', 'Inaccurate Crystal System'], fontsize = 18)
    if returnfig:
        if returnim:
            return fig, ax, im_rgb
        else:
            return fig, ax
    else:
        if returnim:
            return im_rgb
    
def map_to_int(array_of_crystal_sys):
    new_list = []
    for i in array_of_crystal_sys:
        if i == 'cubic':
            new_list.append(int(1))
        if i == 'hexagonal':
            new_list.append(int(2))
        if i == 'monoclinic':
            new_list.append(int(3))
        if i == 'orthorhombic':
            new_list.append(int(4))
        if i == 'tetragonal':
            new_list.append(int(5))
        if i == 'triclinic':
            new_list.append(int(6))
        if i == 'trigonal':
            new_list.append(int(7))
            
    return new_list

def flatten(list1):
    return [item for sublist in list1 for item in sublist]

def build_full_predictions(rf_model, input_test):
    # max_depth=6
    
    # predictions = rf_model.predict(input_test)
    # rf_loss = log_loss(input_test, predictions)
    # print('log loss ' + str(rf_loss))
    # joblib.dump(rf_model, 'just_model_200_ang_full_dataset_80_trees.joblib')
    
    predictions = rf_model.predict(input_test)
    
    predictions_full = []
    trees = rf_model.estimators_
    count = 0
    for tree in trees:
        print(count)
        predictions_full.append(tree.predict(np.asarray(input_test)))
        count += 1
        # print(tree.predict(np.asarray(updated_spectra_test)))
    predictions_ordered = np.asarray(predictions_full).T
    
    return [predictions_ordered, predictions, rf_model]
    
    
    # return rf_model
    
def rf_diffraction(labels_train, labels_test, input_train, input_test, show_cm = True, show_uncertianty = True, num_trees = 500, max_depth = 40,
              index_to_use = 0, max_features = 'sqrt'):
    # max_depth=6
    rf_model = RandomForestClassifier(n_estimators=num_trees, n_jobs=-1, max_features=max_features, random_state=32, verbose = 2, max_depth = max_depth)
    rf_model.fit(np.asarray(input_train), np.asarray(labels_train))
    accuracy = rf_model.score(np.asarray(input_test), np.asarray(labels_test))
    print('accuracy = ' + str(accuracy))
    
    # predictions = rf_model.predict(input_test)
    # rf_loss = log_loss(input_test, predictions)
    # print('log loss ' + str(rf_loss))
    # joblib.dump(rf_model, 'just_model_200_ang_full_dataset_80_trees.joblib')
    
    predictions = rf_model.predict(input_test)
    
    predictions_full = []
    trees = rf_model.estimators_
    count = 0
    for tree in trees:
        print(count)
        predictions_full.append(tree.predict(np.asarray(input_test)))
        count += 1
        # print(tree.predict(np.asarray(updated_spectra_test)))
    predictions_ordered = np.asarray(predictions_full).T
    
    return [predictions_ordered, predictions, rf_model]
    
    
    # return rf_model
def crystal_sys_from_space_group(space_groups):
    crystal_sys = []
    for sg in space_groups:
        # print(sg)
        if sg in [1,2]:
            crystal_sys.append('triclinic')
        if sg >= 3 and sg <= 15:
            crystal_sys.append('monoclinic')
        if sg >= 16 and sg <= 74:
            crystal_sys.append('orthorhombic')
        if sg >= 75 and sg <= 142:
            crystal_sys.append('tetragonal')
        if sg >= 143 and sg <= 167:
            crystal_sys.append('trigonal')
        if sg >= 168 and sg <= 194:
            crystal_sys.append('hexagonal')          
        if sg >= 195 and sg <= 230:
            crystal_sys.append('cubic')   
    return crystal_sys

def update_df(df):
    mat_ids = list(df['mat_id'].unique())
    # mat_ids = mat_ids[0:1000]
    # structures = mpr.get_structure_by_material_id(list(mat_ids))
    with MPRester("CEvsr9tiYxi6MaxfRnSU7V9FCaIAcAZh") as mpr:
        # doc = mpr.materials.search(task_ids = mat_ids[chunk:top], fields=['material_id', 'formula_pretty', 'structure', 'composition'])

        structures = []
        crystal_systems = []
        point_groups = []
        ids = []

        for i in range(0, len(df)):
            mat_id = df.iloc[i]['mat_id']
            if i != 0:
                prev_mat_id = df.iloc[i-1]['mat_id']
            if i == 0: 
                prev_mat_id = None

            if mat_id == prev_mat_id:
                structures.append(structure)
                sg = SpacegroupAnalyzer(structure)
                crystal_systems.append(sg.get_crystal_system())
                point_groups.append(sg.get_point_group_symbol())
                ids.append(mat_id)

            else: 
                structure = mpr.get_structure_by_material_id(mat_id)
                print(i)
                print(mat_id)
                structures.append(structure)
                sg = SpacegroupAnalyzer(structure)
                crystal_systems.append(sg.get_crystal_system())
                point_groups.append(sg.get_point_group_symbol())
                ids.append(mat_id)
    
    df['structure'] = structures
    df['ids'] = ids
    df['crystal system'] = crystal_systems
    df['point group'] = point_groups
    
    return df

def build_dictionary_point_group_mapping(df):
    point_group_dict = {}
    mat_ids = list(df['mat_id'].unique())
    for mat_id in mat_ids:
        subdf = df.loc[df['ids'] == mat_id]
        point_group = subdf.iloc[0]['point group']
        space_group = subdf.iloc[0]['space_group_symbol']
        # print(point_group, space_group)
        if space_group in point_group_dict.keys():
            test = point_group_dict[space_group]
            try:
                assert(test == point_group)
                point_group_dict[space_group] = point_group
                # print(test, space_group, mat_id)
            except AssertionError:
                print('ERROR', test, space_group, mat_id)
        else:
            point_group_dict[space_group] = point_group
    return point_group_dict

def point_group_from_space_group(space_groups, point_group_dict):
    point_group = []
    for sg in space_groups:
         point_group.append(point_group_dict[sg]) 
    return point_group

def crystal_system_from_space_group_full_predictions(predictions_space_group):
    sg_df = pd.DataFrame(predictions_space_group, columns = ['Predictions'])
    val_counts = sg_df['Predictions'].value_counts()
    xs = list(val_counts.index)
    ys = np.asarray(val_counts)
    ys_percentage = 100*ys*(1/len(sg_df))

    crystal_systems = crystal_sys_from_space_group(xs)
    cs_mapping = []
    for i in range(0, len(xs)):
        cs_mapping.append([xs[i], ys_percentage[i], crystal_systems[i]])
    cs_mapping_df = pd.DataFrame(cs_mapping, columns = ['Space Groups', 'Percentages', 'Crystal Systems'])
    crystal_systems = cs_mapping_df['Crystal Systems'].unique()
    percentages = []
    systems_ordered = []
    for system in crystal_systems:
        percentages.append(sum(cs_mapping_df.loc[cs_mapping_df['Crystal Systems'] == system]['Percentages']))
        systems_ordered.append(system)
    print(percentages)
    print(systems_ordered)
    prediction_index = percentages.index(max(percentages))
    prediction = crystal_systems[prediction_index]
    
    return [cs_mapping_df, prediction]

def show_cm_and_uncertianty(predictions_ordered, predictions, rf_model, full_df, labels_test, test_indicies, 
                            show_cm = False, show_uncertianty = False, type_to_show = 'crystal system', point_group_df = None,
                           predicted_quant = 'space_group', savefigure = False, generate_df_with_predictions = False):
    
    if predicted_quant == 'crystal system': 
        
        pred_crystal_system = predictions
                
        
        labels_test_cry_sys = np.asarray(full_df.iloc[test_indicies]['crystal system'])
        mp_ids = np.asarray(full_df.iloc[test_indicies]['mat_id'])

        cm = confusion_matrix(labels_test_cry_sys, pred_crystal_system)
        trues = 0
        for i in range(0, len(cm)):
            trues += cm[i][i]
        
        accuracy = trues/len(predictions)
        print('crystal system ' + str(accuracy))
        
        cm = cm/len(predictions)
        for i in range(0, len(cm)):
            for j in range(0, len(cm[0])):
                cm[i][j] = round(cm[i][j]*100, 5)
            

        
        accuracy = trues/len(predictions)
        print('point group ' + str(accuracy))
        crystal_sys_alph = ['C', 'H', 'M', 'O', 'Te', 'Tr']

        df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)
        
        df_cm.to_pickle('df_cm_percent_unrounded.pkl')
        
        print(df_cm)

        plt.figure(figsize=(15, 20))
        # sn.set(font_scale=1.4) # for label size 
        # cmap=matplotlib.cm.get_cmap('plasma')
        # ax = sn.heatmap(df_cm, annot=True, vmin = 0, vmax = 0.22, cmap = sn.color_palette("rocket_r", as_cmap=True))


        ax = sn.heatmap(df_cm, annot=True, cmap = 'Blues', vmin = 0, vmax = 22, cbar_kws={"ticks":[0.0,5,10,15,20], "location":'bottom', 
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
            plt.savefig('fig3_individual_prediction_accuracy.pdf', bbox_inches="tight")
        plt.show()

        
        crystal_sys_alph = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal']
        
        if generate_df_with_predictions:
            df_with_pred = []
            for i in range(0, len(test_indicies)):
                full_df_index = test_indicies[i]
                label_test_cry_sys = full_df.iloc[test_indicies[i]]['crystal system']
                mp_id = full_df.iloc[test_indicies[i]]['mat_id']
                pred = predictions[i]
                vectorized_pattern = full_df.iloc[test_indicies[i]]['radial_200_ang_Colin_basis']
                
                
                prediction_ordered_cry_sys = []
                for j in predictions_ordered[i]:
                    prediction_ordered_cry_sys.append(rf_model.classes_[int(j)])


                prediction_df_crystal_sys = pd.DataFrame(prediction_ordered_cry_sys, columns=['Predictions'])
                val_counts = prediction_df_crystal_sys['Predictions'].value_counts()

                confidence = max(val_counts) / sum(val_counts)

                df_with_pred.append([label_test_cry_sys, mp_id, pred, confidence, prediction_ordered_cry_sys, full_df_index, vectorized_pattern])
            df_with_pred = pd.DataFrame(df_with_pred, columns = ['true_crystal_system', 'mp_id', 'prediction', 'confidence', 'full_predictions',
                                                                 'full_df_index','radial_200_ang_Colin_basis'])
            df_with_pred.to_pickle('df_with_pred.pkl')                     
        
        if os.path.exists('Model_data/Crystal_sys_outputs/prediction_matrix_confidence.npy') == False:
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

                prediction_mapped = map_predictions(pred_crystal_system[i], list_of_classes = crystal_sys_alph)
                true_mapped = map_predictions(labels_test_cry_sys[i], list_of_classes = crystal_sys_alph)
                predictions_matrix[true_mapped][prediction_mapped].append(confidence)

                predictions_ordered_cry_sys_full.append(prediction_ordered_cry_sys)
                
                
            for k in range(0, len(predictions_matrix)):
                for l in range(0, len(predictions_matrix)):
                    if len(predictions_matrix[k][l]) == 0:
                        predictions_matrix[k][l] = 0
                    else:
                        predictions_matrix[k][l] = np.mean(predictions_matrix[k][l])

            np.save('Model_data/Crystal_sys_outputs/prediction_matrix_confidence.npy', predictions_matrix)  
        
        else:
            predictions_matrix = np.load('Model_data/Crystal_sys_outputs/prediction_matrix_confidence.npy')
        crystal_sys_alph = ['C', 'H', 'M', 'O', 'Te', 'Tr']
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
            plt.savefig('fig3_individual_prediction_confidence.pdf', bbox_inches="tight")
        plt.show()
        
        
        
        if os.path.exists('Model_data/Crystal_sys_outputs/output_df_radial.joblib') == False:
            output_list = []
            for i in range(0, len(labels_test)):
                output_list.append([predictions[i], pred_crystal_system[i], labels_test_cry_sys[i], 
                                    predictions_confidence[i], test_indicies[i], 
                                     predictions_ordered_cry_sys_full[i], mp_ids[i]])



            output_df = pd.DataFrame(np.asarray(output_list),
                                     columns = ['Predictions Space Group', 'Predictions Crystal System', 'True Values Crystal System', 
                                                'Confidence Crystal System', 'Full DF Indicies',
                                                'Full Predictions Crystal System', 'mat_id']

                                    )

            return output_df
    
        else:
            pass

    elif type_to_show in ['crystal system', 'point group']: 
        
        pred_crystal_system = crystal_sys_from_space_group(predictions)
        pred_point_group = point_group_from_space_group(predictions, point_group_df)
        
        labels_test_cry_sys = np.asarray(full_df.iloc[test_indicies]['crystal system'])
        labels_test_point_group = np.asarray(full_df.iloc[test_indicies]['point group'])

        cm = confusion_matrix(labels_test_cry_sys, pred_crystal_system)
        trues = 0
        for i in range(0, len(cm)):
            trues += cm[i][i]
        
        accuracy = trues/len(predictions)
        print('crystal system ' + str(accuracy))
        
        cm_point = confusion_matrix(labels_test_point_group, pred_point_group)
        trues = 0
        for i in range(0, len(cm_point)):
            trues += cm_point[i][i]
        
        accuracy = trues/len(predictions)
        print('point group ' + str(accuracy))
        crystal_sys_alph = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'triclinic', 'trigonal']
        df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)
        plt.figure(figsize=(10, 8))
        # sn.set(font_scale=1.4) # for label size
        ax = sn.heatmap(df_cm, annot=True, cmap='Blues')
        # ax.set_title('Confusion Matrix with labels\n');
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
        
        predictions_ordered_cry_sys_full = []
        predictions_ordered_space_group_full = []
        predictions_ordered_point_group_full = []
        
        prediction_majority_crystal_sys = []
        prediction_majority_point_group = []

        for i in range(0, len(predictions_ordered)):
            prediction_ordered_space_group = []
            for j in predictions_ordered[i]:
                prediction_ordered_space_group.append(rf_model.classes_[int(j)])
            
            predictions_ordered_space_group_full.append(prediction_ordered_space_group)
            
            prediction_ordered_cry_sys = crystal_sys_from_space_group(prediction_ordered_space_group)
            
            prediction_ordered_point_group = point_group_from_space_group(prediction_ordered_space_group, point_group_df)
            
            prediction_df_crystal_sys = pd.DataFrame(prediction_ordered_cry_sys, columns=['Predictions'])
            val_counts = prediction_df_crystal_sys['Predictions'].value_counts()
            prediction_majority_crystal_sys.append(val_counts.index[0])

            prediction_df_point_group = pd.DataFrame(prediction_ordered_point_group, columns=['Predictions'])
            val_counts = prediction_df_point_group['Predictions'].value_counts()
            prediction_majority_point_group.append(val_counts.index[0])
            
            confidence = max(val_counts) / sum(val_counts)
            predictions_confidence.append(confidence)
            
            prediction_mapped = map_predictions(pred_crystal_system[i], list_of_classes = crystal_sys_alph)
            true_mapped = map_predictions(labels_test_cry_sys[i], list_of_classes = crystal_sys_alph)
            predictions_matrix[true_mapped][prediction_mapped].append(confidence)
            
            predictions_ordered_cry_sys_full.append(prediction_ordered_cry_sys)
            predictions_ordered_point_group_full.append(prediction_ordered_point_group)
            
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
        
        cm_maj = confusion_matrix(labels_test_cry_sys, prediction_majority_crystal_sys)
        trues = 0
        for i in range(0, len(cm_maj)):
            trues += cm_maj[i][i]
        
        accuracy = trues/len(predictions)
        print('crystal system Majority ' + str(accuracy))
        
        cm_point_maj = confusion_matrix(labels_test_point_group, prediction_majority_point_group)
        trues = 0
        for i in range(0, len(cm_point_maj)):
            trues += cm_point_maj[i][i]
        
        accuracy = trues/len(predictions)
        print('point group majority ' + str(accuracy))

        df_cm = pd.DataFrame(cm_maj, crystal_sys_alph, crystal_sys_alph)
        plt.figure(figsize=(10, 8))
        # sn.set(font_scale=1.4) # for label size
        ax = sn.heatmap(df_cm, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix with labels\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        plt.show()
        
        print(len(predictions))
        print(len(pred_crystal_system))
        print(len(labels_test))
        print(len(labels_test_cry_sys))
        print(len(predictions_confidence))
        print(len(test_indicies))
        print(len(predictions_ordered_space_group_full))
        print(len(predictions_ordered_cry_sys_full))
        
        output_list = []
        for i in range(0, len(labels_test)):
            output_list.append([predictions[i], pred_crystal_system[i], prediction_majority_crystal_sys[i], 
                                pred_point_group[i], prediction_majority_point_group[i],
                                np.asarray(labels_test)[i], labels_test_cry_sys[i], labels_test_point_group[i],
                                predictions_confidence[i],
                                 test_indicies[i], 
                                 predictions_ordered_space_group_full[i], predictions_ordered_point_group_full[i],
                                 predictions_ordered_cry_sys_full[i]])
        
            
        
        output_df = pd.DataFrame(np.asarray(output_list),
                                 columns = ['Predictions Space Group', 'Predictions Crystal System', 'Majority Crystal System',  
                                            'Predictions Point Group', 'Majority Point Group',
                                            'True Values Space Group', 'True Values Crystal System', 'True Values Point Group',
                                            'Confidence Crystal System', 'Full DF Indicies',
                                           'Full Predictions Space Group', 'Full Predictions Point Group',
                                            'Full Predictions Crystal System']
                                 
                                )
        
        return output_df
    
    
    
    
    if type_to_show == 'thickness': 
        cm = confusion_matrix(np.asarray(labels_test), predictions)
        trues = 0
        for i in range(0, len(cm)):
            trues += cm[i][i]
        
        accuracy = trues/len(predictions)
        print(accuracy)
        # crystal_sys_alph = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'triclinic', 'trigonal']
        df_cm = pd.DataFrame(cm, rf_model.classes_, rf_model.classes_)
        plt.figure(figsize=(10, 8))
        # sn.set(font_scale=1.4) # for label size
        ax = sn.heatmap(df_cm, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix with labels\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        plt.show()
        
        
        predictions_matrix = []

        for j in range(0, len(rf_model.classes_)):
            row = []
            for k in range(0, len(rf_model.classes_)):
                row.append([])
            predictions_matrix.append(row)

        predictions_confidence = []
        
        for i in range(0, len(predictions_ordered)):
            
                        
            prediction_df = pd.DataFrame(predictions_ordered[i], columns=['Predictions'])
            val_counts = prediction_df['Predictions'].value_counts()

            confidence = max(val_counts) / sum(val_counts)
            predictions_confidence.append(confidence)
            
            prediction_mapped = map_predictions(predictions[i], rf_model = rf_model)
            true_mapped = map_predictions(np.asarray(labels_test)[i], rf_model = rf_model)
            predictions_matrix[true_mapped][prediction_mapped].append(confidence)
            
            
        for k in range(0, len(predictions_matrix)):
            for l in range(0, len(predictions_matrix)):
                if len(predictions_matrix[k][l]) == 0:
                    predictions_matrix[k][l] = 0
                else:
                    predictions_matrix[k][l] = np.mean(predictions_matrix[k][l])

        df_cm = pd.DataFrame(predictions_matrix, rf_model.classes_, rf_model.classes_)
        plt.figure(figsize=(10, 8))
        # sn.set(font_scale=1.4) # for label size
        ax = sn.heatmap(df_cm, annot=True, cmap='Blues', vmin=0.35)
        ax.set_title('Average Confidence in Prediction\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        plt.show()
        
        
        output_list = []
        for i in range(0, len(labels_test)):
            output_list.append([predictions[i], np.asarray(labels_test)[i], 
                                predictions_confidence[i],
                                 test_indicies[i], 
                                 predictions_ordered[i]])
        
            
        
        output_df = pd.DataFrame(np.asarray(output_list),
                                 columns = ['Predictions', 'True Values', 'Confidence', 'Full DF Indicies',
                                           'Full Predictions']
                                 
                                )
        
        return output_df
    
def visualize_predictions(index_to_use, output_df, full_df, vis_type = 'crystal system'):
        # temp = []
        # for pred in predictions_ordered[index_to_use]:
            # temp.append(rf_model.classes_[int(pred)])
    if vis_type == 'crystal system':
        row = output_df.iloc[index_to_use]   

        sample_mat_id = full_df.iloc[row['Full DF Indicies']]['mat_id']
        sample_zone_raw = full_df.iloc[row['Full DF Indicies']]['zone']
        sample_zone = []
        for i in sample_zone_raw:
            sample_zone.append(round(i,2))
        sample_zone = tuple(sample_zone)
        true_val_crys_sys = row['True Values Crystal System']
        prediction_crys_sys = row['Predictions Crystal System']
        predictions_crys_sys = row['Full Predictions Crystal System']  

        print(true_val_crys_sys, prediction_crys_sys)
        
        # true_val_point_group = row['True Values Point Group']
        # prediction_point_group = row['Predictions Point Group']
        # predictions_point_group = row['Full Predictions Point Group']  
        
        # true_val_space_group = row['True Values Space Group']
        # prediction_space_group = row['Predictions Space Group']
        # predictions_space_group = row['Full Predictions Space Group']  

        crys_sys_df = pd.DataFrame(predictions_crys_sys, columns = ['Predictions'])
        val_counts = crys_sys_df['Predictions'].value_counts()
        xs = list(val_counts.index)
        ys = np.asarray(val_counts)
        ys_percentage = 100*ys*(1/len(crys_sys_df))

        plt.figure(figsize=(10, 8))
        plt.title('Prediction Histogram ' + sample_mat_id + ' ' + str(sample_zone), fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Prediction', fontsize=24)
        plt.ylabel('Percentage', fontsize=24)
        plt.bar(xs, ys_percentage, edgecolor='k', facecolor='grey', fill=True, linewidth=3)
        plt.xticks(rotation = 270)
        height = max(ys_percentage)
        height_index = list(ys_percentage).index(height)

        plt.bar(xs[height_index], ys_percentage[height_index], edgecolor = 'r', facecolor='r', 
                fill=False, hatch='/', label = 'Prediction')
        true_ind = xs.index(true_val_crys_sys)
        plt.bar(xs[true_ind], ys_percentage[true_ind], edgecolor = 'b', facecolor='b', 
                fill=False, hatch='..', label = 'True')
        # plt.vlines(labels_test[0], 0, height, color='blue', label='True Space Group', linewidth=5)
        # plt.vlines(predictions[0], 0, height, color='red', label='Predicted Space Group',linewidth=5,linestyle=':')
        plt.legend(fontsize=16)
        plt.show()

        
        """
        pg_df = pd.DataFrame(predictions_point_group, columns = ['Predictions'])
        print(pg_df)
        val_counts = pg_df['Predictions'].value_counts()
        xs = list(val_counts.index)
        ys = np.asarray(val_counts)
        ys_percentage = 100*ys*(1/len(pg_df))

        plt.figure(figsize=(10, 8))
        plt.title('Prediction Histogram ' + sample_mat_id + ' ' + str(sample_zone), fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Prediction', fontsize=24)
        plt.ylabel('Percentage', fontsize=24)
        plt.bar(xs, ys_percentage, edgecolor='k', facecolor='grey', fill=True, linewidth=3)
        plt.xticks(rotation = 270)
        height = max(ys_percentage)
        height_index = list(ys_percentage).index(height)

        plt.bar(xs[height_index], ys_percentage[height_index], edgecolor = 'r', facecolor='r', 
                fill=False, hatch='/', label = 'Prediction')
        true_ind = xs.index(true_val_point_group)
        plt.bar(xs[true_ind], ys_percentage[true_ind], edgecolor = 'b', facecolor='b', 
                fill=False, hatch='..', label = 'True')
        # plt.vlines(labels_test[0], 0, height, color='blue', label='True Space Group', linewidth=5)
        # plt.vlines(predictions[0], 0, height, color='red', label='Predicted Space Group',linewidth=5,linestyle=':')
        plt.legend(fontsize=16)
        plt.show()
        """
        
        """
        if sg in [1,2]:
            crystal_sys.append('triclinic')
        if sg >= 3 and sg <= 15:
            crystal_sys.append('monoclinic')
        if sg >= 16 and sg <= 74:
            crystal_sys.append('orthorhombic')
        if sg >= 75 and sg <= 142:
            crystal_sys.append('tetragonal')
        if sg >= 143 and sg <= 167:
            crystal_sys.append('trigonal')
        if sg >= 168 and sg <= 194:
            crystal_sys.append('hexagonal')          
        if sg >= 195 and sg <= 230:
            crystal_sys.append('cubic')   
        """
        
        """
        sg_df = pd.DataFrame(predictions_space_group, columns = ['Predictions'])
        val_counts = sg_df['Predictions'].value_counts()
        xs = list(val_counts.index)
        ys = np.asarray(val_counts)
        ys_percentage = 100*ys*(1/len(sg_df))

        plt.figure(figsize=(12, 8))
        plt.vlines(0, 0,ys_percentage[true_ind], color = 'r', linestyle = '--', label = 'triclinic')
        plt.vlines(3, 0,ys_percentage[true_ind], color = 'r', linestyle = '--')
        plt.hlines(ys_percentage[true_ind], 0,3, color = 'r', linestyle = '--')
        
        plt.vlines(3, 0,ys_percentage[true_ind], color = 'b', linestyle = '--', label = 'monoclinic')
        plt.vlines(16, 0,ys_percentage[true_ind], color = 'b', linestyle = '--')
        plt.hlines(ys_percentage[true_ind], 3,16, color = 'b', linestyle = '--')

        plt.vlines(75, 0,ys_percentage[true_ind], color = 'green', linestyle = '--', label = 'orthorhombic')
        plt.vlines(16, 0,ys_percentage[true_ind], color = 'green', linestyle = '--')
        plt.hlines(ys_percentage[true_ind], 75,16, color = 'green', linestyle = '--')
        
        plt.vlines(143, 0,ys_percentage[true_ind], color = 'purple', linestyle = '--', label = 'tetragonal')
        plt.vlines(75, 0,ys_percentage[true_ind], color = 'purple', linestyle = '--')
        plt.hlines(ys_percentage[true_ind], 143,75, color = 'purple', linestyle = '--')
        
        plt.vlines(143, 0,ys_percentage[true_ind], color = 'k', linestyle = '--', label = 'trigonal')
        plt.vlines(168, 0,ys_percentage[true_ind], color = 'k', linestyle = '--')
        plt.hlines(ys_percentage[true_ind], 143,168, color = 'k', linestyle = '--')
        
        plt.vlines(168, 0,ys_percentage[true_ind], color = 'darkorange', linestyle = '--', label = 'hexagonal')
        plt.vlines(195, 0,ys_percentage[true_ind], color = 'darkorange', linestyle = '--')
        plt.hlines(ys_percentage[true_ind], 168,195, color = 'darkorange', linestyle = '--')
        
        plt.vlines(231, 0,ys_percentage[true_ind], color = 'darkcyan', linestyle = '--', label = 'cubic')
        plt.vlines(195, 0,ys_percentage[true_ind], color = 'darkcyan', linestyle = '--')
        plt.hlines(ys_percentage[true_ind], 231,195, color = 'darkcyan', linestyle = '--')
        plt.legend(fontsize=14, loc='upper right')
        plt.xlim([-5,300])
        plt.ylim([0, 13])

        plt.title('Prediction Histogram ' + sample_mat_id + ' ' + str(sample_zone), fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Prediction', fontsize=24)
        plt.ylabel('Percentage', fontsize=24)
        plt.bar(xs, ys_percentage, edgecolor='k', facecolor='grey', fill=True, linewidth=3)
        print(xs)
        print(ys_percentage)
        crystal_systems = crystal_sys_from_space_group(xs)
        cs_mapping = []
        for i in range(0, len(xs)):
            cs_mapping.append([xs[i], ys_percentage[i], crystal_systems[i]])
        cs_mapping_df = pd.DataFrame(cs_mapping, columns = ['Space Groups', 'Percentages', 'Crystal Systems'])
        
        triclinic = sum(cs_mapping_df.loc[cs_mapping_df['Crystal Systems'] == 'triclinic']['Percentages'])
        monoclinic = sum(cs_mapping_df.loc[cs_mapping_df['Crystal Systems'] == 'monoclinic']['Percentages'])
        orthorhombic = sum(cs_mapping_df.loc[cs_mapping_df['Crystal Systems'] == 'orthorhombic']['Percentages'])
        tetragonal = sum(cs_mapping_df.loc[cs_mapping_df['Crystal Systems'] == 'tetragonal']['Percentages'])
        trigonal = sum(cs_mapping_df.loc[cs_mapping_df['Crystal Systems'] == 'trigonal']['Percentages'])
        hexagonal = sum(cs_mapping_df.loc[cs_mapping_df['Crystal Systems'] == 'hexagonal']['Percentages'])
        cubic = sum(cs_mapping_df.loc[cs_mapping_df['Crystal Systems'] == 'cubic']['Percentages'])
        
        
        sum(cs_mapping_df.loc[cs_mapping_df['Crystal Systems'] == 'orthorhombic']['Percentages'])
        # plt.xticks(rotation = 270)
        height = max(ys_percentage)
        height_index = list(ys_percentage).index(height)

        plt.bar(xs[height_index], ys_percentage[height_index], edgecolor = 'r', facecolor='r', 
                fill=False, hatch='/', label = 'Prediction')
        true_ind = xs.index(true_val_space_group)
        plt.bar(xs[true_ind], ys_percentage[true_ind], edgecolor = 'b', facecolor='b', 
                fill=False, hatch='..', label = 'True')
        # plt.vlines(labels_test[0], 0, height, color='blue', label='True Space Group', linewidth=5)
        # plt.vlines(predictions[0], 0, height, color='red', label='Predicted Space Group',linewidth=5,linestyle=':')
        # plt.legend(fontsize=16)
        plt.show()

        return cs_mapping_df
        # cry_sys_input = input()
        """
        """
        subdf_cry_sys = crys_sys_df.loc[crys_sys_df['Predictions'] == cry_sys_input]
        indicies_to_use = subdf_cry_sys.index
        subdf_sg = sg_df.iloc[indicies_to_use]

        val_counts = subdf_sg['Predictions'].value_counts()
        xs = list(val_counts.index)
        ys = np.asarray(val_counts)
        ys_percentage = 100*ys*(1/len(subdf_sg))

        plt.figure(figsize=(10, 8))
        plt.title('Prediction Histogram ' + sample_mat_id + ' ' + str(sample_zone), fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Prediction', fontsize=24)
        plt.ylabel('Percentage', fontsize=24)
        plt.bar(xs, ys_percentage, edgecolor='k', facecolor='grey', fill=True, linewidth=3)
        # plt.xticks(rotation = 270)
        height = max(ys_percentage)
        height_index = list(ys_percentage).index(height)

        plt.bar(xs[height_index], ys_percentage[height_index], edgecolor = 'r', facecolor='r', 
                fill=False, hatch='/', label = 'Prediction')
        true_ind = xs.index(true_val_space_group)
        plt.bar(xs[true_ind], ys_percentage[true_ind], edgecolor = 'b', facecolor='b', 
                fill=False, hatch='..', label = 'True')
        # plt.vlines(labels_test[0], 0, height, color='blue', label='True Space Group', linewidth=5)
        # plt.vlines(predictions[0], 0, height, color='red', label='Predicted Space Group',linewidth=5,linestyle=':')
        plt.legend(fontsize=16)
        plt.show()
        """
        
    if vis_type == 'thickness':
        row = output_df.iloc[index_to_use]   

        sample_mat_id = full_df.iloc[row['Full DF Indicies']]['mat_id']
        sample_zone = full_df.iloc[row['Full DF Indicies']]['zone']

        true_val = row['True Values']
        prediction = row['Predictions']
        full_predictions = row['Full Predictions']

        thickness_df = pd.DataFrame(full_predictions, columns = ['Predictions'])
        val_counts = thickness_df['Predictions'].value_counts()
        xs = list(val_counts.index)
        ys = np.asarray(val_counts)
        ys_percentage = 100*ys*(1/len(thickness_df))

        plt.figure(figsize=(10, 8))
        plt.title('Prediction Histogram ' + sample_mat_id + ' ' + str(sample_zone), fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Prediction', fontsize=24)
        plt.ylabel('Percentage', fontsize=24)
        plt.bar(xs, ys_percentage, edgecolor='k', facecolor='grey', fill=True, linewidth=3)
        plt.xticks(rotation = 270)
        height = max(ys_percentage)
        height_index = list(ys_percentage).index(height)

        plt.bar(xs[height_index], ys_percentage[height_index], edgecolor = 'r', facecolor='r', 
                fill=False, hatch='/', label = 'Prediction')
        true_ind = xs.index(true_val)
        plt.bar(xs[true_ind], ys_percentage[true_ind], edgecolor = 'b', facecolor='b', 
                fill=False, hatch='..', label = 'True')
        # plt.vlines(labels_test[0], 0, height, color='blue', label='True Space Group', linewidth=5)
        # plt.vlines(predictions[0], 0, height, color='red', label='Predicted Space Group',linewidth=5,linestyle=':')
        plt.legend(fontsize=16)
        plt.show()
        
def visualize_predictions_by_material(output_df, sample_mat_id = 'mp-1005760', random_inds = None, show_all_materials = True, show_individual = False,
                                     show_triclinic = False, savefigure = True, filenames=None):
    
    if show_triclinic:
        crystal_sys_alph = ['Cubic', 'Hexagonal', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Triclinic', 'Trigonal']
    else:
        crystal_sys_alph = ['Cubic', 'Hexagonal', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal']

    
    if show_individual:
        subdf = output_df_radial.loc[output_df_radial['mat_id'] == sample_mat_id]

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



        for mat in output_df['mat_id'].unique():
            # print(mat)
            subdf = output_df.loc[output_df['mat_id'] == mat]
            labels_test_cry_sys.append(subdf.iloc[0]['True Values Crystal System'])

            # subdf = subdf.sample(random_inds, random_state=42)

            predictions_across_zones_cry_sys = subdf['Aggregate Predictions Crystal System']  

            # xs_crystal = np.asarray(predictions_across_zones_cry_sys.value_counts().index)
            # ys_crystal = np.asarray(predictions_across_zones_cry_sys.value_counts())
            pred_crystal_system.append(predictions_across_zones_cry_sys)


        cm = confusion_matrix(labels_test_cry_sys, pred_crystal_system)

        trues = 0
        for i in range(0, len(cm)):
            trues += cm[i][i]

        cm = cm/len(pred_crystal_system)
        for i in range(0, len(cm)):
            for j in range(0, len(cm[0])):
                cm[i][j] = round(cm[i][j]*100, 1)
        accuracy = trues/len(output_df['mat_id'].unique())
        print('crystal system ' + str(accuracy))
        df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)

        # df_cm.to_pickle('Model_data/Crystal_sys_outputs/cm_ag_percent.pkl')

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
            
        crystal_sys_alph = ['C', 'H', 'M', 'O', 'Te', 'Tr']
        df_cm = pd.DataFrame(cm, crystal_sys_alph, crystal_sys_alph)

        
        ax = sn.heatmap(df_cm, annot=True, cmap = 'Blues', vmin = 0.0, vmax = 22, cbar_kws={"ticks":[0.0,5,10,15,20], "location":'bottom', 
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

        crystal_sys_alph = ['Cubic', 'Hexagonal', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal']
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

        for mat in output_df['mat_id'].unique():
            # print(mat)
            subdf = output_df.loc[output_df['mat_id'] == mat]
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

            prediction_mapped = map_predictions(predictions_across_zones_cry_sys, list_of_classes = crystal_sys_alph)
            true_mapped = map_predictions(true_val, list_of_classes = crystal_sys_alph)
            predictions_matrix[true_mapped][prediction_mapped].append(confidence)

                # predictions_ordered_cry_sys_full.append(prediction_ordered_cry_sys)
                # predictions_ordered_point_group_full.append(prediction_ordered_point_group)

        for k in range(0, len(predictions_matrix)):
            for l in range(0, len(predictions_matrix)):
                if len(predictions_matrix[k][l]) == 0:
                    predictions_matrix[k][l] = 0
                else:
                    predictions_matrix[k][l] = np.mean(predictions_matrix[k][l])
        crystal_sys_alph = ['C', 'H', 'M', 'O', 'Te', 'Tr']
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
                
        crystal_sys_alph = ['C', 'H', 'M', 'O', 'Te', 'Tr']
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

def map_predictions(prediction, rf_model = None, list_of_classes = None):
    list_of_classes_uncapped = []
    for c in list_of_classes:
        c_lower = c.lower()
        list_of_classes_uncapped.append(c_lower)
    #print(prediction)
    if rf_model != None: 
        classes = list(rf_model.classes_)
        index = classes.index(prediction)
        return index
    if list_of_classes_uncapped != None:
        index = list_of_classes_uncapped.index(prediction)
        return index
    

def prepare_training_and_test(df, df_type = 'radial', test_fraction = 0.25, column_to_use = '20_ang', 
                              quantity = 'space_group_symbol', split_by_material = True):
    if df_type == 'radial':
        col_name = 'radial_' + column_to_use + '_Colin_basis'
        # col_name = 'radial_' + column_to_use

    if df_type == 'zernike':
        col_name = 'zernike_' + column_to_use     
    
    input_col = np.asarray(df[col_name])
    labels = df[quantity]
    
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
        
        labels_train = df.loc[df['mat_id'].isin(mat_ids_train)][quantity]
        labels_test = df.loc[df['mat_id'].isin(mat_ids_test)][quantity]
        
    else:
        input_train_first, input_test_first, labels_train, labels_test = train_test_split(input_col, labels,
                                                                          test_size=test_fraction,
                                                                          random_state=32)
    if len(input_train_first[0].shape) < 2:
        input_train_temp = input_train_first
        input_test_temp = input_test_first
    
    else:
        input_train_temp = []
        count = 0
        for unit in input_train_first:
            if count in np.arange(0,3000000,10000):
                print(count)
            count += 1
            input_train_temp.append(flatten(unit))

        input_test_temp = []
        count = 0
        for unit in input_test_first:
            if count in np.arange(0,3000000,10000):
                print(count)
            count += 1
            input_test_temp.append(flatten(unit))
    
    input_test_first = None
    input_train_first = None
    
    input_train = []
    count = 0
    for row in input_train_temp:
        if count in np.arange(0,3000000,10000):
            print(count)
        count += 1
        temp = []
        abs_i = np.abs(row)
        angle_i = np.angle(row)
        for i in range(0, len(abs_i)):
            temp.append(abs_i[i])
            temp.append(angle_i[i])
            # print(temp)
        input_train.append(np.asarray(temp))
    
    input_train = np.asarray(input_train)

    count = 0
    input_test = []
    for row in input_test_temp:
        if count in np.arange(0,3000000,10000):
            print(count)
        count += 1
        temp = []
        abs_i = np.abs(row)
        angle_i = np.angle(row)
        for i in range(0, len(abs_i)):
            temp.append(abs_i[i])
            temp.append(angle_i[i])
        input_test.append(np.asarray(temp)) 
    
    input_test = np.asarray(input_test)
    
    return [input_train, input_test, labels_train, labels_test]

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

def remove_triclinic(inputs, labels):
    df_inputs = pd.DataFrame(inputs)
    df_inputs['crystal system'] = np.asarray(labels)
    df_inputs['radial_dataframe_indicies'] = labels.index

    triclinic_indicies = df_inputs.loc[df_inputs['crystal system'] == 'triclinic'].index
    print(len(triclinic_indicies))
    df_no_triclinic = df_inputs.drop(triclinic_indicies)
    # df_no_triclinic.reset_index(inplace = True)
    # df_no_triclinic.drop('index', axis = 1, inplace = True)
    return df_no_triclinic

def visualize_feature_importances(rf_model):
    plt.figure(figsize = (12,10))
    plt.xticks(fontsize = 26)
    plt.yticks(fontsize = 26)
    plt.ylabel('Normalized Variance Reduction', fontsize = 30)
    plt.xlabel('Radial Index', fontsize = 30)
    plt.title('Feature Importances', fontsize = 30)
    plt.plot(np.arange(0, 546, 1), 
             rf_model.feature_importances_, linewidth =3, color = '#ff7f0e')
    loc = 0
    for i in range(0, 21):
        plt.vlines(loc, 0, max(rf_model.feature_importances_), color = 'k', linestyle = '--')
        loc += 26
    plt.show()
    
def lattice_prepare_training_and_test(df, df_type = 'radial', test_fraction = 0.25, column_to_use = '200_ang', split_by_material = True,
                                     use_scaled_cols = False, return_ids = False):
    
    if df_type == 'radial':
        col_name = 'radial_' + column_to_use + '_Colin_basis'
    if df_type == 'zernike':
        col_name = 'zernike_' + column_to_use     
    
    input_col = np.asarray(df[col_name])
    if return_ids == False: 
        labels = df[['a', 'b', 'c', 'alpha', 'beta', 'gamma']]
    else:
        labels = None
    
    if split_by_material: 
        mat_ids = df['mat_id'].unique()
        dummy_output = np.ones((len(mat_ids)))
        mat_ids_train, mat_ids_test, dummy_train, dummy_test = train_test_split(mat_ids, dummy_output,
                                                                          test_size=test_fraction,
                                                                          random_state=32)
        print(len(mat_ids_train))
        print(len(mat_ids_test))
        
        if return_ids:
            return [mat_ids_train, mat_ids_test]
        
        input_train_first = df.loc[df['mat_id'].isin(mat_ids_train)][col_name]
        input_test_first = df.loc[df['mat_id'].isin(mat_ids_test)][col_name]
        if use_scaled_cols:
            print('using scaled cols')
            labels_train = df.loc[df['mat_id'].isin(mat_ids_train)][['a_sorted', 'b_sorted', 'c_sorted', 'alpha', 'beta', 'gamma']]
            labels_test = df.loc[df['mat_id'].isin(mat_ids_test)][['a_sorted', 'b_sorted', 'c_sorted', 'alpha', 'beta', 'gamma']]
            
        else:
            print('using unscaled cols')
            labels_train = df.loc[df['mat_id'].isin(mat_ids_train)][['a', 'b', 'c', 'alpha', 'beta', 'gamma']]
            labels_test = df.loc[df['mat_id'].isin(mat_ids_test)][['a', 'b', 'c', 'alpha', 'beta', 'gamma']]
        
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
            input_train_temp.append(flatten(unit))

        input_test_temp = []
        for unit in input_test_first:
            input_test_temp.append(flatten(unit))
    
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
        
    
    return [input_train, input_test, labels_train, labels_test]

def lattice_rf_diffraction(labels_train, labels_test, input_train, input_test, num_trees = 80, max_depth = 30, max_features = 'sqrt'):
    rf_model = RandomForestRegressor(n_estimators=num_trees, n_jobs=-1, 
                                     max_features=max_features, random_state=32, verbose = 2, max_depth = max_depth)
    rf_model.fit(np.asarray(input_train), np.asarray(labels_train))
    accuracy = rf_model.score(np.asarray(input_test), np.asarray(labels_test))
    print('accuracy = ' + str(accuracy))
    predictions = rf_model.predict(input_test)
    
    predictions_full = []
    trees = rf_model.estimators_
    for tree in trees:
        predictions_full.append(tree.predict(np.asarray(input_test)))
        # print(tree.predict(np.asarray(updated_spectra_test)))
    predictions_ordered = np.asarray(predictions_full).T
        # plt.plot(np.arange(0,max(errors),0.1), np.arange(0,max(errors),0.1), color = 'k', linewidth = 3, linestyle = '--')
        
    return [predictions_ordered, predictions, rf_model]

def lattice_visualize_predictions(predictions_ordered, predictions, rf_model, labels_test, inputs_test, test_indicies):
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
        errors = np.abs(x_labels - x_predictions)
        errors = np.asarray(errors)
        predictions_std = []
        for prediction in predictions_ordered[count]:
            predictions_std.append(np.std(prediction))
        uncertianties.append(predictions_std)
        # print(len(errors))
        # print(len(predictions_std))
        # print(errors)
        # print(predictions_std)
        plt.scatter(errors, predictions_std)
        plt.title('Errors vs Prediction Std', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Error in ' + param + ' Prediction', fontsize=16)
        plt.ylabel('Prediction Std', fontsize=16)
        plt.show()

        MSE = np.square(errors).mean() 
        RMSE = math.sqrt(MSE)
        print('RMSE ' + str(RMSE))

        plt.figure(figsize=(8, 7))
        plt.title('Error Histogram', fontsize=18)
        hist = plt.hist(errors, bins = 50)
        plt.vlines(RMSE, max(hist[0]), min(hist[0]), color='limegreen', linewidth=5, label='RMSE')
        plt.text(RMSE+0.25, max(hist[0])-0.1*max(hist[0]), 'RMSE = ' +str(round(RMSE, 3)), horizontalalignment='center', fontsize = 16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Error', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.show()

        plt.figure(figsize=(8, 7))
        print('R2 Score ' + str(r2_score(x_labels, x_predictions)))
        plt.scatter(x_predictions, x_labels, c=predictions_std, s=50)
        # plt.xlim([1.9, 3.1])
        # plt.ylim([1.9, 3.1])
        cb = plt.colorbar(label='Prediction Std')
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=22)
        text.set_font_properties(font)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(22)
        min_plot = round(min(x_labels)-0.5, 0)
        max_plot = round(max(x_labels)+1.5, 0)
        plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3, linestyle='--')
        plt.title('Predicted vs True ' + param, fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(param + ' Prediction', fontsize=22)
        plt.ylabel('True ' + param, fontsize=22)
        plt.show()
        count += 1 
        
        
    output_list = []
    for i in range(0, len(labels_test)):
        output_list.append([predictions[i], predictions[i][0], predictions[i][1], predictions[i][2],
                            predictions[i][3], predictions[i][4], predictions[i][5],
                            np.asarray(labels_test)[i], np.asarray(labels_test)[i][0], np.asarray(labels_test)[i][1],
                            np.asarray(labels_test)[i][2], np.asarray(labels_test)[i][3], np.asarray(labels_test)[i][4],
                             np.asarray(labels_test)[i][5], predictions_ordered[0][i], predictions_ordered[1][i], 
                            predictions_ordered[2][i], predictions_ordered[3][i], predictions_ordered[4][i], predictions_ordered[5][i], 
                             test_indicies[i], uncertianties[0][i], uncertianties[1][i], uncertianties[2][i], uncertianties[3][i],
                            uncertianties[4][i], uncertianties[5][i]])


    output_df = pd.DataFrame(np.asarray(output_list),
                             columns = ['All Predictions', 'Predictions x', 'Predictions y', 'Predictions z',
                                        'Predictions alpha', 'Predictions beta', 'Predictions gamma', 'Full Labels Test',
                                        'True x', 'True y', 'True z', 'True alpha', 'True beta', 'True gamma', 
                                        'Full Predictions x', 'Full Predictions y', 'Full Predictions z', 
                                        'Full Predictions alpha', 'Full Predictions beta', 'Full Predictions gamma',
                                        'Full DF Index', 'std x', 'std y', 'std z', 'std alpha', 'std beta', 'std gamma']

                            )

    return output_df

def update_df_lattice(df):
    mat_ids = list(df['mat_id'].unique())
    # mat_ids = mat_ids[0:1000]
    # structures = mpr.get_structure_by_material_id(list(mat_ids))
    mpr = MPRester('CEvsr9tiYxi6MaxfRnSU7V9FCaIAcAZh')

    a = []
    b = []
    c = []
    alpha = []
    beta = []
    gamma = []
    volumnes = []
    ids = []
    count = 0
    for i in range(0, len(df)):
        # try:
        mat_id = df.iloc[i]['mat_id']
        # print(mat_id)
        # if mat_id == 'mp-510271':
        #     print(mat_id)
        # else:
        if i != 0:
            prev_mat_id = df.iloc[i-1]['mat_id']
        if i == 0: 
            prev_mat_id = None

        if mat_id == prev_mat_id:

            a.append(round(conventional_structure.lattice.a, 2))
            b.append(round(conventional_structure.lattice.b, 2))
            c.append(round(conventional_structure.lattice.c, 2))

            alpha.append(round(conventional_structure.lattice.alpha, 2))
            beta.append(round(conventional_structure.lattice.beta, 2))
            gamma.append(round(conventional_structure.lattice.gamma, 2))           
            # volumnes.append(round(conventional_structure.lattice.a*conventional_structure.lattice.b*conventional_structure.lattice.c, 2))

            ids.append(mat_id)

        else: 
            if count in np.arange(0,10000,500):
                print(count)
            structure = df.iloc[i]['structure']

            sga = SpacegroupAnalyzer(structure)
            conventional_structure = sga.get_conventional_standard_structure()
            # print(i, count)

            a.append(round(conventional_structure.lattice.a, 2))
            b.append(round(conventional_structure.lattice.b, 2))
            c.append(round(conventional_structure.lattice.c, 2))

            alpha.append(round(conventional_structure.lattice.alpha, 2))
            beta.append(round(conventional_structure.lattice.beta, 2))
            gamma.append(round(conventional_structure.lattice.gamma, 2))           
            # volumnes.append(round(conventional_structure.lattice.a*conventional_structure.lattice.b*conventional_structure.lattice.c, 2))

            ids.append(mat_id)
            # print(x[i],y[i],z[i],alpha[i],beta[i],gamma[i],mat_id)
            count += 1 
        # except:
            # print(mat_id)
            # return [x,y,z,alpha,beta,gamma]
        # print(round(conventional_structure.lattice.a, 2), 
         #      round(conventional_structure.lattice.b, 2), 
          #     round(conventional_structure.lattice.c, 2), 
           #    round(conventional_structure.lattice.alpha, 2), 
            #   round(conventional_structure.lattice.beta, 2), 
             #  round(conventional_structure.lattice.gamma, 2), 
              # round(conventional_structure.lattice.a*conventional_structure.lattice.b*conventional_structure.lattice.c, 2),
              # mat_id)
    
    df['a'] = a
    df['b'] = b
    df['c'] = c
    
    df['alpha'] = alpha
    df['beta'] = beta
    df['gamma'] = gamma
    
    df['ids'] = ids
    # return None
    return df

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

