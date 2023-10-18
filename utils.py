import os 
import json 
import torch 
from torch.autograd import Variable
import numpy as np 
import pandas as pd 
import torch.nn as nn
import models
import sklearn.metrics as metrics
# plot cm matrix 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 100
import seaborn as sns
plt.style.use('bmh')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
legend_properties = {'weight':'bold', 'size': 6}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ce_loss = nn.CrossEntropyLoss()
softmax = torch.nn.Softmax(dim=1) 

def creat_checkpoint_folder(target_path, target_file, data):
    """
    Create a folder to save the checkpoint
    input: target_path,
           target_file,
           data
    output: None
    """
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
            os.makedirs(target_path + '/auc_plots')
            os.makedirs(target_path + '/auc_plots_cv')
            os.makedirs(target_path + '/cm_maps')
            os.makedirs(target_path + '/cm_maps_cv')
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)


def crop_data_target(vital, target_dict, static_dict, mode, target_index):
    '''
    vital: a list of nd array [[200, 81], [200, 93], ...] 
    target_dict: dict of SOFA score: {'train': {30015933: [81, 1], 30016009: [79, 1], ...}, 'dev': }
    static_dict: dict of static variables: {'static_train': a DataFrame (27136, 27) , 'static_dev':}
    variables in static dict: 'gender', 'age', 'hospital_expire_flag', 'max_hours',
       'myocardial_infarct', 'congestive_heart_failure',
       'peripheral_vascular_disease', 'cerebrovascular_disease', 'dementia',
       'chronic_pulmonary_disease', 'rheumatic_disease',
       'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_without_cc',
       'diabetes_with_cc', 'paraplegia', 'renal_disease', 'malignant_cancer',
       'severe_liver_disease', 'metastatic_solid_tumor', 'aids',
       'ethnicity_AMERICAN INDIAN', 'ethnicity_ASIAN', 'ethnicity_BLACK',
       'ethnicity_HISPANIC/LATINO', 'ethnicity_OTHER', 'ethnicity_WHITE'
    return:
    train_filter: [ndarray with shape (200, 8),...
    sofa_tail: [ndarray with shape (8, 1),
    stayids: [39412629, 37943756, 32581623, 37929132,
    train_target: [0, 0, 1, 0, 0, 1]

    '''
    idx = pd.IndexSlice
    length = [i.shape[-1] for i in vital]
    train_filter = [vital[i][:, :-24] for i, m in enumerate(length) if m >24]
    all_train_id = list(target_dict[mode].keys())
    stayids = [all_train_id[i] for i, m in enumerate(length) if m >24]
    sofa_tail = [target_dict[mode][j][24:]/15 for j in stayids]
    static_key = 'static_' + mode
    # for eicu eicu_static['static_train'].loc[141168][1] becomes a value 
    if target_index == 21: # race: 2 is balck, 5 is white 
        # shape [1,6] then use nonzero, after e.g.array([5])
        train_target = [np.nonzero(static_dict[static_key].loc[idx[:, :, j]].iloc[:, 21:].values)[1] for j in stayids]
        sub_ind = [i for i, m in enumerate(train_target) if m == 2 or m == 5]
        race_dict = {2: 1, 5:0}
         # a list of target class
        train_targets = [race_dict[train_target[i][0]]for i in sub_ind]
        train_filters = [train_filter[i] for i in sub_ind]
        sofa_tails = [sofa_tail[i] for i in sub_ind]
        stayidss = [stayids[i] for i in sub_ind]
        return train_filters, sofa_tails, stayidss, train_targets
    elif target_index == 1: # age, binarize it
        # age median is 0.1097
        train_target = [static_dict[static_key].loc[idx[:, :, j]].iloc[:, 1].values[0] for j in stayids]
        train_target = [1 if i >= 0.1097 else 0 for i in train_target]
    else:
        # a list of target class
        train_target = [static_dict[static_key].loc[idx[:, :, j]].iloc[:, target_index].values[0] for j in stayids]
        
    return train_filter, sofa_tail, stayids, train_target

def crop_data_target_e(vital, target_dict, static_dict, mode, target_index):
    '''
    vital: a list of nd array [[200, 81], [200, 93], ...] 
    target_dict: dict of SOFA score: {'train': {30015933: [81, 1], 30016009: [79, 1], ...}, 'dev': }
    static_dict: dict of static variables: {'static_train': a DataFrame (27136, 27) , 'static_dev':}
    variables in static dict: 'gender', 'age', 'hospital_expire_flag', 'max_hours',
       'myocardial_infarct', 'congestive_heart_failure',
       'peripheral_vascular_disease', 'cerebrovascular_disease', 'dementia',
       'chronic_pulmonary_disease', 'rheumatic_disease',
       'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_without_cc',
       'diabetes_with_cc', 'paraplegia', 'renal_disease', 'malignant_cancer',
       'severe_liver_disease', 'metastatic_solid_tumor', 'aids',
       'ethnicity_AMERICAN INDIAN', 'ethnicity_ASIAN', 'ethnicity_BLACK',
       'ethnicity_HISPANIC/LATINO', 'ethnicity_OTHER', 'ethnicity_WHITE'
    return:
    train_filter: [ndarray with shape (200, 8),...
    sofa_tail: [ndarray with shape (8, 1),
    stayids: [39412629, 37943756, 32581623, 37929132,
    train_target: [0, 0, 1, 0, 0, 1]

    '''
    # idx = pd.IndexSlice
    length = [i.shape[-1] for i in vital]
    train_filter = [vital[i][:, :-24] for i, m in enumerate(length) if m >24]
    all_train_id = list(target_dict[mode].keys())
    stayids = [all_train_id[i] for i, m in enumerate(length) if m >24]
    sofa_tail = [target_dict[mode][j][24:]/15 for j in stayids]
    static_key = 'static_' + mode
    # for eicu eicu_static['static_train'].loc[141168][1] becomes a value 
    if target_index == 21: # race: 2 is balck, 5 is white 
        # shape [1,6] then use nonzero, after e.g.array([5])
        train_target = [np.nonzero(static_dict[static_key].loc[j][21:].values)[0] for j in stayids]
        sub_ind = [i for i, m in enumerate(train_target) if m == 2 or m == 5]
        race_dict = {2: 1, 5:0}
         # a list of target class
        train_targets = [race_dict[train_target[i][0]]for i in sub_ind]
        train_filters = [train_filter[i] for i in sub_ind]
        sofa_tails = [sofa_tail[i] for i in sub_ind]
        stayidss = [stayids[i] for i in sub_ind]
        return train_filters, sofa_tails, stayidss, train_targets

    elif target_index == 1: # age, binarize it
        # age median is 0.1097
        train_target = [static_dict[static_key].loc[j][1] for j in stayids]
        train_target = [1 if i >= 0.1097 else 0 for i in train_target]
    else:
        # a list of target class
        train_target = [static_dict[static_key].loc[j][target_index] for j in stayids]
    
    if target_index == 0: # eicu gender has unknown:
        known_ids = [k for k, i in enumerate(stayids) if train_target[k] != 2.0]
        train_filter = [train_filter[i] for i in known_ids]
        sofa_tail = [sofa_tail[i] for i in known_ids]
        stayids= [stayids[i] for i in known_ids]
        train_target= [train_target[i] for i in known_ids]
        
        return train_filter, sofa_tail, stayids, train_target

    return train_filter, sofa_tail, stayids, train_target

def filter_sepsis(vital, sofa, ids, target):
    id_df = pd.read_csv('/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/mimic_sepsis3.csv')
    sepsis3_id = id_df['stay_id'].values # 1d array 
    index_dict = dict((value, idx) for idx,value in enumerate(ids))
    ind = [index_dict[x] for x in sepsis3_id if x in index_dict.keys()]
    vital_sepsis = [vital[i] for i in ind]
    sofa_sepsis = [sofa[i] for i in ind]
    target_sepsis = [target[i] for i in ind]
    return vital_sepsis, sofa_sepsis, [ids[i] for i in ind], target_sepsis

def filter_sepsis_e(vital, sofa, ids, target):
    id_df = pd.read_csv('/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/eicu_sepsis3.csv')
    sepsis3_id = id_df['patientunitstayid'].values # 1d array 
    index_dict = dict((value, idx) for idx,value in enumerate(ids))
    ind = [index_dict[x] for x in sepsis3_id if x in index_dict.keys()]
    vital_sepsis = [vital[i] for i in ind]
    sofa_sepsis = [sofa[i] for i in ind]
    target_sepsis = [target[i] for i in ind]
    return vital_sepsis, sofa_sepsis, [ids[i] for i in ind], target_sepsis