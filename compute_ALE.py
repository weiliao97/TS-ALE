import argparse
import torch
import numpy as np
import pandas as pd
from datetime import date
import os
import json
import glob
import pickle
import utils
import make_optimizer
import prepare_data
import models
import importlib
importlib.reload(utils)
import torch.nn as nn
from sklearn.model_selection import KFold

from plot_metric.functions import BinaryClassification
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import softmax
matplotlib.rcParams["figure.dpi"] = 300

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# matplotlib.rcParams.update({'font.size': 10})
legend_properties = {'weight':'bold', 'size': 4}
# Visualisation with plot_metric
plt.style.use('bmh')

today = date.today()
date = today.strftime("%m%d")
kf = KFold(n_splits=10, random_state=42, shuffle=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for read static info models")

    ## model representation to get from
    parser.add_argument("--sens_index", type=int, default=0, help = "target index to predict")

    args = parser.parse_args()

    # load data 
    meep_mimic = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_compile_0911_2022.npy', \
                    allow_pickle=True).item()
    train_vital = meep_mimic ['train_head']
    dev_vital = meep_mimic ['dev_head']
    test_vital = meep_mimic ['test_head']
    mimic_static = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_static_0922_2022.npy', \
                            allow_pickle=True).item()
    mimic_target = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_target_0922_2022.npy', \
                            allow_pickle=True).item()

    #read eicu data
    meep_eicu = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/eICU_compile_0911_2022_2.npy', \
                    allow_pickle=True).item()
    train_vital_e = meep_eicu['train_head']
    dev_vital_e = meep_eicu ['dev_head']
    test_vital_e = meep_eicu ['test_head']
    eicu_static = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/eICU_static_0922_2022.npy', \
                            allow_pickle=True).item()
    eicu_target = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/eICU_target_0922_2022.npy', \
                            allow_pickle=True).item()

    target_index = [0, 1, 21] + [i for i in range(4, 21)]
    target_name = ['Gender', 'Age', 'Ethnicity', 'MI', 'CHF',
        'PVD', 'CBVD', 'Dementia', 'CPD', 'RD',
        'PUD', 'MLD', 'Diabetes_wo_cc',
        'Diabetes_cc', 'Paraplegia', 'Renal', 'Mal_cancer',
        'SLD', 'MST'] #'AIDS'
    bucket_sizes =  [300, 300, 300, 300, 300, 300, 300, 1200, 300, 1200, 1200, 600, 300, 600, 1200, 300, 300, 1200, 1200]
    # gender 15345 age 27136 hospital_expire_flag 2510 max_hours 27136 myocardial_infarct 4692 congestive_heart_failure 6781
    # peripheral_vascular_disease 3081 cerebrovascular_disease 4302 dementia 991 chronic_pulmonary_disease 6373 rheumatic_disease 891
    # peptic_ulcer_disease 723 mild_liver_disease 2711 diabetes_without_cc 6038 diabetes_with_cc 2281 paraplegia 1401 renal_disease 4930 malignant_cancer 3497
    # severe_liver_disease 1189 metastatic_solid_tumor 1749 aids 133
    # ethnicity_AMERICAN INDIAN 46 ethnicity_ASIAN 791 ethnicity_BLACK 2359 ethnicity_HISPANIC/LATINO 919 ethnicity_OTHER 4547 ethnicity_WHITE 18474
    # for i in range(2, 3):
    arg_dict['target_index'] = target_index[args.sens_ind]
    args.bucket_size = bucket_sizes[args.sens_ind]
    workname = date + '_' + args.database + '_' + target_index[args.sens_ind]
    creat_checkpoint_folder('./checkpoints/' + workname, 'params.json', arg_dict)
    train_head, train_sofa, train_id, train_target =  utils.crop_data_target(train_vital, mimic_target, mimic_static, 'train', target_index[args.sens_ind])
    dev_head, dev_sofa, dev_id, dev_target =  utils.crop_data_target(dev_vital , mimic_target, mimic_static, 'dev', target_index[args.sens_ind])
    test_head, test_sofa, test_id, test_target =  utils.crop_data_target(test_vital, mimic_target, mimic_static, 'test', target_index[args.sens_ind])
    train_head_e, train_sofa_e, train_id_e, train_target_e =  utils.crop_data_target_e(train_vital_e, eicu_target, eicu_static, 'train', target_index[args.sens_ind])
    dev_head_e, dev_sofa_e, dev_id_e, dev_target_e =  utils.crop_data_target_e(dev_vital_e, eicu_target, eicu_static, 'dev', target_index[args.sens_ind])
    test_head_e, test_sofa_e, test_id_e, test_target_e =  utils.crop_data_target_e(test_vital_e, eicu_target, eicu_static, 'test', target_index[args.sens_ind])

    if args.use_sepsis3 == True:
        train_head, train_sofa, train_id, train_target = utils.filter_sepsis(train_head, train_sofa, train_id, train_target)
        dev_head, dev_sofa, dev_id, dev_target = utils.filter_sepsis(dev_head, dev_sofa, dev_id, dev_target)
        test_head, test_sofa, test_id, test_target = utils.filter_sepsis(test_head, test_sofa, test_id, test_target)
        train_head_e, train_sofa_e, train_id_e, train_target_e = utils.filter_sepsis_e(train_head_e, train_sofa_e, train_id_e, train_target_e)
        dev_head_e, dev_sofa_e, dev_id_e, dev_target_e = utils.filter_sepsis_e(dev_head_e, dev_sofa_e, dev_id_e, dev_target_e)
        test_head_e, test_sofa_e, test_id_e, test_target_e = utils.filter_sepsis_e(test_head_e, test_sofa_e, test_id_e, test_target_e)

    # load pretrained model 

