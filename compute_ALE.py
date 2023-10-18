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
import ale 
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
today_date = today.strftime("%m%d")
kf = KFold(n_splits=10, random_state=42, shuffle=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for read static info models")


    # data
    parser.add_argument("--database", type=str, default='mimic_tran_general_sofa', choices=['mimic', 'eicu'])
    parser.add_argument("--use_sepsis3", action = 'store_true', default= False, help="Whethe only use sepsis3 subset")
    parser.add_argument("--input_dim", type = int, default= 200, help="Dimension of variables used to train the extarction model")
    parser.add_argument("--bucket_size", type=int, default=300, help="path to the dataset")
    # TCN
    parser.add_argument("--kernel_size", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout")
    parser.add_argument("--reluslope", type=float, default=0.1, help="Relu slope in the fc model")
    parser.add_argument('--num_channels', nargs='+', type = int, help='num of channels in TCN')
    # FC 
    parser.add_argument("--read_drop", type=float, default=0.2, help="Model dropout in FC read model")
    parser.add_argument("--read_reluslope", type=float, default=0.1, help="Relu slope in the FC read model")
    parser.add_argument("--read_channels", nargs='+', type = int, help='num of channels in FC read model')
    parser.add_argument("--output_classes", type=int, default=2, help="Which static column to target")
    parser.add_argument("--sens_index", type=int, default=0, help = "target index to predict")
    parser.add_argument("--cal_pos_acc", action = 'store_false', default=True, help="Whethe calculate the acc of the positive class")


    # model path 
    parser.add_argument("--model_path", type=str, default='/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/checkpoints/0125_mimic_TCNsepsis_3_256_ks3/fold0_best_loss.pt')
    parser.add_argument("--fc_model_path", type=str, default='/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/checkpoints/0227_mimic_Transformer_general/fold3_best_loss.pt')
  

    args = parser.parse_args()
    arg_dict = vars(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    args.bucket_size = bucket_sizes[args.sens_ind]
    workname = today_date + '_' + args.database + '_' + target_index[args.sens_ind]
    utils.creat_checkpoint_folder('./checkpoints/' + workname, 'params.json', arg_dict)
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
    model = models.TemporalConv(num_inputs = args.input_dim, num_channels=args.num_channels, kernel_size=args.kernel_size, dropout = args.dropout, output_class=1)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    fc_model = models.FCNet(num_inputs=args.num_channels[-1], num_channels=args.read_channels, \
                            dropout=args.read_drop, reluslope=args.read_reluslope, \
                            output_class=args.output_classes)
    fc_model.to(device)
    fc_model.load_state_dict(torch.load(args.fc_model_path))
    fc_model.eval()

    model_c = models.Combined_model(model.network, fc_model)

    # run ale for continous variables
    mimic_mean_std = pd.read_hdf('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MEEP_stats_MIMIC.h5')
    col_means, col_stds = mimic_mean_std.loc[:, 'mean'], mimic_mean_std.loc[:, 'std']
    var_inds = [i for i in range(0, 109, 2)] + [i for i in range(116, 169, 2)]
    keys = list(col_means.keys())
    keys_sim = [i[0] for i in keys]
    ale_df = pd.DataFrame(index=keys_sim)
    ale_df['ale'] = ''
    for var_ind in var_inds:
        ind = var_ind//2 if var_ind <= 108 else (var_ind-6)//2
        key = keys_sim[ind]
        max_ale = []
        quantile_t, ale_t, quantile_nc, ale_nc= ale.get_1d_ale(model_c, test_head, index=var_ind, bins=20, monte_carlo_ratio=0.1, monte_carlo_rep=50, record_flag=1)
        # ind = 55
        fig, ax = plt.subplots(figsize=(5, 4))
        for q, a in zip(quantile_t, ale_t):
            max_ale.append(np.max(a) - np.min(a))
            if len(q) > len(a):
                extra = len(q) - len(a)
                ax.plot(q[extra:]*col_stds[keys[ind]] + col_means[keys[ind]], a, color="#1f77b4", alpha=0.1)
            elif len(q) < len(a):
                extra = len(q) - len(a)
                ax.plot(q*col_stds[keys[ind]] + col_means[keys[ind]], a[extra:], color="#1f77b4", alpha=0.1)
            else:
                ax.plot(q*col_stds[keys[ind]] + col_means[keys[ind]], a, color="#1f77b4", alpha=0.1)
        # print(key, max_ale)
        ale_df.loc[key, 'ale'] = np.mean(max_ale)
        ale_df.loc[key, 'ale_raw'] = max_ale 
        # ax.axvline(x=3, color = '#b45c1f', linestyle='--')
        # ax.axvline(x=5, color = '#b45c1f', linestyle='--')
        ax.set_xlabel('%s'%key, size=18,  fontweight='bold')
        ax.set_ylabel('Prob ALE', size=18,  fontweight='bold')
        plt.savefig('/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/Read/checkpoints/' + workname + '/%s_ale.pdf'%key, format='pdf', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
    ale_df.to_csv('/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/Read/checkpoints/' + workname + '/ale.csv')
    



