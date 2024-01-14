import numpy as np
import pandas as pd 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datetime import date
today = date.today()
date = today.strftime("%m%d")
import models
import prepare_retrain
import utils
import pickle 
import json 
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=None, shuffle=False)
mse_loss = nn.MSELoss()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for Tranformer models")
    # data
    parser.add_argument("--database", type=str, default='mimic', choices=['mimic', 'eicu'])
    parser.add_argument("--col_count", type=int, default=10, help = 'top k cols to drop')
    parser.add_argument("--ale_file", type=str)
    parser.add_argument("--use_random", action = 'store_true', default= False, help="Whethe randomly drop cols")
    parser.add_argument("--use_reverse", action = 'store_true', default= False, help="Whethe reversely drop cols")
    parser.add_argument("--task", type=str, default='sofa', choices=['static', 'sofa'])
    # datapath 
    # parser.add_argument("--td_data", type=str, help = 'path to the time-series data')
    # parser.add_argument("--static_data", type=str, help = 'path to the static data')
    # parser.add_argument("--sofa_data", type=str, help = 'path to the SOFA target data')
    # data grouping and cohort 
    parser.add_argument("--bucket_size", type=int, default=300, help="bucket size to group different length of time-series data")
    parser.add_argument("--use_sepsis3", action='store_false', default=True, help="Whethe only use sepsis3 subset")
    
    # modeling
    parser.add_argument("--model_name", type=str, default='TCN', choices=['TCN', 'RNN', 'Transformer'])

    # TCN
    parser.add_argument("--kernel_size", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout")
    parser.add_argument("--reluslope", type=float, default=0.1, help="Relu slope in the fc model")
    parser.add_argument('--num_channels', nargs='+', help='TCN model channels', type=int)

    # LSTM
    parser.add_argument("--rnn_type", type=str, default='lstm', choices=['rnn', 'lstm', 'gru'])
    parser.add_argument("--hidden_dim", type=int, default=256, help="RNN hidden dim")
    parser.add_argument("--layer_dim", type=int, default=3, help="RNN layer dim")

    # transformer
    parser.add_argument('--warmup', action='store_true', help="whether use learning rate warm up")
    parser.add_argument('--lr_factor', type=int, default=0.1, help="warmup_learning rate factor")
    parser.add_argument('--lr_steps', type=int, default=2000, help="warmup_learning warm up steps")
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model")
    parser.add_argument("--n_head", type=int, default=8, help="Attention head of the model")
    parser.add_argument("--dim_ff_mul", type=int, default=4, help="Dimension of the feedforward model")
    parser.add_argument("--num_enc_layer", type=int, default=2, help="Number of encoding layers")


    # ## FC read model parameters
    # parser.add_argument("--read_drop", type=float, default=0.2, help="Model dropout in FC read model")
    # parser.add_argument("--read_reluslope", type=float, default=0.1, help="Relu slope in the FC read model")
    # parser.add_argument("--read_channels", nargs='+', type = int, help='num of channels in FC read model')
    # parser.add_argument("--output_classes", type=int, default=2, help="Which static column to target")
    # parser.add_argument("--infer_ind", type=int, default=1, help="Which static column to target")
    # parser.add_argument("--cal_pos_acc", action = 'store_false', default=True, help="Whethe calculate the acc of the positive class")

    # learning parameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--data_batching", type=str, default='close', choices=['same', 'close', 'random'], help='How to batch data')
    parser.add_argument("--bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Epochs to wait before stop the training when the loss is not decreasing")

    ## data logging
    parser.add_argument("--checkpoint", type=str, default='test_read_models', help="name of checkpoint model")
    # Parse and return arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = '/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/Read/checkpoints/'

    # read data
    meep_mimic = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_compile_0911_2022.npy', \
                    allow_pickle=True).item()
    train_vital = meep_mimic ['train_head']
    dev_vital = meep_mimic ['dev_head']
    test_vital = meep_mimic ['test_head']
    mimic_static = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_static_0922_2022.npy', \
                            allow_pickle=True).item()
    mimic_target = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_target_0922_2022.npy', \
                            allow_pickle=True).item()
    
    # # gender, age, race #
    # target_index = [0, 1, 21] + [i for i in range(4, 21)]
    # target_name = ['Sex', 'Age', 'Race', 'MI', 'CHF',
    #     'PVD', 'CBVD', 'Dementia', 'CPD', 'RD',
    #     'PUD', 'MLD', 'Diabetes_wo_cc',
    #     'Diabetes_cc', 'Paraplegia', 'Renal', 'Mal_cancer',
    #     'SLD', 'MST'] 
    # bucket_sizes =  [300, 300, 300, 300, 300, 300, 300, 1200, 300, 1200, 1200, 600, 300, 600, 1200, 300, 300, 1200, 1200]
   
    # true_ind = target_index[args.infer_ind]
    # args.bucket_size = bucket_sizes[args.infer_ind]

    train_head_or, train_static, train_sofa, train_id = utils.crop_data_target_sofa(args.database, train_vital, mimic_target, mimic_static, 'train')
    dev_head_or, dev_static, dev_sofa, dev_id = utils.crop_data_target_sofa(args.database, dev_vital, mimic_target, mimic_static, 'dev')
    test_head_or, test_static, test_sofa, test_id = utils.crop_data_target_sofa(args.database, test_vital, mimic_target, mimic_static, 'test')

    if args.use_sepsis3 == True:
        train_head_or, train_static, train_sofa, train_id = utils.filter_sepsis_sofa(args.database, train_head_or, train_static, train_sofa, train_id)
        dev_head_or, dev_static, dev_sofa, dev_id = utils.filter_sepsis_sofa(args.database, dev_head_or, dev_static, dev_sofa, dev_id)
        test_head_or, test_static, test_sofa, test_id = utils.filter_sepsis_sofa(args.database, test_head_or, test_static, test_sofa, test_id)

    ale_df = pd.read_csv(base + args.ale_file)
    ale_df.rename(columns={"Unnamed: 0": "col"}, inplace=True)
    # column name to index 
    mimic_mean_std = pd.read_hdf('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MEEP_stats_MIMIC.h5')
    col_means, col_stds = mimic_mean_std.loc[:, 'mean'], mimic_mean_std.loc[:, 'std']
    var_inds = [i for i in range(0, 109, 2)] + [i for i in range(116, 169, 2)]
    keys = list(col_means.keys())
    keys_sim = [i[0] for i in keys]
    name_col = {name: key for name, key in zip(keys_sim, var_inds)}
    
    col_list = [10, 20, 30, 40, 50]
    for i, col_cnt in enumerate([5, 10, 20, 30, 40, 50]):
        args.col_count = col_cnt
        if args.use_random:
            workname = date + '2024_' + 'sofa_retrain_subset_%d'%args.col_count + '_' + 'random' 
        elif args.use_reverse:
            workname = date + '2024_' + 'sofa_retrain_subset_%d'%args.col_count + '_' + 'reverse' 
        else: 
            workname = date + '2024_' + 'sofa_retrain_subset_%d'%args.col_count 
    
        utils.creat_checkpoint_folder(base + workname, 'params.json', vars(args))
        if not args.use_random: 
            if args.use_reverse: 
                print('Sample reverse cols to zero')
                ale_df.sort_values('ale', ascending=True, inplace=True)
                col_list = ale_df.iloc[:args.col_count, :]['col'].to_list()
                col_to_zero = [name_col[c] for c in col_list]  
            else: 
                print('Use ALE to zero cols')
                ale_df.sort_values('ale', ascending=False, inplace=True)
                col_list = ale_df.iloc[:args.col_count, :]['col'].to_list()
                col_to_zero = [name_col[c] for c in col_list]

        else: 
            print('Randomly zero cols')
            if i >= 1: # load previous level
                prev_col = col_list[i-1]
                prev_work = date + '2024_' + 'sofa_retrain_subset_%d'%prev_col + '_' + 'random' 
                with open(base + workname + '/cols_dopped', "rb")as fp:
                    prev_file= pickle.load(fp)
                new_col = random.sample(var_inds, k=args.col_count - prev_col)
                col_to_zero = prev_file + new_col
            else: 
                col_to_zero = random.sample(var_inds, k=args.col_count)
            
        with open(base + workname + '/cols_dopped', "wb")as fp:
            pickle.dump(col_to_zero, fp)

        print(col_to_zero)
        rows_to_zero = col_to_zero + [i+1 for i in col_to_zero]

        train_head = utils.drop_col(train_head_or, rows_to_zero)
        dev_head = utils.drop_col(dev_head_or, rows_to_zero)
        test_head = utils.drop_col(test_head_or, rows_to_zero)
        
        input_dim =train_head[0].shape[0]
        static_dim = train_static[0].shape[0]

        if args.model_name == 'TCN':
            model = models.TemporalConv(num_inputs=input_dim, num_channels=args.num_channels,
                                        kernel_size=args.kernel_size, dropout=args.dropout, output_class=1)
        elif args.model_name == 'RNN':
            model = models.RecurrentModel(cell=args.rnn_type, input_dim = input_dim, hidden_dim=args.hidden_dim, layer_dim=args.layer_dim, \
                                        output_dim=1, dropout_prob=args.dropout, idrop=args.idrop)

        elif args.model_name == 'Transformer':
            model = models.Trans_encoder(feature_dim=input_dim, d_model=args.d_model, \
                    nhead=args.n_head, d_hid=args.dim_ff_mul * args.d_model, \
                    nlayers=args.num_enc_layer, out_dim=1, dropout=args.dropout)
        
        else: 
            raise ValueError('Please specify a valid model architecture')

        print('Model trainable parameters are: %d' % utils.count_parameters(model))
        torch.save(model.state_dict(), '/content/start_weights.pt')

        model.to(device)

        # loss fn and optimizer
        loss_fn = nn.MSELoss()
        model_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        # 10-fold cross validation
        trainval_head = train_head + dev_head
        trainval_static = train_static + dev_static
        trainval_stail = train_sofa + dev_sofa
        trainval_ids = train_id + dev_id

        for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_head)):
            if c_fold == 0: 
                best_loss = 1e4
                patience = 0
                if c_fold >= 1:
                    model.load_state_dict(torch.load('/content/start_weights.pt'))
                print('Starting Fold %d' % c_fold)
                print("TRAIN:", len(train_index), "TEST:", len(test_index))
                train_head, val_head = utils.slice_data(trainval_head, train_index), utils.slice_data(trainval_head, test_index)
                train_static, val_static = utils.slice_data(trainval_static, train_index), utils.slice_data(trainval_static, test_index)
                train_stail, val_stail = utils.slice_data(trainval_stail, train_index), utils.slice_data(trainval_stail, test_index)
                train_id, val_id = utils.slice_data(trainval_ids, train_index), utils.slice_data(trainval_ids, test_index)

                train_dataloader, dev_dataloader, test_dataloader = prepare_retrain.get_data_loader(args, train_head, val_head,
                                                                                                    test_head, 
                                                                                                    train_stail, val_stail,
                                                                                                    test_sofa,
                                                                                                    train_static=train_static,
                                                                                                    dev_static=val_static,
                                                                                                    test_static=test_static,
                                                                                                    train_id=train_id,
                                                                                                    dev_id=val_id,
                                                                                                    test_id=test_id)

                for j in range(args.epochs):
                    model.train()
                    sofa_list = []
                    sofap_list = []
                    loss_t = []
                    loss_to = []

                    for vitals, static, target, train_ids, key_mask in train_dataloader:
                        # print(label.shape)
                        if args.warmup == True:
                            model_opt.optimizer.zero_grad()
                        else:
                            model_opt.zero_grad()
                        # ti_data = Variable(ti.float().to(device))
                        # td_data = vitals.to(device) # (6, 182, 24)
                        # sofa = target.to(device)
            
                        if args.model_name == 'TCN': 
                            sofa_p = model(vitals.to(device))
                        elif args.model_name == 'RNN':
                            # x_lengths have to be a 1d tensor
                            td_transpose = vitals.to(device).transpose(1, 2)
                            x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                            sofa_p = model(td_transpose, x_lengths)
                        elif args.model_name == 'Transformer':
                            tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                            sofa_p = model(vitals.to(device), tgt_mask, key_mask.bool().to(device))
            

                        loss = utils.mse_maskloss(sofa_p, target.to(device), key_mask.to(device))
                        loss.backward()
                        model_opt.step()

                        sofa_list.append(target)
                        sofap_list.append(sofa_p)
                        loss_t.append(loss)

                    loss_avg = np.mean(torch.stack(loss_t, dim=0).cpu().detach().numpy())

                    model.eval()
                    y_list = []
                    y_pred_list = []
                    ti_list = []
                    td_list = []
                    id_list = []
                    loss_val = []
                    with torch.no_grad():  # validation does not require gradient

                        for vitals, static, target, val_ids, key_mask in dev_dataloader:

                            if args.model_name == 'TCN':
                                sofap_t = model(vitals.to(device))
                            elif args.model_name == 'RNN':
                                # x_lengths have to be a 1d tensor 
                                td_transpose = vitals.to(device).transpose(1, 2)
                                x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                                sofap_t = model(td_transpose, x_lengths)
                            elif args.model_name == 'Transformer':
                                tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                                sofap_t = model(vitals.to(device), tgt_mask, key_mask.bool().to(device))

                            loss_v = utils.mse_maskloss(sofap_t, target.to(device), key_mask.to(device))
                            y_list.append(target.detach().numpy())
                            y_pred_list.append(sofap_t.cpu().detach().numpy())
                            loss_val.append(loss_v)
                            id_list.append(val_ids)

                    loss_te = np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())
                    if loss_te < best_loss:
                        patience = 0
                        best_loss = loss_te
                        torch.save(model.state_dict(),
                                    base + workname + '/' + 'fold%d' % c_fold + '_best_loss.pt')
                    else:
                        patience += 1
                        if patience >= 10:
                            print('Start next fold')
                            break
                    print('Epoch %d, : Train loss is %.4f, test loss is %.4f' % (j, loss_avg, loss_te))
        
        # Load weights and do eval on test set, save mse and its ci 
        if args.task == 'sofa':
            model.load_state_dict(torch.load(base + workname + '/' + 'fold0_best_loss.pt'))
            y_list, y_pred_list, td_list, loss_te = utils.get_eval_results(args, model, test_dataloader)
            y_list_f = [item for sublist in y_list for item in sublist]
            y_pred_list_f = [item for sublist in y_pred_list for item in sublist]
            mse_l, mse_h = utils.get_mse_ci(y_list_f, y_pred_list_f, n_bootstraps = 1000, rng_seed = 42)
            mse_dict = {'mse': str(loss_te*225), 'mse_l': str(mse_l), 'mse_h': str(mse_h)}
            with open(base + workname + '/test_mse.json', "w") as outfile:
                json.dump(mse_dict, outfile)
