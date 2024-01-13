import pandas as pd
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1D ALE
def get_centres(x):
    """Return bin centres from bin edges.
    Parameters
    ----------
    x : array-like
        The first axis of `x` will be averaged.
    Returns
    -------
    centres : array-like
        The centres of `x`, the shape of which is (N - 1, ...) for
        `x` with shape (N, ...).
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3])
    >>> _get_centres(x)
    array([0.5, 1.5, 2.5])
    """
    return (x[1:] + x[:-1]) / 2

def data_piece(train_set_rep, index, record_flag):
    train_set_piece = []
    all_features = []
    # when >=164# something here needs to be fixed
    # INDEX 162, ph urine mean, last numerical
    offset = 1
    # if index<=162 else 0
    for record in train_set_rep:
        true_record = np.where(record[index+offset, :] == record_flag)[0]
        if len(true_record) >= 1:
            for rec_ind in true_record:
                train_set_piece.append(record[:, :rec_ind+1])
                all_features.append(record[index, :rec_ind+1])
    return train_set_piece, np.concatenate(all_features)

def get_1d_ale(args, model, test_head, index, bins, monte_carlo_ratio, monte_carlo_rep, record_flag):

    mc_replicates = np.asarray([
                        [
                            np.random.choice(range(len(test_head)))
                            for _ in range(int(monte_carlo_ratio * len(test_head)))
                        ]
                        for _ in range(monte_carlo_rep)
                    ])

    ale_nc = []
    quantile_nc = []
    ale_t = []
    quantile_t = []
    for k, rep in enumerate(mc_replicates):
        train_set_rep = [test_head[i] for i in rep]
        # [ndarray with shape (200, 1), ndarray(200, 2), ...] and ndarray with shape (1204789,)
        train_set_piece, lin_feature = data_piece(train_set_rep, index, record_flag=record_flag)

        # ndarray with shape (10,), for index 0, it's array([-24.88816841,  -1.39602958,  -1.08280106,  -0.76957254,
            # -0.45634402,  -0.1431155 ,   0.17011301,   0.48334153,
            #  0.79657005,   1.10979857])
        quantiles = np.unique(
            np.quantile(
                lin_feature, np.linspace(0, 1, bins + 1), method="lower"
            )
        )
        bins = len(quantiles) - 1
        piece_shape = [train_set_piece[i].shape[1] for i in range(len(train_set_piece))]
        piece_hist, _ = np.histogram(piece_shape, bins = range(0, 219))

        len_dict = {}
        # key: lenth, value: a list of pieces
        for k, j in enumerate(piece_shape):
            if j in len_dict.keys():
                len_dict[j].append(train_set_piece[k])
            else:
                len_dict[j] = [train_set_piece[k]]

        piece_3d = []
        for i in len_dict.keys():
            # [ndarray with shape (231, 200, 1), ndarray with shape (260, 200, 2),...]
            piece_3d.append(np.stack(len_dict[i]))

        model.eval()
        effects_t = []
        indices_t = []

        for piece in piece_3d:
        # [(200, 5), (200, 6), (200, 7), (200, 5)]
        # train_set_piece could be 20000 pieces, given there are 100 repetitions
        # gdigitalize using [(6, 200, 1), (10, 200, 5) ...] piece_3d, last dim is lenth
            indices = np.clip(
                    np.digitize(piece[:, index, -1], quantiles, right=True) - 1, 0, None
                )
            predictions = []
            for offset in range(2):
                mod_train_set = piece.copy()
                mod_train_set[:, index, -1] = quantiles[indices + offset]
                predictions.append(model(torch.FloatTensor(mod_train_set).to(device)))  # (6, 1, 1) or (6, 5, 1) depending on the length
            # The individual effects.
            # (139, 60, 1) diffrent indices  (139) # (bs, 2)
            effects = np.subtract(predictions[1][:, -1, :].cpu().detach().numpy(), predictions[0][:, -1, :].cpu().detach().numpy())
        # [ndarray with shape (231, 2), ndarray with shape (260, 2)...]
        if args.task == 'static':
            effects_t.append(effects[:, 0])
        else: 
            effects_t.append(effects)
        # [ndarray with shape (231,), ndarray with shape (260, ), ...]
        indices_t.append(indices)

        if args.task == 'sofa':
            index_groupby = pd.DataFrame({"index": np.concatenate(indices_t, axis=0), \
                            "effects": np.concatenate(effects_t, axis=0).squeeze(-1)}).groupby("index")
        else: 
            index_groupby = pd.DataFrame({"index": np.concatenate(indices_t, axis=0), \
                            "effects": np.concatenate(effects_t, axis=0)}).groupby("index")          
        mean_effects = index_groupby.mean().to_numpy().flatten()
        ale = np.array([0, *np.cumsum(mean_effects)])
        ale_nc.append(ale)
        quantile_nc.append(quantiles)
        # The uncentred mean main effects at the bin centres.
        ale = get_centres(ale)

        # Centre the effects by subtracting the mean (the mean of the individual
        # `effects`, which is equivalently calculated using `mean_effects` and the number
        # of samples in each bin).
        ale -= np.sum(ale * index_groupby.size() / len(train_set_piece))
        ale_t.append(ale)
        quantile_t.append(get_centres(quantiles))
    return  quantile_t, ale_t, quantile_nc, ale_nc