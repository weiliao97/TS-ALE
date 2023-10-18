import os 

def creat_checkpoint_folder(target_path, target_file, data):
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