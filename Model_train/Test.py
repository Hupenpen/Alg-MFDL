import csv
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc)
import pandas as pd
import numpy as np
from Train_util import load_data_pkl,load_data_csv,get_CM,normalize_numpyarray
from Model import CNN_Model,BiGRU_Model,AttTransform_Model,CustomDataset


def main(model_type,feature_type,input_channel):
    y, t5 = load_data_pkl('path_p_prottrans',
                          'path_n_prottrans')
    y, esm = load_data_csv('path_p_esm2',
                           'path_n_esm2')
    y, accpssm = load_data_csv('path_p_accpssm',
                               'path_n_accpssm')
    _, dde = load_data_csv('path_dde',
                           'path_dde')


    accpssm = normalize_numpyarray(accpssm)
    dde = normalize_numpyarray(dde)
    accpssm_dde = np.hstack((accpssm, dde))
    esm_accpssm_dde = np.hstack((esm, accpssm_dde))
    t5_esm = np.hstack((t5,esm))
    t5_accpssm_dde = np.hstack((t5,accpssm_dde))
    t5_esm_accpssm_dde = np.hstack((t5, esm_accpssm_dde))
    all_feature = t5_esm_accpssm_dde

    tag = ''
    model_path = ''
    test_roc_path = ''
    test_pr_path =''

    y, t5 = pd.DataFrame(y), pd.DataFrame(t5)
    esm = pd.DataFrame(esm)

    train = pd.read_csv('')

    print(f't5.shape:{t5.shape}')
    print(f'y.shape:{y.shape}')
    print(f'accpssm_dde.shape:{accpssm_dde.shape}')
    print(f'esm.shape:{esm.shape}')

    num_epochs = 100
    batch_size = 64
    num_folds = 5
    num_t = 1
    all_acc = 0
    all_sn = 0
    all_sp = 0
    all_mcc = 0
    all_f1 = 0
    all_recall = 0
    all_val_roc = 0
    all_val_prauc = 0
    num_comp_time = 0

    features = {
        "t5": t5,
        "esm2": esm,
        "handcraft": accpssm_dde,
        "t5_esm": t5_esm,
        "esm2_handcraft": esm_accpssm_dde,
        "t5_handcraft": t5_accpssm_dde,
        "all_feature": all_feature
    }
    Test_feature = features[feature_type]
    Test_y= y
    test_dataset = CustomDataset(Test_feature, Test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    models = {
        "CNN": CNN_Model,
        "Transformer": AttTransform_Model,
        "BiGRU": BiGRU_Model
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')
    model = models[model_type](input_channel)
    model.load_state_dict(torch.load(model_path))

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.eval()
    model.to(device)
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        all_auc = []

        for data, labels in test_loader:
            data = data.to(device).unsqueeze(1)
            final_output = model(data)

            scores = final_output.tolist()
            all_auc.extend(scores)

            final_output = (final_output.data > 0.5).int()
            all_labels.extend(labels.tolist())
            all_predictions.extend(final_output.tolist())

        acc, sn, sp, mcc, pr, f1 = get_CM(all_labels, all_predictions)

        test_accuracy = acc
        test_precision = pr
        test_auc_roc = roc_auc_score(all_labels, all_auc)
        test_recall = recall_score(all_labels, all_predictions)
        test_f1 = f1
        precision, recall, _ = precision_recall_curve(all_labels, all_auc)
        fpr, tpr, _ = roc_curve(all_labels, all_auc)
        test_pr_auc = auc(recall, precision)
        test_roc_curve_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        test_roc_curve_data.to_csv(test_roc_path, index=False)
        test_pr_curve_data = pd.DataFrame({'Precision': precision, 'Recall': recall})
        test_pr_curve_data.to_csv(test_pr_path, index=False)

        print(
            f"acc:{test_accuracy:.4f} pr:{test_precision:.4f} recall{test_recall:.4f} "
            f"f1:{test_f1:.4f} roc:{test_auc_roc:.4f} auc:{test_pr_auc:.4f} "
            f"mcc:{mcc:.4f}sn:{sn:.4f}sp:{sp:.4f}")
        '''save acc'''

        file1 = open('', 'a', newline='')
        content1 = csv.writer(file1, dialect='excel')
        label = tag + ': test'
        content1.writerow([label, acc, sn, sp, mcc, f1, test_recall, test_auc_roc, test_pr_auc])