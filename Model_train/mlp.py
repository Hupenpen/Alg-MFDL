import csv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                             recall_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc)
import pandas as pd
import numpy as np
from Train_util import load_data_pkl,load_data_csv,get_CM,normalize_numpyarray,normalize_dataframe
from Model import CustomDataset


class MLP(nn.Module):
    def __init__(self, input_features):
        super(MLP, self).__init__()
        # 定义网络结构
        self.mlp = nn.Sequential(
            nn.Linear(input_features, 64),  # 第一个隐藏层，64个神经元
            nn.ReLU(),                      # 激活函数
            nn.Dropout(0.2),                # Dropout层，防止过拟合
            nn.Linear(64, 32),              # 第二个隐藏层，32个神经元
            nn.ReLU(),                      # 激活函数
            nn.Dropout(0.2),                # Dropout层，防止过拟合
            nn.Linear(32, 1),               # 输出层，1个神经元，用于二分类
            nn.Sigmoid()                   # Sigmoid激活函数，输出概率
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 确保输入是平展的（flattened）
        output = self.mlp(x)       # 通过MLP
        return output.squeeze()




def main():
    y, t5 = load_data_pkl('',
                          '')
    _, esm = load_data_csv('',
                           '')
    _, accpssm = load_data_csv('',
                               '')
    _, dde = load_data_csv('',
                           '')

    accpssm = normalize_numpyarray(accpssm)
    dde = normalize_numpyarray(dde)
    accpssm_dde = np.hstack((accpssm, dde))

    tag = ''

    model_path = ''
    train_roc_path = ''
    train_pr_path = ''

    aacpssmdde = pd.DataFrame(accpssm_dde)
    y, t5 = pd.DataFrame(y), pd.DataFrame(t5)
    esm = pd.DataFrame(esm)
    t5 = normalize_dataframe(t5)
    esm = normalize_dataframe(esm)

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
    # all_precision = 0
    num_comp_time = 0


    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')

    tprs = []
    fprs = []
    precisions = []
    recalls = []
    best_acc = 0
    best_acc_model = None
    mean_fpr_linspace = np.linspace(0, 1, 100)
    i = 0

    for fold, (train_index, val_index) in enumerate(kf.split(train)):
        n = 0
        print(f"Fold: {fold + 1}")
        i += 1

        Train_t5 = t5.iloc[train_index]
        Train_accpssmdde = aacpssmdde.iloc[train_index]

        Train_esm = esm.iloc[train_index]
        Train_y = y.iloc[train_index]

        val_t5 = t5.iloc[val_index]
        val_accpssmdde = aacpssmdde.iloc[val_index]

        val_esm = esm.iloc[val_index]
        val_y = y.iloc[val_index]

        train_dataset = CustomDataset(Train_t5, Train_esm, Train_accpssmdde, Train_y)
        val_dataset = CustomDataset(val_t5, val_esm, val_accpssmdde, val_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        Model = MLP()
        Model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(Model.parameters(), lr=0.0001)

        final_output_list = []

        for n in range(num_t):
            for epoch in range(num_epochs):
                Model.train()
                for data, labels in train_loader:
                    data = data.to(device).unsqueeze(1)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    final_output = Model(data)
                    loss = criterion(final_output, labels.squeeze(1))
                    loss.backward()
                    optimizer.step()

                all_predictions = []
                all_labels = []
                all_auc = []

                Model.eval()
                for data, labels in val_loader:
                    data = data.to(device).unsqueeze(1)
                    optimizer.zero_grad()
                    optimizer.zero_grad()
                    final_output = Model(data)
                    scores = final_output.tolist()
                    all_auc.extend(scores)

                    final_output = (final_output.data > 0.5).int()

                    all_labels.extend(labels.tolist())
                    all_predictions.extend(final_output.tolist())

                acc, sn, sp, mcc, pr, f1 = get_CM(all_labels, all_predictions)
                val_accuracy = acc
                val_precision = pr

                val_roc = roc_auc_score(all_labels, all_auc)
                val_recall = recall_score(all_labels, all_predictions)
                val_f1 = f1
                precision, recall, _ = precision_recall_curve(all_labels, all_auc)
                fpr, tpr, _ = roc_curve(all_labels, all_auc)
                val_pr_auc = auc(recall, precision)
                num_samples = 100
                precision_sampled = np.linspace(0, 1, num_samples)
                recall_sampled = np.interp(precision_sampled, precision, recall)
                fpr_sampled = np.linspace(0, 1, num_samples)
                tpr_sampled = np.interp(fpr_sampled, fpr, tpr)

                fprs.append(fpr_sampled)
                tprs.append(tpr_sampled)
                precisions.append(precision_sampled)
                recalls.append(recall_sampled)

                all_acc += acc
                all_sp += sp
                all_sn += sn
                all_mcc += mcc
                all_f1 += f1
                # all_recall += recall
                all_val_roc += val_roc
                all_val_prauc += val_pr_auc
                num_comp_time += 1

                # 最佳模型选择
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    best_acc_model = Model.state_dict().copy()
                print(f"acc:{val_accuracy:.4f} pr:{val_precision:.4f} "
                      f"recall{val_recall:.4f} f1:{val_f1:.4f} roc:{val_roc:.4f} "
                      f"auc:{val_pr_auc:.4f} "
                      f"mcc:{mcc:.4f}sn:{sn:.4f}sp:{sp:.4f}")

                '''save acc'''

                file1 = open('', 'a', newline='')
                content1 = csv.writer(file1, dialect='excel')
                label = tag + ': ' + str(i) + 'k'
                content1.writerow([label, acc, sn, sp, mcc, f1, val_recall, val_roc, val_pr_auc])

    file1 = open('', 'a', newline='')
    content1 = csv.writer(file1, dialect='excel')
    label = tag
    content1.writerow([label, all_acc / num_comp_time, all_sn / num_comp_time,
                       all_sp / num_comp_time, all_mcc / num_comp_time, all_f1 / num_comp_time,
                       all_val_roc / num_comp_time,
                       all_val_prauc / num_comp_time])

    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)
    mean_fpr = np.mean(fprs, axis=0)
    mean_tpr = np.mean(tprs, axis=0)

    val_pr_curve_data = pd.DataFrame({'Precision': mean_precision, 'Recall': mean_recall})
    val_pr_curve_data.to_csv(train_roc_path, index=False)

    val_roc_curve_data = pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr})
    val_roc_curve_data.to_csv(train_pr_path, index=False)

    torch.save(best_acc_model, model_path)

if __name__ == '__main__':
    main()



