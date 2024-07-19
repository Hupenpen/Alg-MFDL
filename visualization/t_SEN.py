import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

def load_data_csv(pos_path, neg_path):
    pos = pd.read_csv(pos_path)
    pos = np.array(pos)
    pos = np.insert(pos, 0, values=[1 for _ in range(pos.shape[0])], axis=1)
    print("pos:", pos.shape)
    neg = pd.read_csv(neg_path)
    neg = np.array(neg)
    neg = np.insert(neg, 0, values=[0 for _ in range(neg.shape[0])], axis=1)
    print("neg:", neg.shape)
    data = np.row_stack((pos, neg))
    print("data:", data.shape)
    data_Y, data_X = data[:, 0], data[:, 1:]
    print("标签:", data_Y.shape)  # 标签
    print("特征:", data_X.shape)  # 特征
    return data_Y, data_X

def load_data_pkl(pos_path, neg_path,p=1,n=0):
    with open(pos_path, "rb") as tf:
        feature_dict = pickle.load(tf)
    pos = np.array([item for item in feature_dict.values()])
    pos = np.insert(pos, 0, values=[p for _ in range(pos.shape[0])], axis=1)
    print("pos:", pos.shape)
    with open(neg_path, "rb") as tf:
        feature_dict = pickle.load(tf)
    neg = np.array([item for item in feature_dict.values()])
    neg = np.insert(neg, 0, values=[n for _ in range(neg.shape[0])], axis=1)
    print("neg:", neg.shape)
    data = np.row_stack((pos, neg))
    print("data", data.shape)
    data_Y, data_X = data[:, 0], data[:, 1:]
    print("标签:", data_Y.shape)  # 标签
    print("特征:", data_X.shape)  # 特征
    return data_Y, data_X


def main():
    X_train_accpssm_dde = ''
    X_test_accpssm_dde = ''
    y_train_accpssm_dde = ''
    y_test_accpssm_dde = ''

    X_train = X_train_accpssm_dde
    X_test = X_test_accpssm_dde
    y_train = y_train_accpssm_dde
    y_test = y_test_accpssm_dde

    tag = 'AAC-PSSM+DDE'

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  # normalize X to 0-1 range
    X_test = scaler.transform(X_test)
    # concatenate the dataset
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    from sklearn.manifold import TSNE
    import pandas as pd
    tsne = TSNE(n_components=2, verbose=0, perplexity=25, learning_rate='auto', n_iter=5000, random_state=123)
    z = tsne.fit_transform(X)
    df = pd.DataFrame()
    df["Dimension-1"] = z[:, 0]
    df["Dimension-2"] = z[:, 1]
    y_new_label = []
    for i in y:
        if i == 0:
            y_new_label.append('Non-allergenic')
        if i == 1:
            y_new_label.append('Allergenic')
    df["y"] = y_new_label

    fig, ax = plt.subplots()
    color_map = {"Non-allergenic": "#FF6347", "Allergenic": "#4682B4"}  # 定义颜色映射

    for category, color in color_map.items():
        subset = df[df['y'] == category]
        ax.scatter(subset['Dimension-1'], subset['Dimension-2'], c=color, label=category,
                   alpha=0.6, edgecolors='white', linewidth=1.0, s=20)

    ax.set_xlabel('Dimension-1')
    ax.set_ylabel('Dimension-2')
    ax.set_title('2D t-SNE visualization of ' + tag)
    ax.legend()

    fig.savefig('', dpi=600)
    plt.show()

    df.to_excel('' + tag + '_t-SNE.xlsx')
if __name__ == '__main__':
    main()