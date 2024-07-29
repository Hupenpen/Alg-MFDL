import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    last_fold_path = ''
    last_fold_label_path = ''
    tensors = pd.read_csv(last_fold_path)
    tensors = np.array(tensors)
    tensors_label = pd.read_csv(last_fold_label_path)
    tensors_label = np.array(tensors_label)

    tag = 'All features after Transformer networks'

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(tensors)
    X_train = scaler.transform(tensors)
    X = tensors
    y = tensors_label

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
    color_map = {"Non-allergenic": "#FF6347", "Allergenic": "#4682B4"}

    for category, color in color_map.items():
        subset = df[df['y'] == category]
        ax.scatter(subset['Dimension-1'], subset['Dimension-2'], c=color, label=category,
                   alpha=0.6, edgecolors='white', linewidth=1.0, s=20)

    ax.set_xlabel('Dimension-1')
    ax.set_ylabel('Dimension-2')
    ax.set_title(tag)
    ax.legend()

    fig.savefig('./t_sen/' + tag + '_t-SNE.png', dpi=600)
    plt.show()

    df.to_excel('./t_sen_excel/' + tag + '_t-SNE.xlsx')


if __name__ == '__main__':
    main()
