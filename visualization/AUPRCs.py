import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np

directory = ('')
file_list = glob.glob(os.path.join(directory, '*.csv'))
plt.figure(figsize=(10,8 ))
plt.title('PR Curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0, 1])
plt.ylim([0, 1])

for file in file_list:
    df = pd.read_csv(file)
    recall = df['Recall']
    precision = df['Precision']

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    filename = os.path.basename(file).replace('.csv', '')

    plt.plot(recall, precision, label=f' {filename}')
    plt.plot([1, 0], [0, 1], linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

handles, labels = plt.gca().get_legend_handles_labels()

labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
plt.legend(handles, labels, loc='best', fontsize='small')


plt.savefig(os.path.join(''), dpi=600)

plt.show()
