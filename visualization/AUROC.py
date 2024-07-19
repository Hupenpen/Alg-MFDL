
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

folder_path = '../path'

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))


plt.figure(figsize=(10, 8))
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], 'k--')

for csv_file in csv_files:

    data = pd.read_csv(csv_file)

    filename = os.path.splitext(os.path.basename(csv_file))[0]

    if 'FPR' in data.columns and 'TPR' in data.columns:

        plt.plot(data['FPR'], data['TPR'], label=f'{filename}')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

handles, labels = plt.gca().get_legend_handles_labels()

labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
plt.legend(handles, labels, loc='best', fontsize='small')


plt.savefig(os.path.join('./rocpr/all_Feature_ROC.png'), dpi=600)

plt.show()


