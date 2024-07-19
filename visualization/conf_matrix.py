import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


conf_matrix = np.array([[707, 3], [35, 675]])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=True, ax=ax, annot_kws={"size": 30})

ax.set_xlabel('Predicted label', fontsize=20)
ax.set_ylabel('True label', fontsize=20)

ax.set_xticklabels(['Non-allergenic', 'Allergenic'], fontsize=16)
ax.set_yticklabels(['Non-allergenic', 'Allergenic'], fontsize=16)
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
