import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
import const

fig, ax = plt.subplots()

'''
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

fig.tight_layout()
plt.show()

'''
f, ax = plt.subplots(figsize=[28, 21])

df = pd.read_csv(os.path.join(const.DATAPATH, 'data_kenlm_paopao_v2.1.csv'))

sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax,
            cbar_kws={'label': 'Correlation Coefficient'}, cmap='viridis')

ax.set_title("Correlation Matrix for KENLM vs. PAOPAO and New Features ", fontsize=18)

plt.show()
'''
0.8-1.0 极强相关
0.6-0.8 强相关
0.4-0.6 中等程度相关
0.2-0.4 弱相关
0.0-0.2 极弱相关或无相关
'''