import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# set font size of plots
plt.rcParams.update({'font.size': 18})

df = pd.read_csv('output/tuning.csv', index_col=0)

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True, vmin=-1, vmax=1, center=0)
plt.show()
