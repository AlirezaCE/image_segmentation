#%% package
import torch 
import numpy as np
import seaborn as sns 

#%%
def y_function(val):
    return (val-3)*(val-6)*(val-4)

x_range = np.linspace(0,10,101)
print (x_range)
y_range = [y_function(i) for i in x_range]
sns.lineplot(x=x_range, y = y_range)
# %%
