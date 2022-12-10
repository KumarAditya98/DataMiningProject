# %%
# # Importing Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mlp
import rfit
import os


# Reading DataSet
#%%
OnlineNewsdf = pd.read_csv('Dataset/OnlineNewsPopularity_Viz.csv')
print(OnlineNewsdf.head())


# %%
print(len(OnlineNewsdf))
#There are total 39644 rows in the entire dataset


# %%
OnlineNewsdf=OnlineNewsdf.drop_duplicates()
print(OnlineNewsdf.isna().sum())

# Any duplicates values in the data set are removed, and there are no null values the data set.


# Describing the DataSet
# %%
OnlineNewsdf.describe()


#The n_tokens_content columns which contains the 
# value 0 is removed
# %%
OnlineNewsdf = OnlineNewsdf[OnlineNewsdf['n_tokens_content']!=0]


# Since URL is a non-numeric attribute and will not add value to our analysis so dropping it from the dataset
# Also timedelta is a non-predictive attribute and not a feature of the data set so we can drop it from the dataset
# We observe multicollinearity variables "n_non_stop_unique_tokens","n_non_stop_words" and "kw_avg_min", hence dropping these variables.
# %%
OnlineNewsdf = OnlineNewsdf.drop('url',axis=1)
OnlineNewsdf = OnlineNewsdf.drop('timedelta',axis=1)
OnlineNewsdf= OnlineNewsdf.drop(["n_non_stop_unique_tokens","n_non_stop_words","kw_avg_min"],axis=1)

# %%
OnlineNewsdf.head()


# # Correlation Heatmap
# %%
plt.figure(figsize=(42, 15))
heatmap = sns.heatmap(OnlineNewsdf.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);



#Dropping Unnamed Index column, as it dosen't make sense.
# %%
OnlineNewsdf = OnlineNewsdf.reset_index(drop=True)
OnlineNewsdf.head()

# creating a grading criteria for the shares
# %%
share_data = OnlineNewsdf['shares']
OnlineNewsdf['shares'].describe() 

