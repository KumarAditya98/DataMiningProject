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



# Segregating Shares into different Categorical Factors.
# Exceptional - Greater than 75000 shares
# Excellent   - Between 50k to 75K shares
# Good        - Between 20k to 50k
# Average     - Between 7k to 20k
# Poor        - Less than 7k
# %%
share_label = list()
for share in share_data:
    if share <= 7000:
        share_label.append('Poor')
    elif share > 7000 and share <= 20000:
        share_label.append('Average')
    elif share > 20000 and share <= 50000:
        share_label.append('Good')
    elif share > 50000 and share <= 75000:
        share_label.append('Excellent')
    else:
        share_label.append('Exceptional')

# Update this class label into the dataframe
OnlineNewsdf = pd.concat([OnlineNewsdf, pd.DataFrame(share_label, columns=['popularity'])], axis=1)
OnlineNewsdf.head(4)




# Plots 

# shares vs n_tokens_title
# %%
sns.set_theme(style="ticks")
palette = sns.color_palette("rocket_r")
sns.relplot(
    data = OnlineNewsdf,
    x = "n_tokens_title", y = "shares",
    hue = "popularity", kind = "line", palette = palette,
    height = 5, aspect = .75, facet_kws = dict(sharex = False),
)

 
 
# %%
sns.set_style(style='whitegrid')
sns.scatterplot(
    data=OnlineNewsdf, 
    x='n_tokens_content', 
    y='shares', 
    hue='popularity',
    palette='Paired_r'
    )
plt.title('Analysing Popularity based on Shares')
plt.xlabel('No of Tokens Content')
plt.ylabel('Shares')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()


# %%
sns.countplot(x ='popularity', hue = "Data_Channel", data = OnlineNewsdf)
plt.show()


# %%
sns.countplot(x ='popularity', data = OnlineNewsdf)
plt.show()


#%%
df = OnlineNewsdf[OnlineNewsdf['popularity'] == "Exceptional"]
len(df)


# %%
sns.countplot( x= "Publish_DOW", hue="Data_Channel", data=OnlineNewsdf)
plt.show()

# %%
#df1 = OnlineNewsdf[OnlineNewsdf['shares'] > 1400]
#len(df1)


# %%
sns.set_theme(style="whitegrid")
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
g = sns.relplot(
    data=OnlineNewsdf,
    x="num_hrefs", y="shares",
    hue="Data_Channel",
    sizes=(10, 200),
)
g.set(xscale="log", yscale="log")
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)



# %%
sns.set_theme(style="white") 
sns.relplot(x="num_imgs", y="shares", hue="Publish_DOW", size="Data_Channel",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data = OnlineNewsdf)


# %%
sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 5))
sns.despine(f)
sns.histplot(
    OnlineNewsdf,
    x="shares", hue="title_subjectivity",
    multiple="stack",
    palette="light:m_r",
    edgecolor=".3",
    linewidth=.5,
    log_scale=True,
)
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks([500, 1000, 2000, 5000, 10000])