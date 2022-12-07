# %%
# # Initital Dataset Cleaning and Manipulation 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import rfit
import os

# %%
# # %%
cwd = os.getcwd()

print(cwd)

dataset = pd.read_csv('Dataset/OnlineNewsPopularity.csv')
# # %%
# Basic checks

rfit.dfchk(dataset)

# 61 total attributes, 39644 rows

# No missing values/ No nulls

# First identifying what columns have these extra spaces
bad_columns = [x for x in dataset.columns if x.endswith(' ') or x.startswith(' ')]
print('Number of Columns with unwanted spaces: ',len(bad_columns))

# Almost all columns have this problem, so we'll fix this
dataset.columns = dataset.columns.str.strip()

# Checking to see if the problem is resolved
bad_columns_validation = [x for x in dataset.columns if x.endswith(' ') or x.startswith(' ')]
print('\nAfter Fix:\nNumber of Columns still with issue: ',len(bad_columns_validation))

#%%
# Running the code again

dataset_viz = dataset.copy()

dataset_viz['Data_Channel'] = np.where(dataset_viz['data_channel_is_lifestyle']==1,'Lifestyle',np.where(dataset_viz['data_channel_is_entertainment']==1,"Entertainment",np.where(dataset_viz['data_channel_is_bus']==1,'Business',np.where(dataset_viz['data_channel_is_socmed']==1,'Social Media',np.where(dataset_viz['data_channel_is_tech']==1,'Technology','World')))))
# dataset.head()

#%%
# Now doing the same thing for Day of the Week
dataset_viz['Publish_DOW'] = np.where(dataset_viz['weekday_is_monday']==1,'Monday',np.where(dataset_viz['weekday_is_tuesday']==1,"Tuesday",np.where(dataset_viz['weekday_is_wednesday']==1,'Wednesday',np.where(dataset_viz['weekday_is_thursday']==1,'Thursday',np.where(dataset_viz['weekday_is_friday']==1,'Friday',np.where(dataset_viz['weekday_is_saturday'],'Saturday','Sunday'))))))
# dataset.head()

#%%
# We can go ahead and remove the columns that have been utilized
dataset_viz = dataset_viz.drop(['weekday_is_saturday','weekday_is_friday','weekday_is_sunday','weekday_is_thursday','weekday_is_wednesday','weekday_is_tuesday','weekday_is_monday','data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world'],axis=1)
# dataset_viz.shape

#%%
# Saving out this dataset for collaboration
dataset_viz.to_csv('Dataset\\OnlineNewsPopularity_Viz.csv')

# We're going to use dataset_viz for visualizations and dataset for modeling

#%%
# Some of the features are dependent of particularities of the Mashable service (whose articles have been used as data source): articles often reference other articles published in the same service; and articles have meta-data, such as keywords, data channel type and total number of shares (when considering Facebook, Twitter, Google+, LinkedIn, Stumble-Upon and Pinterest). The minimum, average and maximum number of shares was determined of all Mashable links cited in the article were extracted to prepare the data. Similarly, rank of all article keyword average shares was determined, in order to get the worst, average and best keywords. For each of these keywords, the minimum, average and maximum number of shares was extracted as a feature. [Reference: Research Paper]

# Several features are extracted by performing natural language processing on the original articles. The Latent Dirichlet Allocation (LDA) algorithm was applied to all Mashable articles in order to first identify the five top relevant topics and then measure the closeness of current article to such topics. To compute the subjectivity and polarity sentiment analysis, Pattern web mining module was adopted, allowing the computation of sentiment polarity and subjectivity scores. These are such features:
# 1. Closeness to top 5 LDA topics 
# 2. Title subjectivity ratio 
# 3. Article text subjectivity score and its absolute difference to 0.5 
# 4. Title sentiment polarity 
# 5. Rate of positive and negative words 
# 6. Pos. words rate among non-neutral words 
# 7. Neg. words rate among non-neutral words 
# 8. Polarity of positive words (min./avg./max.) 
# 9. Polarity of negative words (min./avg./max.) 
# 10. Article text polarity score and its absolute difference to 0.5
# [Reference: Research Paper]

# Even though we don't yet understand what these variables represent exactly, we will keep them for the purpose of model building.

#%%
OnlineNewsdf = pd.read_csv('Dataset/OnlineNewsPopularity_Viz.csv')
print(OnlineNewsdf.head())


# %%
print(len(OnlineNewsdf))

#There are total 39644 rows in the entire dataset

# %%
OnlineNewsdf=OnlineNewsdf.drop_duplicates()
print(OnlineNewsdf.isna().sum())

# Any duplicates values in the data set are removed, and there are no 
#null values the data set.

# %%
OnlineNewsdf.describe()


# %%
OnlineNewsdf = OnlineNewsdf[OnlineNewsdf['n_tokens_content']!=0]

#The n_tokens_content columns which contains the 
# value 0 is removed
