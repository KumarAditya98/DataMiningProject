#%%
# # Initital Dataset Cleaning and Manipulation 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import rfit
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
# # %%
cwd = os.getcwd()

print(cwd)

dataset = pd.read_csv('Dataset/OnlineNewsPopularity.csv')
# # %%
# Basic checks

rfit.dfchk(dataset)

# 61 total attributes, 39644 rows

# No missing values/ No nulls

# %%
# Info on Dataset Features:
#
# Some of the attributes in the dataset have already been encoded for machine learning. However, we will decode it into a single column for visualization purposes. Such columns include: 
# 1. Data_Channel : Type of article (Entertainment, lifestyle, Media, Technology, World etc.)
# 2. Publish Day : Day the article was pubished (Monday, Tuesday, etc.)

# dataset['Data_Channel'] = np.where(dataset['data_channel_is_lifestyle']==1,'Lifestyle',np.where(dataset['data_channel_is_entertainment']==1,"Entertainment",np.where(dataset['data_channel_is_bus']==1,'Business',np.where(dataset['data_channel_is_socmed']==1,'Social Media',np.where(dataset['data_channel_is_tech']==1,'Technology','World')))))

# For some reason, above code was giving key-error. After further checking, i foudn out that several column titles in the dataset have leading or trailing empty spaces. Needed to fix this
#%%
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
dataset_viz.to_csv('Dataset/OnlineNewsPopularity_Viz.csv', index=False)

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
# Reading the csv file

sharedf = pd.read_csv('Dataset/OnlineNewsPopularity_Viz.csv')
print(sharedf.head())

# %%
print(len(sharedf))

#There are total 39644 rows in the entire dataset

# %%
sharedf=sharedf.drop_duplicates()
print(sharedf.isna().sum())

# Any duplicates values in the data set are removed, and there are no 
#null values the data set.

# %%
sharedf.describe()

#%%
sharedf = sharedf[sharedf['n_tokens_title']!=0]

# %%
sharedf = sharedf[sharedf['n_tokens_content']!=0]

#The n_tokens_title and n_tokens_content columns which contains the 
# value 0 is removed

#%%
print(len(sharedf))
#After removing these 0 values, the length of the dataframe is 38463.
#1180 rows are removed and the dataset is stored.
 
#%%
print(sharedf['shares'].mean())
print(sharedf['shares'].median())

# The mean value of the shares is 3355.56, and the median value of the shares is 1400

# %%
#correlation between the columns
plt.figure(figsize=(15,15))

correlations = sharedf.corr()

print(correlations)

sns.heatmap(correlations, cmap="Blues")

# From the heat map(correlation plot) we can observe that n_non_stop_unique_tokens, n_non_stop_words, kw_avg_min

# has the high correlation

#%%
sharedf = sharedf.drop('url',axis=1)

#%%
#From the collerations we can observe that n_non_stop_words, n_non_stop_unique_tokens, kw_avg_min has high correlations, we are dropping these columns

sharedf= sharedf.drop(["n_non_stop_unique_tokens","n_non_stop_words","kw_avg_min"],axis=1)

# %%
print(sharedf.head())
# %%
plt.figure(figsize=(15,10))

sns.scatterplot( x='n_tokens_content', y='shares', data=sharedf)
# %%
plt.figure(figsize=(15,10))

sns.scatterplot( x='n_tokens_title', y='shares', data=sharedf)

# %%
group_1= pd.DataFrame(sharedf.groupby("Publish_DOW").mean()["shares"])

sns.barplot(x= group_1.index, y="shares", data=group_1)
# %%
group_2= pd.DataFrame(sharedf.groupby("Data_Channel").mean()["shares"])

sns.barplot(x= group_2.index, y="shares", data=group_2)
# %%
fig = plt.subplots(figsize=(10,10))


sns.scatterplot(x='avg_positive_polarity', y='shares', data=sharedf, alpha=0.5)
# %%
fig = plt.subplots(figsize=(10,10))
sns.scatterplot(x='num_imgs', y='shares', data=sharedf)

# %%
#pair plots between all the kw values

plt.figure(figsize=(30,30),dpi=200)

columnskw = ['kw_min_min', 'kw_max_min',  'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg', 'shares']
sns.pairplot(data = sharedf, vars=columnskw, diag_kind="kde")
# %%
plt.figure(figsize=(15,10))
sns.scatterplot(y = "shares", x = "num_imgs", data=sharedf)
plt.title("scatter plot between shares and number of images")

# %%
plt.figure(figsize=(15,10))
sns.scatterplot(y = "shares", x = "num_videos", data=sharedf)
plt.title("scatter plot between shares and number of videos")

# %%
group_3= pd.DataFrame(sharedf.groupby("is_weekend").mean()["shares"])
sns.barplot(x= group_3.index, y="shares", data=group_3)
plt.title("bar plot for shares based on whether day is weekend or not")

# %%
group_4= pd.DataFrame(sharedf.groupby("is_weekend").count()['shares'])
print(group_4)
sns.barplot(x= group_4.index, y="shares", data=group_4)
plt.ylabel("count of the shares for weekend vs weekday")
plt.title("count of the shares between weekend or weekday")

# %%
#model building
#Decision Tree Regression
from sklearn.tree import plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error
X = pd.get_dummies(sharedf.drop('shares',axis=1),drop_first=True)

y = sharedf['shares']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
Scaled_Xtrain = scaler.fit_transform(X_train)
Scaled_Xtest= scaler.transform(X_test)
model = DecisionTreeRegressor(random_state=0, max_depth= 5)
model.fit(Scaled_Xtrain, y_train)
predicted_values = model.predict(Scaled_Xtest)


plt.figure(figsize=(20,15), dpi=200)
plot_tree(model, feature_names=X.columns, filled=True);
#%%
print(f"The mean value of the shares is {sharedf['shares'].mean()}")
features = model.feature_importances_
pd.DataFrame(index=X.columns, data=features, columns=['Feature Importance']).sort_values('Feature Importance', ascending=False)
print(mean_absolute_error(y_test, predicted_values))
print(np.sqrt(mean_squared_error(y_test, predicted_values)))

#%%
pd.set_option('display.float_format', lambda x: '%.3f' % x)

model_2 = DecisionTreeRegressor(random_state=0, max_depth=3)
model_2.fit(Scaled_Xtrain, y_train)
predicted_values1 = model_2.predict(Scaled_Xtest)
features1 = model_2.feature_importances_
pd.DataFrame(index=X.columns, data= features1, columns=['Feature Importance']).sort_values('Feature Importance', ascending=False)
print(mean_absolute_error(y_test, predicted_values1))
print(np.sqrt(mean_squared_error(y_test, predicted_values1)))
plot_tree(model_2, feature_names= X.columns, filled=True)
#%%
X_test.head()
# %%
#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(Scaled_Xtrain, y_train)
predictons = model_lr.predict(Scaled_Xtest)
#%%
print(mean_absolute_error(y_test, predictons))
print(np.sqrt(mean_squared_error(y_test, predictons)))
print(min(sharedf['shares']))

# %%
print(sharedf.columns)
# %%


# %%
# %%
