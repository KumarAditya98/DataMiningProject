#%%
import pandas as pd
import numpy as np
import rfit
import os
import matplotlib.pyplot as plt
import seaborn as sns
# %%
dfviz = pd.read_csv('Dataset/OnlineNewsPopularity_Viz.csv')
#%%
dfviz.head()
rfit.dfchk(dfviz)
#%%
# An interesting point was raised by Prof Edwin Lo in his feeddback on the project proposal. Is the number of shares a moving target? Is it a function of the number of days that has passed since when an article was published that determines the number of shares it receives. 

# After reading through a couple of research papers on this topic, we were able to understand that the two are correlated only for the initial period. One such paper suggested that there is a convergence of shares reached and the number of days it took that to happen was 21.

# Interestingly, the dataset that we've obtained consists of articles published between 2015 to 2017, and the most recent articles were discarded from the dataset. Most recent being any article published within the last 21 days of data acquisition. 

# we have a feature in the dataset which is known as timedelta which gives us the No. of days between the article was published and this dataset was collected. To reinforce the above assumption, it was important to see whether there was any relation between the timedelta and the number of shares the article recieved within our dataset. We'll plot this using a scatter plot.

#%%
plt.scatter(x='timedelta',y='shares',data=dfviz,alpha=0.5)
plt.xlabel("Time delta (No. of Days)")
plt.ylabel("Number of Shares")
plt.show()

#%%
# There are many outliers in the dataset, so i will subset it to get a better view
dfviz1 = dfviz[dfviz['shares']<200000]
plt.scatter(x='timedelta',y='shares',data=dfviz1,alpha=0.5)
plt.xlabel("Time delta (No. of Days)")
plt.ylabel("Number of Shares")
plt.show()

#%%
# Subsetting it further
dfviz2 = dfviz[dfviz['shares']<25000]
plt.scatter(x='timedelta',y='shares',data=dfviz2,alpha=0.1)
plt.xlabel("Time delta (No. of Days)")
plt.ylabel("Number of Shares")
plt.show()

#%%
# and further..
dfviz3 = dfviz[dfviz['shares']<5000]
plt.scatter(x='timedelta',y='shares',data=dfviz3,alpha=0.1)
plt.xlabel("Time delta (No. of Days)")
plt.ylabel("Number of Shares")
plt.show()

#%%
print("The number of articles within 0 - 5000 shares: ",len(dfviz3))
print("The number of articles within 0 - 25000 shares: ",len(dfviz2))
print("The number of articles within 0 - 200000 shares: ",len(dfviz1))
print("The number of articles in the dataset: ",len(dfviz))
# Two Observations : 
# 1. It seems like there is no concrete relationship between when the article was published and the number of shares it receives if data is acquired after 3 weeks, that is, there appears to be some convergence in the number of shares an article receives a definite number of days (21) after it has been published. 
# 2. In the dataset at hand, majority of the (~87%) of the articles have shares in the range of 0-5000. Whereas the maximum number of shares received are as high as 843300.

#%%
# Update: Our dataset does contain <21 timedelta. Therefore subsetting it.

dfviz = dfviz[dfviz['timedelta']>21]

#%%
# Another clarity that we received from Prof Edwin Lo was to either focus on a classification problem or regression problem as it was redundant to do both using the same response variable. 

# So we have decided to convert our scope into only a classification problem. We'll do this by first calculating the mean of number of shares in our dataset and then engineering a response variable. If share <= mean then 'unpopular' :0. Else if share > mean then 'popular':1

#%%
# Commented out these codes as it was a one time operation
# df = pd.read_csv('Dataset/OnlineNewsPopularity.csv')
# threshold = df.shares.mean()
# df['target'] = np.where(df.shares>threshold,1,0)
# df.head(20)
# Checking to see the proportion of 0's and 1's
# print(df.target.value_counts())

# This is highly disproportionate, instead we will choose a lower threshold. Therefore, instead of calculating the mean, we will use the median value as threshold to get better proportion of 0's and 1's.

# threshold = df.shares.median()
# df['target'] = np.where(df.shares>threshold,1,0)
# print(df.target.value_counts())

# Now we have a better distribution to train our model, almost 50-50

# Will be saving back this file as the dataset. Doing the sae for dataset_viz file.
# df.to_csv('Dataset/OnlineNewsPopularity.csv',index=False)
# For convenience sake, I have just rerun my initial dataset file to get the target variable in the actual dataset as well.

#%%
# Now onto more EDA!!
dfviz.head()
# While looking at the dataset, I had a few curious questions. I will help answer them with EDA.

# Q1. What is the effect of number of images/number of videos on an articles popularity (hence the number of shares it receives)?
# Answer

#%%
# First seeing the ditribution of num_videos
sns.set(style="white", palette="muted")
sns.distplot(dfviz['num_videos'], color="b")
plt.show()

# It seems like majority of the num_videos is dsitributed among 0 and 1. Looking at their value_counts

print(dfviz.num_videos.isin([0,1]).value_counts())
# This goes to show that out of 39644 rows, 34509 rows have num_video as 0 or 1 and the rest have only 5135 count. This column is highly skewed and does not follow a normal distribution.

#%%
# However, lets see whether between having 0 videos or 1 video, which has a better impact on number of shares.

# Since shares value have many outliers and the mean is <5000 , I'm using the dataframe above that was subset to having shares < 25000
dfviz0 = dfviz2[dfviz2['num_videos']==0]
dfviz1 = dfviz2[dfviz2['num_videos']==1]

sns.set(style="whitegrid")
f, axes = plt.subplots(1, 2, figsize=(5, 5), sharey=True)
sns.despine(left=True)

sns.boxplot(data=dfviz0,y='shares',ax=axes[0])
axes[0].set(xlabel='For num_videos = 0')

sns.boxplot(data=dfviz1,y='shares',ax=axes[1])
axes[1].set(xlabel='For num_videos = 1')

plt.show()

# From the above plot we see that the average of num_video = 0 and 1 is about the same. We can assume that having greater number of videos does not have a direct effect on number of shares.

# Repeating the exercise for number of images

#%%
# Looking at the distribution first
sns.set(style="white", palette="muted")
sns.distplot(dfviz['num_imgs'], color="r")
plt.show()

# Looking at the value count in the interactive session, i found that majority of the num_imgs are distributed among 0,1,2,3,11. Looking at their value_counts.

print(dfviz.num_imgs.isin([0,1,2,3,11]).value_counts())

#%%
# Plotting a boxplot with these number of images vs the rest.
dfviz0 = dfviz3[dfviz3['num_imgs'].isin([0,1,2,3,11,10])]
dfviz1 = dfviz3[~dfviz3['num_imgs'].isin([0,1,2,3,11,10])]

sns.set(style="whitegrid")
f, axes = plt.subplots(1, 2, figsize=(5, 5), sharey=True)
sns.despine(left=True)

sns.boxplot(data=dfviz0,y='shares',ax=axes[0])
axes[0].set(xlabel='For num_videos = 0')

sns.boxplot(data=dfviz1,y='shares',ax=axes[1])
axes[1].set(xlabel='For num_videos = 1')

plt.show()

# We see that number if images not equal to [0,1,2,3,11,10] has a slightly higher average number of shares. This EDA does not lead to any conclusion though. 
# The distribution of num_imgs is also not normal, it is safe to assume that there is no relation between num_imgs and number of shares. 

# However, just to reinforce this point, i will plot a scatter matrix to see any relation.

#%%
plt.scatter(dfviz.num_imgs,dfviz.shares,alpha=0.3)
plt.xlabel('Number of Images')
plt.ylabel('Number of Shares')
plt.show()

# No trend can be observed in this. Similarly, lets see for num_videos
#%%
plt.scatter(dfviz.num_videos,dfviz.shares,alpha=0.3)
plt.xlabel('Number of Videos')
plt.ylabel('Number of Shares')
plt.show()

# No trend

#%%
# Q2. Is there a relationship between the number of words in the content and number of words in the title in the article popularity.

sns.set(style="white", palette="muted")
bins = np.linspace(0,20,21)

plt.hist(dfviz.n_tokens_title, bins, alpha=0.5, edgecolor='black', linewidth=1)
plt.xticks(np.arange(0,21, step=1))
plt.xlabel('Number of words in title')
plt.ylabel('Density')
plt.show() 

# Wow, n_token_titles seems to have a perfect normal distribution in the dataset with average value at around 10

#%%
# Looking at distribution of n_tokens_content

sns.set(style="white", palette="muted")
#bins = np.linspace(0,20,21)

plt.hist(dfviz.n_tokens_content, alpha=0.5, edgecolor='black', linewidth=1)
#plt.xticks(np.arange(0,21, step=1))
plt.xlabel('Number of words in content')
plt.ylabel('Density')
plt.show() 

# Seems like 0-2000 is heavily populated. So lets split the distribution and have a look.

#%%
sns.set(style="white", palette="muted")
#bins = np.linspace(0,20,21)
# For 0-1000
plt.hist(dfviz[dfviz.n_tokens_content<2000].n_tokens_content, alpha=0.5, edgecolor='black', linewidth=1)
#plt.xticks(np.arange(0,21, step=1))
plt.xlabel('Number of words in content')
plt.ylabel('Density')
plt.show() 
# Looks almost normally distributed.

#%%
# Having a look at the value_counts of this column
print(dfviz.n_tokens_content.value_counts())

# It seems like there are 1181 articles that have 0 words in their columns. Just to be sure if this is possible, let me have an actual look at these articles through their links. I'll look at at least 5 articles.

a = dfviz[dfviz['n_tokens_content']==0].url.reset_index().drop('index',axis=1)
print(a.values)

# After having a look at these URLs, i can see that each of them do have some words in their content, not 0. Therefore, these are erroneous rows that i will have to subset from my data. 

dfviz = dfviz[dfviz['n_tokens_content']!=0]

# Making sure that such a thing does not exist in title as well.

#%%
print(dfviz.n_tokens_title.value_counts())
# Do not see 0, therefore no errors in title

# Now lets see their relationship with each other and then with number of shares.


 



# %%
