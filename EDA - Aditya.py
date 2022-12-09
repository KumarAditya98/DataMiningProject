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

