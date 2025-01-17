#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Loading Dataset

# In[2]:


df = pd.read_csv("dataset.csv")  # to load csv file to a varible as data_frame


# ### Overview of the dataset 

# In[3]:


df.head()  # prints 1st five rows of the dataset


# In[4]:


df.shape  # returns dimensions of the dataframe


# In[5]:


df.describe() # displays some of the statistical measures of the every column (every feature)


# In[6]:


plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
ax = sns.heatmap(df.corr(), xticklabels=2, annot=True ,yticklabels=True)


# ## Mutual information gain of attributes with ouput variable

# In[7]:


from sklearn.feature_selection import mutual_info_classif

mu_info = mutual_info_classif(df.drop(['Diabetes_012'],axis=1),df['Diabetes_012'])
mu_info


# In[10]:


mu_info = pd.Series(mu_info)
mu_info.index = df.drop(['Diabetes_012'],axis=1).columns
mu_info.sort_values(ascending=False)


# In[11]:


mu_info.sort_values(ascending=False).plot.bar(figsize=(20,8))


# # Data Preprocessing

# ## 1. Handling Null values

# In[6]:


df.isnull().sum() # returns count of null values in every column


# 

# ## 2. Handling Duplicates

# In[7]:


# checking duplicates

df.duplicated().sum()  # returns total count of duplicate samples


# In[8]:


df = df.drop_duplicates(subset=None,keep='first') # removes duplicates and keeps only the 1st occurence of the sample


# In[9]:


df.duplicated().sum()  # checking duplicates after removal


# In[10]:


df.shape


# In[11]:


df.columns


# ## 3.Feature selection

# In[17]:


# Education and Income are irrelevent features so they are dropped

df = df.drop(['Education','Income'],axis=1)
df.head()


# In[18]:


df.shape


# ## 4. Handling Outliers

# In[19]:


plt.figure(figsize=(20,50))
for i,col in enumerate(df.columns):
    plt.subplot(12,2, i+1)
    sns.boxplot(x=col , data = df)
plt.show()


# ### Checking skewness of features

# In[20]:


sk_frs = df[(df.columns)].skew()


# In[21]:


sk_frs


# In[ ]:





# Handling outliers of the features ['BMI','GenHlth','PhysHlth','MentHlth'] using Capping with the help of IQR( InterQuartile Range )

# In[22]:


# 1. BMI


# In[23]:


per_25 = df['BMI'].quantile(0.25)
per_75 = df['BMI'].quantile(0.75)

print("25th perc",per_25)
print("75th perc",per_75)

iqr = per_75-per_25
print("IQR",iqr)

up_limit = per_75 + 1.5*iqr
low_limit = per_25 - 1.5*iqr

print("Upper limit ",up_limit)
print("lower limit ",low_limit)


# In[24]:


# finding outliers in BMI 

df[df['BMI']<low_limit]


# In[25]:


df[df['BMI']>up_limit]


# In[26]:


# Hanlding outliers in BMI  using Capping

df2 = df.copy()

df2['BMI'] = np.where(df2['BMI']>up_limit ,
                         up_limit ,
                         np.where(
                             df2['BMI']<low_limit ,
                             low_limit ,
                             df2['BMI']))
df2.shape


# In[27]:


plt.subplot(2,2,1)
sns.distplot(df,x=df['BMI'])

plt.subplot(2,2,2)
sns.distplot(df2,x=df2['BMI'])

plt.subplot(2,2,3)
sns.boxplot(df,x=df['BMI'])

plt.subplot(2,2,4)
sns.boxplot(df2,x=df2['BMI'])


plt.show()


# In[28]:


# Handling outlier for the feature GenHlth

per_25 = df2['GenHlth'].quantile(0.25)
per_75 = df2['GenHlth'].quantile(0.75)

print("25th perc",per_25)
print("75th perc",per_75)

iqr = per_75-per_25
print("IQR",iqr)

up_limit = per_75 + 1.5*iqr
low_limit = per_25 - 1.5*iqr

print("Upper limit ",up_limit)
print("lower limit ",low_limit)


# In[29]:


df2[df2['GenHlth']>up_limit]


# In[30]:


df2[df2['GenHlth']<low_limit]


# In[31]:


df2['GenHlth'] = np.where(df2['GenHlth']>up_limit ,
                         up_limit ,
                         np.where(
                             df2['GenHlth']<low_limit ,
                             low_limit ,
                             df2['GenHlth']))
df2.shape


# In[32]:


plt.subplot(2,2,1)
sns.distplot(df,x=df['GenHlth'])

plt.subplot(2,2,2)
sns.distplot(df2,x=df2['GenHlth'])

plt.subplot(2,2,3)
sns.boxplot(df,x=df['GenHlth'])

plt.subplot(2,2,4)
sns.boxplot(df2,x=df2['GenHlth'])

plt.show()


# In[33]:


# Handling outlier for the feature PhysHlth

per_25 = df2['PhysHlth'].quantile(0.25)
per_75 = df2['PhysHlth'].quantile(0.75)

print("25th perc",per_25)
print("75th perc",per_75)

iqr = per_75-per_25
print("IQR",iqr)

up_limit = per_75 + 1.5*iqr
low_limit = per_25 - 1.5*iqr

print("Upper limit ",up_limit)
print("lower limit ",low_limit)


# In[34]:


df2[df2['PhysHlth']>up_limit]


# In[35]:


df2[df2['PhysHlth']<low_limit]


# In[36]:


df2['PhysHlth'] = np.where(df2['PhysHlth']>up_limit ,
                         up_limit ,
                         np.where(
                             df2['PhysHlth']<low_limit ,
                             low_limit ,
                             df2['PhysHlth']))
df2.shape


# In[37]:


plt.subplot(2,2,1)
sns.distplot(df,x=df['PhysHlth'])

plt.subplot(2,2,2)
sns.distplot(df2,x=df2['PhysHlth'])

plt.subplot(2,2,3)
sns.boxplot(df,x=df['PhysHlth'])

plt.subplot(2,2,4)
sns.boxplot(df2,x=df2['PhysHlth'])

plt.show()


# In[ ]:





# In[38]:


# Handling outlier for the feature MentHlth


per_25 = df2['MentHlth'].quantile(0.25)
per_75 = df2['MentHlth'].quantile(0.75)

print("25th perc",per_25)
print("75th perc",per_75)

iqr = per_75-per_25
print("IQR",iqr)

up_limit = per_75 + 1.5*iqr
low_limit = per_25 - 1.5*iqr

print("Upper limit ",up_limit)
print("lower limit ",low_limit)


# In[39]:


df2[df2['MentHlth']>up_limit]


# In[40]:


df2[df2['MentHlth']<low_limit]


# In[41]:


df2['MentHlth'] = np.where(df2['MentHlth']>up_limit ,
                         up_limit ,
                         np.where(
                             df2['MentHlth']<low_limit ,
                             low_limit ,
                             df2['MentHlth']))
df2.shape


# In[42]:


plt.subplot(2,2,1)
sns.distplot(df,x=df['MentHlth'])

plt.subplot(2,2,2)
sns.distplot(df2,x=df2['MentHlth'])

plt.subplot(2,2,3)
sns.boxplot(df,x=df['MentHlth'])

plt.subplot(2,2,4)
sns.boxplot(df2,x=df2['MentHlth'])

plt.show()


# ## 6. Data Splitting

# In[43]:


x = df2.drop(labels=['Diabetes_012'],axis=1)
y = df2['Diabetes_012']


# In[46]:


from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

X_train.shape , X_test.shape


# In[47]:


y_train.value_counts()


# In[48]:


y_test.value_counts()


# ## 7.Modelling the Data

# In[49]:


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


# In[50]:


rf_model = RandomForestClassifier(n_estimators=500,random_state=42)  # n_estimators = no of decision trees 
rf_model.fit(X_train,y_train)


# ## 8. Model Evaluation

# In[56]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[57]:


y_pred = rf_model.predict(X_test)
print("-------------------------------------------------------------------------")
print(f"The accuraccy score is: ------>>  {accuracy_score(y_test,y_pred)}")
print("-------------------------------------------------------------------------")
print(f"The Confusion Matrix is: ------>> \n{confusion_matrix(y_test,y_pred)}")
print("-------------------------------------------------------------------------")
print(f"The Classification Report is: ---->> {classification_report(y_test,y_pred)}")


# In[ ]:




