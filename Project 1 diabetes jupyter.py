#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("C:\\Users\\DELL\\Downloads\\diabetes (2).csv")


# In[3]:


df.head(10)


# In[4]:


df.tail()


# # Data Preprocessing

# In[6]:


df.shape


# In[9]:


df.isnull().any()


# In[11]:


df.isnull().sum()


# In[12]:


df['Glucose']=df['Glucose'].fillna(df['Glucose'].median())


# In[13]:


df['BloodPressure']=df['BloodPressure'].fillna(df['BloodPressure'].median())


# In[14]:


df['Insulin']=df['Insulin'].fillna(df['Insulin'].median())


# In[15]:


df['DiabetesPedigreeFunction']=df['DiabetesPedigreeFunction'].fillna(df['DiabetesPedigreeFunction'].median())


# In[16]:


df['Age']=df['Age'].fillna(df['Age'].median())


# In[17]:


df.isnull().sum()


# # EDA
# 

# In[18]:


import seaborn as sns


# In[19]:


tc=df.corr()


# In[20]:


tc


# In[23]:


sns.heatmap(tc)


# In[26]:


sns.pairplot(tc)


# In[24]:


#feature selection


# # define x andd y

# In[30]:


df.columns


# In[31]:


x=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]


# In[32]:


y=df[['Outcome']]


# In[33]:


x.head()


# In[34]:


y.head()


# # Model Development

# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[39]:


df.shape


# In[40]:


x_train.shape


# In[41]:


x_test.shape


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[47]:


model=LogisticRegression()


# In[48]:


model.fit(x_train,y_train)


# In[49]:


y_pred=model.predict(x_test)


# In[50]:


y_pred


# In[51]:


from sklearn import metrics


# In[52]:


cm=metrics.confusion_matrix(y_test,y_pred)


# In[53]:


cm


# In[56]:


metrics.accuracy_score(y_test,y_pred)


# In[57]:


metrics.recall_score(y_test,y_pred)


# In[58]:


metrics.precision_score(y_test,y_pred)


# # Feature selection using RFE

# In[132]:


from sklearn.feature_selection import RFE


# In[133]:


rfe=RFE(model,7)


# In[134]:


fit=rfe.fit(x_train,y_train)


# In[135]:


fit.ranking_


# In[136]:


x.columns


# In[137]:


x_new3=x[['Pregnancies','Glucose','BloodPressure' , 'SkinThickness',
       'BMI', 'DiabetesPedigreeFunction','Age']]


# In[138]:


x_new3_train,x_new3_test,y_train,y_test=train_test_split(x_new3,y,test_size=0.3)


# In[139]:


model.fit(x_new3_train,y_train)


# In[140]:


y_pred3=model.predict(x_new3_test)


# In[141]:


y_pred3


# In[142]:


cm=metrics.confusion_matrix(y_test,y_pred3)


# In[143]:


cm


# In[144]:


metrics.accuracy_score(y_test,y_pred3)


# In[131]:


metrics.recall_score(y_test,y_pred3)


# In[ ]:




