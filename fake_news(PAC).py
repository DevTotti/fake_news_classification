#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('news.csv')


# In[3]:


sns.pairplot(df)


# In[4]:


df.describe()


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


X = df.text


# In[9]:


X


# In[10]:


y = df.label


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=7)


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[15]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[16]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.8)


# In[17]:


tfidf_train = tfidf_vectorizer.fit_transform(X_train)


# In[18]:


tfidf_test = tfidf_vectorizer.transform(X_test)


# In[21]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[23]:


PAC = PassiveAggressiveClassifier(C = 0.5, random_state = 5)


# In[24]:


PAC.fit(tfidf_train, y_train)


# In[25]:


PAC.score(tfidf_test, y_test)


# In[26]:


y_pred = PAC.predict(tfidf_test)


# In[29]:


pac_acc = accuracy_score(y_test, y_pred)


# In[30]:


pac_acc


# In[31]:


conf_mat = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])


# In[32]:


conf_mat


# In[33]:


clf_report = classification_report(y_test, y_pred)


# In[35]:


print(clf_report)


# In[ ]:




