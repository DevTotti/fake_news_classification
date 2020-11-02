#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[69]:


df = pd.read_csv('news.csv')


# In[70]:


sns.pairplot(df)


# In[71]:


df.describe()


# In[72]:


df.head()


# In[73]:


df.info()


# In[74]:


df.shape


# In[75]:


X = df.text


# In[76]:


X


# In[77]:


y = df.label


# In[78]:


y


# In[79]:


from sklearn.model_selection import train_test_split


# In[80]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=7)


# In[81]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[82]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[83]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.8)


# In[84]:


tfidf_train = tfidf_vectorizer.fit_transform(X_train)


# In[85]:


tfidf_test = tfidf_vectorizer.transform(X_test)


# In[88]:


# from sklearn.naive_bayes import BernoulliNB


# In[89]:


# clf = BernoulliNB(binarize=False)


# In[90]:


from sklearn.svm import SVC


# In[91]:


clf = SVC()


# In[92]:


clf.fit(tfidf_train, y_train)


# In[93]:


clf.score(tfidf_test, y_test)


# In[94]:


y_pred = clf.predict(tfidf_test)


# In[96]:


clf_acc = accuracy_score(y_test, y_pred)


# In[97]:


clf_acc


# In[98]:


conf_mat = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])


# In[99]:


conf_mat


# In[ ]:




