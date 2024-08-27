#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# In[4]:


df=pd.read_csv("Restaurant_Reviews.tsv",delimiter = '\t' ,quoting=3)#quoting=3 to ignore all the quotes in the reviews  
# X=df.iloc[:,:-1].values
# y=df.iloc[:,-1].values
df


# In[17]:


# removing all the unrequired punctuations 
import re 
import nltk
nltk.download("stopwords")
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus=[]
for i in range(0,1000):
    review=re.sub("^a-zA-Z"," ",df["Review"][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words("English")
    all_stopwords.remove("not")
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)] #remove words which wont help us 
    review=" ".join(review)
    corpus.append(review)


# In[5]:


df["Review"][1]


# In[18]:


print(corpus)


# In[64]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1300)
X=cv.fit_transform(corpus).toarray() #all the words will be taken by fit and transform will put it in a column
y=df.iloc[:,-1].values


# In[65]:


len(X[0])


# In[66]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[67]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,y_train)


# In[68]:


y_predict=lr.predict(X_test)


# In[69]:


y_1=y_predict.reshape(len(y_predict),1).astype(int)
y_2=y_test.reshape(len(y_test),1).astype(int)
print(np.concatenate([y_1,y_2],axis=1))


# In[70]:


from sklearn.metrics import confusion_matrix , accuracy_score
cm=confusion_matrix(y_test,y_predict)
print(cm)
print(accuracy_score(y_predict,y_test))


# In[ ]:





# In[84]:


import re 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download("stopwords")
from nltk.corpus import stopwords
new_rev="I hate the food here"
new_rev=re.sub("^a-zA-Z"," ",df["Review"][i])
new_rev.lower()
new_rev.split()
all_stopwords=stopwords.words("English")
all_stopwords.remove("not")
new_rev=[ps.stem(words) for words in new_rev if not words in set(all_stopwords)]
new_rev=" ".join(new_rev)
new_corpus=[new_rev]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = lr.predict(new_X_test)
print(new_y_pred)


# In[ ]:




