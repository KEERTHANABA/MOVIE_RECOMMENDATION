#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv') 


# In[3]:


credits.head(1)


# In[4]:


movies = movies.merge(credits,on='title')


# In[5]:


movies.shape
movies.head()
movies.info()


# In[6]:


movies = movies[['title','overview','genres','keywords']]


# In[7]:


movies.isnull().sum()


# In[8]:


movies.dropna(inplace=True)


# In[9]:


movies.duplicated().sum()


# In[10]:


credits.head()


# In[11]:


movies = movies.merge(credits,on='title')


# In[12]:


movies.head()


# In[13]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[14]:


movies.head()


# In[15]:


movies.isnull().sum()


# In[16]:


movies.iloc[0].genres


# In[17]:


import ast


# In[24]:


def convert(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L


# In[25]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[26]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[27]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[28]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


# In[29]:


movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()


# In[30]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head()


# In[31]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies.head()


# In[32]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head()


# In[33]:


new_df=movies[['movie_id','title','tags']]


# In[34]:


new_df


# In[35]:


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()


# In[47]:


new_df['tags'][1]


# In[37]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
new_df.head()


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer


# In[39]:


cv = CountVectorizer(max_features=5000,stop_words='english')


# In[40]:


vector = cv.fit_transform(new_df['tags']).toarray()
vector.shape


# In[41]:


vector[0]


# In[42]:


cv.get_feature_names()


# In[43]:


from sklearn.metrics.pairwise import cosine_similarity


# In[44]:


similarity = cosine_similarity(vector)


# In[45]:


similarity.shape


# In[49]:


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
recommend('Spider-Man 3')


# In[ ]:




