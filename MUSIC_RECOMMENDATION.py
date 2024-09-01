#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv(r"G:\DHARSHNI WORKS\MUSIC_RECOMMENDATION_SYSTEM\spotify_millsongdata.csv")
df


# In[3]:


df.isnull().sum()


# In[4]:


df =df.sample(5000).drop('link', axis=1).reset_index(drop=True)


# In[5]:


df['text'][0]


# In[6]:


df.shape


# In[7]:


df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex = True)


# In[8]:


import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)


# In[9]:


df['text'] = df['text'].apply(lambda x: tokenization(x))


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[11]:


tfidvector = TfidfVectorizer(analyzer='word',stop_words='english')
matrix = tfidvector.fit_transform(df['text'])
similarity = cosine_similarity(matrix)


# In[12]:


similarity[0]


# In[13]:


df[df['song'] == 'Crying Over You']


# In[14]:


def recommendation(song_df):
    idx = df[df['song'] == song_df].index[0]
    distances = sorted(list(enumerate(similarity[idx])),reverse=True,key=lambda x:x[1])
    
    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]].song)
        
    return songs


# In[15]:


recommendation('Crying Over You')


# In[16]:


import pickle
pickle.dump(similarity,open('similarity.pkl','wb'))
pickle.dump(df,open('df.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




