
# coding: utf-8

# #  Spooky Author Classification: Naive bayes

# **the objective is to build a machine learning model that predicts the author of a given piece of text. The possible authors are Edgar Allan Poe (EAP), HP Lovecraft (HPL), and Mary Wollstonecraft Shelley (MWS).**

# In[95]:


# Importing required libraries
import numpy as np 
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import os
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# ## Loading Data

# In[96]:


train = pd.read_csv('C:/Users/Kushal/Desktop/kaggle datasets/train.csv')
test = pd.read_csv('C:/Users/Kushal/Desktop/kaggle datasets/test.csv')


# In[97]:


print(train.shape)
train.head()


# ## Spooky Authors

# In[98]:


print(train['author'].unique())


# ### edgar ellen poe
# ![eap.jpg](attachment:eap.jpg)

# ### HP lovecraft
# ![hpl.jpg](attachment:hpl.jpg)

# ### Mary Wollstonecraft Shelley
# ![mws.jpg](attachment:mws.jpg)

# In[99]:


train.info()


# In[100]:


plt.figure(figsize=(8,6))
sb.countplot(train.author)
plt.title('Count of text for each Author in dataset',fontsize = 20)
plt.show()


# In[101]:


train['length'] = train.text.str.count(' ')
train.head()


# In[102]:


train[train["author"]=="MWS"]["length"].describe()


# In[103]:


train[train["author"]=="HPL"]["length"].describe()


# In[104]:


train[train["author"]=="EAP"]["length"].describe()


# **WORDCLOUD**

# In[105]:


eap = train[train.author=="EAP"]["text"].values
hpl = train[train.author=="HPL"]["text"].values
mws = train[train.author=="MWS"]["text"].values


# In[106]:


# importing mask images
wave_mask1 = np.array(Image.open(os.path.join( "C:/Users/Kushal/Desktop/wordpress masks/alien.png")))
wave_mask2 = np.array(Image.open(os.path.join( "C:/Users/Kushal/Desktop/wordpress masks/oct.png")))
wave_mask3 = np.array(Image.open(os.path.join( "C:/Users/Kushal/Desktop/wordpress masks/frank.png")))


# In[107]:


plt.figure(figsize=(15,10))
wc = WordCloud(width=1080, height=720, background_color="black", max_words=1000,mask=wave_mask1, colormap="Greens_r",
               stopwords=STOPWORDS, max_font_size= 40)
wc.generate(" ".join(eap))
plt.title(" Edgar Ellen Poe", fontsize=20)
plt.imshow(wc, alpha =0.98)
plt.axis('off', interpolation="bilinear")
plt.show()


# In[108]:


plt.figure(figsize=(15,10))
wc = WordCloud(width=1080, height=720, background_color="black", max_words=1000,mask=wave_mask2, colormap="Reds_r",
               stopwords=STOPWORDS, max_font_size= 60)
wc.generate(" ".join(hpl))
plt.title("HP lovecraft", fontsize=20)
#plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)
plt.imshow(wc, alpha =0.98)
plt.axis('off', interpolation="bilinear")
plt.show()


# In[109]:


plt.figure(figsize=(15,10))
wc = WordCloud(width=1080, height=720, background_color="black", max_words=1000,mask=wave_mask3, colormap="GnBu",
               stopwords=STOPWORDS, max_font_size= 60)
wc.generate(" ".join(mws))
plt.title("Mary Wollstonecraft Shelley", fontsize=20)
#plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)
plt.imshow(wc, alpha =0.98)
plt.axis('off', interpolation="bilinear")
plt.show()


# In[110]:


#Encoding the categorical values
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)
x = train['text']


# In[111]:


# data which we are going to predict
test_data = test['text']


# In[113]:


# spliting the data, 0.9 = training and 0.1 = testing
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify= y, test_size = 0.20, random_state = 66)


# In[114]:


#converting text data, Countvectors 
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

ctv.fit(list(x_train) + list(x_test))
x_train_cv =  ctv.transform(x_train) 
x_test_cv = ctv.transform(x_test)

data_to_predict = ctv.transform(test_data) # data to be predicted.


# In[115]:


# ML Model Naive bayes
clf=MultinomialNB(alpha=2.0, class_prior=None, fit_prior=True)

clf.fit(x_train_cv, y_train) 
pred = clf.predict(x_test_cv)

clf.score(x_test_cv,y_test)


# ### Classification Report

# In[116]:


# Classification Report and confusion matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))


# ### Predticting Author 

# In[ ]:


predicted_values = clf.predict(data_to_predict)
predicted_values


# In[117]:


#Data frame for newly predicted text values
predicted_authors = pd.DataFrame()
predicted_authors['id'] = test['id']
predicted_authors['authors'] = predicted_values
predicted_authors.head()


# In[118]:


predicted_authors['author'] = predicted_authors.authors.map({0:'EAP', 1:'HPL', 2:'MWS' })


# In[119]:


predicted_authors.drop(['authors'],axis=1).head()

