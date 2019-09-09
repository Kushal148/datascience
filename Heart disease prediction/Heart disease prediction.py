
# coding: utf-8

# # Heart Disease Prediction

# This dataset is about heart disease study, predicting weather the patient is suffering from a heart disease based on the give data.

# ## Dataset:
# 1. Systolic Blood Pressure: blood pressure of the patient/person.
# 2. Cholestrol: Cholestro level of the patient/person.
# 3. Family History: either 1 or 0.
# 4. BMI: body to mass index.
# 5. Age: age of the patient/person.
# 6. Class: heart disease(1,0).

# In[21]:


# importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
plt.rcParams['figure.figsize'] = (9.0,9.0)


# In[22]:


# Reading the the dataset as heart_data.
heart_data = pd.read_csv('F:/intern/HeartDiseasePrediction/HeartDisease_Training_Data.csv')
heart_data.shape


# In[23]:


heart_data.head()


# ### Renaming the columns

# In[24]:


heart_data.rename(columns = {'Systolic Blood Pressure':'Systolic_Blood_Pressure',
                            'Family History':'family_history'} , inplace = True)
heart_data.head()


# In[25]:


heart_data.describe()


# In[26]:


heart_data.info()


# the dataset has no null values.

# ## Data visualization

# In[27]:


sb.set_style('whitegrid')
sb.countplot(heart_data.Class, palette = 'autumn')
plt.title('Count of people with and without heart disease', fontsize = 20)
plt.show()


# In[28]:


sb.boxplot(data = heart_data, x = 'Class', y = 'Age')
plt.title('box plot of age vs class', fontsize = 20)
plt.show()


# In[29]:


sb.countplot(heart_data['family_history'], hue = heart_data['Class'])
plt.title('countplot family history vs Class', fontsize = 20)
plt.show()


# In[30]:


sb.violinplot(heart_data.Class, heart_data.BMI)
plt.title('class vs BMI', fontsize = 20)
plt.show()


# In[31]:


plt.scatter(heart_data.Systolic_Blood_Pressure, heart_data.Cholestrol)
plt.title('relation between blood pressure and colestrol level', fontsize = 20)
plt.show()


# In[32]:


sb.boxplot( heart_data.Class, heart_data.Cholestrol)
plt.title('box plot of Cholestrol level vs class', fontsize = 20)
plt.show()


# In[33]:


sb.boxplot( heart_data.Class, heart_data.Systolic_Blood_Pressure)
plt.title('box plot of blood pressure vs class', fontsize = 20)
plt.show()


# ### Train test data

# In[34]:


from sklearn.model_selection import train_test_split
x = heart_data.drop(['Class'], axis=1).values
y = heart_data['Class'].values


# In[47]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 62)


# ### decision tree classifier

# In[48]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train )


# In[49]:


y_pred = tree.predict(x_test)


# In[50]:


y_pred


# In[51]:


tree.predict([[110,4,1,20,50]])


# In[52]:


print('accuraccy of taining set:{:.2f}'.format(tree.score(x_train,y_train)))
print('accuraccy of testing set:{:.2f}'.format(tree.score(x_test,y_test)))

