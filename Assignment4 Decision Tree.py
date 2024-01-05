#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


from IPython.display import Image
Image(filename='IRIS.jpg', height=450, width=400)


# In[4]:


iris = datasets.load_iris()


# In[5]:


data=pd.DataFrame(iris['data'],columns=["Petal length","Petal Width","Sepal Length","Sepal Width"])


# In[6]:


data['Species']=iris['target']


# In[7]:


data['Species']=data['Species'].apply(lambda x: iris['target_names'][x])


# In[8]:


data.head()


# In[9]:


sns.pairplot(data, hue = 'Species')
plt.show()


# In[10]:


plt.figure(figsize=(10,11))
sns.heatmap(data.corr(),annot=True)
plt.plot()


# In[11]:


plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.boxplot(x="Species",y="Sepal Length",data=data)
plt.subplot(2,2,2)
sns.boxplot(x="Species",y="Sepal Width",data=data)
plt.subplot(2,2,3)
sns.boxplot(x="Species",y="Petal length",data=data)
plt.subplot(2,2,4)
sns.boxplot(x="Species",y="Petal Width",data=data)


# In[12]:


from sklearn.model_selection import train_test_split

train,test=train_test_split(data,test_size=0.3)


# In[13]:


train_X=train[['Sepal Length',"Sepal Width","Petal length","Petal Width"]]
train_y=train.Species


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


DT = DecisionTreeClassifier(random_state=12)
model = DT.fit(train_X, train_y)


# In[17]:


DT.predict(train_X)


# In[19]:


DT.score(train_X, train_y)


# In[21]:


y_pred = DT.predict(train_X)


# In[23]:


from sklearn import metrics
print('Accuracy Score:', metrics.accuracy_score(train_y, y_pred))


# In[24]:


get_ipython().system('pip install pydotplus')


# In[25]:


from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# Visualize the graph
dot_data = StringIO()
export_graphviz(DT, out_file=dot_data, feature_names=iris.feature_names,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[26]:


DT.predict([[3.5, 1.9, 5.2, 2.4]])


# In[27]:


DT.predict([[4.6, 2.9, 3.6, 5.9]])


# In[ ]:




