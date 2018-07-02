
# coding: utf-8

# In[3]:

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#used for Confusion Matrix
from sklearn import metrics

get_ipython().magic('matplotlib inline')


# In[4]:

digits = load_digits()


# In[5]:

digits.data.shape


# In[6]:

digits.target.shape


# In[7]:

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1,5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training %i\n' % label, fontsize = 20)


# # Splitting Data into Training and Test Sets
# * Will be using 75% for the training set and 25% for the testing set

# In[8]:

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 0)


# In[9]:

print(x_train.shape)


# # Scikit-Learn 4-Step Modelling Pattern

# __Step 1:__ Import the model you want to use
# 
# In sklearn, all machine learning models are implemented as Python classes

# In[10]:

from sklearn.linear_model import LogisticRegression 


# __Step 2:__ Make an instance of the model

# In[11]:

logisticRegr = LogisticRegression()


# __Step 3:__ Train the model on the data, store information learned from the data
# 
# * Model is learning from the relationship between x (digits) and y (labels)

# In[12]:

logisticRegr.fit(x_train, y_train)


# __Step 4:__ Predict the labels of new data (new pictures)
# 
# Uses information the model learned during model training process

# In[13]:

# Returns a NumPy Array
# Predict for One Observation (Image)

logisticRegr.predict(x_test[0].reshape(1,-1))


# In[14]:

#Predict for Multiple Observations (Images) at once
logisticRegr.predict(x_test[0:10])


# In[15]:

#Predictions on the entire test data
predictions = logisticRegr.predict(x_test)


# In[16]:

predictions.shape


# # Measuring Model Performance (Digits Dataset)
# 
# * In this case, I'll be using accuracy as my metric
# * Accuracy is defined as- (fraction of correct predictions): correct predictions / total number of data points

# In[18]:

score = logisticRegr.score(x_test,y_test)
print(score)


# ### Confusion Matrix (Digits Dataset)

# In[20]:

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# In[21]:

#Seaborn
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt = ".3f", linewidths = .5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[53]:

#matplotlib
plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape

for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
  horizontalalignment='center',
  verticalalignment='center')


# ### Downloading Data( MNIST Data)
# 

# In[25]:

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')


# In[26]:

# These are the images
# There are 70,000 images (28 by 28 images for a dimensionality of 784)

print(mnist.data.shape)
# These are the labels
print(mnist.target.shape)


# ### Splitting Data Into Training and test Sets (MNIST Data)

# In[27]:

from sklearn.model_selection import train_test_split

train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size = 1/7.0, random_state = 0)


# In[35]:

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


# In[39]:

from sklearn.linear_model import LogisticRegression


# In[40]:

logisticRegr = LogisticRegression(solver = 'lbfgs')


# In[41]:

logisticRegr.fit(train_img, train_lbl)


# In[42]:

logisticRegr.predict(test_img[0].reshape(1,-1))


# In[44]:

logisticRegr.predict(test_img[0:10])


# In[45]:

logisticRegr.predict(test_img)


# In[46]:

score = logisticRegr.score(test_img, test_lbl)
print(score)


# ### Display Misclassified Images with Predicted Labels (MNIST)

# In[50]:

import numpy as np 
import matplotlib.pyplot as plt

index = 0 
misclassifiedIndexes = []
for label, predict in zip(test_lbl, predictions):
    if label != predict:
        misclassifiedIndexes.append(index)
        index += 1


# In[52]:

plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
 plt.subplot(1, 5, plotIndex + 1)
 plt.imshow(np.reshape(test_img[badIndex], (28,28)), cmap=plt.cm.gray)
 plt.title('Predicted: {}, Actual: {}' .format(predictions[badIndex], test_lbl[badIndex]), fontsize = 15)


# In[ ]:



