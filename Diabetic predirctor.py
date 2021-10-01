#!/usr/bin/env python
# coding: utf-8

# # Importing data and data preprocessing

# In[29]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('diabetes.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
display(data.head())


# # Data exploration 
# 

# In[30]:


n_records = data.shape[0]
n_of_diabetic =data[data["Outcome"]==1].shape[0]
n_of_non_diabetic =data[data["Outcome"]==0].shape[0]
greater_percent = (n_of_diabetic/n_records)*100
print("Total number of records: {}".format(n_records))
print("Individuals who are diabetic: {}".format(n_of_diabetic))

print("Individuals who are not diabetic: {}".format(n_of_non_diabetic))
print("Percentage of individuals who are diabetic: {}%".format(greater_percent))


# In[31]:


data.groupby('Outcome').mean()


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[33]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)


# In[34]:


print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# In[35]:


TP=np.sum(y)
FP= n_records - TP
TN=0
FN=0
accuracy = TP/(TP+FP)
recall = TP/(TP+FN)
precision = TP/(TP+FP)
 
# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore =(1+ beta**2)*(precision * recall)/((beta**2*precision)+recall) 


# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# In[36]:


# using support vector


# In[37]:


from sklearn.svm import SVC
model = SVC(C=10, gamma=0.01, random_state=42,kernel='linear')
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)


# # Gridsearching 

# In[38]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
clf = SVC(random_state=42)

# TODO: Create the parameters list you wish to tune.
parameters = {'C':[0.1,0.001,0.5,1,2,10],'kernel':['rbf'], 'gamma':[0.01,0.1,27,10,50,100]}
cv_sets=ShuffleSplit(n_splits=10, random_state=42, test_size=0.2, train_size=None)
# TODO: Make an fbeta_score scoring object.
scorer = make_scorer(f1_score)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer,cv=cv_sets)

# TODO: Fit the grid search object to the training data and find the optimal parameters.
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator.
best_clf = grid_fit.best_estimator_

# Fit the new model.
best_clf.fit(X_train, y_train)

# Make predictions using the new model.
best_train_predictions = best_clf.predict(X_train)
best_test_predictions = best_clf.predict(X_test)

# Calculate the f1_score of the new model.
print('The training F1 Score is', f1_score(best_train_predictions, y_train))
print('The testing F1 Score is', f1_score(best_test_predictions, y_test))
train_accuracy = accuracy_score(y_train, best_train_predictions)
test_accuracy = accuracy_score(y_test,best_test_predictions )
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)

# Let's also explore what parameters ended up being used in the new model.
best_clf


# # predective system 

# In[39]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = sc.transform(input_data_reshaped)
print(std_data)

prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




