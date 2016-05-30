
# coding: utf-8

# In[ ]:

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[ ]:

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


# In[ ]:

lr = LogisticRegression()
scaler = StandardScaler()

pipe = Pipeline([('scaler', scaler),                 ('lr', lr)])

param_grid = {"lr__penalty": ['l1', 'l2'],
              "lr__max_iter": [100, 200],
              "lr__C": [1.0, .2]}

cv = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring='accuracy')


# In[ ]:

cv.fit(train.loc[:, 'pixel0':], train.loc[:, 'label'])


# In[ ]:

cv.best_params_


# In[ ]:

print(classification_report(train.loc[:, 'label'],                            cv.predict(train.loc[:, 'pixel0':])))

