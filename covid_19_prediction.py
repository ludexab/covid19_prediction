# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 01:32:36 2022

@author: boss
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (LabelEncoder, StandardScaler)
from sklearn.metrics import (confusion_matrix, accuracy_score)

#   loading dataset
dataset = pd.read_csv('covid19.csv')

#   splitting dataset into features and dependent variable(label)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#   Label encoding
label_encoder_y = LabelEncoder()
Y = label_encoder_y.fit_transform(Y)

#   Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

#   splitting dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0, shuffle=True)

classifier = RandomForestClassifier(n_estimators=11, criterion='entropy',
                                    random_state=0, oob_score=True, n_jobs=-1)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test, y_pred)