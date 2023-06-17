import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('spam.csv')

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

x = df["message"].values
y = df["class"].values
cv = CountVectorizer()
x = cv.fit_transform(x)
v = x.toarray()

first_col = df.pop('message')
df.insert(0, 'message', first_col)
train_x = x[:4179]
train_y = y[:4179]

test_x = x[4179:]
test_y = y[4179:]
bnb = MultinomialNB()
model = bnb.fit(train_x, train_y)
y_pred_train= bnb.predict(train_x)
y_pred_test = bnb.predict(test_x)

print(bnb.score(train_x, train_y)*100)
print(bnb.score(test_x, test_y)*100)

from sklearn.metrics import classification_report
print(classification_report(train_y, y_pred_train))