import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import linear_model
from feature import Bag, Gram

df = pd.read_csv("train.tsv",delimiter='\t')
x_train, x_test, y_train, y_test = train_test_split(df['Phrase'], df['Sentiment'], test_size=0.2)

tf = TfidfVectorizer()
x_train = tf.fit_transform(x_train)
x_test = tf.transform(x_test)

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=0.01, solver="sag").fit(x_train, y_train)
predicted = clf.predict(x_test)
print(metrics.classification_report(y_test, predicted))
print('accuracy_score: %0.5fs' % (metrics.accuracy_score(y_test, predicted)))