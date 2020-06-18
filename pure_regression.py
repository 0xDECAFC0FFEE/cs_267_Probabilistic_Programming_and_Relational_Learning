import numpy as np 
import pandas as pd 
import pickle
import re

import time

from os import path
from joblib import dump,load

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing, utils

from sklearn.metrics import mean_squared_error
from math import sqrt

#Load ratings data
ratingsdata = pd.read_csv('./ml-1m/ratings.dat', names=['userId','movieId','rating','timestamp'], skiprows=1, sep="::", engine='python')
ratingsdata = ratingsdata.drop(columns='timestamp')
ratingstraining = ratingsdata.sample(frac=0.8, random_state=1)
ratingstest = ratingsdata.drop(ratingstraining.index)

ratingstrainingscore = ratingstraining['rating']
ratingstrainingtest = ratingstest['rating']


# Load LDA if it exists

number_components = 5 # number of possible ratings

clf = LogisticRegression(random_state=1, max_iter=10000)

encoder = preprocessing.LabelEncoder()
rts_encoded = encoder.fit_transform(ratingstrainingscore)

original_encoded = encoder.fit_transform(ratingstrainingtest)

print(rts_encoded)

ratingstraining_drop = ratingstraining.drop(columns='rating')
clf = clf.fit(ratingstraining_drop, rts_encoded)


print(clf.score(ratingstraining_drop, rts_encoded))


predictions = clf.predict(ratingstest.drop(columns='rating'))
print(predictions)

rms = sqrt(mean_squared_error(predictions, original_encoded))

print(rms)
