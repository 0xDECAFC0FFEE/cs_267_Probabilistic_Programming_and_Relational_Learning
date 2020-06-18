# Using LDA on words https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
# sklearn page on LDA https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
# https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28 coding example of combining lda

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

number_components = 10 # number of possible ratings
print("Starting")
start_time = time.time()
if path.exists('modelratings.joblib'):
    lda = load('modelratings.joblib')
# If no LDA exists, fit new one
else:
    lda = LatentDirichletAllocation(n_components=number_components)
    lda.fit(ratingstraining) # try fit transform
    print("training took", time.time() - start_time)
    dump(lda, 'modelratings.joblib')


training_lda = lda.transform(ratingstraining)
test_lda = lda.transform(ratingstest)
print(training_lda)


clf = LogisticRegression(random_state=1, max_iter=10000)

encoder = preprocessing.LabelEncoder()
rts_encoded = encoder.fit_transform(ratingstrainingscore)
original_encoded = encoder.fit_transform(ratingstrainingtest)

print(rts_encoded)

clf = clf.fit(training_lda, rts_encoded)


print(clf.score(training_lda, rts_encoded))

predictions = clf.predict(test_lda)
print(predictions)

rms = sqrt(mean_squared_error(predictions, original_encoded))

print(rms)
