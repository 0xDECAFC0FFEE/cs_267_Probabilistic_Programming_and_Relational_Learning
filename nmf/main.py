from utils import *

import pickle
import random
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm import tqdm
from itertools import chain
from collections import Counter, defaultdict
from pathlib import Path
from sklearn import metrics
from datetime import datetime
import csv
from datetime import datetime
import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain

movieid_mid_lookup = get_movieid_mid_lookup(recompute=True)

user_Xs, movie_Xs, ys = get_dataset(train_set, include_ys=True, recompute=True)
user_val_Xs, movie_val_Xs, val_ys = get_dataset(val_set, include_ys=True, recompute=True)
user_test_Xs, movie_test_Xs = get_dataset(test_set, include_ys=False, recompute=True)
movie_genres_one_hot = get_movie_genres_one_hot(recompute=True)

train_genres = get_dataset_genres(train_set, dataset_includes_ys=True, recompute=True)
val_genres = get_dataset_genres(val_set, dataset_includes_ys=True, recompute=True)
test_genres = get_dataset_genres(test_set, dataset_includes_ys=False, recompute=True)

preserved_variance = 1

train_tags = get_tags(movie_Xs, preserved_variance, recompute=True)
val_tags = get_tags(movie_val_Xs, preserved_variance, recompute=True)
test_tags = get_tags(movie_test_Xs, preserved_variance, recompute=True)

NUM_PROJ_TAGS = train_tags[0].shape[0]
print(f"projected down to {NUM_PROJ_TAGS} dims from {NUM_TAGS} dimensions (reduced by {1-NUM_PROJ_TAGS/NUM_TAGS})")

def model_v1(): 
    # smaller model used for hyperparameter tuning and testing

    embedding_dim = 40

    movie_genre_embeddings = tf.placeholder(dtype=tf.float64, shape=[None, 20], name="movie_genre_placeholder")
    movie_embeddings = tf.Variable(tf.contrib.layers.xavier_initializer(dtype=tf.float64)([NUM_MOVIES, embedding_dim]))
    user_embeddings = tf.Variable(tf.contrib.layers.xavier_initializer(dtype=tf.float64)([NUM_USERS, embedding_dim]))

    user_slice_idxs = tf.placeholder(dtype=tf.int64, shape=[None, 1], name="uids") # columns vectors to do tensor slicing
    movie_slice_idxs = tf.placeholder(dtype=tf.int64, shape=[None, 1], name="mids") # columns vectors to do tensor slicing

    user_embedding_columns = tf.reshape(tf.gather_nd(user_embeddings, user_slice_idxs), [-1, embedding_dim])
    movie_embedding_rows = tf.reshape(tf.gather_nd(movie_embeddings, movie_slice_idxs), [-1, embedding_dim])

    mult_input = movie_embedding_rows * user_embedding_columns

    input_layer = tf.concat((
        movie_embedding_rows,
        movie_genre_embeddings,
        user_embedding_columns,
    ), axis=1)
    print(movie_embedding_rows.shape, user_embedding_columns.shape)
    print("input layer shape", input_layer.shape)

    W1 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[int(input_layer.shape[1]), 60], dtype=tf.float64))
    b1 = tf.Variable(initial_value=np.zeros(shape=[60], dtype=np.float64))
    l1 = tf.nn.relu(tf.matmul(input_layer, W1) + b1)

    W2 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[60, 20], dtype=tf.float64))
    b2 = tf.Variable(initial_value=np.zeros(shape=[20], dtype=np.float64))
    l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

    W3 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[int(l2.shape[1]), 2], dtype=tf.float64))
    b3 = tf.Variable(initial_value=np.zeros(shape=[2], dtype=np.float64))
    l3 = tf.matmul(l2, W3) + b3
    pred_y = tf.nn.sigmoid(l3)

    all_weights = [W1, b1, W2, b2, W3, b3]

    return all_weights, movie_genre_embeddings, user_slice_idxs, movie_slice_idxs, pred_y


def model_v2():
    # larger model with higher acc but generally takes longer to train

    embedding_dim = 40

    movie_embeddings = tf.Variable(tf.contrib.layers.xavier_initializer(dtype=tf.float64)([NUM_MOVIES, embedding_dim]))
    user_embeddings = tf.Variable(tf.contrib.layers.xavier_initializer(dtype=tf.float64)([NUM_USERS, embedding_dim]))

    user_slice_idxs = tf.placeholder(dtype=tf.int64, shape=[None, 1], name="uids") # columns vectors to do tensor slicing
    movie_slice_idxs = tf.placeholder(dtype=tf.int64, shape=[None, 1], name="mids") # columns vectors to do tensor slicing

    user_embedding_columns = tf.reshape(tf.gather_nd(user_embeddings, user_slice_idxs), [-1, embedding_dim])
    movie_embedding_rows = tf.reshape(tf.gather_nd(movie_embeddings, movie_slice_idxs), [-1, embedding_dim])

    movie_genre_embeddings = tf.placeholder(dtype=tf.float64, shape=[None, 20], name="movie_genre_placeholder")
    tags = tf.placeholder(dtype=tf.float64, shape=[None, NUM_PROJ_TAGS], name="tags_placeholder")

    content_filtering_embedding = tf.concat((
        tags,
        movie_genre_embeddings
    ), axis=1)
    content_W1 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[NUM_PROJ_TAGS, 150], dtype=tf.float64))
    content_b1 = tf.Variable(initial_value=np.zeros(shape=[150], dtype=np.float64))
    content_l1 = tf.nn.relu(tf.matmul(content_filtering_embedding, content_W1) + content_b1)

    content_W2 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[150, 40], dtype=tf.float64))
    content_b2 = tf.Variable(initial_value=np.zeros(shape=[40], dtype=np.float64))
    content_embedding = tf.nn.relu(tf.matmul(content_l1, content_W2) + content_b2)

    matrix_factorization_layer = movie_embedding_rows * user_embedding_columns

    mlp_input_layer = tf.concat((
        content_embedding,
        movie_embedding_rows,
        user_embedding_columns,
    ), axis=1)

    W1 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[int(mlp_input_layer.shape[1]), 60], dtype=tf.float64))
    b1 = tf.Variable(initial_value=np.zeros(shape=[60], dtype=np.float64))
    l1 = tf.nn.relu(tf.matmul(mlp_input_layer, W1) + b1)

    W2 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[60, 20], dtype=tf.float64))
    b2 = tf.Variable(initial_value=np.zeros(shape=[20], dtype=np.float64))
    l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

    W3 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[int(l2.shape[1]), 2], dtype=tf.float64))
    b3 = tf.Variable(initial_value=np.zeros(shape=[2], dtype=np.float64))
    l3 = tf.matmul(tf.concat((
            l2,
            matrix_factorization_layer
        ), axis=1), W3) + b3

    pred_y = tf.nn.sigmoid(l3)

    all_weights = [W1, b1, W2, b2, W3, b3, content_W1, content_b1, content_W2, content_b2]

    return all_weights, tags, movie_genre_embeddings, user_slice_idxs, movie_slice_idxs, pred_y


# training and running the model on validation data

with tf.Session() as sess:
    all_weights, movie_genre_embeddings, user_slice_idxs, movie_slice_idxs, pred_y = model_v1()
    y_true = tf.placeholder(dtype=tf.float64, shape=[None, 2])

    learning_rate=.001
    epochs=60
    l2_loss_term = .001 * sum([tf.reduce_sum(tf.reshape(weight*weight, [-1])) for weight in all_weights])
    mse_loss_term = tf.reduce_mean(tf.squared_difference(pred_y, y_true))
    ce_loss_term = -(tf.reduce_mean(((y_true+1)/2)*tf.math.log((pred_y+1)/2)+(1-(y_true+1)/2)*tf.math.log(1-(pred_y+1)/2)))
    loss = ce_loss_term# + l2_loss_term
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    flat_val_ys = [(1 if y[0] > y[1] else 0) for y in val_ys]

    min_acc_auc = (math.inf, math.inf)

    sess.run(init)

    val_accs = []
    val_aucs = []
    train_losses = []
    val_losses = []

    cur_best_auc = 0
    cur_best_train_loss = math.inf

    for epoch in tqdm(range(epochs), leave=False):
        print("epoch", epoch)
        print("training")
        for b_user_Xs, b_movie_Xs, b_genres, b_ys in batchify(user_Xs, movie_Xs, train_genres, ys, batch_size=746661, shuffle=False):
            feed_dict = {user_slice_idxs: b_user_Xs, 
                        movie_slice_idxs: b_movie_Xs, 
                        movie_genre_embeddings: b_genres,
    #                      tags: b_tags,
                        y_true: b_ys}
            outs = (train_step, loss, l2_loss_term, ce_loss_term)
            _, lossval, l2_lossval, mse_lossval = sess.run(outs, feed_dict=feed_dict)
    #         print("pred_ys", pred_y_val, "true_ys", b_ys[:5])
            print("train loss", lossval, "l2", l2_lossval, "mse", mse_lossval)

    #         with train_file_writer as writer:
    #             writer.add_summary(tf.summary.scalar("loss", loss))

        feed_dict = {user_slice_idxs: user_val_Xs,
                    movie_slice_idxs: movie_val_Xs,
                    movie_genre_embeddings: val_genres,
    #                 tags: val_tags,
                    y_true: val_ys}
        val_y_pred, val_loss_val = sess.run((pred_y, loss), feed_dict=feed_dict)
        flat_pred_y_floats = [y[0]/(y[0]+y[1]) for y in val_y_pred]
        flat_pred_y_bools = [(1 if y[0] > y[1] else 0) for y in val_y_pred]

        print("val loss", val_loss_val)
        acc = metrics.accuracy_score(flat_val_ys, flat_pred_y_bools)
        fpr, tpr, _ = metrics.roc_curve(flat_val_ys, flat_pred_y_floats)
        auc = metrics.auc(fpr, tpr)
        print("val acc", acc, "val auc", auc)
        min_acc_auc = min(min_acc_auc, (acc, auc))
        val_accs.append(acc)
        val_aucs.append(auc)
        val_losses.append(val_loss_val)

        plt.clf()
        plt.title("val aucs per epoch")
        plt.grid(b=True)
        plt.minorticks_on()
        plt.plot(val_aucs, marker='o', color="black")
        plt.savefig("cur_val_aucs.png")

        train_loss = lossval
        train_losses.append(train_loss)
        plt.clf()
        fig, ax = plt.subplots()
        plt.title("losses per epoch")
        ax.grid(b=True)
        ax.minorticks_on()
        train_losses_plot = ax.plot(train_losses, marker='o', color="blue", label="train losses")
        val_losses_plot = ax.plot(val_losses, marker='o', color="green", label="val losses")
        ax.legend()

        plt.savefig("cur_train_loss.png")

        if auc > cur_best_auc:
            print(f"NEW BEST AUC: {auc} @ epoch {epoch}")
        if train_loss < cur_best_train_loss:
            print(f"NEW BEST TRAIN LOSS: {train_loss} @ epoch {epoch}")
