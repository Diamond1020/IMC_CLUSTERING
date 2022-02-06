import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import cv2
import os, glob, shutil
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    input_dir = 'pets'
    glob_dir = input_dir + '/*.jpg'

    images = [cv2.resize(cv2.imread(file), (100, 100)) for file in glob.glob(glob_dir)]
    paths = [file for file in glob.glob(glob_dir)]
    images = np.array(np.float32(images).reshape(len(images), -1)/255)
    images = StandardScaler().fit_transform(images)

    model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(100, 100, 3))
    predictions = model.predict(images.reshape(-1, 100, 100, 3))
    pred_images = predictions.reshape(images.shape[0], -1)

    # neighbors = NearestNeighbors(10)
    # neighbors_fit = neighbors.fit(pred_images)
    # distances, indices = neighbors_fit.kneighbors(pred_images)

    # distances = np.sort(distances, axis=0)
    # distances = distances[:,1]
    # plt.plot(distances)
    # plt.show()
    
    dbmodel = DBSCAN(eps=200, min_samples=1).fit(pred_images)
    labels = dbmodel.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(n_clusters_)
    print(n_noise_)
    print(labels)

    dbpredictions = dbmodel.fit_predict(pred_images)
    shutil.rmtree('output\dbscan')
    os.makedirs("output\dbscan\cluster-1")
    for i in range(n_clusters_):
        os.makedirs("output\dbscan\cluster" + str(i))
    for i in range(len(paths)):
        shutil.copy2(paths[i], "output\dbscan\cluster"+str(dbpredictions[i]))


