import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil

input_dir = 'pets'
glob_dir = input_dir + '/*.jpg'

images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
paths = [file for file in glob.glob(glob_dir)]
images = np.array(np.float32(images).reshape(len(images), -1)/255)

model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
predictions = model.predict(images.reshape(-1, 224, 224, 3))
pred_images = predictions.reshape(images.shape[0], -1)

sil = []
kl = []

kmax = 6

for k in range(2, kmax+1):
  kmeans2 = KMeans(n_clusters = k).fit(pred_images)
  labels = kmeans2.labels_
  print(labels)
  sil.append(silhouette_score(pred_images, labels, metric = 'euclidean'))
  kl.append(k)

plt.plot(kl, sil)
plt.ylabel('Silhoutte Score')
plt.ylabel('K')
plt.show()

# You can decide the optimal value of K. For the pets example, its 2.

k = 5
kmodel = KMeans(n_clusters=k, n_jobs=-1, random_state=728)
kmodel.fit(pred_images)
kpredictions = kmodel.predict(pred_images)
shutil.rmtree('output\kmean')
for i in range(k):
	os.makedirs("output\kmean\cluster" + str(i))
for i in range(len(paths)):
	shutil.copy2(paths[i], "output\kmean\cluster"+str(kpredictions[i]))