# BIRCH

from sklearn.cluster import Birch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil
import pandas as pd


input_dir = 'pets'
glob_dir = input_dir + '/*.jpg'

images = [cv2.resize(cv2.imread(file), (128, 128)) for file in glob.glob(glob_dir)]
paths = [file for file in glob.glob(glob_dir)]
images = np.array(np.float32(images).reshape(len(images), -1)/255)

brc = Birch(n_clusters=6)
brc.fit(images)
brc.predict(images)
print(brc.labels_)

sil = []
b1 = []

nmax = 6

for n in range(2, nmax+1):
  brc = Birch(n_clusters = n).fit(images)
  labels = brc.labels_
  print("k="+str(n))
  print(labels)
  sil.append(silhouette_score(images, labels, metric = 'euclidean'))
  b1.append(n)

plt.plot(b1, sil)
plt.ylabel('Silhoutte Score')
plt.ylabel('N')
plt.show()

n = 5
brcmodel = Birch(n_clusters = 5)
brcmodel.fit(images)
brcpredictions = brcmodel.predict(images)

frame = pd.DataFrame(images)
frame['cluster'] = brcpredictions
print(frame['cluster'].value_counts())

shutil.rmtree('./output/birch')
cluster_cnt = {}
for i in range(n):
  os.makedirs("./output/birch/cluster" + str(i))  
  cluster_cnt[i] = 0

for i in range(len(paths)): 
  shutil.copy2(paths[i], "./output/birch/cluster"+str(brcpredictions[i]))
  cluster_cnt[brcpredictions[i]] += 1

print(cluster_cnt)
