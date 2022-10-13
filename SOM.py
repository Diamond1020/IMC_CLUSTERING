#SOM

from sklearn_som.som import SOM
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil
 
input_dir = 'pets'
glob_dir = input_dir + '/*.jpg'

images = [cv2.resize(cv2.imread(file), (128, 128)) for file in glob.glob(glob_dir)]
paths = [file for file in glob.glob(glob_dir)]
images = np.array(np.float32(images).reshape(len(images), -1)/255)

model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
predictions = model.predict(images.reshape(-1, 128, 128, 3))
pred_images = predictions.reshape(images.shape[0], -1)
dim = len(pred_images[0])

kmax = 6
sil = []
k1 = []
for k in range(2, kmax+1):
    iris_som = SOM(k,1,dim)
    iris_som.fit(pred_images)   
    predictions = iris_som.predict(pred_images)
    sil.append(silhouette_score(pred_images, predictions, metric = 'euclidean'))
    k1.append(k)

plt.plot(k1, sil)
plt.ylabel('Silhoutte Score')
plt.ylabel('K')
plt.show()

k = 6
iris_som = SOM(6,1,dim)
iris_som.fit(pred_images)
predictions = iris_som.predict(pred_images)

frame = pd.DataFrame(pred_images)
frame['cluster'] = predictions
print(frame['cluster'].value_counts())

shutil.rmtree('output/som')
for i in range(k):
	os.makedirs("output/som/cluster" + str(i))
for i in range(len(paths)):
	shutil.copy2(paths[i], "output/som/cluster"+str(predictions[i]))
