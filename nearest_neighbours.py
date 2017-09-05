import glob
import cv2
import numpy as np
import annoy
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



train_dir = 'data/train/'
test_dir = 'data/test/'

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
 
	cv2.normalize(hist, hist)
 
	# return the flattened histogram as the feature vector
	return hist.flatten()


def get_data(train_dir, test_dir):

	train = glob.glob(train_dir+'*')

	train_classes = []
	train_matrix = []
	for d in train:
		cl = d.split('/')[-1]
	
		for f in glob.glob(d+'/*'):
			train_classes.append(cl)
			train_matrix.append(extract_color_histogram(f))

	train_matrix = np.asarray(train_matrix)						

	np.save('train_matrix.npy', train_matrix)

	test = glob.glob(test_dir+'*')

	test_classes = []
	test_matrix = []
	for d in train:
		cl = d.split('/')[-1]
	
		for f in glob.glob(d+'/*'):
			test_classes.append(cl)
			test_matrix.append(extract_color_histogram(f))

	test_matrix = np.asarray(test_matrix)						

	np.save('test_matrix.npy', test_matrix)

	return train_classes, test_classes


def nearest_neighbours(train_labels, test_labels):

	train_data = np.load('train_matrix.npy')
	test_data  = np.load('test_matrix.npy')

	ann = annoy.AnnoyIndex(512, metric='euclidean')
	for i, v in enumerate(train_data):
		ann.add_item(i, v.ravel())

	ann.build(20)

	acc = 0
	preds = []
	for i, t in enumerate(test_data):

		true_class = test_labels[i]
		pred = Counter(train_labels[i] for i in 
			ann.get_nns_by_vector(t, 10)).most_common()[0][0]
	
		preds.append(pred)	
		if pred == true_class:
			acc += 1

	print('Accuracy {}'.format(acc / len(test_data))) 

	confusion(preds, test_labels)


def confusion(y_p, y_t):

	le = preprocessing.LabelEncoder()
	
	y_p = le.fit_transform(y_p)
	y_t = le.fit_transform(y_t)
	
	cij = confusion_matrix(y_p, y_t)
	cij = (cij.T / cij.sum(1)).T 

	df = pd.DataFrame(cij, index=le.classes_, columns=le.classes_)

	sns.set(font_scale=0.7)
	sns.heatmap(df, annot=True, linewidths=0.2)

	plt.show()



train_cl, test_cl = get_data(train_dir, test_dir)
nearest_neighbours(train_cl, test_cl)
