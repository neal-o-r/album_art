import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import confusion_matrix


img_width, img_height = 350, 350

train_data_dir = 'data/train'
test_data_dir  = 'data/test'
nb_train_samples = 6755
nb_validation_samples = 3058
epochs = 25
batch_size = 16


def create_bottlebeck_features():

	datagen = ImageDataGenerator(rescale=1/255.)

	model = applications.VGG16(include_top=False, weights='imagenet')

	generator = datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)

	print('Predicting training features...')    
	bottleneck_features_train = model.predict_generator(
		generator, nb_train_samples // batch_size, verbose=1)

	print('Saving...')
	
	np.save('bottleneck_features_train.npy',
		bottleneck_features_train)

	train_labels = to_categorical(generator.classes)


	generator = datagen.flow_from_directory(
		test_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)

	test_labels = to_categorical(generator.classes)
	
	print('Predicting test features...')
	bottleneck_features_validation = model.predict_generator(
		generator, nb_validation_samples // batch_size, verbose=1)
	
	
	print('Saving...')
	np.save('bottleneck_features_validation.npy',
		bottleneck_features_validation)

	return train_labels, test_labels


def train_model(train_labels, test_labels):

	print('Loading features...')
	train_data = np.load('bottleneck_features_train.npy')

	test_data = np.load('bottleneck_features_validation.npy')

	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))

	rmsprop = optimizers.RMSprop(lr=0.01)
	model.compile(optimizer=rmsprop,
		  loss='categorical_crossentropy', metrics=['accuracy'])

	print('Fitting Model.')
	model.fit(train_data, train_labels,
	      epochs=epochs,
	      batch_size=batch_size)
	
	p = model.predict_proba(test_data)
	
	confusion(p, test_labels)

	return model


def confusion(y_p, y_t):

	classes = [i.split('/')[-1].split('_')[0] 
			for i in glob.glob(test_data_dir+'/*')]

	cij = confusion_matrix(y_p.argmax(1), y_t.argmax(1))
	
	cij = (cij.T / cij.sum(1)).T 

	df = pd.DataFrame(cij, index=classes, columns=classes)

	sns.set(font_scale=0.7)
	sns.heatmap(df, annot=True, linewidths=0.2)

	plt.show()


train_labels, test_labels = create_bottlebeck_features()
m = train_model(train_labels, test_labels)








