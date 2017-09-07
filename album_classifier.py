from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras import regularizers


base_dir = 'data/'

train_dir = os.path.join(base_dir, 'train')
test_dir  = os.path.join(base_dir, 'test')


def extract_features(directory, sample_count):
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 20

    conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

    i = 0
    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch, verbose=1)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch.argmax()
        i += 1
        if i * batch_size >= sample_count:
            break

    return features, labels


train_features, train_labels = extract_features(train_dir, 6500)
test_features, test_labels = extract_features(test_dir, 3000)

train_features = np.reshape(train_features, (-1, 4 * 4 * 512))
test_features = np.reshape(test_features, (-1, 4 * 4 * 512))

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

np.save('npy/train_features.npy', train_features)
np.save('npy/test_features.npy', test_features)
np.save('npy/train_labels.npy', train_labels)
np.save('npy/test_labels.npy', test_labels)


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512, 
		kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))

model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=5,
                    batch_size=20,
		    validation_data=(test_features, test_labels))
