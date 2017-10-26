import os
import nearest_neighbours 
import numpy as np
np.random.seed(123)
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras import regularizers
from keras.applications import VGG16


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
    labels = np.zeros(shape=(sample_count, 10))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
       
        features_batch = conv_base.predict(inputs_batch, verbose=1)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break

    return features, labels, generator.class_indices

def net(outshape):

        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(outshape, activation='softmax'))

        model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])

        return model


if __name__ == '__main__':

        train_features, train_labels, classes = extract_features(train_dir, 6000)
        test_features, test_labels, classes = extract_features(test_dir, 3000)

        train_features = np.reshape(train_features, (-1, 4 * 4 * 512))
        test_features = np.reshape(test_features, (-1, 4 * 4 * 512))

        model = net(train_labels.shape[1])
        history = model.fit(train_features, train_labels,
                        epochs=20,
                        batch_size=64,
                        validation_data=(test_features, test_labels))

        classes = dict([[v,k] for k,v in classes.items()])
        preds = model.predict(test_features)

        nearest_neighbours.confusion([classes[p] for p in preds.argmax(1)], 
                        [classes[t] for t in test_labels.argmax(1)])
