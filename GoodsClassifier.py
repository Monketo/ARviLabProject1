# -*- coding: utf-8 -*-

<<<<<<< HEAD
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
=======
import os

from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from annoy import AnnoyIndex

dataset_path = "/mnt/course/classifieds/"
vectors_path = "data/vectors_resnet/"
features_dict_path = vectors_path + 'feature_dict.npy'

annoy_counter = 0
>>>>>>> Yura


# модель = ResNet50 без голови з одним dense шаром для класифікації об'єктів на nb_classes
def get_model(cls=100):
    feature_extractor = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    flat = Flatten()(feature_extractor.output)
    # можна додати кілька dense шарів:
    # d = Dense(nb_classes*2, activation='relu')(flat)
    # d = Dense(nb_classes, activation='softmax')(d)
    d = Dense(cls, activation='softmax')(flat)
    m = Model(inputs=feature_extractor.input, outputs=d)

    # "заморозимо" всі шари ResNet50, крім кількох останніх
    # базові ознаки згорткових шарів перших рівнів досить універсальні, тому ми не будемо міняти їх ваги
    # кількість шарів, які ми "заморожуємо" - це гіперпараметр
    for layer in m.layers[:-12]:
        layer.trainable = False

    # для finetuning ми використаємо звичайний SGD з малою швидкістю навчання та моментом
    m.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    m.summary()
    return m


<<<<<<< HEAD
# кількість класів, підставте ваше значення
nb_classes = 42

model = get_model(nb_classes=nb_classes)
=======
def get_classes_imageVecs_dict(dataset_path, image_batch_size=64):
    if not os.path.isdir(vectors_path):
        os.makedirs(vectors_path)
        labels_features = {}
        dirs = os.listdir(dataset_path)

        for dir in dirs[:-1]:
            files = os.listdir(dataset_path + dir)
            features = extract_feature_vectors(files, image_batch_size)
            labels_features[dir.name] = features

        np.save(features_dict_path, labels_features)
        annoy.build(n_trees)
        annoy.save('data/resnet_%d.idx' % n_trees)
        return labels_features
    return np.load(features_dict_path).item()


def extract_feature_vectors(imgPaths, batch_size=64):
    resultVectors = list()
    i = 0
    batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float)
    for imgPath in imgPaths:
        img = image.img_to_array(image.load_img(imgPath, target_size=(224, 224)))
        img = preprocess_input(img, data_format=np.float16)
        batch[i] = img
        i += 1
        if i == batch_size:
            i = 0
            batchVectors = model.predict_on_batch(batch).astype(np.float16)
            for vector in batchVectors:
                annoy.add_item(annoy_counter, vector)
                resultVectors.append(vector)
    return np.array(resultVectors)


# кількість класів, підставте ваше значення
nb_classes = 844
vector_size = 2048
n_trees = 256

model = get_model(nb_classes=nb_classes)
annoy = AnnoyIndex(vector_size, metric='angular')
>>>>>>> Yura

# при необхідності завантажити ваги:
# model.load_weights('weights_finetuned.h5')

img_height = 224
img_width = 224
batch_size = 8

# розділити датасет на тренувальний та тестовий
# у пропорції 90/10
train_dir = ''
test_dir = ''

# зробити генератор за рекомендаціями статті:
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

train_gen = ImageDataGenerator(
<<<<<<< HEAD
    rescale=1./255,
=======
    rescale=1. / 255,
>>>>>>> Yura
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    generator=train_generator,
    # validation_data= напишіть генератор для тестових даних
    steps_per_epoch=42,
    nb_epoch=42,
    callbacks=[ModelCheckpoint('weights_finetuned.h5', save_best_only=True, monitor='val_loss')])

<<<<<<< HEAD
model.save_weights('weights_finetuned.h5')
=======
model.save_weights('weights_finetuned.h5')
>>>>>>> Yura
