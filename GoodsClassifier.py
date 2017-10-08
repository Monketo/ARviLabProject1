
# coding: utf-8

# In[11]:


# %load GoodsClassifier.py

import os
# -*- coding: utf-8 -*-
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from annoy import AnnoyIndex


# In[68]:


features_path = "data/"
index_path = features_path + "resnet_256.idx"
pretrained_features_path = features_path + "finetuned_weights.h5"

img_height = 224
img_width = 224
batch_size = 32

nb_train_samples = 611889
nb_validation_samples = 67799

nb_classes = 844

dataset_path = "classifieds/"
train_dir = dataset_path + 'train'
validation_dir = dataset_path + 'validation'


# In[69]:


# модель = ResNet50 без голови з одним dense шаром для класифікації об'єктів на nb_classes
def get_model(nb_classes=844, fine_tune=False, weights_path=None, layers_unfreeze=3, lr=1e-4, epochs=10):
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    
    if not fine_tune:        
        optimizer = Adam(lr=lr)
    else:
        optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
    for layer in model.layers[:-layers_unfreeze]:
        layer.trainable = False
        
    flat = Flatten()(model.output)  
    # можна додати кілька dense шарів:
    d = Dense(2048, activation='relu')(flat)
    d = Dropout(0.2)(d)
    d = Dense(nb_classes, activation='softmax')(d)
    model = Model(inputs=model.input, outputs=d)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    if weights_path:
        model.load_weights(weights_path)
        
    model.summary()
    return model


# In[70]:


def train_model(nb_classes, layers_unfreeze=3, lr=1e-4, fine_tune=False, weights_path=None, epochs=10):

    model = get_model(nb_classes, fine_tune=fine_tune, weights_path=weights_path,
                      layers_unfreeze=layers_unfreeze, lr=lr)
#     train_gen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
        
    validation_gen = ImageDataGenerator(rescale=1. / 255)
    train_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_gen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_gen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    
    model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        validation_steps=500,
        steps_per_epoch=1000,
        nb_epoch=epochs,
        shuffle=True,
        callbacks=[ModelCheckpoint(pretrained_features_path, save_best_only=True, monitor='val_loss')])

    model.save_weights(pretrained_features_path)
    
    return model


# In[75]:


def startTraining(nb_classes):
    train_model(nb_classes, layers_unfreeze=3, lr=1e-3, epochs=20)
    train_model(nb_classes, weights_path=pretrained_features_path,
                layers_unfreeze=8, lr=1e-3, epochs=20)
    model = train_model(nb_classes, fine_tune=True, weights_path=pretrained_features_path, 
                        layers_unfreeze=12, lr=1e-5, epochs=20)
    
    return model    


# In[76]:


model = startTraining(nb_classes)


# In[ ]:


vector_size = 2048
n_trees = 256


# In[ ]:


def predict_n_neighbours(targetImagePath, topn=5):
    annoy = AnnoyIndex(vector_size, metric='angular')
    annoy.load(index_path)
    


# In[56]:


