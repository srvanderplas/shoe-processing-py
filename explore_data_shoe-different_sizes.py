#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install gast


# In[2]:


import os
import glob
from collections import Counter
from random import sample
import numpy as  np
import random
from sklearn.preprocessing import MultiLabelBinarizer
import progressbar
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
# import gast
# from tf.keras.preprocessing.image import ImageDataGenerator


# In[3]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[4]:


tf.__version__


# In[5]:


image_path = r"D:\Shoes\shoe_data\data_all\images1\images"


# In[6]:


os.chdir(image_path)


# In[7]:


os.getcwd()


# In[8]:


folders = os.listdir()


# In[9]:


# write a function to get the class labels
# write a function that will take in the file paths
# and give the class distributions

def get_class_labels(obj):
    classes_to_keep = ["bowtie", "circle","line","quad","triangle","chevron","other","star","text","polygon"]
    classes = [i.replace("(E)","").replace("(R)","").replace("logo","other").split("-")[0].split("_") for i in obj]
    classes = [tuple(np.array(i)) for i in classes if len(set(i).intersection(set(classes_to_keep))) == len(i)]
    return(classes)


# In[10]:


all_labels = get_class_labels(folders)


# In[11]:


mlb = MultiLabelBinarizer()


# In[12]:


all_labels_binarized =  mlb.fit_transform(all_labels)


# In[13]:


shoe_names = set([i.split("-")[-1] for i in folders])


# In[14]:


shoe_names = set([i.split("-")[-1] for i in folders])


# In[15]:


shoe_names


# In[16]:


len(shoe_names)


# In[17]:


# randomly get some shoes in train, test, and validation
random.seed(42)
train_files = sample(set(shoe_names), k = round(len(shoe_names)*0.6))


# In[18]:


# get the rest
valid_and_test = [i for i in shoe_names if i not in train_files]


# In[19]:


# validation files
valid_files = sample(set(valid_and_test), k = round(len(valid_and_test)*0.5))


# In[20]:


# test files
test_files = [i for i in valid_and_test if i not in valid_files]


# In[21]:


# .replace("(E)","").replace("(R)","").replace("logo","other")


# In[22]:


folders


# In[23]:


# get the respective images
train_images = [i for i in folders if i.split("-")[-1] in train_files]


# In[24]:


test_images = [i for i in folders if i.split("-")[-1] in test_files]


# In[25]:


valid_images = [i for i in folders if i.split("-")[-1] in valid_files]


# In[26]:


# indexes
train_index = [i for i,n in enumerate(folders) if n.split("-")[-1] in train_files]


# In[27]:


test_index = [i for i,n in enumerate(folders) if n.split("-")[-1] in test_files]


# In[28]:


valid_index = [i for i,n in enumerate(folders) if n.split("-")[-1] in valid_files]


# In[29]:


train_labels = [n for i,n in enumerate(all_labels_binarized) if i in train_index]


# In[30]:


test_labels = [n for i,n in enumerate(all_labels_binarized) if i in test_index]


# In[31]:


valid_labels = [n for i,n in enumerate(all_labels_binarized) if i in valid_index]


# In[32]:


# should now get the class level information


# In[33]:


# write a function that will take in the file paths
# and give the class distributions

def get_class_distribution(obj):
    classes_to_keep = ["bowtie", "circle","line","quad","triangle","chevron","other","star","text","polygon"]
    classes = [i.replace("(E)","").replace("(R)","").replace("logo","other").split("-")[0].split("_") for i in obj]
    classes = [i for i in classes if len(set(i).intersection(set(classes_to_keep))) >= 1]
    return(Counter([j for i in classes for j in i if j in classes_to_keep]))


# In[34]:


get_class_distribution(train_images)


# In[35]:


get_class_distribution(test_images)


# In[36]:


get_class_distribution(valid_images)


# In[37]:


get_class_distribution(folders)


# In[38]:


# make dataframe objects for train, test, and valid

# write a function to get the class labels
# write a function that will take in the file paths
# and give the class distributions

def get_class_labels(obj):
    classes_to_keep = ["bowtie", "circle","line","quad","triangle","chevron","other","star","text","polygon"]
    classes = [i.replace("(E)","").replace("(R)","").replace("logo","other").split("-")[0].split("_") for i in obj]
    paths = [n for i,n in enumerate(obj) if len(set(classes[i]).intersection(set(classes_to_keep))) == len(classes[i])]
    classes = [list(np.array(i)) for i in classes if len(set(i).intersection(set(classes_to_keep))) == len(i)]
    if len(paths) == len(classes):
        dataframe_object = {"filename": paths, "labels": classes}
        df_object = pd.DataFrame(dataframe_object)
        return(df_object)
    else:
        print("the lengths were not equal")


# In[39]:


get_class_labels(train_images)


# In[40]:


img_gen = ImageDataGenerator(rescale=1/255,samplewise_std_normalization = True,
    rotation_range = 40,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    shear_range = 60,
    zoom_range = 0.1,
    channel_shift_range = .1,
    zca_whitening = True,
    vertical_flip = True,
    horizontal_flip = True)


# In[41]:


img_iter = img_gen.flow_from_dataframe(
    get_class_labels(train_images),
    shuffle=True,
    directory= r"D:\Shoes\shoe_data\data_all\images1\images",
    x_col='filename',
    y_col='labels',
    class_mode='categorical',
    target_size=(None, None),
    batch_size=1
)


# In[42]:


img_iter_val = img_gen.flow_from_dataframe(
    get_class_labels(valid_images),
    shuffle=False,
    directory=r"D:\Shoes\shoe_data\data_all\images1\images",
    x_col='filename',
    y_col='labels',
    class_mode='categorical',
    target_size=(None, None),
    batch_size=1
)


# In[43]:


import keras


# In[44]:


from tensorflow.keras.applications.vgg16 import VGG16


# In[49]:


from tensorflow.keras.layers import Dense, Flatten, GlobalMaxPool2D
from tensorflow.keras.models import Model


# In[46]:


# base_model = ResNet50(
#     include_top=False,
#     weights='imagenet',
#     input_shape=None,
#     pooling='avg'
# )

# base_model = tf.keras.applications.VGG16(input_shape=(256, 256, 3),
#                                                include_top=False,
#                                                weights='imagenet')


# for layer in base_model.layers:
#     layer.trainable = False

# predictions = Dense(10, activation='sigmoid')(base_model.output)
# model = Model(inputs=base_model.input, outputs=predictions)


# In[47]:


# baseModel = VGG16(input_shape=(256, 256, 3),include_top=False,weights='imagenet')

# headModel = baseModel.output
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(256, activation="relu")(headModel)
# # headModel = Dropout(0.5)(headModel)

# # add a softmax layer
# headModel = Dense(10, activation="sigmoid")(headModel)
# model = Model(inputs=baseModel.input, outputs=headModel)

# # loop over all layers in the base model and freeze them so they
# # will *not* be updated during the training process
# for layer in baseModel.layers:
#     layer.trainable = False
# # opt = RMSprop(lr=0.001)
# # opt = SGD(lr=0.01, decay=0.01 / 100, momentum=0.9, nesterov=True)
# # model.compile(optimizer=opt,
# #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
# #               weighted_metrics=['accuracy'])
# base_learning_rate = 0.0001
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
#               loss= 'binary_crossentropy',
#             metrics = [tf.keras.metrics.AUC(multi_label = True)])


# In[51]:


baseModel = VGG16(input_shape=(None, None, 3),include_top=False,weights='imagenet')

headModel = baseModel.output
headModel = GlobalMaxPool2D()(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dense(256, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)

# add a softmax layer
headModel = Dense(10, activation="sigmoid")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False
# opt = RMSprop(lr=0.001)
# opt = SGD(lr=0.01, decay=0.01 / 100, momentum=0.9, nesterov=True)
# model.compile(optimizer=opt,
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               weighted_metrics=['accuracy'])
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss= 'binary_crossentropy',
            metrics = [tf.keras.metrics.AUC(multi_label = True)])


# In[ ]:


# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
#               loss= 'binary_crossentropy',
#             metrics = [tf.keras.metrics.AUC()])


# In[ ]:


# model.compile(
#     loss='binary_crossentropy',
#     optimizer='rmsprop', metrics = [tf.keras.metrics.AUC(multi_label = True)]
# )


# In[52]:


img_metadata = get_class_labels(train_images)


# In[53]:


all_labels = [label for lbs in img_metadata['labels'] for label in lbs]
labels_count = Counter(all_labels)


# In[54]:


total_counts = sum(labels_count.values())
class_weights = {img_iter.class_indices[cls]: total_counts / count for cls, count in labels_count.items()}
class_weights


# In[55]:


STEP_SIZE_TRAIN=img_iter.n//img_iter.batch_size
STEP_SIZE_VALID=img_iter_val.n//img_iter_val.batch_size
# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


# In[56]:


es = EarlyStopping(monitor='val_auc', mode='max', verbose=1,patience=2, restore_best_weights = True)


# In[57]:


model.fit_generator(generator=img_iter,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=img_iter_val,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=200, class_weight = class_weights, callbacks = [es]
)


# In[ ]:


for layer in baseModel.layers[-3:]: 
    layer.trainable =  True


# In[ ]:


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss= 'binary_crossentropy',
            metrics = [tf.keras.metrics.AUC(multi_label = True)])


# In[ ]:


# img_gen = ImageDataGenerator(rescale=1/255,samplewise_std_normalization = True,
#     rotation_range = 40,
#     width_shift_range = 0.05,
#     height_shift_range = 0.05,
#     shear_range = 60,
#     zoom_range = 0.1,
#     channel_shift_range = .1,
#     zca_whitening = True,
#     vertical_flip = True,
#     horizontal_flip = True)

# img_iter = img_gen.flow_from_dataframe(
#     get_class_labels(train_images),
#     shuffle=True,
#     directory= r"D:\Shoes\shoe_data\data_all\images1\images",
#     x_col='filename',
#     y_col='labels',
#     class_mode='categorical',
#     target_size=(256, 256),
#     batch_size=64
# )

# img_iter_val = img_gen.flow_from_dataframe(
#     get_class_labels(valid_images),
#     shuffle=False,
#     directory=r"D:\Shoes\shoe_data\data_all\images1\images",
#     x_col='filename',
#     y_col='labels',
#     class_mode='categorical',
#     target_size=(256, 256),
#     batch_size=64
# )


# In[ ]:


STEP_SIZE_TRAIN=img_iter.n//img_iter.batch_size
STEP_SIZE_VALID=img_iter_val.n//img_iter_val.batch_size
# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


# In[ ]:


es = EarlyStopping(monitor='val_auc', mode='max', verbose=1,patience=20, restore_best_weights = True)


# In[ ]:


model.fit_generator(generator=img_iter,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=img_iter_val,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=200, class_weight = class_weights, callbacks = [es]
)

