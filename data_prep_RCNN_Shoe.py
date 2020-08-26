#!/usr/bin/env python
# coding: utf-8

# In[149]:


import os
import glob
from collections import Counter
from random import sample
import numpy as  np
import random
# from sklearn.preprocessing import MultiLabelBinarizer
# import progressbar
import cv2
import pandas as pd
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.callbacks import EarlyStopping
# import gast
# from tf.keras.preprocessing.image import ImageDataGenerator


# In[150]:


tf.__version__


# In[3]:


import keras


# In[4]:


from tensorflow.keras.applications.vgg19 import VGG19


# In[5]:


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# In[6]:


from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Model


# In[7]:


physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], False)


# In[8]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[9]:


import pandas as pd


# In[10]:


from tensorflow.keras.applications.xception import Xception, preprocess_input


# In[11]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)


# In[ ]:


# read in the proposal file
proposals = pd.read_csv(r"D:\Shoes\Latest_Iteration\Data\bbox_data.csv")


# In[ ]:


proposals["class"].value_counts()


# In[ ]:


proposals.shape


# In[ ]:


# can we work with just the positives
# how to select negative data

positives = proposals[proposals.iou >= 0.6]


# In[ ]:


positives.shape


# In[ ]:


frequencies = positives["class"].value_counts()


# In[ ]:


frequencies[:50]


# In[ ]:


frequencies.index


# In[ ]:


import numpy as np


# In[ ]:


# check unique classes
np.unique([j for i  in frequencies.index for j in i.split(",")])


# In[ ]:


# define the correct classes
correct_classes = ["bowtie", "chevron", "circle","quadrilateral", "triangle", "hexagon", "pentagon", "octagon", "polygon", "other", "ribbon",
                   "star", "text", "quad", "logo", "line"]


# In[ ]:


# retain only these
correct_indices = [i for i,n in enumerate(positives["class"]) if set(n.split(",")).issubset(set(correct_classes))]


# In[ ]:


positives_correct = positives.iloc[correct_indices,:]


# In[ ]:


positives_correct.shape


# In[ ]:


positives_correct.head()


# In[ ]:


positives_correct_crop_out = positives_correct.iloc[:,[0,2,-8,-7,-6,-5]]


# In[ ]:


positives_correct_crop_out.head()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


from tqdm.notebook import tqdm


# In[ ]:


# make a dataframe from positives_correct that has the filenames
# and the label in the form of text
# that is crop the image, and store it somewhere
images_path = r"D:\Shoes\Shoes_with_annotations\\"
save_path = r"D:\Shoes\Latest_Iteration\Data_RCNN\\"
counter = 0
catch_dict = []
for i in tqdm(positives_correct_crop_out.iterrows()):
    counter = counter + 1
    filename = i[1][0]
    image1 = plt.imread(images_path + filename)
    class_label = i[1][1]
    xmin, ymin, xmax, ymax = i[1][2], i[1][3], i[1][4], i[1][5]
    roi =image1[ymin:ymax,xmin:xmax].copy()
    plt.imsave(save_path + filename.split(".")[0] + "_"  + str(counter) +  "_" + class_label + ".jpg", roi)
    dict1 = {"shoe_name": save_path + filename.split(".")[0] + "_"  + str(counter) +  "_" + class_label + ".jpg",
             "class_label": class_label}
    catch_dict.append(dict1)


# In[ ]:


shoe_cropped_images = pd.DataFrame(catch_dict)


# In[ ]:


shoe_cropped_images.head()


# In[ ]:


from random import sample


# In[ ]:


shoe_cropped_images["shoe_name"][0].split("_")


# In[ ]:


shoe_cropped_images["name"] = ["_".join(i.split("_")[:-2]) for i in shoe_cropped_images["shoe_name"]]


# In[ ]:


len(np.unique(shoe_cropped_images["name"]))


# In[ ]:


random.seed(42)


# In[ ]:


train_shoes = sample(set(shoe_cropped_images.name), k = round(0.6*len(set(shoe_cropped_images.name))))


# In[ ]:


test_shoes = set(shoe_cropped_images.name).difference(train_shoes)


# In[ ]:


len(train_shoes)


# In[ ]:


len(test_shoes)


# In[ ]:


len(train_shoes) + len(test_shoes)


# In[ ]:


# test_shoes = [i for i in shoe_cropped_images.shoe_name if i not in train_shoes]


# In[12]:


train_data = shoe_cropped_images[shoe_cropped_images.name.isin(train_shoes)]


# In[ ]:


test_data = shoe_cropped_images[shoe_cropped_images.name.isin(test_shoes)]


# In[ ]:


# from sklearn.model_selection import train_test_split


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(shoe_cropped_images["shoe_name"], shoe_cropped_images["class_label"],
#                                                     test_size=0.25, random_state=42)


# In[ ]:


# train_data = pd.DataFrame({"shoe_name" : X_train, 
#                           "class_label" : y_train})


# In[ ]:


# test_data = pd.DataFrame({"shoe_name" : X_test, 
#                           "class_label" : y_test})


# In[ ]:


train_data["class_label"] = [cl.replace(" ", "").split(",") for cl in train_data["class_label"]]


# In[ ]:


test_data["class_label"] = [cl.replace(" ", "").split(",") for cl in test_data["class_label"]]


# In[ ]:


catch_catch_j = []
for i in train_data["class_label"]: 
    catch_j = []
    for j in i: 
        if j == "quad": 
            j = "quadrilateral"
        if j in ["polygon", "hexagon", "octagon", "pentagon"]: 
            j = "polygon"
        if j in ["logo", "text"]: 
            j = "text"
        else:
            j = j
        catch_j.append(j)
    catch_catch_j.append(catch_j)


# In[ ]:


np.unique([j for i in catch_catch_j for j in i ])


# In[ ]:


len(np.unique([j for i in catch_catch_j for j in i ]))


# In[ ]:


train_data["class_label"] = catch_catch_j


# In[ ]:


catch_catch_j = []
for i in test_data["class_label"]: 
    catch_j = []
    for j in i: 
        if j == "quad": 
            j = "quadrilateral"
        if j in ["polygon", "hexagon", "octagon", "pentagon"]: 
            j = "polygon"
        if j in ["logo", "text"]: 
            j = "text"
        else:
            j = j
        catch_j.append(j)
    catch_catch_j.append(catch_j)


# In[ ]:


np.unique([j for i in catch_catch_j for j in i ])


# In[ ]:


len(np.unique([j for i in catch_catch_j for j in i ]))


# In[ ]:


test_data["class_label"] = catch_catch_j


# In[ ]:


train_data["shoe_name"] = [j.split("\\")[-1] for j in train_data["shoe_name"]]


# In[ ]:


train_data = train_data.reset_index(drop = True)


# In[ ]:


train_data.head()


# In[ ]:


test_data["shoe_name"] = [j.split("\\")[-1] for j in test_data["shoe_name"]]


# In[ ]:


test_data = test_data.reset_index(drop = True)


# In[ ]:


test_data.head()


# In[ ]:


set(train_data.name).intersection(set(test_data.name))


# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


# train_data.to_csv(r"D:\Shoes\Latest_Iteration\Data\train_data.csv", index = False)


# In[ ]:


# test_data.to_csv(r"D:\Shoes\Latest_Iteration\Data\test_data.csv", index = False)


# In[13]:


train_data = pd.read_csv(r"D:\Shoes\Latest_Iteration\Data\train_data.csv",  converters={'class_label': eval})


# In[ ]:


train_data.head()


# In[14]:


test_data = pd.read_csv(r"D:\Shoes\Latest_Iteration\Data\test_data.csv",  converters={'class_label': eval})


# In[ ]:


test_data.head()


# In[15]:


img_gen = ImageDataGenerator(preprocessing_function = preprocess_input,
    samplewise_std_normalization = True,
    rotation_range = 40,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    shear_range = 60,
    zoom_range = 0.1,
    channel_shift_range = .1,
    zca_whitening = True,
    vertical_flip = True,
    horizontal_flip = True)


# In[16]:


img_iter = img_gen.flow_from_dataframe(
    train_data,
    shuffle=True,
    directory= r"D:\Shoes\Latest_Iteration\Data_RCNN",
    x_col='shoe_name',
    y_col='class_label',
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=64
)


# In[ ]:


img_iter_val = img_gen.flow_from_dataframe(
    test_data,
    shuffle=True,
    directory=r"D:\Shoes\Latest_Iteration\Data_RCNN",
    x_col='shoe_name',
    y_col='class_label',
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=64
)


# In[ ]:


all_labels = [label for lbs in train_data['class_label'] for label in lbs]
labels_count = Counter(all_labels)


# In[ ]:


img_iter.class_indices


# In[ ]:


total_counts = sum(labels_count.values())
class_weights = {img_iter.class_indices[cls]: total_counts / count for cls, count in labels_count.items()}
class_weights


# In[ ]:


baseModel = VGG19(input_shape=(256, 256, 3),include_top=False,weights='imagenet')


# baseModel = Xception(input_shape=(256, 256, 3),include_top=False,weights='imagenet')


headModel = baseModel.output

headModel = Flatten(name="flatten")(headModel)

# headModel = BatchNormalization()(headModel)
# headModel = Dropout(0.5)(headModel)

headModel = Dense(1024, activation="relu")(headModel)
headModel = BatchNormalization()(headModel)
# headModel = Dropout(0.5)(headModel)

headModel = Dense(1024, activation="relu")(headModel)
headModel = BatchNormalization()(headModel)
# headModel = Dropout(0.5)(headModel)

# headModel = Dense(2048, activation="relu")(headModel)
# headModel = BatchNormalization()(headModel)
# headModel = BatchNormalization()(headModel)
# headModel = Dropout(0.5)(headModel)

# headModel = Dense(1024, activation="relu")(headModel)
# headModel = BatchNormalization()(headModel)

# headModel = Dropout(0.5)(headModel)
# headModel = Dense(4096, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2048, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2048, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)

# add a softmax layer
headModel = Dense(11, activation="sigmoid")(headModel)
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
base_learning_rate = 1e-5
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss= 'binary_crossentropy',
            metrics = [tf.keras.metrics.AUC(multi_label = True)])


# In[ ]:


STEP_SIZE_TRAIN=img_iter.n//img_iter.batch_size
STEP_SIZE_VALID=img_iter_val.n//img_iter_val.batch_size
# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

es = EarlyStopping(monitor='val_auc_3', mode='max', verbose=1,patience=5, restore_best_weights = True)

# model = tf.keras.models.load_model(r"D:\Shoes\pretrained_1.h5")  


# In[ ]:


model.fit_generator(generator=img_iter,
                    steps_per_epoch=100,
                    validation_data=img_iter_val,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=200, class_weight = class_weights, callbacks = [es], verbose = 1
)


# In[ ]:


# model.fit_generator(generator=img_iter,
#                     steps_per_epoch=STEP_SIZE_TRAIN,
#                     validation_data=img_iter_val,
#                     validation_steps=STEP_SIZE_VALID,
#                     epochs=200, class_weight = class_weights, callbacks = [es], verbose = 1
# )


# In[ ]:


# model.save("D:\Shoes\Latest_Iteration\Models\connor_1.h5")


# In[ ]:


model.save("D:\Shoes\Latest_Iteration\Models\connor_1_quick.h5")


# In[ ]:


model = tf.keras.models.load_model("D:\Shoes\Latest_Iteration\Models\connor_1.h5")


# In[ ]:


baseModel = VGG16(input_shape=(256, 256, 3),include_top=False,weights='imagenet')

headModel = baseModel.output

headModel = Flatten(name="flatten")(headModel)

# headModel = BatchNormalization()(headModel)
# headModel = Dropout(0.5)(headModel)

headModel = Dense(1024, activation="relu")(headModel)
headModel = BatchNormalization()(headModel)
headModel = Dropout(0.5)(headModel)

headModel = Dense(1024, activation="relu")(headModel)
headModel = BatchNormalization()(headModel)
headModel = Dropout(0.5)(headModel)

# headModel = Dense(2048, activation="relu")(headModel)
# headModel = BatchNormalization()(headModel)
# headModel = BatchNormalization()(headModel)
# headModel = Dropout(0.5)(headModel)

# headModel = Dense(1024, activation="relu")(headModel)
# headModel = BatchNormalization()(headModel)

# headModel = Dropout(0.5)(headModel)
# headModel = Dense(4096, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2048, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2048, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)

# add a softmax layer
headModel = Dense(11, activation="sigmoid")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers[-4:]:
    layer.trainable = True
# opt = RMSprop(lr=0.001)
# opt = SGD(lr=0.01, decay=0.01 / 100, momentum=0.9, nesterov=True)
# model.compile(optimizer=opt,
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               weighted_metrics=['accuracy'])
base_learning_rate = 1e-6
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss= 'binary_crossentropy',
            metrics = [tf.keras.metrics.AUC(multi_label = True)])


# In[ ]:


model = tf.keras.models.load_model("D:\Shoes\Latest_Iteration\Models\connor_1.h5")


# In[ ]:


img_iter = img_gen.flow_from_dataframe(
    train_data,
    shuffle=True,
    directory= r"D:\Shoes\Latest_Iteration\Data_RCNN",
    x_col='shoe_name',
    y_col='class_label',
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=32
)


# In[ ]:


img_iter_val = img_gen.flow_from_dataframe(
    test_data,
    shuffle=True,
    directory=r"D:\Shoes\Latest_Iteration\Data_RCNN",
    x_col='shoe_name',
    y_col='class_label',
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=32
)


# In[ ]:


all_labels = [label for lbs in train_data['class_label'] for label in lbs]
labels_count = Counter(all_labels)


# In[ ]:


total_counts = sum(labels_count.values())
class_weights = {img_iter.class_indices[cls]: total_counts / count for cls, count in labels_count.items()}
class_weights


# In[ ]:


STEP_SIZE_TRAIN=img_iter.n//img_iter.batch_size
STEP_SIZE_VALID=img_iter_val.n//img_iter_val.batch_size


# In[ ]:


es = EarlyStopping(monitor='val_auc', mode='max', verbose=1,patience=2, restore_best_weights = True)


# In[ ]:


model.fit_generator(generator=img_iter,
                    steps_per_epoch=100,
                    validation_data=img_iter_val,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=200, class_weight = class_weights, callbacks = [es], verbose = 1
)


# In[ ]:


model.save("D:\Shoes\Latest_Iteration\Models\connor_4.h5")


# In[ ]:


model = tf.keras.models.load_model("D:\Shoes\Latest_Iteration\Models\connor_4.h5")


# In[ ]:


for layer in baseModel.layers[-15:]:
    layer.trainable = True
# opt = RMSprop(lr=0.001)
# opt = SGD(lr=0.01, decay=0.01 / 100, momentum=0.9, nesterov=True)
# model.compile(optimizer=opt,
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               weighted_metrics=['accuracy'])
base_learning_rate = 1e-6
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss= 'binary_crossentropy',
            metrics = [tf.keras.metrics.AUC(multi_label = True)])


# In[ ]:


es = EarlyStopping(monitor='val_auc_4', mode='max', verbose=1,patience=5, restore_best_weights = True)


# In[ ]:


model.fit_generator(generator=img_iter,
                    steps_per_epoch=100,
                    validation_data=img_iter_val,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=200, class_weight = class_weights, callbacks = [es], verbose = 1
)


# In[ ]:


model.save("D:\Shoes\Latest_Iteration\Models\connor_5_quick.h5")


# In[ ]:


# model.save("D:\Shoes\Latest_Iteration\Models\connor_5.h5")


# In[ ]:


# img_iter_val.filepaths


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


file_names = glob.glob("D:\Shoes\Shoes_with_annotations\*.jpg")


# In[19]:


import random


# In[20]:


index = random.sample(set(file_names),k = 1)[0]


# In[ ]:


index


# In[21]:


# read one file and do selective search
img_test = plt.imread(index)


# In[22]:


plt.imshow(img_test)


# In[23]:


def get_rpn(filename,method = "quality"): 
#     base_path = r"D:\Shoes\Shoes_with_annotations\\"
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#     base_path = r"D:\Shoes\Shoes_with_annotations\\"
    filename = filename
    samp_path = filename
    im = cv2.imread(samp_path)
    ss.setBaseImage(im)
    if method == "fast":
        print(method)
        ss.switchToSelectiveSearchFast()
    if method == "quality":
        print(method)
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    coords = []
    for (x, y, w, h) in rects: 
        xmin, ymin, xmax, ymax = x,y,x+w, y+h
        coords.append([xmin, ymin, xmax, ymax])
    return(coords)


# In[24]:


coords = get_rpn(index,method = "fast")


# In[25]:


len(coords)


# In[26]:


from scipy import ndimage, misc


# In[27]:


from PIL import Image


# In[28]:


im = Image.open(index)


# In[29]:


image_array = np.zeros((len(coords), 256,256,3))


# In[30]:


image_array.shape


# In[31]:


from tqdm.notebook import tqdm


# In[32]:


counter = 0
for i in tqdm(coords):
    xmin, ymin, xmax, ymax = i[0], i[1], i[2], i[3]
    cropped = np.array(im)[ymin:ymax,xmin:xmax]
    cropped = Image.fromarray(cropped)
    cropped = cropped.resize((256,256),Image.ANTIALIAS)
    
    cropped = np.array(cropped)
    cropped = cropped/255.0
    image_array[counter] = cropped
    counter =counter + 1
#     print(cropped.shape)
#     plt.imshow(cropped)
#     plt.show()


# In[33]:


image_array.shape


# In[ ]:


# del(image_array)


# In[34]:


model = tf.keras.models.load_model("D:\Shoes\Latest_Iteration\Models\connor_5.h5")


# In[35]:


predicted_proba = model.predict(image_array, verbose = 1, batch_size = 32)


# In[36]:


del(image_array)


# In[37]:


indexes = predicted_proba >= 0.5


# In[38]:


len(indexes)


# In[39]:


img_iter.class_indices


# In[40]:


new_dict = {}
for k, v in img_iter.class_indices.items():
    new_dict[v] = k


# In[41]:


new_dict


# In[42]:


catch_catch = []
for i1 in indexes: 
#     catch = []
    catch = [new_dict[i] for i,n in enumerate(i1) if n == True]
    catch_catch.append(catch)    


# In[43]:


len(catch_catch)


# In[ ]:


from collections import Counter


# In[ ]:


Counter([j for i in catch_catch for j in i])


# In[ ]:


len(coords)


# In[44]:


pred_df = pd.DataFrame({"coords":coords, 
             "preds": catch_catch})


# In[45]:


pred_df.head()


# In[ ]:


pred_df["coords"]


# In[46]:


idx = [i for i,n in enumerate(pred_df["preds"]) if len(n) > 0]


# In[47]:


subsetted = pred_df.iloc[idx,:].reset_index(drop = True)


# In[48]:


subsetted


# In[144]:


pred_classes = np.unique([j for i in np.unique(subsetted["preds"]) for j in i])


# In[117]:


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return (boxes[pick].astype("int"), pick)


# In[120]:


col_scheme = {'bowtie': (0,0,255),
 'chevron': (0,255,0),
 'circle': (255,0,0),
 'line': (255,0,255),
 'other': (0,0,128),
 'polygon': (255,255,0),
 'quadrilateral': (0,0,0),
 'ribbon': (0,128,128),
 'star': (192,192,192),
 'text': (128,128,0),
 'triangle': (147,20,255)}


# In[145]:


img_test_1 = img_test.copy()
for class_name in pred_classes: 
    idxs = [i for i,n in enumerate(subsetted["preds"]) if class_name in n]
    for i in non_max_suppression_fast(np.array(list(subsetted.iloc[idxs, :]["coords"])), 0.5)[0]: 
        coords = i
        xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
    #     name = i[1][1]
        cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), col_scheme[class_name], 2)
        cv2.putText(img_test_1, class_name, (int(round((xmin+xmax)/2)), ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, col_scheme[class_name], 2) 


# In[146]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

plt.imshow(img_test_1)


# In[114]:


class_name = "triangle"


# In[115]:


idxs = [i for i,n in enumerate(subsetted["preds"]) if class_name in n]


# In[116]:


subsetted.iloc[idxs, :]


# In[94]:


img_test_1 = img_test.copy()


# In[95]:


for i in subsetted.iloc[idxs, :]["coords"]: 
    coords = i
    xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
#     name = i[1][1]
    cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#     plt.title(name)
#     plt.imshow(img)
#     plt.show()


# In[96]:


plt.imshow(img_test_1)


# In[139]:


img_test_1 = img_test.copy()


# In[119]:


non_max_suppression_fast(np.array(list(subsetted.iloc[idxs, :]["coords"])), 0.5)


# In[121]:


round((xmin+xmax)/2)


# In[140]:


for i in non_max_suppression_fast(np.array(list(subsetted.iloc[idxs, :]["coords"])), 0.5)[0]: 
    coords = i
    xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
#     name = i[1][1]
    cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), col_scheme[class_name], 2)
    cv2.putText(img_test_1, class_name, (int(round((xmin+xmax)/2)), ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, col_scheme[class_name], 2) 
#     plt.title(name)
#     plt.imshow(img)
#     plt.show()


# In[136]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)


# In[141]:


plt.imshow(img_test_1)

