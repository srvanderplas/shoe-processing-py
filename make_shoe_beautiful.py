#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add, Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization, RepeatVector, Reshape
# from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
# from tensorflow import set_seed
# set_random_seed(2)
# from tensorflow import set_seed
# set_random_seed(2)
np.random.seed(1)
# print(os.listdir("../input"))
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import glob
import random
from random import sample, choices,shuffle
plt.rcParams["figure.figsize"] = (20,20)
import gc
from joblib import Parallel, delayed


# In[2]:


import tensorflow as tf


# In[3]:


InputPath=r"D:\Shoes\Shoes_with_annotations\\"


# In[4]:


InputPath=r"D:\Shoes\New_Images\\"


# In[5]:


def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.0001
        sigma = var**0.05
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy =  gauss + image
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 1.0
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i , int(num_pepper))
              for i in image.shape]
        out[coords] = 1
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy


# In[6]:


img = cv.imread(InputPath+"1-state-bennie-black-1_product_9330538_color_125647.jpg")  
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
l = img.max()
plt.imshow(img)
l


# In[7]:


Noise = noisy("s&p",img)
plt.imshow(Noise)


# In[8]:


hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) #convert it to hsv
hsv[...,2] = hsv[...,2]*0.2
img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
Noise2 = noisy("s&p",img1)

plt.imshow(Noise2)


# In[9]:


shape_size = 128


# In[ ]:


# InputPath=r"D:\Shoes\Shoes_with_annotations\\"


# In[10]:


total_files = glob.glob(InputPath + "*.jpg")


# In[11]:


total_files = sample(total_files, k = int(1*len(total_files)))


# In[12]:


total_files = total_files[:15000]


# In[ ]:


random.seed(42)
train_files = sample(total_files, k = int(0.70*len(total_files)))


# In[ ]:


test_files = [i for i in total_files if i not in train_files]


# In[ ]:


len(train_files)


# In[ ]:


len(test_files)


# In[ ]:


len(total_files)


# In[ ]:


# train_files = train_files[:50]
# test_files = test_files[:20]


# In[ ]:


round(0.8895,2)


# In[ ]:


fracs = np.arange(0,100)/100


# In[ ]:


# to create the noise and the correct image
def PreProcessData(ImagePath):
    X_=[]
    y_=[]
    count=0
    for imageDir in os.listdir(ImagePath):
        if count<10001:
            print(count)
            try:
                count=count+1
                img = cv.imread(ImagePath + imageDir)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_y = cv.resize(img,(shape_size,shape_size))
                print(img_y.shape)
                hsv = cv.cvtColor(img_y, cv.COLOR_BGR2HSV) #convert it to hsv
                hsv[...,2] = hsv[...,2]*sample(set(fracs),1)[0]
                img_1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                Noisey_img = (noisy("s&p",img_1)/255.0).round(3)
                X_.append(Noisey_img)
                y_.append((img_y/255.0).round(3))
            except:
                pass
    X_ = np.array(X_)
    y_ = np.array(y_)
    
    return X_,y_


# In[ ]:


# to create the noise and the correct image
def PreProcessData(ImagePath,shape_size = 128):
    img = cv.imread(ImagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_y = cv.resize(img,(shape_size,shape_size))
#     print(img_y.shape)
    hsv = cv.cvtColor(img_y, cv.COLOR_BGR2HSV) #convert it to hsv
    hsv[...,2] = hsv[...,2]*sample(set(fracs),1)[0]
    img_1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    Noisey_img = (noisy("s&p",img_1)/255.0).round(3)
    X_ = Noisey_img
    y_ = (img_y/255.0).round(3)
    X_ = np.array(X_)
    y_ = np.array(y_)
    
    return X_,y_


# In[13]:


# to create the noise and the correct image
def PreProcessData(ImagePath,shape_size = 128):
    img = cv.imread(ImagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_y = cv.resize(img,(shape_size,shape_size))
#     print(img_y.shape)
    hsv = cv.cvtColor(img_y, cv.COLOR_BGR2HSV) #convert it to hsv
    hsv[...,2] = hsv[...,2]*0.2
    img_1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    Noisey_img = (noisy("s&p",img_1)/255.0).round(3)
    X_ = Noisey_img
    y_ = (img_y/255.0).round(3)
    X_ = np.array(X_)
    y_ = np.array(y_)
    
    return X_,y_


# In[14]:


obj = Parallel(n_jobs=7, backend = "threading", verbose = 5)(delayed(PreProcessData)(i) for i in total_files)


# In[15]:


X_ = [i[0] for i in obj ]


# In[16]:


X_ = np.array(X_)


# In[17]:


y_ = [i[1] for i in obj]


# In[18]:


y_ = np.array(y_)


# In[19]:


del(obj)


# In[ ]:


# X_,y_ = PreProcessData(InputPath)


# In[36]:


K.clear_session()
def InstantiateModel(in_):
    
    model_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
    model_1 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_1)
    
    model_2 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_2 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_2_0 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_add = add([model_1,model_2,model_2_0])
    
    model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
    model_3 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3)
    
    model_3_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3_1 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3_1)
    
    model_3_2 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add)
    
    model_add_2 = add([model_3_1,model_3_2,model_3])
    
    model_4 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
    model_4_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add)
    #Extension
    model_add_3 = add([model_4_1,model_add_2,model_4])
    
    model_5 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_3)
    model_5 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add_3)
    
    model_5 = Conv2D(3,(3,3), activation='relu',padding='same',strides=1)(model_5)
    
    return model_5


# In[37]:


Input_Sample = Input(shape=(shape_size, shape_size,3))
Output_ = InstantiateModel(Input_Sample)
Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)


# In[38]:


Model_Enhancer.compile(optimizer="adam", loss='mean_squared_error',  metrics=[tf.keras.metrics.RootMeanSquaredError()])
Model_Enhancer.summary()


# In[33]:


get_ipython().system('pip install pydot')


# In[43]:


get_ipython().system(' pip install graphviz ')


# In[44]:


tf.keras.utils.model_to_dot(
    Model_Enhancer,
    show_shapes=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    subgraph=False,
)


# In[41]:


from keras.utils.vis_utils import plot_model
plot_model(Model_Enhancer,to_file='model_.png',show_shapes=True, show_layer_names=True)
from IPython.display import Image
Image(retina=True, filename='model_.png')


# In[ ]:


# garbage collection
gc.enable()
gc.collect()


# In[ ]:


es = EarlyStopping(monitor='val_root_mean_squared_error', mode='min', verbose=1,patience=15, restore_best_weights = True)
Model_Enhancer.fit(X_[:12000],y_[:12000],epochs=1000,verbose=1,batch_size = 2,
                            validation_data=(X_[12000:],y_[12000:]),callbacks = [es])


# In[ ]:


# Model_Enhancer.save(r"D:\Shoes\beautiful_model") 


# In[ ]:


Model_Enhancer.save(r"D:\Shoes\beautiful_model.h5") 


# In[20]:


new_model = tf.keras.models.load_model(r"D:\Shoes\beautiful_model.h5")


# In[ ]:


# new_model = tf.keras.models.load_model(r"D:\Shoes\beautiful_model")


# In[21]:


# fixing the multiplier for the v channel of hsv
def ExtractTestInput(ImagePath):
    img = cv.imread(ImagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_ = cv.resize(img,(1000,1000))
    hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV) #convert it to hsv
    hsv[...,2] = hsv[...,2]*0.2
    img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    Noise = noisy("s&p",img1)
    Noise = Noise.reshape(1,1000,1000,3)
    Noise = (Noise/255.0).round(3)
    return Noise


# In[ ]:


# def ExtractTestInput(ImagePath):
#     img = cv.imread(ImagePath)
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     img_ = cv.resize(img,(1000,1000))
#     hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV) #convert it to hsv
#     hsv[...,2] = hsv[...,2]*sample(set(fracs),1)[0]
#     img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#     Noise = noisy("s&p",img1)
#     Noise = Noise.reshape(1,1000,1000,3)
#     Noise = (Noise/255.0).round(3)
#     return Noise


# In[ ]:


# index_to_predict = 4687


# In[ ]:


# new_model = tf.keras.models.load_model(r"D:\Shoes\beautiful_model")


# In[45]:


index_to_predict = sample(range(3493,4988),1)[0]


# In[46]:


image_for_test = X_[index_to_predict]
image_for_test = image_for_test.reshape(1,shape_size,shape_size,3)


# In[ ]:


# Prediction = Model_Enhancer.predict(image_for_test)


# In[47]:


Prediction = new_model.predict(image_for_test)


# In[48]:


Prediction = Prediction.reshape(shape_size,shape_size,3)
# plt.imshow(img_[:,:,:])


# In[49]:


Prediction[Prediction > 1.0] = 1.0


# In[ ]:


# Prediction = Prediction.round().astype(int)


# In[50]:


plt.imshow(Prediction)


# In[51]:


plt.figure(figsize=(30,30))
plt.subplot(5,5,1)
# img_1 = cv.imread(Image_test)
# img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
img_1 = y_[index_to_predict]
plt.title("Ground Truth",fontsize=20)
plt.imshow(img_1)

plt.subplot(5,5,1+1)
# img_ = ExtractTestInput(Image_test)
img_ = X_[index_to_predict]
plt.title("Low Light Image",fontsize=20)
plt.imshow(img_)

plt.subplot(5,5,1+2)
img_[:,:,:] = Prediction[:,:,:]
plt.title("Enhanced Image",fontsize=20)
plt.imshow(img_)


# ## Experiments

# In[ ]:


base_model = VGG16(weights = "imagenet", include_top = False, input_tensor = Input(shape = (128,128,3)))


# In[ ]:


headmodel = base_model.layers[2].output


# In[ ]:


headmodel = Conv2D(3,(3,3),strides = 1, padding = "same", activation = "relu")(headmodel)


# In[ ]:


# Output_ = InstantiateModel(Input_Sample)
model = Model(inputs=base_model.input, outputs=headmodel)


# In[ ]:


for layers in base_model.layers:
    layers.trainable = False


# In[ ]:


model.compile(optimizer="adam", loss='mean_squared_error',  metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()


# In[ ]:


def GenerateInputs(X,y):
    for i in range(len(X))[:12000]:
        X_input = X[i].reshape(1,128,128,3)
        y_input = y[i].reshape(1,128,128,3)
        yield (X_input,y_input)


# In[ ]:


def GenerateInputs1(X,y):
    for i in range(len(X))[12000:]:
        X_input = X[i].reshape(1,128,128,3)
        y_input = y[i].reshape(1,128,128,3)
        yield (X_input,y_input)


# In[ ]:


model.fit(GenerateInputs(X_,y_),epochs=20,verbose=1,steps_per_epoch=12000,shuffle=True, validation_data = (X_[12000:], y_[12000:]))


# In[ ]:


es = EarlyStopping(monitor='val_root_mean_squared_error', mode='min', verbose=1,patience=15, restore_best_weights = True)
model.fit(X_[:12000],y_[:12000],epochs=1000,verbose=1,batch_size = 2,
                            validation_data=(X_[12000:],y_[12000:]),callbacks = [es])


# In[ ]:


K.clear_session()
def InstantiateModel(in_):
    
    model_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
    model_1 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_1)
    
    model_2 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_2 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_2_0 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_add = add([model_1,model_2,model_2_0])
    
    model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
    model_3 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3)
    
    model_3_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3_1 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3_1)
    
    model_3_2 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add)
    
    model_add_2 = add([model_3_1,model_3_2,model_3])
    
    model_4 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
    model_4_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add)
    #Extension
    model_add_3 = add([model_4_1,model_add_2,model_4])
    
    model_5 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_3)
    model_5 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add_3)
    
    model_5 = Conv2D(3,(3,3), activation='relu',padding='same',strides=1)(model_5)
    
    return model_5

Input_Sample = Input(shape=(shape_size, shape_size,3))
Output_ = InstantiateModel(Input_Sample)
Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)

Model_Enhancer.compile(optimizer="adam", loss='mean_squared_error',  metrics=[tf.keras.metrics.RootMeanSquaredError()])
Model_Enhancer.summary()

# garbage collection
gc.enable()
gc.collect()

es = EarlyStopping(monitor='val_root_mean_squared_error', mode='min', verbose=1,patience=15, restore_best_weights = True)
Model_Enhancer.fit(X_[:12000],y_[:12000],epochs=1000,verbose=1,batch_size = 2,
                            validation_data=(X_[12000:],y_[12000:]),callbacks = [es])

