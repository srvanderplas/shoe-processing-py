#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
from shutil import copyfile
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from xml.etree import ElementTree
import glob
import os.path, time
import datetime
from tqdm import tqdm_notebook
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import random
from collections import Counter
from random import sample, choices
import cv2
from tqdm.notebook import tqdm
plt.rcParams["figure.figsize"] = [10,10]


# In[2]:


# base path for xml
xml_base_path = r"D:\Shoes\Shoes\\"


# In[3]:


# annotation files
annotation_files = glob.glob(xml_base_path + "*.xml")
annotation_file_names =[i.split("\\")[-1].split(".")[0] for i in annotation_files]


# In[4]:


def check_dim(name):
    name = name
    xml_path =  r"D:\Shoes\Shoes\\" + name + ".xml"
    img = plt.imread(r"D:\Shoes\Shoes_with_annotations\\"  +name + ".jpg")
    soup = BeautifulSoup(open(xml_path).read())
    x_dim = img.shape[1]
    y_dim = img.shape[0]
    x_xml = soup.find_all(["x","xmin","xmax"])
    y_xml = soup.find_all(["y","ymin","ymax"])
    x_txt = [float(i.text) for i in x_xml]
    y_txt = [float(i.text) for i in y_xml]
    ind_x = sum([i for i in x_txt if i > x_dim])
    ind_y = sum([i for i in y_txt if i > y_dim])
    if ((ind_x > 0) or (ind_y >0)):
        return "nf"
    else:
        return name


# In[5]:


# uncomment to get data
# catch_images_shoes = Parallel(n_jobs=6, verbose = 10, backend = "threading")(delayed(check_dim)(file) for file in annotation_file_names)


# In[6]:


# paths_correct = [i for i in catch_images_shoes if i !="nf"]


# In[7]:


# len(paths_correct)


# In[8]:


# len([i for i in paths_correct])/len(annotation_files)


# In[9]:


def parse_xmls(name): 
    # read the file name
    xml_path =  r"D:\Shoes\Shoes\\" + name + ".xml"
    img = plt.imread(r"D:\Shoes\Shoes_with_annotations\\"  +name + ".jpg")
    width = img.shape[1]
    height = img.shape[0]
    # make soup
    # pass the path to the beautiful soup function
    soup = BeautifulSoup(open(xml_path).read())
    soup_obj = soup.find_all("object")
    catch = []
    for item in range(len(soup_obj)):
        class_name = soup_obj[item].find_all("name")[0].text
        x_pts = [int(round(float(i.text),0)) for i in soup_obj[item].find_all(["x","xmin","xmax"])]
        y_pts = [int(round(float(i.text),0)) for i in soup_obj[item].find_all(["y","ymin","ymax"])]
        xmin = np.min(x_pts)
        ymin = np.min(y_pts)
        xmax = np.max(x_pts)
        ymax = np.max(y_pts)
        catch_obj = {"class": class_name, "xmin":xmin, "ymin":ymin,"xmax":xmax,"ymax":ymax}
        catch_obj["filename"] = name + ".jpg"
        catch_obj["width"] = width
        catch_obj["height"] = height
        catch.append(catch_obj)
    return(catch)
        


# In[10]:


# uncomment to get data
# catch_images_shoes = Parallel(n_jobs=6, verbose = 10, backend = "threading")(delayed(parse_xmls)(file) for file in paths_correct)


# In[11]:


# uncomment to get data
# catch_images_shoes = [j for i in catch_images_shoes for j in i]


# In[12]:


# uncomment to get data
# catch_images_shoes_df = pd.DataFrame(catch_images_shoes)


# In[13]:


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# In[14]:


# uncomment to get data
# catch_images_shoes_df.to_csv("D:\Shoes\shoe_annotations.csv", index = False)


# In[15]:


# uncomment to get data
catch_images_shoes_df = pd.read_csv("D:\Shoes\shoe_annotations.csv")


# In[16]:


catch_images_shoes_df.head()


# In[ ]:


# write a function to take an image and plot the box and labels

def plot_rectangles(filename): 
    base_path = r"D:\Shoes\Shoes_with_annotations\\"
    filename = filename
    samp_path = base_path + filename
    samp_image1 = plt.imread(samp_path)
    samp_image = samp_image1.copy()
    temp_data = catch_images_shoes_df[catch_images_shoes_df["filename"] == filename]
    temp_data = temp_data[temp_data["class"] != "exclude"]
    categories = temp_data["class"]
    # get the unique categories
    unique_cat = np.unique(categories)
    unique_cat = [i for i in unique_cat if i != "exclude"]
    for i in unique_cat:
        temp_data_cat = temp_data[temp_data["class"] == i]
        for i1 in temp_data_cat.iterrows(): 
#     print(i)
            xmin, ymin, xmax, ymax = i1[1]["xmin"], i1[1]["ymin"], i1[1]["xmax"], i1[1]["ymax"]
#     samp_image = np.array(samp_image[:,:,::-1])
            cv2.rectangle(samp_image, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
    plt.title("region proposals from humans")
    plt.imshow(samp_image)
    return(temp_data, samp_image,samp_image1)


# In[ ]:


index = random.sample(set(np.unique(catch_images_shoes_df["filename"])), k = 1)[0]


# In[ ]:


file_details, human_image, og_image = plot_rectangles(index)


# In[ ]:


# write a function to take an image and plot the box and labels

def store_human_annotated(filename): 
    base_path = r"D:\Shoes\Shoes_with_annotations\\"
    store_path = r"D:\Shoes\Latest_Iteration\Human_Annotated\\"
    filename = filename
    samp_path = base_path + filename
    samp_image1 = plt.imread(samp_path)
    samp_image = samp_image1.copy()
    temp_data = catch_images_shoes_df[catch_images_shoes_df["filename"] == filename]
    temp_data = temp_data[temp_data["class"] != "exclude"]
    categories = temp_data["class"]
    # get the unique categories
    unique_cat = np.unique(categories)
    unique_cat = [i for i in unique_cat if i != "exclude"]
    for i in unique_cat:
        temp_data_cat = temp_data[temp_data["class"] == i]
        for i1 in temp_data_cat.iterrows(): 
#     print(i)
            xmin, ymin, xmax, ymax = i1[1]["xmin"], i1[1]["ymin"], i1[1]["xmax"], i1[1]["ymax"]
#     samp_image = np.array(samp_image[:,:,::-1])
            cv2.rectangle(samp_image, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
#     plt.title("region proposals from humans")
    plt.imsave(store_path + filename,samp_image)
    return(1)


# In[ ]:


# uncomment following three chunks to get human annotated images


# In[ ]:


file_names = np.unique(catch_images_shoes_df["filename"])


# In[ ]:


# _ = store_human_annotated(index)


# In[ ]:


# _ = Parallel(n_jobs=6, verbose = 10, backend = "threading")(delayed(store_human_annotated)(file) for file in file_names)


# In[ ]:


file_details


# In[ ]:


# base_path = r"D:\Shoes\Shoes_with_annotations\\"
# filename = catch_images_shoes_df["filename"][2356]
# samp_path = base_path + filename
# image = cv2.imread(samp_path)
# ss.setBaseImage(image)


# In[ ]:


# write a function to get all the region proposals

def get_rpn(filename,method = "quality"): 
    base_path = r"D:\Shoes\Shoes_with_annotations\\"
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    base_path = r"D:\Shoes\Shoes_with_annotations\\"
    filename = filename
    samp_path = base_path + filename
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


# In[ ]:


coords = get_rpn(index, method = "quality")


# In[ ]:


robot_dict = []
for i in coords:
    human_annotated_dict = {"x1": i[0], "x2": i[2], 
                       "y1": i[1],"y2": i[3]}
    robot_dict.append(human_annotated_dict)


# In[ ]:


best_catches = []
max_value_catch = []
for i in file_details[["xmin","ymin","xmax","ymax"]].values:
    human_annotated = i
    human_annotated_dict = {"x1": human_annotated[0], "x2": human_annotated[2], 
                       "y1": human_annotated[1],"y2": human_annotated[3]}
    iou_human_vs_robot = [get_iou(n,human_annotated_dict) for i,n in enumerate(robot_dict)]
    max_value = np.max(iou_human_vs_robot)
    max_value_catch.append(max_value)
    best_rpn = robot_dict[np.argmax(iou_human_vs_robot)]
    xmin, ymin, xmax, ymax = best_rpn["x1"], best_rpn["y1"], best_rpn["x2"], best_rpn["y2"]
    obj = [xmin, ymin, xmax, ymax]
    best_catches.append(obj)


# In[ ]:


best_catches


# In[ ]:


max_value_catch


# In[ ]:


np.mean(max_value_catch)


# In[ ]:


np.mean(max_value_catch)


# In[ ]:


counter = 0
output = og_image.copy()
for i1 in range(0, len(best_catches)):
    counter = counter + 1
    # clone the original image so we can draw on it
    
    
    # loop over the current subset of region proposals
    i = best_catches[i1]
    
    xmin = i[0]
    ymin = i[1]
    xmax = i[2]
    ymax = i[3]
    # draw the region proposal bounding box on the image
#     color = [random.randint(0, 255) for j in range(0, 3)]
    cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (0,0,255), 3)


# In[ ]:


plt.title("region proposal from selective search")
plt.imshow(output)


# In[ ]:


plt.title("region proposal from humans")
plt.imshow(human_image)


# In[ ]:


# write a function to get all the region proposals

def get_rpn_store(filename,method = "quality"):
    store_path = r"D:\Shoes\Latest_Iteration\Robot_Annotated\\"
    base_path = r"D:\Shoes\Shoes_with_annotations\\"
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    base_path = r"D:\Shoes\Shoes_with_annotations\\"
    filename = filename
    samp_path = base_path + filename
    im = cv2.imread(samp_path)
    ss.setBaseImage(im)
    if method == "fast":
        print(method)
        ss.switchToSelectiveSearchFast()
    if method == "quality":
#         print(method)
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    coords = []
    for (x, y, w, h) in rects: 
        xmin, ymin, xmax, ymax = x,y,x+w, y+h
        coords.append([xmin, ymin, xmax, ymax])
    robot_dict = []
    for i in coords:
        human_annotated_dict = {"x1": i[0], "x2": i[2], 
                       "y1": i[1],"y2": i[3]}
        robot_dict.append(human_annotated_dict)
    best_catches = []
    max_value_catch = []
    for i in file_details[["xmin","ymin","xmax","ymax"]].values:
        human_annotated = i
        human_annotated_dict = {"x1": human_annotated[0], "x2": human_annotated[2], 
                       "y1": human_annotated[1],"y2": human_annotated[3]}
        iou_human_vs_robot = [get_iou(n,human_annotated_dict) for i,n in enumerate(robot_dict)]
        max_value = np.max(iou_human_vs_robot)
        max_value_catch.append(max_value)
        best_rpn = robot_dict[np.argmax(iou_human_vs_robot)]
        xmin, ymin, xmax, ymax = best_rpn["x1"], best_rpn["y1"], best_rpn["x2"], best_rpn["y2"]
        obj = [xmin, ymin, xmax, ymax]
        best_catches.append(obj)
    mean_iou = np.mean(max_value_catch)
    counter = 0
    output = im.copy()
    for i1 in range(0, len(best_catches)):
        counter = counter + 1
    # clone the original image so we can draw on it
    
    
    # loop over the current subset of region proposals
        i = best_catches[i1]
    
        xmin = i[0]
        ymin = i[1]
        xmax = i[2]
        ymax = i[3]
    # draw the region proposal bounding box on the image
#     color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
    plt.imsave(store_path + filename,output[:,:,::-1])
    return({"filename":filename, "mean_IOU":mean_iou})


# In[ ]:


# ious  = get_rpn_store(index)


# In[ ]:


ious = Parallel(n_jobs=6, verbose = 10, backend = "threading")(delayed(get_rpn_store)
                                                               (file) for file in file_names)

