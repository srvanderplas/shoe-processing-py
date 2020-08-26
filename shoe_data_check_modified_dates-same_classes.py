#!/usr/bin/env python
# coding: utf-8

# In[146]:


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
plt.rcParams["figure.figsize"] = [10,10]


# In[147]:


# base path for xml
xml_base_path = r"D:\Shoes\Shoes\\"


# In[148]:


# annotation files
annotation_files = glob.glob(xml_base_path + "*.xml")
annotation_file_names =[i.split("\\")[-1].split(".")[0] for i in annotation_files]


# In[149]:


# base path for images corresponding to the annotation files
# images_base_path = r"D:\Shoes\Shoes_with_annotations"


# In[150]:


# image files
# image_files = glob.glob(images_base_path + "*.jpg")


# In[151]:


# files in wrong format folder - does this mean the ones which have wrong annotation files
# wrong_format_train = glob.glob(r"Z:\LabelMe_data\Images\Textures\wrongformat\*.png")
# wrong_format_train_names =[i.split("\\")[-1].split(".")[0] for i in wrong_format_train]


# In[152]:


# wrong_format_test = glob.glob(r"Z:\LabelMe_data\Images\test\wrongformat\*.png")
# wrong_format_test_names =[i.split("\\")[-1].split(".")[0] for i in wrong_format_test]


# In[153]:


# is there anything in annotation file names that is in wrong 
# wrong_file_names = [i for i in annotation_file_names if i in wrong_format_train_names or i in wrong_format_test_names]


# In[154]:


# not wrong file names
# correct_file_names = [i for i in annotation_file_names if i not in wrong_file_names]


# In[155]:


# correct_file_names[:10]


# In[156]:


# function to do the same but for all the images
# def annotation_details(file):
#     filename = file.split("\\")[-1].split(".")[0]
#     date_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file))
#     dict1 = {"file_name":filename, 
#              "date_modified": date_modified}
#     return(dict1)
                               


# In[157]:


# catch_annotations = Parallel(n_jobs=6, verbose = 3, backend = "threading")(delayed(annotation_details)(file) for file in annotation_files)


# In[158]:


# annotations_df = pd.DataFrame(catch_annotations)


# In[159]:


# annotations_df.columns = ["filename", "annotation_mod_date"]


# In[160]:


# annotations_df.head()


# In[161]:


# annotations_df.to_csv(r"D:\Shoes\annotation.csv", index = False)


# In[162]:


# image_files_from_putty = glob.glob("Z:\LabelMe_data\Images\Textures\*.jpg")


# In[163]:


# check if the annotation files are in these files
# len([i for i in image_files_from_putty if i.split("\\")[-1].split(".")[0] in annotation_file_names]) == len(annotation_file_names)


# In[164]:


# function to do the same but for all the images
# def image_details(file):
#     filename = file.split("\\")[-1].split(".")[0]
#     if filename in annotation_file_names:
#         date_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file))
#         dict1 = {"file_name":filename, 
#              "date_modified": date_modified}
#     else:
#         dict1 = {"file_name":np.nan, 
#              "date_modified": np.nan}
#     return(dict1)
                               


# In[165]:


# catch_images = Parallel(n_jobs=6, verbose = 3, backend = "threading")(delayed(image_details)(file) for file in image_files_from_putty)


# In[166]:


# images_df = pd.DataFrame(catch_images)


# In[167]:


# images_df = images_df.dropna()


# In[168]:


# images_df.columns = ["filename", "image_mod_date"]


# In[169]:


# images_df.to_csv(r"D:\Shoes\images_annotations.csv", index = False)


# In[170]:


# join the two dataframes
# combo_df = pd.merge(images_df, annotations_df, how = "inner", left_on = "filename", right_on = "filename")


# In[171]:


# combo_df["annotations_mod_minus_image_mod"] = combo_df["annotation_mod_date"] - combo_df["image_mod_date"] 


# In[172]:


# combo_df


# In[173]:


# combo_df["annotations_mod_minus_image_mod"] = [i.seconds for i in combo_df["annotations_mod_minus_image_mod"]]


# In[174]:


# np.sum(combo_df["annotations_mod_minus_image_mod"] < 0)


# In[175]:


### check from the other folder that also has images


# In[176]:


# image_files_from_putty_shoes = glob.glob("Z:\LabelMe_data\Images\Shoes\*.jpg")


# In[177]:


# check if the annotation files are in these files
# len([i for i in image_files_from_putty_shoes if i.split("\\")[-1].split(".")[0] in annotation_file_names]) == len(annotation_file_names)


# In[178]:


# function to do the same but for all the images
# def image_details_shoes(file):
#     filename = file.split("\\")[-1].split(".")[0]
#     if filename in annotation_file_names:
#         date_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file))
#         dict1 = {"file_name":filename, 
#              "date_modified": date_modified}
#     else:
#         dict1 = {"file_name":np.nan, 
#              "date_modified": np.nan}
#     return(dict1)
                               


# In[179]:


# catch_images_shoes = Parallel(n_jobs=6, verbose = 3, backend = "threading")(delayed(image_details_shoes)(file) for file in image_files_from_putty_shoes)


# In[180]:


# images_df_shoes = pd.DataFrame(catch_images_shoes)


# In[181]:


# images_df_shoes = images_df_shoes.dropna()


# In[182]:


# images_df_shoes.columns = ["filename", "image_mod_date"]


# In[183]:


# images_df_shoes.to_csv(r"D:\Shoes\images_annotations_shoes.csv", index = False)


# In[184]:


# join the two dataframes
# combo_df = pd.merge(images_df_shoes, annotations_df, how = "inner", left_on = "filename", right_on = "filename")


# In[185]:


# combo_df["annotations_mod_minus_image_mod"] = combo_df["annotation_mod_date"] - combo_df["image_mod_date"] 


# In[186]:


# combo_df


# In[187]:


# combo_df["annotations_mod_minus_image_mod"] = [i.seconds for i in combo_df["annotations_mod_minus_image_mod"]]


# In[188]:


# combo_df.head()


# In[189]:


# np.sum(combo_df["annotations_mod_minus_image_mod"] < 0)


# In[190]:


################################# checking modified date ends ################################


# In[191]:


################################ Keep images for which the annotation dimensions are all within dimensions of the image#####


# In[192]:


################################## Miscallaneous stuff #######################################


# In[193]:


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


# In[194]:


catch_images_shoes = Parallel(n_jobs=6, verbose = 10, backend = "threading")(delayed(check_dim)(file) for file in annotation_file_names)


# In[195]:


paths_correct = [i for i in catch_images_shoes if i !="nf"]


# In[196]:


len(paths_correct)


# In[197]:


len([i for i in paths_correct])/len(annotation_files)


# In[198]:


# xml_path = xml_base_path + paths_correct[0] + ".xml"


# In[199]:


# function to visualize the xml
# def indent(elem, level=0):
#     i = "\n" + level*"  "
#     j = "\n" + (level-1)*"  "
#     if len(elem):
#         if not elem.text or not elem.text.strip():
#             elem.text = i + "  "
#         if not elem.tail or not elem.tail.strip():
#             elem.tail = i
#         for subelem in elem:
#             indent(subelem, level+1)
#         if not elem.tail or not elem.tail.strip():
#             elem.tail = j
#     else:
#         if level and (not elem.tail or not elem.tail.strip()):
#             elem.tail = j
#     return elem        

# root = ElementTree.parse(xml_path).getroot()
# indent(root)
# ElementTree.dump(root)


# In[200]:


# paths_correct[0]


# In[201]:


# xml_path =  r"D:\Shoes\Shoes\\" + "converse-skate-ctas-pro-hi-skate-black-black-white-2_product_8354725_color_398774" + ".xml"


# In[202]:


# img = plt.imread(r"D:\Shoes\Shoes_with_annotations\\"  +"converse-skate-ctas-pro-hi-skate-black-black-white-2_product_8354725_color_398774" + ".jpg")


# In[203]:


#     width = img.shape[1]
#     height = img.shape[0]


# In[204]:


# width


# In[205]:


# soup = BeautifulSoup(open(xml_path).read())


# In[206]:


# soup_obj =soup.find_all("object")


# In[207]:


# soup.find_all(["x","xmax","xmin"])


# In[208]:


# soup_obj.find_all(["x","xmax"])


# In[209]:


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
        


# In[210]:


# def parse_soup(item): 
#     class_name = soup_obj[item].find_all("name")[0].text
#     x_pts = [int(round(float(i.text),0)) for i in soup_obj[item].find_all("x")]
#     y_pts = [int(round(float(i.text),0)) for i in soup_obj[item].find_all("y")]
#     xmin = np.min(x_pts)
#     ymin = np.min(y_pts)
#     xmax = np.max(x_pts)
#     ymax = np.max(y_pts)
#     catch_obj = {"class": class_name, "xmin":xmin, "ymin":ymin,"xmax":xmax,"ymax":ymax}
#     return(catch_obj)


# In[211]:


catch_images_shoes = Parallel(n_jobs=6, verbose = 10, backend = "threading")(delayed(parse_xmls)(file) for file in paths_correct)


# In[212]:


catch_images_shoes = [j for i in catch_images_shoes for j in i]


# In[213]:


catch_images_shoes_df = pd.DataFrame(catch_images_shoes)


# In[214]:


catch_images_shoes_df.head()


# In[215]:


catch_images_shoes_df["filename"][0]


# In[216]:


# get a sample path
base_path = r"D:\Shoes\Shoes_with_annotations\\"
filename = catch_images_shoes_df["filename"][2356]
samp_path = base_path + filename
samp_image = plt.imread(samp_path)
temp_data = catch_images_shoes_df[catch_images_shoes_df["filename"] == filename]
# get the unique categories
categories = temp_data["class"]
unique_cat = np.unique(categories)
unique_cat = [i for i in unique_cat if i != "exclude"]


# In[217]:


unique_cat


# In[218]:


temp_data


# In[219]:


import cv2


# In[220]:


for i in unique_cat:
    temp_data_cat = temp_data[temp_data["class"] == i]
    for i1 in temp_data_cat.iterrows(): 
#     print(i)
        xmin, ymin, xmax, ymax = i1[1]["xmin"], i1[1]["ymin"], i1[1]["xmax"], i1[1]["ymax"]
#     samp_image = np.array(samp_image[:,:,::-1])
        cv2.rectangle(samp_image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)


# In[221]:


plt.imshow(samp_image)


# In[222]:


# write a function to take an image and plot the box and labels

def plot_rectangles(filename): 
    base_path = r"D:\Shoes\Shoes_with_annotations\\"
    filename = filename
    samp_path = base_path + filename
    samp_image = plt.imread(samp_path)
    temp_data = catch_images_shoes_df[catch_images_shoes_df["filename"] == filename]
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
            cv2.rectangle(samp_image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    plt.imshow(samp_image)
    return(temp_data)


# In[223]:


plot_rectangles(catch_images_shoes_df["filename"][2356])


# In[224]:


# base_path = r"D:\Shoes\Shoes_with_annotations\\"
# filename = catch_images_shoes_df["filename"][2356]
# samp_path = base_path + filename
# image = cv2.imread(samp_path)
# ss.setBaseImage(image)


# In[225]:


# cv2.setUseOptimized(True);
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
base_path = r"D:\Shoes\Shoes_with_annotations\\"
filename = catch_images_shoes_df["filename"][2356]
samp_path = base_path + filename
im = cv2.imread(samp_path)
ss.setBaseImage(im)
ss.switchToSelectiveSearchFast()
rects = ss.process()


# In[226]:


len(rects)


# In[228]:


# loop over the region proposals in chunks (so we can better
# visualize them)
for i in range(0, len(rects), 100):
    # clone the original image so we can draw on it
    output = im.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in rects[i:i + 100]:
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)


# In[229]:


plt.imshow(output)


# In[ ]:


# 


# In[13]:


catch_images_shoes_df = catch_images_shoes_df[["filename","width","height","class","xmin","ymin","xmax","ymax"]]


# In[14]:


keep_classes = [(k,v) for k,v in catch_images_shoes_df["class"].value_counts().items() if len(k.split(",")) ==1]


# In[18]:


catch_images_shoes_df["class"].value_counts()[:50]


# In[19]:


keep_classes


# In[20]:


# what are the main classes
keep_classes[:20]


# In[18]:


keep_classes[0][:4]


# In[21]:


catch_images_shoes_df["class"] = ["quadrilateral" if i[:2] == "qu" else i for i in catch_images_shoes_df["class"] ]


# In[22]:


catch_images_shoes_df["class"] = ["lines" if i[:4] == "line" else i for i in catch_images_shoes_df["class"] ]


# In[23]:


catch_images_shoes_df["class"] = ["circles" if i[:6] == "circle" else i for i in catch_images_shoes_df["class"] ]


# In[24]:


catch_images_shoes_df["class"] = ["triangles" if i[:8] == "triangle" else i for i in catch_images_shoes_df["class"] ]


# In[25]:


catch_images_shoes_df["class"] = ["stars" if i[:4] == "star" else i for i in catch_images_shoes_df["class"] ]


# In[29]:


catch_images_shoes_df["class"] = ["polygon" if i in ["pentagon", "hexagon"] else i for i in catch_images_shoes_df["class"] ]


# In[33]:


catch_images_shoes_df["class"] = ["other" if i in ["logo", "other"] else i for i in catch_images_shoes_df["class"] ]


# In[34]:


keep_classes_these  = [(k) for k,v in catch_images_shoes_df["class"].value_counts()[:50].items() if 
                       len(k.split(",")) == 1 and len(k.split(" ")) == 1]


# In[35]:


keep_classes_these = [i for i in keep_classes_these if i not in ["exclude","ribbon"]]


# In[36]:


keep_classes_these


# In[37]:


catch_images_shoes_df = catch_images_shoes_df[catch_images_shoes_df["class"].isin(keep_classes_these)].reset_index(drop = True)


# In[38]:


catch_images_shoes_df


# In[39]:


# how many unique images
catch_images_shoes_df["filename"].nunique()


# In[40]:


unique_files = np.unique(catch_images_shoes_df["filename"])


# In[41]:


catch_images_shoes_df[catch_images_shoes_df["filename"] == unique_files[0]]


# In[42]:


# don't just randomly split - there will be leakage
# learn some more about the files

def file_details(item):
    subset = catch_images_shoes_df[catch_images_shoes_df["filename"] == item]
    count = Counter(subset["class"])
    return({"filename":item, "freq" : count})
    


# In[43]:


file_freqs = Parallel(n_jobs=6, verbose = 10, backend = "threading")(delayed(file_details)(file) for file in unique_files)


# In[44]:


# actually split by different files - you can't control the test population

catch_images_shoes_df = catch_images_shoes_df.sample(frac = 1, random_state = 42).reset_index(drop = True)


# In[45]:


catch_images_shoes_df


# In[46]:


len(unique_files)


# In[47]:


int(0.75*len(unique_files))


# In[48]:


unique_files = set(unique_files)


# In[49]:


# randomly select 75 % of the filenames for train
# 25% for test
random.seed(42)
train_files = sample(unique_files, k = int(0.75*len(unique_files)))


# In[50]:


len(train_files)


# In[51]:


test_files = [i for i in unique_files if i not in train_files]


# In[52]:


len(test_files)


# In[53]:


len(train_files)


# In[54]:


len(test_files)+len(train_files) == len(unique_files)


# In[55]:


train_data = catch_images_shoes_df[catch_images_shoes_df["filename"].isin(train_files)]


# In[56]:


test_data = catch_images_shoes_df[catch_images_shoes_df["filename"].isin(test_files)]


# In[57]:


train_data.head()


# In[ ]:


# write out the csv files


# In[58]:


train_data["class"].unique()


# In[60]:


Counter(train_data["class"])


# In[59]:


test_data["class"].unique()


# In[61]:


Counter(test_data["class"])


# In[62]:


train_data.to_csv(r"C:\Users\19169\Documents\tensorflow1\workspace\training_demo\annotations\train_labels.csv", index = False)


# In[63]:


test_data.to_csv(r"C:\Users\19169\Documents\tensorflow1\workspace\training_demo\annotations\test_labels.csv", index = False)


# In[ ]:


# put the respective images as well


# In[64]:


[copyfile(r"D:\Shoes\Shoes_with_annotations\\"+ i,
          r"C:\Users\19169\Documents\tensorflow1\workspace\training_demo\images\train\\"+ i ) for i in train_files]


# In[65]:


[copyfile(r"D:\Shoes\Shoes_with_annotations\\"+ i,
          r"C:\Users\19169\Documents\tensorflow1\workspace\training_demo\images\test\\"+ i ) for i in test_files]


# In[354]:


# xml_path = r"D:\Shoes\Shoes\\" + name + ".xml"


# In[ ]:


# xml_path


# In[89]:


# pass the path to the beautiful soup function
# soup = BeautifulSoup(open(xml_path).read())


# In[90]:


#     soup_obj = soup.find_all("object")


# In[91]:


# len(soup_obj)


# In[92]:


# name =  "converse-skate-ctas-pro-hi-skate-black-black-white-2_product_8354725_color_398774"


# In[93]:


#     catch = []
#     for item in range(len(soup_obj)):
#         class_name = soup_obj[item].find_all("name")[0].text
#         x_pts = [int(round(float(i.text),0)) for i in soup_obj[item].find_all(["x","xmin","xmax"])]
#         y_pts = [int(round(float(i.text),0)) for i in soup_obj[item].find_all(["y","ymin","ymax"])]
#         xmin = np.min(x_pts)
#         ymin = np.min(y_pts)
#         xmax = np.max(x_pts)
#         ymax = np.max(y_pts)
#         catch_obj = {"class": class_name, "xmin":xmin, "ymin":ymin,"xmax":xmax,"ymax":ymax}
#         catch_obj["filename"] = name + ".jpg"
#         catch_obj["width"] = width
#         catch_obj["height"] = height
#         catch.append(catch_obj)


# In[94]:


# len(catch)


# In[95]:


# name = "converse-skate-ctas-pro-hi-skate-black-black-white-2_product_8354725_color_398774" 


# In[96]:


# item


# In[97]:


# parse_soup(item)


# In[98]:


# dict1


# In[99]:


#     catch = []
#     for item in range(len(soup_obj)):
#         dict1 = parse_soup(item)
#         dict1["filename"] = name + ".jpg"
#         dict1["width"] = width
#         dict1["height"] = height
#         catch.append(dict1)


# In[100]:


# catch


# In[101]:


# paths_correct[0]


# In[102]:


# read in the corresponding image
# img = plt.imread(r"D:\Shoes\Shoes_with_annotations\\"  +name + ".jpg")


# In[103]:


# soup.find_all("object")[0].find_all("name")[0].text


# In[104]:


# [int(round(float(i.text),0)) for i in soup.find_all("object")[0].find_all("x")]


# In[105]:


# soup.find_all("object")


# In[106]:


# soup.find_all("object")[0].find_all("name")[0].text


# In[ ]:


# soup.find_all("object")[0].find_all("x")


# In[107]:


# soup.find_all("object")[0].find_all("name")[0].text


# In[108]:


# soup.find_all("object")[1].find_all("name")[0].text


# In[109]:


# soup.find_all("object")[2].find_all("name")[0].text


# In[110]:


# soup.find_all("object")[3].find_all("name")[0].text


# In[111]:


# [int(i.text) for i in soup.find_all("x")]


# In[112]:


# soup.find_all("x", text = True)


# In[113]:


# soup.find_all("object")


# In[114]:


# soup.find_all("object")[0].find_all("pt")


# In[115]:



# x1 = [int(i.find_all("x")[0].text) for i in soup.find_all("object")[0].find_all("pt")]


# In[116]:


# x1


# In[117]:


# y1


# In[118]:


# y1 = [int(i.find_all("y")[0].text) for i in soup.find_all("object")[0].find_all("pt")]


# In[119]:


# list(zip(x1, y1))


# In[120]:


# soup.find_all("object")[0].find_all("name")


# In[121]:


# pts = [921,73,1498,550]


# In[122]:


# plt.imshow(cv2.rectangle(img, (921, 73), (1498, 550), (255,0,0), 2))


# In[123]:


# pts = np.array(list(zip(x1, y1)))


# In[124]:


# pts = pts.reshape((-1,1,2))
# cv2.polylines(img,[pts],True,(0,255,255),5)


# In[125]:


# plt.imshow(img)


# In[126]:


# plt.imshow(cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2))


# In[127]:


# [np.array([i.find_all("x"), i.find_all("y")]) for i in soup.find_all("object")[0].find_all("pt")]


# In[128]:


# soup.find_all("object")[1].find_all("name")[0].text


# In[129]:


# ind = 1


# In[130]:


# x_pts = [int(round(float(i.text),0)) for i in soup.find_all("object")[ind].find_all("polygon")[0].find_all("x")]


# In[131]:


# y_pts = [int(round(float(i.text),0)) for i in soup.find_all("object")[ind].find_all("polygon")[0].find_all("y")]


# In[132]:


# x_pts


# In[133]:


# img.shape


# In[134]:


# xmin = np.min(x_pts)


# In[135]:


# xmax = np.max(x_pts)


# In[136]:


# ymin = np.min(y_pts)


# In[137]:


# ymax = np.max(y_pts)


# In[138]:


# import cv2


# In[139]:


# img = plt.imread("Z:\\LabelMe_data\\Images\\Shoes\\"  +paths_correct[0] + ".jpg")
# plt.imshow(cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2))


# In[140]:


# plt.imshow(img)


# In[141]:


# import glob


# In[142]:


# xml_files = [i.split("\\")[-1].split(".")[0] for i in glob.glob(r"Z:\LabelMe_data\Annotations\Images\*.xml")]


# In[143]:


# len(xml_files)


# In[144]:


# xml_files1 = [i.split("\\")[-1].split(".")[0] for i in glob.glob(r"Z:\LabelMe_data\Annotations\Shoes\*.xml")]


# In[ ]:


[copyfile(r"Z:\LabelMe_data\Images\Shoes\\"+ i + ".jpg",
          r"D:\Shoes\Shoes_with_annotations\\"+ i + ".jpg") for i in xml_files]


# In[ ]:


# len(xml_files1)


# In[145]:


# xml_files[59]


# In[146]:


# [i for i,n in enumerate(img_files) if n == xml_files[59]]


# In[147]:


# img_files[1634]


# In[148]:


# img_files = [i.split("\\")[-1].split(".")[0] for i in glob.glob(r"Z:\LabelMe_data\Images\Shoes\*.jpg")]


# In[149]:


# len(set(xml_files).intersection(xml_files))/len(xml_files)

