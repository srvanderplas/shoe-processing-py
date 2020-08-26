#!/usr/bin/env python
# coding: utf-8

# In[201]:


import cv2
import pandas
import matplotlib.pyplot as plt
import numpy as np


# In[202]:


import pandas as pd


# In[203]:


from tqdm.notebook import tqdm


# In[204]:


from PIL import Image


# In[205]:


plt.rcParams["figure.figsize"] = (20,20)


# In[206]:


path_1 = r"D:\Shoes\Shoes_with_annotations\adidas-originals-nmd-r1-grey-five-grey-five-black_product_8808124_color_758542.jpg"


# In[207]:


path_2 = r"D:\Shoes\Shoes_with_annotations\adidas-originals-nmd-r1-w-light-granite-light-granite-clear-orange_product_9075941_color_776232.jpg"


# In[208]:


img_1 = plt.imread(path_1)


# In[209]:


img_1.shape


# In[210]:


img_2= plt.imread(path_2)


# In[211]:


img_2.shape


# In[212]:


# img_2 = np.array(Image.fromarray(img_2).resize((img_1.shape[1], img_1.shape[0]), Image.ANTIALIAS))


# In[213]:


# img_2.shape


# In[214]:


plt.imshow(img_1)


# In[215]:


plt.imshow(img_2)


# In[216]:


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


# In[217]:


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


# In[218]:


coords_1 = get_rpn(path_1, 
                  method = "fast")


# In[219]:


coords_2 = get_rpn(path_2, 
                  method = "fast")


# In[220]:


len(coords_1)


# In[221]:


len(coords_2)


# In[222]:


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


# In[223]:


catch = []
for boxA in tqdm(coords_1):
    catch_ious = []
    for boxB in coords_2: 
        iou = bb_intersection_over_union(boxA, boxB)
        catch_ious.append(iou)
    catch.append(catch_ious)


# In[224]:


argmax_catch = []
max_catch = []
for iou in tqdm(catch): 
    argmax = np.argmax(iou)
    argmax_catch.append(argmax)
    max_catch.append(np.max(iou))


# In[225]:


np.mean(max_catch)


# In[226]:


# make a dataframe of coords_1 indexes and max iou corresponding to them
# and the index of that max iou
df_compare = pd.DataFrame({"index": range(len(coords_1)), 
             "max_iou": max_catch, 
             "max_iou_index": argmax_catch})


# In[227]:


df_compare.head()


# In[111]:


# non max suppression
# _, index = non_max_suppression_fast(np.array(coords_1), 0.5)


# In[228]:


# subset these coords from img 1 coords
max_coords_1 = [coords_1[i] for i in df_compare.sort_values("max_iou", ascending = False).head(25).index ]


# In[113]:


# max_coords_2 = df_compare[df_compare.index.isin(index)].max_iou_index


# In[114]:


# max_coords_2


# In[115]:


# max_coords_2 = [coords_2[i] for i in max_coords_2 ]


# In[282]:


# subset these coords from img 2 coords
max_coords_2 = [coords_2[i] for i in df_compare.sort_values("max_iou", ascending = False).head(50).max_iou_index ]


# In[283]:


img_test_1 = img_1.copy()

for i in max_coords_1: 
    coords = i
    xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
#     name = i[1][1]
    cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
#     plt.title(name)
#     plt.imshow(img)
#     plt.show()

plt.imshow(img_test_1)


# In[231]:


img_test_2 = img_2.copy()

for i in max_coords_2: 
    coords = i
    xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
#     name = i[1][1]
    cv2.rectangle(img_test_2, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
#     plt.title(name)
#     plt.imshow(img)
#     plt.show()

plt.imshow(img_test_2)


# In[232]:


from scipy import ndimage, misc

from PIL import Image

# im = Image.open(index)

image_array = np.zeros((len(coords_1), 256,256,3))

image_array.shape


# In[233]:



from tqdm.notebook import tqdm

counter = 0
for i in tqdm(coords_1):
    xmin, ymin, xmax, ymax = i[0], i[1], i[2], i[3]
    cropped = img_1[ymin:ymax,xmin:xmax]
    cropped = Image.fromarray(cropped)
    cropped = cropped.resize((256,256),Image.ANTIALIAS)
    
    cropped = np.array(cropped)
    cropped = cropped/255.0
    image_array[counter] = cropped
    counter =counter + 1
#     print(cropped.shape)
#     plt.imshow(cropped)
#     plt.show()

image_array.shape


# In[234]:


import tensorflow as tf


# In[235]:


model = tf.keras.models.load_model("D:\Shoes\Latest_Iteration\Models\connor_5.h5")


# In[236]:


# del(image_array)


predicted_proba = model.predict(image_array, verbose = 1, batch_size = 32)

del(image_array)

indexes = predicted_proba >= 0.5

len(indexes)


# In[237]:


new_dict = {0: 'bowtie',
 1: 'chevron',
 2: 'circle',
 3: 'line',
 4: 'other',
 5: 'polygon',
 6: 'quadrilateral',
 7: 'ribbon',
 8: 'star',
 9: 'text',
 10: 'triangle'}


# In[238]:


catch_catch = []
for i1 in indexes: 
#     catch = []
    catch = [new_dict[i] for i,n in enumerate(i1) if n == True]
    catch_catch.append(catch)    


# In[239]:


from collections import Counter


# In[240]:


Counter([j for i in catch_catch for j in i])


# In[241]:


pred_df_1 = pd.DataFrame({"coords":coords_1, 
             "preds": catch_catch})


# In[242]:


pred_df_1.head()


# In[243]:


idx = [i for i,n in enumerate(pred_df_1["preds"]) if len(n) > 0]


# In[244]:


subsetted = pred_df_1.iloc[idx,:].reset_index(drop = True)


# In[245]:


subsetted


# In[246]:


np.unique(subsetted["preds"])


# In[247]:


idxs = [i for i,n in enumerate(subsetted["preds"]) if "line" in n]


# In[248]:


subsetted.iloc[idxs, :]


# In[249]:


img_test_1 = img_1.copy()


# In[250]:


for i in subsetted.iloc[idxs, :]["coords"]: 
    coords = i
    xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
#     name = i[1][1]
    cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#     plt.title(name)
#     plt.imshow(img)
#     plt.show()


# In[251]:


plt.imshow(img_test_1)


# In[252]:


img_test_1 = img_1.copy()


# In[253]:


non_max_suppression_fast(np.array(list(subsetted.iloc[idxs, :]["coords"])), 0.5)


# In[254]:


for i in non_max_suppression_fast(np.array(list(subsetted.iloc[idxs, :]["coords"])), 0.5)[0]: 
    coords = i
    xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
#     name = i[1][1]
    cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#     plt.title(name)
#     plt.imshow(img)
#     plt.show()


# In[255]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)


# In[256]:


plt.imshow(img_test_1)


# In[257]:


from scipy import ndimage, misc

from PIL import Image

# im = Image.open(index)

image_array = np.zeros((len(coords_2), 256,256,3))

image_array.shape


# In[259]:



from tqdm.notebook import tqdm

counter = 0
for i in tqdm(coords_2):
    xmin, ymin, xmax, ymax = i[0], i[1], i[2], i[3]
    cropped = img_2[ymin:ymax,xmin:xmax]
    cropped = Image.fromarray(cropped)
    cropped = cropped.resize((256,256),Image.ANTIALIAS)
    
    cropped = np.array(cropped)
    cropped = cropped/255.0
    image_array[counter] = cropped
    counter =counter + 1
#     print(cropped.shape)
#     plt.imshow(cropped)
#     plt.show()

image_array.shape


# In[260]:


# del(image_array)


predicted_proba = model.predict(image_array, verbose = 1, batch_size = 32)

del(image_array)

indexes = predicted_proba >= 0.5

len(indexes)


# In[261]:


new_dict = {0: 'bowtie',
 1: 'chevron',
 2: 'circle',
 3: 'line',
 4: 'other',
 5: 'polygon',
 6: 'quadrilateral',
 7: 'ribbon',
 8: 'star',
 9: 'text',
 10: 'triangle'}


# In[262]:


catch_catch = []
for i1 in indexes: 
#     catch = []
    catch = [new_dict[i] for i,n in enumerate(i1) if n == True]
    catch_catch.append(catch)    


# In[263]:


from collections import Counter


# In[264]:


Counter([j for i in catch_catch for j in i])


# In[266]:


pred_df_1 = pd.DataFrame({"coords":coords_2, 
             "preds": catch_catch})


# In[267]:


pred_df_1.head()


# In[268]:


idx = [i for i,n in enumerate(pred_df_1["preds"]) if len(n) > 0]


# In[269]:


subsetted = pred_df_1.iloc[idx,:].reset_index(drop = True)


# In[270]:


subsetted


# In[271]:


np.unique(subsetted["preds"])


# In[272]:


idxs = [i for i,n in enumerate(subsetted["preds"]) if "line" in n]


# In[273]:


subsetted.iloc[idxs, :]


# In[274]:


img_test_1 = img_2.copy()


# In[275]:


for i in subsetted.iloc[idxs, :]["coords"]: 
    coords = i
    xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
#     name = i[1][1]
    cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#     plt.title(name)
#     plt.imshow(img)
#     plt.show()


# In[276]:


plt.imshow(img_test_1)


# In[285]:


img_test_1 = img_2.copy()


# In[287]:


non_max_suppression_fast(np.array(list(subsetted.iloc[idxs, :]["coords"])), 0.1)


# In[288]:


for i in non_max_suppression_fast(np.array(list(subsetted.iloc[idxs, :]["coords"])), 0.2)[0]: 
    coords = i
    xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
#     name = i[1][1]
    cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#     plt.title(name)
#     plt.imshow(img)
#     plt.show()


# In[289]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)


# In[290]:


plt.imshow(img_test_1)


# In[ ]:





# In[ ]:




