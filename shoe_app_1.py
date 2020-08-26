import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import cv2
from tqdm import tqdm
# import sys
import gc
import pandas as pd

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], False)



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




def get_rpn(filename,method = "quality"):
    gc.collect()
#     base_path = r"D:\Shoes\Shoes_with_annotations\\"
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#     base_path = r"D:\Shoes\Shoes_with_annotations\\"
    # filename = filename
    # samp_path = filename
    im = filename
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



model = tf.keras.models.load_model("D:\Shoes\Latest_Iteration\Models\connor_5.h5")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
   dcc.Input(
        id='number-in',
        value=0.5,
        style={'fontSize':28}
    ), 
   
   html.Label('Multi-Select Dropdown'),
    dcc.Dropdown(id = "class_type",
        options=[{'label': 'all', 'value': 'all'},
            {'label': 'bowtie', 'value': 'bowtie'},
            {'label': "chevron", 'value': 'chevron'},
            {'label': 'circle', 'value': 'circle'}, 
            {'label': 'line', 'value': 'line'}, 
            {'label': 'other', 'value': 'other'}, 
            {'label': 'polygon', 'value': 'polygon'}, 
            {'label': 'quadrilateral', 'value': 'quadrilateral'}, 
            {'label': 'ribbon', 'value': 'ribbon'},
            {'label': 'star', 'value': 'star'},
            {'label': 'text', 'value': 'text'},
            {'label': 'triangle', 'value': 'triangle'}
        ],
        value=["all"],
        multi=True
    ),
    
   dcc.Input(
        id='number-in1',
        value=0.5,
        style={'fontSize':28}
    ),

   
    html.Button(
        id='submit-button',
        n_clicks=0,
        children='Submit',
        style={'fontSize':28}
    ),

    html.Div(id='output-image-upload'),
])


def parse_contents(contents, cutoff1,class_types ,cutoff2,filename):
    print(filename[0])
    # print(datetime.datetime.fromtimestamp(date[0]))
    print(cutoff1)
    print(class_types)
    # print(len(contents))
    print(len(contents[0].split(",")))
    print(contents[0].split(",")[0])
    string_part = contents[0].split(",")[-1]
    # r = base64.decodebytes(string_part)
    # print(string_part)
    imgdata = base64.b64decode(string_part)
    del(string_part)
    im = Image.open(io.BytesIO(imgdata))
    # print(np.array(im).shape)
    image_array_1 = np.array(im)[:,:,:3]
    # del(im)
    coords = get_rpn(image_array_1,method = "fast")
    image_array = np.zeros((len(coords), 256,256,3)).astype("float16")
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
    print(image_array.shape)
    predicted_proba = model.predict(image_array, verbose = 1, batch_size = 32)
    del(image_array)
    indexes = predicted_proba >= cutoff1
    len(indexes)
    catch_catch = []
    for i1 in indexes: 
#     catch = []
        catch = [new_dict[i] for i,n in enumerate(i1) if n == True]
        catch_catch.append(catch)
    pred_df = pd.DataFrame({"coords":coords, 
             "preds": catch_catch})
    print(pred_df.head())
    idx = [i for i,n in enumerate(pred_df["preds"]) if len(n) > 0]
    subsetted = pred_df.iloc[idx,:].reset_index(drop = True)
    pred_classes = np.unique([j for i in np.unique(subsetted["preds"]) for j in i])
    img_test_1 = image_array_1.copy()
    print(np.max(img_test_1))
    if "all" in class_types:
        for class_name in pred_classes: 
            idxs = [i for i,n in enumerate(subsetted["preds"]) if class_name in n]
            for i in non_max_suppression_fast(np.array(list(subsetted.iloc[idxs, :]["coords"])), cutoff2)[0]: 
                coords = i
                xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
    #     name = i[1][1]
                cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), col_scheme[class_name], 2)
                cv2.putText(img_test_1, class_name, (int(round((xmin+xmax)/2)), ymin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, col_scheme[class_name], 1)
    else: 
        pred_classes = list(set(pred_classes).intersection(set(class_types)))
        for class_name in pred_classes: 
            idxs = [i for i,n in enumerate(subsetted["preds"]) if class_name in n]
            for i in non_max_suppression_fast(np.array(list(subsetted.iloc[idxs, :]["coords"])), 0.5)[0]: 
                coords = i
                xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
    #     name = i[1][1]
                cv2.rectangle(img_test_1, (xmin, ymin), (xmax, ymax), col_scheme[class_name], 2)
                cv2.putText(img_test_1, class_name, (int(round((xmin+xmax)/2)), ymin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, col_scheme[class_name], 1)
        
    im_pil = Image.fromarray(img_test_1)
    if im_pil.mode != 'RGB':
        im_pil = im_pil.convert('RGB')
    buff = io.BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
#     print(cropped.shape)
#     plt.imshow(cropped)
#     plt.show()
    # print(len(coords))
    # predicted_proba = model.predict(image_array, verbose = 1, batch_size = 32)
    # base64_img_bytes = string_part.encode('utf-8')
    # decoded_image_data = base64.decodebytes(base64_img_bytes)
    # q = np.frombuffer(decoded_image_data, dtype=np.float64)
    # print
    # r = base64.decodebytes(decoded_image_data)
    # print(decoded_image_data.shape)
    # r = base64.decodebytes(contents)
    # q = np.frombuffer(r, dtype=np.float64)
    # q.shape
    # img = Image.open(io.BytesIO(contents))
    return html.Div([
         html.H5(filename[0]),
        # html.H6(datetime.datetime.fromtimestamp(date[0])),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src='data:image/png;base64,{}'.format(im_b64)),
        html.Hr(),
        html.Img(src='data:image/png;base64,{}'.format(im_b64))
    ])


@app.callback(Output('output-image-upload', 'children'),
               [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('number-in', 'value'),
               State('class_type', 'value'),
               State('number-in1', 'value'),
               State('upload-image', 'filename')
               ])
def update_output(n_clicks, list_of_contents, cutoff1,class_type ,cutoff2,list_of_names):
    if list_of_contents is not None:
        children = [parse_contents(list_of_contents, cutoff1,class_type, cutoff2,list_of_names)]
        return children


if __name__ == '__main__':
    app.run_server()