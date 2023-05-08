from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import cv2 
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
import scipy.misc
import joblib
import sklearn
import io
from pdf2image import convert_from_path
from pdf2image import convert_from_bytes
from io import BytesIO


app = Flask(__name__)
cors = CORS(app)


# load the model
model = joblib.load('Wmlpmodel.pkl')

def convert_to_jpg(file):
    # Convert PDF or PNG file to JPG
    if file.filename.lower().endswith('.pdf'):
        # Convert PDF to JPG
        image = convert_pdf_to_jpg(file)
    elif file.filename.lower().endswith('.png'):
        # Convert PNG to JPG
        image = convert_png_to_jpg(file)

    return image
def convert_png_to_jpg(file):
    # Convert PNG file to JPG using OpenCV
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert RGBA to RGB if PNG has transparency channel
    if img_rgb.shape[2] == 4:
        alpha_channel = img_rgb[:, :, 3]
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
        img_rgb[alpha_channel == 0] = [255, 255, 255]  # Convert transparent pixels to white

    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img_rgb)

    # Save the PIL Image as JPG in memory
    jpg_io = io.BytesIO()
    img_pil.save(jpg_io, 'JPEG')
    jpg_io.seek(0)

    # Return the JPG image
    return Image.open(jpg_io)

def convert_pdf_to_jpg(file_storage):
    # Read the PDF file from FileStorage
    pdf_data = file_storage.read()

    # Convert PDF to a list of PIL Image objects
    images = convert_from_bytes(pdf_data)

    # Convert the first image to JPEG and return as bytes
    jpg_data = BytesIO()
    images[0].save(jpg_data, format='JPEG')
    jpg_data.seek(0)

    return jpg_data

def extract_words(img):
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        words_lst = []
        kernel = np.ones((3, 30), np.uint8)
        roi = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        roi_cnts = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_cnts = roi_cnts[0] if len(roi_cnts) == 2 else roi_cnts[1]
        #
        count = 1
        for c in roi_cnts:
            x, y, w, h = cv2.boundingRect(c)
            pad = 10
            if w > 30 and h > 21:
                word = thresh[y+pad:y+h+pad, x+pad:x+w+pad]
                words_lst.append(word)
                count += 1
        return words_lst

def preprocess_image(img):
    # resize the image to (150, 150) using cv2
    #img = cv2.resize(img, (150, 150))

    # convert the image to an array

    # resize the image to (150, 150) using cv2
    img = cv2.resize(img, (150, 150))

    # if the image is grayscale, convert it to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # reshape the array to have the required dimensions
    img_array = np.reshape(img, (1, 150, 150, 3))

    # convert the array values to float32
    img_array = img_array.astype('float32')

    # normalize the pixel values to be between 0 and 1
    img_array /= 255.

    return img_array
# --------------------- FEATURES ---------------------------------------------------------------------

def word_curvature(img):
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return None
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    curvature_sum = 0
    curvature_list = []
    for contour in contours:
        if len(contour) > 2:
            poly = cv2.approxPolyDP(contour, epsilon=0.01*cv2.arcLength(contour,True), closed=True)
            curve = cv2.arcLength(poly,True)
            area = cv2.contourArea(poly)
            curvature = 4*np.pi*(area/curve**2)
            curvature_sum += curvature
            curvature_list.append(curvature)

    smoothness = None
    if contours:
        smoothness = curvature_sum / len(contours)
    return smoothness

def word_strokewidth(img):
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return None
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    stroke_widths = []
    for contour in contours:
        if len(contour) > 2:
            poly = cv2.approxPolyDP(contour, epsilon=0.01*cv2.arcLength(contour,True), closed=True)
            distances = []
            for i in range(len(poly)):
                for j in range(i+1, len(poly)):
                    distances.append(np.linalg.norm(poly[i]-poly[j]))
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            stroke_width = max_dist - min_dist
            stroke_widths.append(stroke_width)
    smoothness = None
    if stroke_widths:
        smoothness = np.mean(stroke_widths)
    return smoothness

def density(img):
    '''Density based on ratio of black and white - can be used for chars and words. Returns single value.'''
    grey_levels = 256
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return None
    hist, bin_edges = np.histogram(img.ravel(),bins=grey_levels)
    return np.mean(hist) #return density value

def distances_spatial(img):
    '''Takes word or char image as input and returns list of averaged values for x,y,w,h, and distance as a list. Returns -1 if no contours found.'''
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return None
    W, H = img.shape[:2]
    result = img.copy()

    contours = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1] #??
    cv2.line(result, (W, W), (H, W), (255, 255, 255), 4)
    ROI_number = 0
    avg_dist = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10:
            x_lst = []
            y_lst = []
            w_lst = []
            h_lst = []
            dist_lst = []
            x,y,w,h = cv2.boundingRect(c)
            cX = x + w//2
            cY = y + h
            D = dist.euclidean((cX,   cY), (cX, W))
            avg_dist += D
            x_lst.append(x)
            y_lst.append(y)
            w_lst.append(w)
            h_lst.append(h)
            ROI = 255 - img[y:y+h, x:x+w]
            ROI_number += 1
            dist_lst.append(D)
        else:
            return None
    if contours == []:
        return None
    return [np.mean(x_lst), np.mean(y_lst), np.mean(w_lst), np.mean(h_lst), np.mean(dist_lst)]

#------------------------------------
def extract_features(words_lst):
    df = pd.DataFrame(columns=['curvature','strokewidth','density','x','y','w','h','distance'])
    for word in words_lst:
        # print(filename) # debugging
        curv = word_curvature(word)
        sw = word_strokewidth(word)
        dens = density(word)
        spatial = distances_spatial(word)
        if spatial is None:
             x=y=w=h=dist=None
        else:
            x,y,w,h,dist = spatial

        new_row = [curv, sw, dens, x,y,w,h,dist]
        df.loc[len(df)] = new_row
    return df

    #------------------------------------

def cleaning(img):
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #horizontal lines
        h_kernel = np.ones((1, 50), np.uint8)
        remove_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        cnts_h = cv2.findContours(remove_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_h = cnts_h[0] if len(cnts_h) == 2 else cnts_h[1]
        for c in cnts_h:
            cv2.drawContours(img, [c], -1, (255, 255, 255), 2)
        #vertical lines
        v_kernel = np.ones((50, 1), np.uint8)
        remove_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        cnts_v = cv2.findContours(remove_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_v = cnts_v[0] if len(cnts_v) == 2 else cnts_v[1]
        for c in cnts_v:
            cv2.drawContours(img, [c], -1, (255, 255, 255), 2)
        return img
# define route for image upload
@app.route('/predict', methods=['POST'])
def predict():
    # check if request contains file
    if 'file' not in request.files:
        return 'No file uploaded', 40001


    # get uploaded file
    file = request.files['file']

    # check file extension
    if file.filename.lower().endswith(('.pdf', '.png')):
        # convert PDF or PNG to JPG
        img = convert_to_jpg(file)
    else:
        # read image file and convert to numpy array
        img = Image.open(file.stream)

    # preprocess image and run model prediction
    # replace the following with your actual model code
    #img_arr = preprocess_image(img_arr) # replace with your preprocessing code
    # resize the image to the required dimensions
    #img = img.resize((150, 150))
    page_arr=np.array(img)
    img = cv2.cvtColor(page_arr, cv2.COLOR_RGB2GRAY)
    cleaned_img = cleaning(img)
    # words.extend(extract_words(cleaned_img))
    predLst=[]
    # img = np.array(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wrdLst=extract_words(cleaned_img)
    df = extract_features(wrdLst)

    df = df.fillna(0)
    df=df.astype('float32')
    df_test=df.loc[:, :]
    df_test=df_test.to_numpy()


    # for wrd in wrdLst:
    #     img_array = preprocess_image(wrd)

    prediction = model.predict_proba(df_test)
    #print(prediction)
    finalP=((np.sum(prediction)-np.min(prediction))/(np.max(prediction)))/100
    secP=np.sum(prediction)/len(prediction)
    print(finalP)
    print(secP)

    # return prediction as JSON
    response = jsonify({'prediction': [[float(finalP)]]})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
