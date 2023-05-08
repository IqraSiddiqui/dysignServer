from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
import io
from pdf2image import convert_from_path, convert_from_bytes
from io import BytesIO
from tensorflow.keras.models import load_model


app = Flask(__name__)
cors = CORS(app)

# load the model
model = load_model('dysignIncModel.h5')

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
    # Convert PNG file to JPG using PIL
    img = Image.open(file)
    img = img.convert('RGB')

    # Save the image as JPG in memory
    jpg_io = io.BytesIO()
    img.save(jpg_io, format='JPEG')
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


# Your other functions...
def cleaning(thresh):
    
    thresh = thresh.point(lambda x: 0 if x < 128 else 255, '1')

    # Horizontal lines
    h_kernel = Image.new('1', (50, 1), 1)
    remove_h = thresh.filter(ImageMorphology.Open((2, 2), kernel=h_kernel))
    cnts_h = remove_h.find_contours()
    draw = ImageDraw.Draw(img_pil)
    for c in cnts_h:
        draw.line(c, fill=255, width=2)

    # Vertical lines
    v_kernel = Image.new('1', (1, 50), 1)
    remove_v = thresh.filter(ImageMorphology.Open((2, 2), kernel=v_kernel))
    cnts_v = remove_v.find_contours()
    for c in cnts_v:
        draw.line(c, fill=255, width=2)

    return np.array(img_pil)

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
        # read image file
        img = Image.open(file)

    # preprocess image and run model prediction
    # replace the following with your actual model code
    # img_arr = preprocess_image(img_arr)  # replace with your preprocessing code
    img = img.resize((150, 150))
    img = img.convert('RGB')  # Ensure the image has three channels (R, G, B)
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)  # Add a batch dimension

    # img_pil = Image.fromarray(page_arr)
    # img = img_pil.convert('L')

    # cleaned_img = cleaning(img)
    # wrdLst = extract_words(cleaned_img)
    # df = extract_features(wrdLst)

    # df = df.fillna(0)
    # df = df.astype('float32')
    # df_test = df.to_numpy()

    prediction = model.predict(img_arr)
    # finalP = prediction[0][0]
    finalP = str(prediction[0][0]*100000000000000000000000000000000000000000)
    print(finalP)
    # print(secP)

    # return prediction as JSON
    response = jsonify({'prediction': [[float(finalP[0:2])/100]]})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
