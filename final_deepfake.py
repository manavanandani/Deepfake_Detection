from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB0 #EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.utils as image
from tensorflow.keras.models import load_model

# Flask constructor
app = Flask(__name__)  


# UPLOAD_FOLDER_VIDEO = 'uploadVid'
# app.config['UPLOAD_FOLDER_VIDEO'] = UPLOAD_FOLDER_VIDEO
UPLOAD_FOLDER_IMAGE = 'uploadImg'
app.config['UPLOAD_FOLDER_IMAGE'] = UPLOAD_FOLDER_IMAGE


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/image', methods=['POST'])
def upload_image():
    
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        
    # # Define the path to the uploadImg folder
    # upload_path = '.\\uploadImg\\'
    # checkpoint_filepath = '.\\model'

    # # List the files in the uploadImg folder
    # upload_files_image = [os.path.join(upload_path, filename) for filename in os.listdir(upload_path)]
    # print(upload_files_image)
    # # load the saved model that is considered the best
    # best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

    # # Create an empty list to store prediction resuto store prediction results
    # predictions = []
    # input_size = 128
    # images_to_predict = []  # Create a list to accumulate images
    # for upload_file in upload_files_image:
    #     img = tf.keras.utils.load_img(upload_file, target_size=(input_size, input_size))
    #     img = tf.keras.utils.img_to_array(img)
    #     img = img / 255.0
    #     img = np.expand_dims(img, axis=0)
    #     images_to_predict.append(img)  # Accumulate images to predict

    # # Make predictions using the model
    #     prediction = best_model.predict(img)
    #     prediction_real=round(prediction[0][0]*100,2)
    #     prediction_fake=round(100-prediction_real,2)
    #     print(type(prediction[0][0]))
    #     predictions.append({
    #         "Filename": os.path.basename(upload_file),
    #         "Real (%)": prediction_real,  # Assuming a binary output
    #         "Fake (%)": prediction_fake
    #     })

    # # # Create a DataFrame from the predictions
    # # prediction_df = pd.DataFrame(predictions)

    # # Print or save the prediction results
    # print(predictions)
    
    return redirect(url_for('image'))

# @app.route('/video')
# def video():
#     return render_template('video.html')

# @app.route('/video', methods=['POST','GET'])
# def upload_video():
    
#     uploaded_file_video = request.files['file']
#     if uploaded_file_video.filename != '':
#         uploaded_file_video.save(os.path.join(app.config['UPLOAD_FOLDER_VIDEO'], uploaded_file_video.filename))
        
#     base_path_video = '.\\uploadVid\\'

#     def get_filename_only(file_path):
#         file_basename = os.path.basename(file_path)
#         filename_only = file_basename.split('.')[0]
#         return filename_only

#     upload_files_folder = [os.path.join(base_path_video, filename) for filename in os.listdir(base_path_video)]
#     print(upload_files_folder)

#     upload_files=[]
    
#     for file in upload_files_folder:
#         file_base=os.path.basename(file)
#         upload_files.append(file_base)

#     print(upload_files)
    
#     for filename in upload_files:
#         print(filename)
#         if (filename.endswith(".mp4")):
#             print('Converting Video to Images...')
#             count = 0
#             video_file = os.path.join(base_path_video, filename)
#             cap = cv2.VideoCapture(video_file)
#             frame_rate = cap.get(5) #frame rate
#             while(cap.isOpened()):
#                 frame_id = cap.get(1) #current frame number
#                 ret, frame = cap.read()
#                 if (ret != True):
#                     break
#                 if (frame_id % math.floor(frame_rate) == 0):
#                     print('Original Dimensions: ', frame.shape)
#                     if frame.shape[1] < 300:
#                         scale_ratio = 2
#                     elif frame.shape[1] > 1900:
#                         scale_ratio = 0.33
#                     elif frame.shape[1] > 1000 and frame.shape[1] <= 1900 :
#                         scale_ratio = 0.5
#                     else:
#                         scale_ratio = 1
#                     print('Scale Ratio: ', scale_ratio)

#                     width = int(frame.shape[1] * scale_ratio)
#                     height = int(frame.shape[0] * scale_ratio)
#                     dim = (width, height)
#                     new_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#                     print('Resized Dimensions: ', new_frame.shape)

#                     new_filename = '{}-{:03d}.png'.format(os.path.join(base_path_video, get_filename_only(filename)), count)
#                     count = count + 1
#                     cv2.imwrite(new_filename, new_frame)
#             cap.release()
#             print("Done!")
#         else:
#             continue

#     predictions_video = []
#     input_size = 128
#     # Iterate through the files in the uploadImg folder
#     sum_real=0
#     count=0
#     for upload_file in upload_files_folder:
#         if (upload_file.endswith(".png")):
#             count+=1
#             # Load and preprocess the image
#             vid = tf.keras.utils.load_img(upload_file, target_size=(input_size, input_size))
#             vid = tf.keras.utils.img_to_array(vid)
#             vid =vid / 255.0  # Normalize the image
#             vid = np.expand_dims(vid, axis=0)  # Add a batch dimension
#         # Make predictions using the model
            
#             prediction_video = best_model.predict(vid)
#             prediction_video_real=round(prediction_video[0][0]*100,2)
#             sum_real=sum_real+prediction_video_real
#             prediction_fake=round(100-prediction_video_real,2)
#             print(type(prediction_video[0][0]))
#             predictions_video.append({
#                 "Filename": os.path.basename(upload_file),
#                 "Real (%)": prediction_video_real,  # Assuming a binary output
#                 "Fake (%)": prediction_fake
#         })
#             output_real=round(float(sum_real)/count,2)
#             output_fake=round(100-output_real,2)
            
#             print("Real: ",output_real,"\nFake: ",output_fake)

#     return redirect(url_for('video'))

if __name__=='__main__':
    app.run()