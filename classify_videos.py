import os
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

base_path = '.\\uploadVid\\'
checkpoint_filepath = '.\\model'

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

def get_filename(file_path):
    file_basename = os.path.basename(file_path)
    return file_basename


upload_files_folder = [os.path.join(base_path, filename) for filename in os.listdir(base_path)]
print(upload_files_folder)

upload_files=[]
for file in upload_files_folder:
    file_base=os.path.basename(file)
    upload_files.append(file_base)

print(upload_files)
for filename in upload_files:
    print(filename)
    if (filename.endswith(".mp4")):
        print('Converting Video to Images...')
        count = 0
        video_file = os.path.join(base_path, filename)
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5) 
        while(cap.isOpened()):
            frame_id = cap.get(1) 
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frame_id % math.floor(frame_rate) == 0):
                print('Original Dimensions: ', frame.shape)
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif frame.shape[1] > 1000 and frame.shape[1] <= 1900 :
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1
                print('Scale Ratio: ', scale_ratio)

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                print('Resized Dimensions: ', new_frame.shape)

                new_filename = '{}-{:03d}.png'.format(os.path.join(base_path, get_filename_only(filename)), count)
                count = count + 1
                cv2.imwrite(new_filename, new_frame)
        cap.release()
        print("Done!")
    else:
        continue


best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))
predictions = []
input_size = 128

sum_real=0
count=0
for upload_file in upload_files_folder:
    if (upload_file.endswith(".png")):
        count+=1
        img = image.load_img(upload_file, target_size=(input_size, input_size))
        img = image.img_to_array(img)
        img = img / 255.0 
        img = np.expand_dims(img, axis=0) 
        
        prediction = best_model.predict(img)
        prediction_real=round(prediction[0][0]*100,2)
        sum_real=sum_real+prediction_real
        prediction_fake=round(100-prediction_real,2)
        print(type(prediction[0][0]))
        predictions.append({
            "Filename": os.path.basename(upload_file),
            "Real (%)": prediction_real,  
            "Fake (%)": prediction_fake
    })
        output_real=round(float(sum_real)/count,2)
        output_fake=round(100-output_real,2)
        

print("Real (%) = ",output_real,"\nFake (%) = ",output_fake)
    

