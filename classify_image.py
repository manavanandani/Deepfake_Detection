import os
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

upload_path = '.\\uploadImg\\'
checkpoint_filepath = '.\\model'

upload_files = [os.path.join(upload_path, filename) for filename in os.listdir(upload_path)]
print(upload_files)

best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))


predictions = []
input_size = 128

for upload_file in upload_files:

    img = image.load_img(upload_file, target_size=(input_size, input_size))
    img = image.img_to_array(img)
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)  
    prediction = best_model.predict(img)
    prediction_real=round(prediction[0][0]*100,2)
    prediction_fake=round(100-prediction_real,2)
    print(type(prediction[0][0]))
    predictions.append({
        "Filename": os.path.basename(upload_file),
        "Real (%)": prediction_real,  
        "Fake (%)": prediction_fake
    })


print(predictions)