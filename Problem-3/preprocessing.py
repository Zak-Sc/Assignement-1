# -*- coding: utf-8 -*-
import cv2
from tqdm import tqdm
import numpy as np
import os
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
TEST_DIR='./dataset/test'
IMG_SIZE=256

def process_test_data():
        testing_data=[]
        for img in tqdm(os.listdir(TEST_DIR)):
            img_id=int(img.split(".")[0])
            path=os.path.join(TEST_DIR,img)
            img=cv2.resize(cv2.imread(path),(IMG_SIZE,IMG_SIZE))
            testing_data.append([np.array(img),img_id])
        np.save('test_data.npy',testing_data)
        return testing_data
test_data=process_test_data()
sorted_test=sorted(test_data, key=lambda a_entry: a_entry[1]) 
model=load_model('model-10')

with open('submission-file-6-02-model10.csv','w') as f: f.write('id,label\n')
with open('submission-file-6-02-model10.csv','a') as f:
   for data in tqdm(sorted_test):
    img_num=data[1]
    img_data=data[0]
    orig=img_data
    data=img_data.reshape(1,IMG_SIZE,IMG_SIZE,3)
    model_out=model.predict(data)
    print(np.argmax(model_out))
    if np.argmax(model_out)==0:
        f.write('{},{}\n'.format(img_num,'Dog'))
    else: 
        f.write('{},{}\n'.format(img_num,'Cat'))

   