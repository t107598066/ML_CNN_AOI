# -*- coding: utf-8 -*-
import os 
import tensorflow as tf 
from PIL import Image  
import pandas as pd

train_path = '../aoi/train_images'
writer = tf.python_io.TFRecordWriter('train.tfrecords') 
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

for img_name in os.listdir(train_path):
    
    img_path = train_path +'/'+ img_name    #每個圖片的地址
    img = Image.open(img_path)
    #label
    record = pd.read_csv('../aoi/train.csv')

    label_record = record[record['ID'] == img_name]
    label_index = label_record.iloc[0]['Label']
#     img = img.resize((512, 512))
    img_raw = img.tobytes()  #將圖片轉化為二進制格式
    example = tf.train.Example(features = tf.train.Features(feature = {
                                                                       "label": _int64_feature(label_index),
                                                                       "img_raw": _bytes_feature(img_raw),                                                                          
                                                                       }))
    writer.write(example.SerializeToString())  #序列化為字符串
writer.close()
