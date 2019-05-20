import tensorflow as tf 
def read_and_decode(filename): # 讀入tfrecords
    filename_queue = tf.train.string_input_producer([filename],shuffle=True)#生成一個queue隊列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#將image數據和label取出來
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [ 512, 512 ,1])  #reshape為512*512的單通道圖片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中拋出img張量
    label = tf.cast(features['label'], tf.int32) #在流中拋出label張量
    return img, label

def read_and_decode_test(filename): # 讀入tfrecords
    filename_queue = tf.train.string_input_producer([filename],shuffle=False)#生成一個queue隊列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#將image數據和label取出來
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [ 512, 512 ,1])  #reshape為512*512的單通道圖片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中拋出label張量
    return img
