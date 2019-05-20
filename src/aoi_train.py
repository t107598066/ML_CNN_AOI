import tensorflow as tf 
import numpy as np
import readTFrecord
import math
import time

batch_size = 4
def write_to_file(output_data,filename = '../aoi/test.csv'):
    with open(filename,'w') as f:
        f.write('ID,Label\n')
        for i in range(len(output_data)):
            f.write('test_%05d.png' %(i) +','+str(output_data[i])+'\n')

def sec_time(exc_time):
    m=0
    s=0
    while (exc_time/60)>0:
        m += 1
        s = exc_time%60
        exc_time %= 60        
    return m,s

def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
    return one_hot_label

#initial weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.02)
    return tf.Variable(initial)
#initial bias
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

#convolution layer
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#max_pool layer
def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

def max_pool_2x2_stride1(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, [batch_size,512,512,1],name="x")
y_ = tf.placeholder(tf.float32, [batch_size,6],name="y_")

#first convolution and max_pool layer
W_conv1 = weight_variable([3,3,1,64]) #patch = 5*5, in_size = 1, out_size = 32
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) #output_size = 512*512*32

W_conv1_b = weight_variable([3,3,64,64]) #patch = 5*5, in_size = 1, out_size = 32
b_conv1_b = bias_variable([64])
h_conv1_b= tf.nn.relu(conv2d(h_conv1, W_conv1_b) + b_conv1_b) #output_size = 512*512*32
h_pool1_b = max_pool_2x2_stride1(h_conv1_b) #output_size = 256*256*32

#second convolution and max_pool layer
W_conv2 = weight_variable([3,3,64,128]) #patch = 5*5, in_size = 32, out_size = 64
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_conv1_b, W_conv2) + b_conv2) #output_size = 256*256*64

W_conv2_b = weight_variable([3,3,128,128]) #patch = 5*5, in_size = 32, out_size = 64
b_conv2_b = bias_variable([128])
h_conv2_b = tf.nn.relu(conv2d(h_conv2, W_conv2_b) + b_conv2_b) #output_size = 256*256*64
h_pool2_b = max_pool_2x2(h_conv2_b) #out_put_size = 128*128*64

#third convolution and max_pool layer
W_conv3 = weight_variable([3,3,128,256]) #patch = 5*5, in_size = 64, out_size = 128
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_pool2_b, W_conv3) + b_conv3) #output_size = 128*128*128

W_conv3_b = weight_variable([3,3,256,256]) #patch = 5*5, in_size = 64, out_size = 128
b_conv3_b = bias_variable([256])
h_conv3_b = tf.nn.relu(conv2d(h_conv3, W_conv3_b) + b_conv3_b) #output_size = 128*128*128

W_conv3_c = weight_variable([3,3,256,256]) #patch = 5*5, in_size = 64, out_size = 128
b_conv3_c = bias_variable([256])
h_conv3_c = tf.nn.relu(conv2d(h_conv3_b, W_conv3_c) + b_conv3_c) #output_size = 128*128*128

W_conv3_d = weight_variable([3,3,256,256]) #patch = 5*5, in_size = 64, out_size = 128
b_conv3_d = bias_variable([256])
h_conv3_d = tf.nn.relu(conv2d(h_conv3_c, W_conv3_d) + b_conv3_d) #output_size = 128*128*128
h_pool3_d = max_pool_2x2(h_conv3_d) #out_put_size = 64*64*128

#fourth convolution and max_pool layer
W_conv4 = weight_variable([3,3,256,512]) #patch = 5*5, in_size = 128, out_size = 256
b_conv4 = bias_variable([512])
h_conv4 = tf.nn.relu(conv2d(h_pool3_d, W_conv4) + b_conv4) #output_size = 64*64*256

W_conv4_b = weight_variable([3,3,512,512]) #patch = 5*5, in_size = 128, out_size = 256
b_conv4_b = bias_variable([512])
h_conv4_b = tf.nn.relu(conv2d(h_conv4, W_conv4_b) + b_conv4_b) #output_size = 64*64*256

W_conv4_c = weight_variable([3,3,512,512]) #patch = 5*5, in_size = 128, out_size = 256
b_conv4_c = bias_variable([512])
h_conv4_c = tf.nn.relu(conv2d(h_conv4_b, W_conv4_c) + b_conv4_c) #output_size = 64*64*256

W_conv4_d = weight_variable([3,3,512,512]) #patch = 5*5, in_size = 128, out_size = 256
b_conv4_d = bias_variable([512])
h_conv4_d = tf.nn.relu(conv2d(h_conv4_c, W_conv4_d) + b_conv4_d) #output_size = 64*64*256
h_pool4_d = max_pool_4x4(h_conv4_d) #out_put_size = 32*32*256

#fourth convolution and max_pool layer
W_conv5 = weight_variable([3,3,512,512]) #patch = 5*5, in_size = 128, out_size = 256
b_conv5 = bias_variable([512])
h_conv5 = tf.nn.relu(conv2d(h_pool4_d, W_conv5) + b_conv5) #output_size = 32*32*512

W_conv5_b = weight_variable([3,3,512,512]) #patch = 5*5, in_size = 128, out_size = 256
b_conv5_b = bias_variable([512])
h_conv5_b = tf.nn.relu(conv2d(h_conv5, W_conv5) + b_conv5) #output_size = 32*32*512

W_conv5_c = weight_variable([3,3,512,512]) #patch = 5*5, in_size = 128, out_size = 256
b_conv5_c = bias_variable([512])
h_conv5_c = tf.nn.relu(conv2d(h_conv5_b, W_conv5_c) + b_conv5_c) #output_size = 32*32*512

W_conv5_d = weight_variable([3,3,512,512]) #patch = 5*5, in_size = 128, out_size = 256
b_conv5_d = bias_variable([512])
h_conv5_d = tf.nn.relu(conv2d(h_conv5_c, W_conv5_d) + b_conv5_d) #output_size = 32*32*512
h_pool5_d = max_pool_4x4(h_conv5_d) #out_put_size = 16*16*512

#變成全連接層 Layer1
reshape = tf.reshape(h_pool5_d,[batch_size, -1])
dim = reshape.get_shape()[1].value
W_fc1 = weight_variable([dim, 4096])
b_fc1 = bias_variable([4096])
h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
#dropout
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#輸出層 Layer2
W_fc3 = weight_variable([4096,6])
b_fc3 = bias_variable([6])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)#prediction
logits = y_conv

#(小處理)將logits乘以1賦值給logits_eval，定義name，方便在後續調用模型時通過tensor名字調用輸出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval') 

#損失函數及優化算法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv+ 1e-10), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


image, label = readTFrecord.read_and_decode("train.tfrecords")

img_test = readTFrecord.read_and_decode_test("test.tfrecords")

#使用shuffle_batch可以隨機打亂輸入
img_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)

img_test_batch = tf.train.batch([img_test],
                                batch_size=batch_size,capacity=2000)


one_hot_label = []
all_acc = 0
all_loss = 0
cout_acc = []
cout_loss = []
epoch = 100
cou_test_img = 0
output_data = []
with tf.Session() as sess:      
    #saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    #定義多線程
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    #定義訓練圖像和標籤
    train_img = 2527
    batch_index = int(train_img/batch_size)
    example=np.zeros((batch_size,512,512,1))
    l=np.zeros((batch_size,6))
    test_img = 10141
    batch_test_index = int(math.ceil(test_img/batch_size))
    
    test_example=np.zeros((batch_size,512,512,1))
    try:
        print("start training-------------------------------------------")
        for j in range(epoch):            
            #將數據存入example和l
            train_acc = 0
            train_loss = 0
            start = time.clock()
            for i in range(batch_index):
                example,l=sess.run([img_batch,label_batch])
                l = one_hot(l,6)
                one_hot_label.append(l)
            #開始訓練
                sess.run(train_step,feed_dict={x:example,y_:l,keep_prob:0.5})
                sess_acc = sess.run(accuracy,feed_dict={x:example,y_:l,keep_prob:0.5})
                sess_loss = sess.run(cross_entropy,feed_dict={x:example,y_:l,keep_prob:0.5})
                train_acc += sess_acc
                train_loss += sess_loss
                print('train epoch : [%2d/%2d] '%(j+1,epoch),' step : [%04d/%04d]' %(i+1,batch_index),'Accuracy= %f'%sess_acc)
                print('loss: %f' % sess_loss)
                print("-" * 60)
            print('train batch Accuracy = %f' % (train_acc / batch_index),'train batch Loss = %f' % (train_loss / batch_index))
            end = time.clock()
            print(end-start)    
            print("-" * 60)
            cout_acc.append(train_acc / batch_index)
            cout_loss.append(train_loss / batch_index)
            all_acc += float(train_acc / batch_index)
            all_loss += float(train_loss / batch_index)  
        all_acc = float(all_acc/epoch)
        all_loss = float(all_loss/epoch)
        print("-" * 60)
        print("sum of acc: %7f   sum of loss: %7f " % (all_acc,all_loss))  
        #save model
        #saver.save(sess,'training_model/model.ckpt',global_step=epoch)
        print("training finish!")
        print("-" * 60)
        
        #
        #test_path = '../aoi/test_images'
        #data = read_img(test_path)
        
        print("start testing")
        for i in range (batch_test_index):
            test_example=sess.run(img_test_batch)            
            classification_result = sess.run(logits, feed_dict={x:test_example,keep_prob:1.})
            print(classification_result)
            #print(tf.argmax(classification_result,1).eval())
            output = []
            output = tf.argmax(classification_result,1).eval()
            for j in range(len(output)):
 
                if cou_test_img <= test_img:            
                    output_data.append(output[j])
                    print('第 %5d' % (cou_test_img) ,'AOI预测: %3d' % (output[j]))
                cou_test_img+=1 
        #寫入test.csv
        write_to_file(output_data) 
        print("predict finish---------")
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:        
        coord.request_stop()
        coord.join(threads)
        sess.close()
