import tensorflow as tf
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

x = tf.placeholder(tf.float32,shape=[None,10800])
y_true = tf.placeholder(tf.float32,shape=[None,101])

x_image = tf.reshape(x,[-1,60,60,3])

convo_1 = convolutional_layer(x_image,shape=[1,1,3,32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[1,1,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling,[-1,15*15*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout,101)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

#----------------------------------------------------------------------------------
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,20])

x_image = tf.reshape(x,[-1,28,28,1])

convo_1 = convolutional_layer(x_image,shape=[6,6,3,32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout,20)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()


#---------------------------------------<< Run >> -------------------------------

epochs = 4000

with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(epochs):
        
        batch_x , batch_y = train_X[i:i+50],train_y[i:i+50]
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:test_X,y_true:test_y,hold_prob:1.0}))
            print('\n')


#------------------------------------------------<<get pixels >>-----------------------------------


import numpy as np
from PIL import Image
from PIL import ImageOps
import glob
import os
dir = 'fruits/Training'
ls=[]
labels = []
subdirs = [x[1] for x in os.walk(dir)]
k=0
for subdir in subdirs[0]:
    k+=1
    print(str(subdir))
    c=0
    for filename in glob.glob('fruits/Training/'+str(subdir) +'/*.jpg'):
        if(c!=100):
            image = Image.open(filename)
            size = (60, 60)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)#.convert('LA')
            pix = np.array(image)
            pix = pix.flatten()
            pix = pix.tolist()
            ls.append(pix)
            labels.append(k)
            c+=1
        else:
            break


#------------------------------------------------------------------------------


dataset = np.array(ls)
lab     = np.array(lan)

#---------------------------------------------------------------------------




lan=[]
for k in labels:
    lab=[]
    for x in range(1,102):   
        if(x==k):
            lab.append(1)
        else:
            lab.append(0)
    lan.append(lab)
lab = np.array(lan)
train_X,test_X,train_y,test_y = train_test_split(dataset,lab,random_state=0)

#----------------------------------------------------------------------------------