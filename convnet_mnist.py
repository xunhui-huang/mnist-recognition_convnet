####################tf.get_variable()版本,这版本训练有点问题，请用下面tf.Variable()代码
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

lr=0.005
BATCH_SIZE=100
NUM_EPOCHS=3000
train_keep_prob=0.8
test_keep_prob=1.0
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

x=tf.placeholder(shape=[None,784],dtype=tf.float32,name='x')
y=tf.placeholder(shape=[None,10],dtype=tf.float32,name='y')


def weights_init(shape,name):
    return tf.get_variable(name,shape=shape,dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1.0))

def bias_init(shape,name):
    return tf.get_variable(name,shape=shape,dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1.0)

def conv2d(x,conv_w):
    return tf.nn.conv2d(x,conv_w,strides=[1,1,1,1],padding='SAME')

def maxpool(x,k_size):
    return tf.nn.max_pool(x,k_size,strides=[1,2,2,1],padding='SAME')

def dropout(x,keep_prob):
    return tf.nn.dropout(x,keep_prob)

def inference(x,keep_prob):
    x_images=tf.reshape(x,[-1,28,28,1])
    with tf.variable_scope('conv_1',reuse=tf.AUTO_REUSE):
        weights_conv1=weights_init([3,3,1,32],'conv1_w')
        bias_conv1=bias_init([32],'conv1_b')
        h_conv1=tf.nn.relu(conv2d(x_images,weights_conv1)+bias_conv1)
        h_pool1=maxpool(h_conv1,[1,2,2,1])
    with tf.variable_scope('conv_2',reuse=tf.AUTO_REUSE):
        weights_conv2=weights_init([3,3,32,64],'conv2_w')
        bias_conv2=bias_init([64],'conv2_b')
        h_conv2=tf.nn.relu(conv2d(h_pool1,weights_conv2)+bias_conv2)
        h_pool2=maxpool(h_conv2,[1,2,2,1])
    with tf.variable_scope('fc1_layer',reuse=tf.AUTO_REUSE):
        weights_fc1=weights_init([7*7*64,1024],'fc1_w')
        bias_fc1=bias_init([1024],'fc1_b')
        h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,weights_fc1)+bias_fc1)
        h_fc1_drop=dropout(h_fc1,keep_prob)
    with tf.variable_scope('output_layer',reuse=tf.AUTO_REUSE):
        weights_fc2=weights_init([1024,10],'fc2_w')
        bias_fc2=bias_init([10],'fc2_b')
        output=tf.nn.softmax(tf.matmul(h_fc1_drop,weights_fc2)+bias_fc2)
    return output

train_y_pred=inference(x,train_keep_prob)
train_cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(train_y_pred,0.0001,1)),name='train_loss'))
train_correct_prediction=tf.equal(tf.argmax(train_y_pred,1),tf.argmax(y,1))
train_accuracy=tf.reduce_mean(tf.cast(train_correct_prediction,tf.float32),name='train_acc')
tf.get_variable_scope().reuse_variables()
test_y_pred=inference(x,test_keep_prob)
test_cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(test_y_pred,0.00001,1))),name='test_loss')
test_correct_prediction=tf.equal(tf.argmax(test_y_pred,1),tf.argmax(y,1))
test_accuracy=tf.reduce_mean(tf.cast(test_correct_prediction,tf.float32),name='test_acc')

train_op=tf.train.AdagradOptimizer(lr).minimize(train_cross_entropy)
init_op=tf.global_variables_initializer()
local_init_op=tf.local_variables_initializer()

gpu_options=tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)

tf.summary.scalar('train_loss',train_cross_entropy)
tf.summary.scalar('train_acc',train_accuracy)
summary_op=tf.summary.merge_all()
summary_writer=tf.summary.FileWriter('./convnet_mnist')
summary_writer.add_graph(tf.get_default_graph())

with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    for i in range(NUM_EPOCHS):
        train_batch_x,train_batch_y=mnist.train.next_batch(BATCH_SIZE)
        train_loss_value,train_summary,_=sess.run([train_cross_entropy,summary_op,train_op],feed_dict={x:train_batch_x,y:train_batch_y})
        summary_writer.add_summary(train_summary,i)
        if i%500==0:
            test_batch_x,test_batch_y=mnist.test.next_batch(BATCH_SIZE)
            test_loss_value,test_acc_value=sess.run([test_cross_entropy,test_accuracy],feed_dict={x:test_batch_x,y:test_batch_y})
            print('Epoch %d,test loss is %f' % (i,test_loss_value))
            print('Epoch %d,test accuracy is %f' % (i,test_acc_value))
            tf.Variable()
'''

# -*- coding: utf-8 -*-
###################tf.Variable()版本
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

lr=0.001
BATCH_SIZE=50
NUM_EPOCHS=5000
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

x=tf.placeholder(shape=[None,784],dtype=tf.float32,name='x')
y=tf.placeholder(shape=[None,10],dtype=tf.float32,name='y')
x_images=tf.reshape(x,[-1,28,28,1])

def weights_init(shape,name):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=0.1,name=name))

def bias_init(shape,name):
    return tf.Variable(initial_value=tf.constant(0.1,shape=shape,name=name))

def conv2d(x,conv_w):
    return tf.nn.conv2d(x,conv_w,strides=[1,1,1,1],padding='SAME')

def maxpool(x,k_size):
    return tf.nn.max_pool(x,k_size,strides=[1,2,2,1],padding='SAME')

def dropout(x,keep_prob):
    return tf.nn.dropout(x,keep_prob)

def inference(x,keep_prob):
    weights_conv1=weights_init([5,5,1,32],'conv1_w')
    bias_conv1=bias_init([32],'conv1_b')
    h_conv1=tf.nn.relu(conv2d(x,weights_conv1)+bias_conv1)
    h_pool1=maxpool(h_conv1,[1,2,2,1])
    weights_conv2=weights_init([5,5,32,64],'conv2_w')
    bias_conv2=bias_init([64],'conv2_b')
    h_conv2=tf.nn.relu(conv2d(h_pool1,weights_conv2)+bias_conv2)
    h_pool2=maxpool(h_conv2,[1,2,2,1])
    weights_fc1=weights_init([7*7*64,1024],'fc1_w')
    bias_fc1=bias_init([1024],'fc1_b')
    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,weights_fc1)+bias_fc1)
    h_fc1_drop=dropout(h_fc1,keep_prob)
    weights_fc2=weights_init([1024,10],'fc2_w')
    bias_fc2=bias_init([10],'fc2_b')
    output=tf.nn.softmax(tf.matmul(h_fc1_drop,weights_fc2)+bias_fc2)
    return output

keep_prob=tf.placeholder(tf.float32)
y_pred=inference(x_images,keep_prob)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(y_pred,0.0001,1.0))))
correct_prediction=tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_op=tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
init_op=tf.global_variables_initializer()
local_init_op=tf.local_variables_initializer()

gpu_options=tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    for i in range(NUM_EPOCHS):
        train_batch_x,train_batch_y=mnist.train.next_batch(BATCH_SIZE)
        train_loss_value,_=sess.run([cross_entropy,train_op],feed_dict={x:train_batch_x,y:train_batch_y,keep_prob:0.8})
        if i%500==0:
            print('Epoch %d,train_loss is :%f' % (i,train_loss_value))
            test_batch_x,test_batch_y=mnist.test.next_batch(BATCH_SIZE)
            test_loss_value,test_acc_value=sess.run([cross_entropy,acc],feed_dict={x:test_batch_x,y:test_batch_y,keep_prob:1.0})
            print('Epoch %d,test loss is %f' % (i,test_loss_value))
            print('Epoch %d,test accuracy is %f' % (i,test_acc_value))
    print('-----Program end------')

