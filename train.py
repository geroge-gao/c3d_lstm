#coding=utf-8
import os
import io
import sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import inputs as input
import c3d_clstm as net
import time
import random
import queue
import threading
from datetime import datetime

seq_len = 16
batch_size = 5
n_epoch = 1000
learning_rate = 0.1
decay_steps = 15000
decay_rate  = 0.1
weight_decay= 0.004
print_freq = 20
queue_num = 4
start_step = 1

num_classes = 249
dataset_name = 'isogr_ rgb'
training_datalist = 'trte_splits/IsoGD_Image/train_rgb_list.txt'
testing_datalist = 'trte_splits/IsoGD_Image/valid_rgb_list.txt'

sess = tf.InteractiveSession()
#x=[batch_size,seq_len,height,width,channels]
x = tf.placeholder(tf.float32, [batch_size, seq_len, 112, 112, 3],name='datas')
# y=[batch_size,label]
y = tf.placeholder(tf.int32, shape=[batch_size, ],name='labels')
# get the output of the layer
networks=net.c3d_clstm(x,num_classes,False,True)
networks_y=networks.outputs

print(networks)

networks_y_op=tf.argmax(tf.nn.softmax(networks_y),1)
networks_cost = tl.cost.cross_entropy(networks_y, y,name="cost_network")
tf.summary.scalar("network loss",networks_cost)
correct_pred = tf.equal(tf.cast(networks_y_op, tf.int32), y)
networks_accu = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accary",networks_accu)

# test the model files
# predictions = net.c3d_clstm(x,num_classes,True,False)
# predictions_y_op = tf.argmax(tf.nn.softmax(predictions.outputs),1)
# predictions_accu = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predictions_y_op),tf.int32),y),tf.float32)

l2_cost = tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[0]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[2]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[4])+\
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[6]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[8]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[10]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[12])
tf.summary.scalar("l2 loss",l2_cost)

cost = networks_cost + l2_cost
tf.summary.scalar("loss",cost)
tf.summary.scalar("cross_entropy",networks_cost)

global_step = tf.Variable(start_step, trainable=False)

lr = tf.train.exponential_decay(learning_rate,
                                global_step,
                                decay_steps,
                                decay_rate,
                                staircase=True)

tf.summary.scalar("learning rate",lr)
#get the network parameters'
train_parms=networks.all_params
#optimizer
train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost,
                                                          var_list=train_parms,
                                                          global_step=global_step)

#init the parameters
sess.run(tf.global_variables_initializer())

# create a summary
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train_log', sess.graph)

# load the parameters from the pre-trained model
if start_step>0:
    load_params = tl.files.load_npz(path='models/',name='model-1000.npz')
    tl.files.assign_params(sess,load_params,networks)
  # tl.files.load_ckpt(sess,var_list=networks.all_params,mode_name="model-18000.ckpt",save_dir="models",printable=True)

networks.print_params(True)

# define the data queue
data_x=queue.Queue(maxsize=20)
data_y=queue.Queue(maxsize=20)

## load the file path and the labels
data, label = input.load_video_list(training_datalist)
X_tridx = np.asarray(np.arange(0, len(label)), dtype=np.int32)
y_train = np.asarray(label, dtype=np.int32)
X_data_a  = np.empty((batch_size, seq_len, 112, 112, 3),float)
y_label_a = np.empty((batch_size,),int)

def read_data():
    #get image and labels from the file path
    for i in range(n_epoch):
        # 获取对应批次的数据
        for X_indices, y_labels in tl.iterate.minibatches(X_tridx,
                                                          y_train,
                                                          batch_size,
                                                          shuffle=True):

            image_path = []
            image_fcnt = []
            image_olen = []
            is_training = []
            for data_a in range(batch_size):
                X_index_a = X_indices[data_a]
                # X_index_a = X_indices
                key_str = '%06d' % X_index_a
                image_path.append(data[key_str]['videopath'])
                image_fcnt.append(data[key_str]['framecnt'])
                image_olen.append(seq_len)
                is_training.append(True)  # Training
            # pack the path fcnt and so on
            image_info = zip(image_path, image_fcnt, image_olen, is_training)
            X_data_a = tl.prepro.threading_data([_ for _ in image_info], input.prepare_isogr_rgb_data)
            y_label_a = y_labels
            data_x.put(X_data_a)
            data_y.put(y_label_a)
            time.sleep(random.randrange(1))

def training():
    step=0
    total_accu=0
    total_loss=0
    training_time=0
    for epoch in range(n_epoch):
        for x_indices,label in tl.iterate.minibatches(X_tridx,
                                                      y_train,
                                                      batch_size,
                                                      shuffle=True):
            # get the data from quene
            data=data_x.get()
            label=data_y.get()
            # feed the data to feed_dict
            feed_dict = {x:data,y:label}
            feed_dict.update(networks.all_drop)
            #start training
            start_time = time.time()
            summary,acc,loss,lr_value,op = sess.run([merged,networks_accu,cost,lr,train_op],feed_dict=feed_dict)            
            duration=time.time()-start_time
            step+=1
            total_accu+=acc
            total_loss+=loss
            training_time+=duration
            # print("step %d ,accuary:%.6f"%(step,acc))

            # print the information
            if step%print_freq==0:
                training_bps=batch_size * print_freq / training_time
                average_acc = total_accu / print_freq
                average_loss = total_loss / print_freq
                format_str = ('%s: iter = %d, lr=%f, average_loss = %.2f average_acc = %.6f (training: %.1f batches/sec)')
                print(format_str % (datetime.now(), step, lr_value, average_loss, average_acc, training_bps))
                train_writer.add_summary(summary,step)
                training_time=0
                total_accu=0
                total_loss=0

            # save the model
            if step%1000==0:
                # tl.files.save_ckpt(sess=sess,
                #                    mode_name="model-%d.ckpt"%(step),
                #                    var_list=networks.all_params,
                #                    save_dir="models",
                #                    printable=True)
                tl.files.save_npz(networks.all_params,name="models/model-%d.npz"%step,sess=sess)
                print("Model saved in file %s_model.ckpt")



    sess.close()

if __name__=="__main__":
    t1 = threading.Thread(target=read_data, name="read_data")
    t2 = threading.Thread(target=training, name="training")
    t1.start()
    t2.start()
    t1.join()
    t2.join()
