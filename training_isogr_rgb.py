#coding=utf-8
import os
import io
import sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import inputs as data
import c3d_clstm as net
import time
from datetime import datetime
import threading
import logging
import queue
import random

seq_len = 16
batch_size = 4
n_epoch = 1000
learning_rate = 0.1
decay_steps = 15000
decay_rate  = 0.1
weight_decay= 0.004
print_freq = 20
queue_num = 4
start_step = 0


num_classes = 249
dataset_name = 'isogr_ rgb'
training_datalist = 'trte_splits/IsoGD_Image/train_rgb_list.txt'
testing_datalist = 'trte_splits/IsoGD_Image/valid_rgb_list.txt'

sess = tf.InteractiveSession()

#输入张量=[batch_size,seq_len,height,width,channels]
x = tf.placeholder(tf.float32, [batch_size, seq_len, 112, 112, 3],name='datas')
# y=[batch_size,label],每个时刻的标签
y = tf.placeholder(tf.int32, shape=[batch_size, ],name='labels')

#得到输出
networks = net.c3d_clstm(x, num_classes, False, True)
#outputs，当前层网络的输出，即预测结果
networks_y = networks.outputs
'''
softmax将值归一化到[0,1]之间
argmax返回每一行最大值，每一个时刻序列的类别
'''
networks_y_op = tf.argmax(tf.nn.softmax(networks_y), 1)
#计算softmax交叉熵
networks_cost = tl.cost.cross_entropy(networks_y, y,name="cost_network")
tf.summary.scalar("cross entropy",networks_cost)
'''
首先将x转换成int32类型，然后判断是否与y相等
最后利用tf.equal判断结果是否相等，返回值为bool
利用tf.cast将correct_pre转换成[0,1]的浮点数，然后求平均值得到准确率
'''
correct_pred = tf.equal(tf.cast(networks_y_op, tf.int32), y)
networks_accu = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accucay",networks_accu)

'''
这一步没有训练数据，所以应该是用于评测训练数据
'''
# predictons = net.c3d_clstm(x, num_classes, True, False)
# predicton_y_op = tf.argmax(tf.nn.softmax(predictons.outputs),1)
# predicton_accu = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predicton_y_op, tf.int32), y), tf.float32))

#l2正则化
l2_cost = tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[0]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[6]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[12]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[14]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[20]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[22])+\
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[24])
tf.summary.scalar("L2 cost",l2_cost)

#损失函数并且加上L2正则
cost = networks_cost + l2_cost
tf.summary.scalar("loss",cost)


global_step = tf.Variable(start_step, trainable=False)
#学习衰减率，按照一定比率衰减
lr = tf.train.exponential_decay(learning_rate,
                                global_step,
                                decay_steps,
                                decay_rate,
                                staircase=True)
# tf.summary.scalar("learning rate",lr)

#获取输出层网络参数
train_params = networks.all_params
#梯度下降法优化
train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost,
                                       var_list=train_params,
                                       global_step=global_step)


sess.run(tf.global_variables_initializer())
if start_step>0:
  load_params = tl.files.load_npz(name='%s_model_iter_%d.ckpt'%(dataset_name, start_step))
  tl.files.assign_params(sess, load_params, networks)
networks.print_params(True)

# Data Reading
X_train,y_train = data.load_video_list(training_datalist)
X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
y_train = np.asarray(y_train, dtype=np.int32)
X_test,y_test = data.load_video_list(testing_datalist)
X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
y_test  = np.asarray(y_test, dtype=np.int32)

X_data_a  = np.empty((batch_size, seq_len, 112, 112, 3),float)
y_label_a = np.empty((batch_size,),int)

#进程标志
full_flg  = np.zeros((queue_num, 1))
#线程锁
rdwr_lock = threading.Lock()


data_x=queue.Queue(maxsize=20)
data_y=queue.Queue(maxsize=20)

def training_data_read():
  wr_pos = 0
  for i in range(n_epoch):
    #获取对应批次的数据
    for X_indices, y_labels in tl.iterate.minibatches(X_tridx,
                                                      y_train,
                                                      batch_size,
                                                      shuffle=True):
      '''
      和下面训练过程对应，多线程控制读数据与写数据
      之前读取不到数据应该是由于线程死锁造成的。
      这里应该借鉴了生产者消费者模式
      '''

      image_path = []
      image_fcnt = []
      image_olen = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        # X_index_a = X_indices
        key_str = '%06d' % X_index_a
        image_path.append(X_train[key_str]['videopath'])
        image_fcnt.append(X_train[key_str]['framecnt'])
        image_olen.append(seq_len)
        is_training.append(True) # Training
      #将一个batch_size的image_path和image_fcnt和image_olen打包成一个元组
      image_info = zip(image_path,image_fcnt,image_olen,is_training)
      '''
      通过给定的数据，返回一个批次的结果
      _ for _ in image_info:使用的是链表推导式
      利用threading_data函数,创建一个线程，将一个批次的数据输入到函数prepare_data中
      实际是对Threading.thread()的封装
      '''

      X_data_a = tl.prepro.threading_data([_ for _ in image_info],data.prepare_isogr_rgb_data)
      y_label_a = y_labels
      data_x.put(X_data_a)
      data_y.put(y_label_a)
      time.sleep(random.randrange(1))




# Output the saved logs to stdout and the opened log file
# sys.stdout = saved_stdout
# mem_log.seek(0)
# print(mem_log.read())
# mem_log.seek(0)
# log.writelines(['%s' % mem_log.read()])
# log.flush()
# mem_log.close()

step = start_step
count=0

def training():
  global count,step
  # create a summary writer
  merged = tf.summary.merge_all()
  train_writer=tf.summary.FileWriter("train_log",sess.graph)
  for epoch in range(n_epoch):
    # Train Stage
    for _,_ in tl.iterate.minibatches(X_tridx,
                                      y_train,
                                      batch_size,
                                      shuffle=True):
      # 1. Read data for each batch
      # while True:
      #   rdwr_lock.acquire()
      #   if full_flg[rd_pos] == 0:
      #     rdwr_lock.release()
      #     time.sleep(1)
      #     continue
      #   rdwr_lock.release()
      #   break
      # 2. Training
      '''
      这一步没有数据加载进去，不知道为何,猜测前面线程造成死锁
      fuck居然不是？,mmp居然是最后全连接层写错了。
      '''
      # data_a=q.get()
      #print("\033[32;0m: %d")
      count+=1
      # X_data=data_a[0]
      # y_label=data_a[1]
      # feed_dict = {x: X_data, y: y_label}
      x_data=data_x.get()
      y_label=data_y.get()
      #print(y_label.shape)
      feed_dict={x:x_data,y:y_label}

      feed_dict.update(networks.all_drop)
      start_time = time.time()
      summary,_,loss_value,lr_value,acc= sess.run([merged,train_op,cost,lr,networks_accu], feed_dict=feed_dict)
      duration = time.time() - start_time


      # 4. Statistics
      if step%print_freq == 0:
        average_acc = acc
        total_loss = loss_value
        training_time = duration
      else:
        average_acc += acc
        total_loss = total_loss + loss_value
        training_time = training_time + duration
      if (step+1)%print_freq == 0:
        training_bps = batch_size*print_freq / training_time
        average_loss = total_loss / print_freq
        average_acc = average_acc / print_freq
        format_str = ('%s: iter = %d, lr=%f, average_loss = %.2f average_acc = %.6f (training: %.1f batches/sec)')
        print(format_str % (datetime.now(), step+1, lr_value, average_loss, average_acc, training_bps))

        # write the summary to the disk
        train_writer.add_summary(summary,merged)

        # log.writelines([format_str % (datetime.now(), step+1, lr_value, average_loss, average_acc, training_bps),
        #                 '\n'])
        # log.flush()
      step = step + 1
      time.sleep(random.randrange(1))

      # save the model
      if step % 1000 == 0:
          tl.files.save_ckpt(sess=sess,
                             mode_name='model.ckpt' % step,
                             save_dir="models",
                             printable=True)
          print("Model saved in file model_iter_%d.ckpt" % step)



    # tl.files.save_npz(networks.all_params,
    #                   name='%s_model_iter_%d.npz'%(dataset_name, step),
    #                   sess=sess)
    # print("Model saved in file: %s_model_iter_%d.npz" %(dataset_name, step))

    # Test Stage
    # average_accuracy = 0.0
    # test_iterations = 0;
    # for X_indices, y_label_t in tl.iterate.minibatches(X_teidx,
    #                                                    y_test,
    #                                                    batch_size,
    #                                                    shuffle=True):
    #   # Read data for each batch
    #   image_path = []
    #   image_fcnt = []
    #   image_olen = []
    #   is_training = []
    #   for data_a in range(batch_size):
    #     X_index_a = X_indices[data_a]
    #     key_str = '%06d' % X_index_a
    #     image_path.append(X_test[key_str]['videopath'])
    #     image_fcnt.append(X_test[key_str]['framecnt'])
    #     image_olen.append(seq_len)
    #     is_training.append(False) # Testing
    #   image_info = zip(image_path,image_fcnt,image_olen,is_training)
    #   X_data_t = tl.prepro.threading_data([_ for _ in image_info],
    #                                       data.prepare_isogr_rgb_data)
    #   feed_dict = {x: X_data_t, y: y_label_t}
    #   #在这一步关闭drop层
    #   #将所有的概率设置为1，表示禁用drop_out层
    #   dp_dict = tl.utils.dict_to_one(predictons.all_drop)
    #   feed_dict.update(dp_dict)
    #   _,accu_value = sess.run([predicton_y_op, predicton_accu], feed_dict=feed_dict)
    #   average_accuracy = average_accuracy + accu_value
    #   test_iterations = test_iterations + 1
    # average_accuracy = average_accuracy / test_iterations
    # format_str = ('%s: epoch = %d, average_accuracy = %.6f')
    # print (format_str % (datetime.now(), epoch, average_accuracy))
    # log.writelines([format_str % (datetime.now(), epoch, average_accuracy), '\n'])
    # log.flush()

# In the end, close TensorFlow session.

t1 = threading.Thread(target=training_data_read,name="train_read_data")
t2=threading.Thread(target=training,name="training")
t1.start()
t2.start()
t1.join()
t2.join()
sess.close()