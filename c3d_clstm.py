import io
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import tensorlayer as tl


def c3d_clstm(inputs, num_classes, reuse, is_training):
    """Builds the Conv3D ConvLSTM Networks."""

  with tf.device('/gpu:0'):
    with tf.variable_scope('C3D ConvLSTM', reuse=reuse):
      tl.layers.set_name_reuse(reuse)
      if inputs.get_shape().ndims!=5:
        raise Exception("The input dimension of 3DCNN must be rank 5")
      # Input Layer
      network_input = tl.layers.InputLayer(inputs, name='input_layer')

      # convluation layer 1
      Conv1a = tl.layers.Conv3dLayer(prev_layer=network_input,
                                        act=tf.nn.relu,
                                        shape=[3,3,3,3,64],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv1a')
      Pool1 = tl.layers.PoolLayer(prev_layer=Conv1a,
                                     ksize=[1,1,2,2,1],
                                     strides=[1,1,2,2,1],
                                     padding='SAME',
                                     pool = tf.nn.max_pool3d,
                                     name='Pool1')

      # Convluation Layer 2
      Conv2a = tl.layers.Conv3dLayer(prev_layer=Pool1,
                                        act=tf.nn.relu,
                                        shape=[3,3,3,64,128],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv2a')
      Pool2 = tl.layers.PoolLayer(prev_layer=Conv2a,
                                     ksize=[1,2,2,2,1],
                                     strides=[1,2,2,2,1],
                                     padding='SAME',
                                     pool = tf.nn.max_pool3d,
                                     name='Pool2')

      # Convluation Layer  3
      Conv3a = tl.layers.Conv3dLayer(prev_layer=Pool2,
                                            act=tf.nn.relu,
                                            shape=[3,3,3,128,256],
                                            strides=[1,1,1,1,1],
                                            padding='SAME',
                                            name='Conv3a')
      Conv3b = tl.layers.Conv3dLayer(prev_layer=Conv3a,
                                     act=tf.nn.relu,
                                     shape=[3,3,3,256,256],
                                     strides=[1,1,1,1,1],
                                     padding='SAME',
                                     name='Conv3b')

      Pool3 = tl.layers.PoolLayer(prev_layer=Conv3b,
                                  ksize=[1,2,2,2,1],
                                  strides=[1,2,2,2,1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool3d,
                                  name='Pool3')
      # Pool3 = tf.transpose(Pool3,perm=[0,1,4,2,3])

      # ConvLstm Layer
      shape3d = Pool3.outputs.get_shape().as_list()
      num_steps = shape3d[1]
      convlstm1=tl.layers.ConvLSTMLayer(prev_layer=Pool3,
                                        cell_shape=[14,14],
                                        filter_size=[3,3],
                                        feature_map=256,
                                        initializer=tf.random_uniform_initializer(-0.1,0.1),
                                        n_steps=num_steps,
                                        return_last=False,
                                        return_seq_2d=False,
                                        name='clstm_layer_1')

      convlstm2 = tl.layers.ConvLSTMLayer(prev_layer=convlstm1,
                                          cell_shape=[14,14],
                                          filter_size=[3,3],
                                          feature_map=384,
                                          initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                          n_steps=num_steps,
                                          return_last=True,
                                          return_seq_2d=False,
                                          name='clstm_layer_2')
      # FC Layer 1
      convlstm2 = tl.layers.FlattenLayer(convlstm2,name='flatten')
      dense = tl.layers.DenseLayer(convlstm2,
                                   n_units=num_classes,
                                   act=tf.identity,
                                   name='NetWorks')
      return dense


