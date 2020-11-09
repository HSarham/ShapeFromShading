import tensorflow as tf
import numpy as np
import h5py
import time
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.test.is_gpu_available()

weight_names = ["W11", "W12", "W21", "W22", "W31", "W32", "W41", "W42", "W51", "W52", "W61", "W62", "W71", "W72",
                "W81", "W82", "W91", "W92", "W93", "WU5", "WU6", "WU7", "WU8"]

tf.random.set_random_seed(5678)

def init_tf_var(name, shape, Ws):
    Ws[name] = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(seed=1))

def init_weights():
    weights = {}

    init_tf_var("W11", [3, 3, 3, 32], weights)
    init_tf_var("W12", [3, 3, 32, 32], weights)

    init_tf_var("W21", [3, 3, 32, 64], weights)
    init_tf_var("W22", [3, 3, 64, 64], weights)

    init_tf_var("W31", [3, 3, 64, 128], weights)
    init_tf_var("W32", [3, 3, 128, 128], weights)

    init_tf_var("W41", [3, 3, 128, 256], weights)
    init_tf_var("W42", [3, 3, 256, 256], weights)

    init_tf_var("W51", [3, 3, 256, 512], weights)
    init_tf_var("W52", [3, 3, 512, 512], weights)

    init_tf_var("WU5", [2, 2, 512, 256], weights)

    init_tf_var("W61", [3, 3, 512, 256], weights)
    init_tf_var("W62", [3, 3, 256, 256], weights)

    init_tf_var("WU6", [2, 2, 256, 128], weights)

    init_tf_var("W71", [3, 3, 256, 128], weights)
    init_tf_var("W72", [3, 3, 128, 128], weights)

    init_tf_var("WU7", [2, 2, 128, 64], weights)

    init_tf_var("W81", [3, 3, 128, 64], weights)
    init_tf_var("W82", [3, 3, 64, 64], weights)

    init_tf_var("WU8", [2, 2, 64, 32], weights)

    init_tf_var("W91", [3, 3, 64, 32], weights)
    init_tf_var("W92", [3, 3, 32, 32], weights)

    init_tf_var("W93", [1, 1, 32, 1], weights)

    return weights


def double_conv2d(input_, W1, W2):
    Z1 = tf.nn.conv2d(input_, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    Z2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    return A2


def up_conv2d(input_, up_shape, W, A):
    U = tf.image.resize(input_, up_shape, method='nearest')
    UC = tf.nn.conv2d(U, W, strides=[1, 1, 1, 1], padding='SAME')
    C = tf.concat([A, UC], 3)  # concatenate the channels

    return C


def feed_forward(X, Ws):
    # Step 1 input 128x128x3
    A1 = double_conv2d(X, Ws["W11"], Ws["W12"])
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Step 2 input 64x64x32
    A2 = double_conv2d(P1, Ws["W21"], Ws["W22"])
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Step 3 input 32x32x64
    A3 = double_conv2d(P2, Ws["W31"], Ws["W32"])
    P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Step 4 input 16x16x128
    A4 = double_conv2d(P3, Ws["W41"], Ws["W42"])
    P4 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Step 5 input 8x8x256 output 8x8x512
    A5 = double_conv2d(P4, Ws["W51"], Ws["W52"])
    # Step 5 Up-convolution input 8x8x512 output 16x16x512
    C5 = up_conv2d(A5, [16, 16], Ws["WU5"], A4)
    # Step 6 input 16x16x512 output 16x16x256
    A6 = double_conv2d(C5, Ws["W61"], Ws["W62"])
    # Step 6 Up-convolution input 16x16x256
    C6 = up_conv2d(A6, [32, 32], Ws["WU6"], A3)
    # Step 7 input 32x32x256 output 32x32x128
    A7 = double_conv2d(C6, Ws["W71"], Ws["W72"])
    # Step 7 Up-convolution input 32x32x128 output 64x64x128
    C7 = up_conv2d(A7, [64, 64], Ws["WU7"], A2)
    # Step 8 input 64x64x128
    A8 = double_conv2d(C7, Ws["W81"], Ws["W82"])
    # Step 8 Up-convolution input 64x64x64
    C8 = up_conv2d(A8, [128, 128], Ws["WU8"], A1)
    # Step 9 input 128x128x64
    A8 = double_conv2d(C8, Ws["W91"], Ws["W92"])
    # Step 9 input 128x128x32
    Z9 = tf.nn.conv2d(A8, Ws["W93"], strides=[1, 1, 1, 1], padding='SAME')

    return Z9


def cost_function(in_Y, in_ff):
    diff = in_Y - in_ff
    mask = in_Y >= 0
    return tf.reduce_mean(tf.abs(diff[mask]))


def train(dataset, minibatch_size=64, num_iters=5):
    X = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 128, 128, 1], name='Y')

    Ws = init_weights()

    ff = feed_forward(X, Ws)
    cost = cost_function(Y, ff)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=cost)

    inputs = dataset["input"]
    targets = dataset["target"]

    num_elems = 200000#inputs.shape[0]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_minibatches = int((num_elems - num_elems % minibatch_size) / minibatch_size + 1)

    for i in range(num_iters):
        log_file = open("model_log_"+str(i)+".txt", "w")
        before = time.time()
        for minibatch_index in range(num_minibatches):
            start_index = minibatch_index * minibatch_size
            end_index = min(num_elems, start_index + minibatch_size)

            if (start_index < end_index):
                X_minibatch = inputs[start_index:end_index, ...]
                Y_minibatch = targets[start_index:end_index, ...]
                _, cost_val = sess.run(fetches=[optimizer, cost], feed_dict={X: X_minibatch, Y: Y_minibatch})
                log_file.write(str(cost_val)+"\n")
                print(cost_val)

        duration = time.time()-before
        log_file.write("time passed (seconds): "+str(duration)+"\n")

        w_file = h5py.File('weights_'+str(i)+'.h5py', 'w')
        for name in weight_names:
            w_file.create_dataset(name, data=np.array(weights[name].eval(session=sess)))

        w_file.close()
        log_file.close()

    return Ws, sess
