#%%
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
from get_data import get_data

train_data, train_label = get_data(r'data\lstm_train_file\train_sentences.pickle')
test_data , test_label  = get_data(r'data\lstm_train_file\test_sentences.pickle')

batch_size = 128
n_steps = 113 # timesteps, number of words in one sentence
embedding_size = 311
n_classes = 5
num_layers =2
n_hidden = 512
learning_rate = 0.001
keep_prob = 0.5

x = tf.placeholder("float", [None, n_steps, embedding_size])
y = tf.placeholder("float", [None, n_steps, n_classes])
# RNN output node weights and biases
weights = tf.Variable(tf.random_normal([n_hidden*2, n_classes],stddev=0.01))
biases =  tf.Variable(tf.random_normal([n_classes]))

def lstm_cell():
    # With the latest TensorFlow source code (as of Mar 27, 2017),
    # the BasicLSTMCell will need a reuse parameter which is unfortunately not
    # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
    # an argument check here:
    return tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)

def attn_cell():
    return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=keep_prob)


def BiRnn(x):
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, embedding_size)
    x = tf.unstack(x, n_steps, 1)
    lstm_fw_cells = rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)] , state_is_tuple=True)
    lstm_bw_cells = rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)] , state_is_tuple=True)
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cells, lstm_bw_cells, x,dtype=tf.float32)
    return outputs

def get_sentence_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def get_cost(prediction,label,length):
    cross_entropy = label * tf.log(prediction)
    cross_entropy = -1 * tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(label), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.cast(length, tf.float32)
    return tf.reduce_mean(cross_entropy)

def get_accuracy(pred,label,length):
    mistakes = tf.equal(tf.argmax(label, 2), tf.argmax(pred, 2))
    mistakes = tf.cast(mistakes, tf.float32)
    mask = tf.sign(tf.reduce_max(tf.abs(label), reduction_indices=2))
    mistakes *= mask
    # Average over actual sequence lengths.
    mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
    mistakes /= tf.cast(length, tf.float32)
    return tf.reduce_mean(mistakes)

#%%

# with padding word, we should be careful about the loss
outputs = BiRnn(x)
outputs = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])
outputs = tf.reshape(outputs,[-1,2*n_hidden])
length = get_sentence_length(x)
pred = tf.nn.softmax( tf.matmul(outputs, weights) + biases )
pred = tf.reshape(pred, [-1, n_steps, n_classes])
cost = get_cost(pred,y,length)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = get_accuracy(pred,y,length)

display_step =10

init = tf.global_variables_initializer()

with tf.Session() as sess:

    num_sentences = len(train_data)
    num_iters = num_sentences//batch_size
    num_epochs = 2
    sess.run(init)
    print('variable initialized')
    print('num of sentences: ', num_sentences)
    print('num of iters: ', num_iters)
    offset = 0
    endset = 0+batch_size
    for e in range(num_epochs):
        for i in range(num_iters):
            batch_x = train_data[offset:endset]
            batch_y = train_label[offset:endset]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            offset = endset
            endset = offset + batch_size
            
            if  endset - num_sentences > 0:
                batch_x = train_data[offset:]
                batch_x = np.concatenate((batch_x, train_data[0:endset - num_sentences]), axis=0)
                batch_y = train_label[offset:]
                batch_y = np.concatenate((batch_y, train_label[0:endset - num_sentences]), axis=0)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                offset = endset - num_sentences
                endset = offset + batch_size

            if i%display_step ==0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
        
        test_data = np.array(test_data)
        batch_x = test_data[0:128,0:113,:]
        batch_y = test_label[0:128,0:113,:]
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        pred = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
        print("epoche " + str(e) + ",testbatch Loss= " + \
                    "{:.6f}".format(loss) + ", testing Accuracy= " + \
                    "{:.5f}".format(acc))
                
#%%
test_data = np.array(test_data)
batch_x = test_data[0:128,0:113,:]
print(batch_x.shape)


