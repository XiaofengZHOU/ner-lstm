#%%
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
from get_data import get_data

def get_max_length(data):
    temp_len = 0
    max_length = 0
    for sentence in data:
        temp_len = len(sentence)
        if temp_len > max_length:
            max_length =temp_len
    return max_length

train_data, train_label = get_data(r'data\lstm_train_file\train_sentences_without_padding.pickle')
valid_data, valid_label = get_data(r'data\lstm_train_file\train_sentences_without_padding.pickle')
test_data , test_label  = get_data(r'data\lstm_train_file\test_sentences_without_padding.pickle')

max_length_train = get_max_length(train_data)
max_length_valid = get_max_length(valid_data)
max_length_test  = get_max_length(test_data)
print(max_length_train,max_length_valid,max_length_test)
max_length = max_length_test


#%%
batch_size = 32
n_steps = max_length # timesteps, number of words in one sentence
embedding_size = 311
n_classes = 5
num_layers =2
n_hidden = 512
learning_rate = 0.001
keep_prob = 0.5

display_step =10

x = tf.placeholder("float", [None, n_steps, embedding_size])
y = tf.placeholder("float", [None, n_steps, n_classes])
# RNN output node weights and biases
weights = tf.Variable(tf.random_normal([n_hidden*2, n_classes],stddev=0.01))
biases =  tf.Variable(tf.random_normal([n_classes]))

def lstm_cell(n_hidden):
    # With the latest TensorFlow source code (as of Mar 27, 2017),
    # the BasicLSTMCell will need a reuse parameter which is unfortunately not
    # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
    # an argument check here:
    return tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)

def attn_cell(n_hidden):
    return tf.contrib.rnn.DropoutWrapper(lstm_cell(n_hidden), output_keep_prob=keep_prob)


def BiRnn(x,n_hidden):
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, embedding_size)
    x = tf.unstack(x, n_steps, 1)
    lstm_fw_cells = rnn.MultiRNNCell([attn_cell(n_hidden) for _ in range(num_layers)] , state_is_tuple=True)
    lstm_bw_cells = rnn.MultiRNNCell([attn_cell(n_hidden) for _ in range(num_layers)] , state_is_tuple=True)
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cells, lstm_bw_cells, x,dtype=tf.float32)
    outputs = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])
    outputs = tf.reshape(outputs,[-1,2*n_hidden])
    return outputs

def get_sentence_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def get_cost(prediction,label):
    cross_entropy = label * tf.log(prediction)
    cross_entropy = -1 * tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(label), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_entropy)

def get_accuracy(pred,label):
    mistakes = tf.equal(tf.argmax(label, 2), tf.argmax(pred, 2))
    mistakes = tf.cast(mistakes, tf.float32)
    mask = tf.sign(tf.reduce_max(tf.abs(label), reduction_indices=2))
    mistakes *= mask
    # Average over actual sequence lengths.
    mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
    mistakes /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(mistakes)

def generate_batch(data,label,length,offset,endset):
    if endset > len(data):
        batch_data = data[offset:]
        batch_label= label[offset:]
        batch_data = batch_data + data[0:endset - len(data)]
        batch_label= batch_label+ label[0:endset - len(data)]
        offset = endset - len(data)
        endset = offset + batch_size
    else:
        batch_data  = data[offset:endset]
        batch_label = label[offset:endset]
        offset = endset
        endset = offset + batch_size
    
    for i in range(len(batch_data)):
        if len(batch_data[i]) < length:
            padding_word = np.array([0 for _ in range(311)])
            padding_label = np.array([0 for _ in range(5)])
            padding_words = np.array([padding_word   for _ in range(length-len(batch_data[i]))])
            padding_labels = np.array([padding_label for _ in range(length-len(batch_data[i]))])

            batch_data[i]  = np.concatenate( (batch_data[i],padding_words),  axis=0 )
            batch_label[i] = np.concatenate( (batch_label[i],padding_labels),  axis=0 )

    return batch_data,batch_label,offset,endset

def generate_test_data(data,label,length):
    for i in range(len(data)):
        if len(data[i]) < length:
            padding_word = np.array([0 for _ in range(311)])
            padding_label = np.array([0 for _ in range(5)])
            padding_words = np.array([padding_word   for _ in range(length-len(data[i]))])
            padding_labels = np.array([padding_label for _ in range(length-len(data[i]))])
            data[i]  = np.concatenate( (data[i],padding_words),  axis=0 )
            label[i] = np.concatenate( (label[i],padding_labels),  axis=0 )
    return data,label


def get_entity_accuracy(prediction,label):
    result_mat = np.zeros((5,6))
    for i in range(len(prediction)):
        for j in range(len(label[i])):
            result_label = np.argmax(label[i][j])
            result_pred  = np.argmax(prediction[i][j])
            result_mat[result_label][5] = result_mat[result_label][5] +1
            result_mat[result_label][result_pred] = result_mat[result_label][result_pred] + 1
    return result_mat
    


#%%
outputs = BiRnn(x,n_hidden)
pred = tf.nn.softmax( tf.matmul(outputs, weights) + biases )
pred = tf.reshape(pred, [-1, n_steps, n_classes])
cost = get_cost(pred,y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = get_accuracy(pred,y)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#%%
with tf.Session() as sess:
    sess.run(init)
    num_iters = len(train_data)//batch_size
    num_epochs = 10
    print('variable initialized')
    print('num of iters: ', num_iters)
    offset = 0
    endset = 0+batch_size

    for e in range(num_epochs):
        for i in range(num_iters):
            batch_x,batch_y,offset,endset= generate_batch(train_data,train_label,max_length,offset,endset)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if i%display_step ==0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
        saver.save(sess,'tf_model/model.ckpt')
    
 
#%%
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'tf_model/model.ckpt')
    test_offset = 0
    test_endset = test_offset +batch_size
    for i in range( len(test_data)//batch_size ):
        batch_x,batch_y,test_offset,test_endset= generate_batch(test_data,test_label,max_length,test_offset,test_endset)
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        pred = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
        print("Test Iter " + str(i) + ", Minibatch Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc))


        


    