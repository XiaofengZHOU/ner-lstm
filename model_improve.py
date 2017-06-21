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
        batch_label= label[label:]
        batch_data = np.concatenate( (batch_data,  data[0:endset - len(data)]),  axis=0 )
        batch_label= np.concatenate( (batch_label, label[0:endset - len(data)]), axis=0 )
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


def get_entity_accuracy(prediction,label):
    num_o =0
    num_o_true = 0
    num_loc = 0
    num_loc_true = 0
    num_per = 0
    num_per_true = 0
    num_misc = 0
    num_misc_true = 0
    num_org = 0 
    num_org_true = 0
    for i in range(len(prediction)):
        for j in range(len(label[i])):
            if np.argmax(label[i][j]) ==0:
                num_per = num_per +1
                if np.argmax(prediction[i][j]) ==0:
                    num_per_true = num_per_true +1
            if np.argmax(label[i][j]) ==1:
                num_loc = num_loc +1
                if np.argmax(prediction[i][j]) ==1:
                    num_loc_true = num_loc_true +1
            if np.argmax(label[i][j]) ==2:
                num_org = num_org +1
                if np.argmax(prediction[i][j]) ==3:
                    num_org_true = num_org_true +1
            if np.argmax(label[i][j]) ==3:
                num_misc = num_misc +1
                if np.argmax(prediction[i][j]) ==3:
                    num_misc_true = num_misc_true +1
            if np.argmax(label[i][j]) ==4:
                num_o = num_o +1
                if np.argmax(prediction[i][j]) ==4:
                    num_o_true = num_o_true +1

    return [num_o,num_o_true,num_loc,num_loc_true,num_per,num_per_true,num_misc,num_misc_true,num_org,num_org_true]
    

#%%

# offset = 0
# endset = offset + batch_size
# batch_x,batch_y,offset,endset= generate_batch(train_data,train_label,max_length,offset,endset)



#%%
outputs = BiRnn(x)
outputs = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])
outputs = tf.reshape(outputs,[-1,2*n_hidden])
length = get_sentence_length(x)
pred = tf.nn.softmax( tf.matmul(outputs, weights) + biases )
pred = tf.reshape(pred, [-1, n_steps, n_classes])
cost = get_cost(pred,y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = get_accuracy(pred,y)
init = tf.global_variables_initializer()

num_iters = len(train_data)//batch_size
num_epochs = 5

#%%
with tf.Session() as sess:
    sess.run(init)
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

    
        


        


    