#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import tensorflow.compat.v1 as tfc
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tfc.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import pandas as pd


# In[2]:


# load NPS data from Amazon Instant Video Reviews section
#use pd.read_json(datapath, lines=True)
df=pd.read_json('..\\speech\\Amazon_Instant_Video_5.json' , lines=',')

df.head()


# In[3]:


# filter out columns other than feedback and rating
df = df.filter(items=["reviewText", "overall"])

# drop scores without feedback
df = df.dropna()

df[:10]


# # Data preprocessing

# In[4]:


# removing punctuation
from string import punctuation
df.reviewText = df.reviewText.str.replace('[^\w\s]','')


# In[5]:


# join all feedback and split into words
all_text = ' '.join(df.reviewText)
words = all_text.split()


# In[6]:


words[:10]


# # Encoding the words

# In[7]:


# making a dictionary mapping words to integers
from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)} # starting at 1, as we'll later pad input vectors with 0s

# converting feedback to integers
df['feedback_ints'] = [list(map(lambda w: vocab_to_int[w], f)) for f in list(map(str.split, df.reviewText))]


# In[8]:


feedback_lens = Counter([len(x) for x in df.feedback_ints])
print("Zero-length feedback: {}".format(feedback_lens[0]))
print("Maximum feedback length: {}".format(max(feedback_lens)))


# In[9]:


seq_len = 200

# remove zero-length feedback
df = df[df.feedback_ints.apply(lambda x: len(x) > 0)]

# truncate feedback to 200 words
df.feedback_ints = df.feedback_ints.apply(lambda x: x[:seq_len])

# pad feedback longer than 200 words with 0s, add to a features array
features = np.zeros((len(df.feedback_ints), seq_len), dtype=int)
df = df.reset_index(drop=True)
for i, row in df.iterrows():
    features[i, -len(row.feedback_ints):] = row.feedback_ints[:seq_len]


# In[10]:


#reshape labels for sparse_softmax_cross_entropy
labels = df.overall.values.reshape([-1, 1])


# In[11]:


features[:10, 150:200]


# # Training, validation, test

# In[12]:


split_frac = 0.8 # fraction of data to keep in the training set
split_idx = int(len(features)*0.8)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# # Building the graph
# 
# • lstm_size: Number of units in the hidden layers in the LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
# • lstm_layers: Number of LSTM layers in the network. Start with 1, then add more if underfitting.
# • batch_size: The number of feedback messages to feed the network in one training pass. Typically this should be set as high as you can go without running out of memory.
# • learning_rate: Learning rate

# In[13]:


lstm_size = 512
lstm_layers = 5
batch_size = 150
learning_rate = 0.01


# We'll be passing in our 200 element long feedback vectors. Each batch will be batch_size vectors. We'll also be using dropout on the LSTM layer with keep_prob.

# In[14]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

n_words = len(vocab_to_int) + 1 # adding 1 because we use 0s for padding, dictionary started at 1

# create the graph object
graph = tf.Graph()
# add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int64, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int64, [None, 1], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# # Embedding

# In[15]:


# size of the embedding vectors (number of units in the embedding layer)
embed_size = 300 

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)


# # LSTM cell

# In[16]:


#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

with graph.as_default():
    # Your basic LSTM cell
    #tf.compat.v1.nn.rnn_cell.BasicLSTMCell
    #Old code for version below Tensorflow 2.0 
    def lstm_cell():
        lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(lstm_size)
        return lstm
    #lstm = tf.keras.layers.LSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_cell(), output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    #cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_layers)
    #cell = tf.keras.layers.StackedRNNCells([drop] * lstm_layers)
    
    Input=tf.placeholder(tf.float64,shape=(None,embed_size,batch_size),name='input')
    Expected_o=tf.placeholder(tf.float64,shape=(None,embed_size,lstm_size),name='Expected_o')
    #creation of the network

    initializer = tf.random_uniform_initializer(-1, 1)      
    #layers = [tf.nn.rnn_cell.GRUCell([lstm_layer,kernel_initializer=initializer)]
             
    
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)]) 
  
    
    # Getting an initial state of all zeros
    initial_state = rnn_cells.zero_state(batch_size, tf.float32)


# # RNN forward pass

# In[17]:


with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(rnn_cells, embed,initial_state=initial_state)
    #outputs, states = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    #self._output, out_state = tf.nn.dynamic_rnn(cell=rnn_cells,inputs= self._Input, dtype=tf.float64)


# # Output

# In[18]:


with graph.as_default():
    logits = tf.layers.dense(inputs=outputs[:, -1], units=11)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_, logits=logits)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# # Validation accuracy

# In[19]:


#Here we add a few nodes to calculate the accuracy which we'll use in the validation pass

with graph.as_default():
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)


# Batching
# This is a simple function for returning batches from the data. First it removes data such that we only have full batches. Then it iterates through the x and y arrays and returns slices out of those arrays with size [batch_size].

# In[20]:


def get_batches(x, y, batch_size=150):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# # Training

# In[21]:


epochs = 60

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y,
                    keep_prob: 0.5,
                    initial_state: state}
            loss_val, state, _ = sess.run([loss, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss_val))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(rnn_cells.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y,
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "checkpoints/nps.ckpt")


# In[22]:


test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(rnn_cells.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y,
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


# In[ ]:




