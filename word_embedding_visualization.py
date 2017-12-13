#Reference : https://gist.github.com/lampts/026a4d6400b1efac9a13a3296f16e655
#GLOVE download url : http://nlp.stanford.edu/data/glove.6B.zip
#Tensorflow CPU : 1.4

import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


# Path to glove embeddings file.
glove_file = '../data/glove/glove.6B.100d.txt' 
# loading your gensim
with open(glove_file,'r') as f:
	glove_file_data = f.readlines()

glove_embd = []
word2idx = {}
idx2word = []
for line in glove_file_data:
	line_split = line.split()
	# position at which embedding will be added
	word2idx[line_split[0]] = len(glove_embd)
	idx2word.append(line_split[0])
	glove_embd.append(list(map(float,line_split[1:])))

n_words = 100000

print("Total number of words :",len(glove_embd))
# project part of vocab, 10K of 300 dimension
glove_embd = np.asarray(glove_embd[:n_words])
print("Total words being visualized :",len(glove_embd))

with open("./projector/prefix_metadata.tsv", 'wb') as file_metadata:
    for word in idx2word[:n_words]:
        if word == '':
        	print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
        	file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
        else:
        	#print(type("{0}".format(word).encode('utf-8')),type(b'\n'))
        	file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')
# define the model without training
sess = tf.InteractiveSession()

with tf.device("/cpu:0"):
    embedding = tf.Variable(glove_embd, trainable=False, name='prefix_embedding')

tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter('./projector', sess.graph)

# adding into projector
config = projector.ProjectorConfig()
embed= config.embeddings.add()
embed.tensor_name = 'prefix_embedding'
embed.metadata_path = './projector/prefix_metadata.tsv'

# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)

saver.save(sess, './projector/prefix_model.ckpt', global_step=10000)

print("Save")
# open tensorboard with logdir, check localhost:6006 for viewing your embedding.
# tensorboard --logdir="./projector/"
