# Enable future and backwards compatability:
from __future__ import absolute_import, division, generators, unicode_literals, print_function, nested_scopes, with_statement

import os
import numpy as np
import collections
import random
import math
import tensorflow as tf

def tf_model(data, reverse_dictionary, n_words, skip_window, learning_rate, num_skips, batch_size, model_name):

  vocabulary_size = 3600
  valid_size = 16     # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)
  num_sampled = 64    # Number of negative examples to sample.

  embedding_size = 300  # Dimension of the embedding vector.

  graph = tf.Graph()

  with graph.as_default():

    # Input data.
    name_scope = 'input_data'
    with tf.name_scope(name_scope):
      train_inputs = tf.placeholder(
          tf.int32, shape=[batch_size], name='train_inputs')
      train_outputs = tf.placeholder(
          tf.int32, shape=[batch_size, 1], name='train_outputs')
      valid_dataset = tf.constant(
          valid_examples, dtype=tf.int32, name='validation_set')

    # Look up embeddings for inputs.
    # embeddings is the weights of the connections to the linear hidden layer. We initialize
    # the variable with a random uniform distribution between -1.0 to 1.0
    name_scope = 'embedding_layer'
    with tf.name_scope(name_scope):
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')

      embed = tf.nn.embedding_lookup(embeddings, train_inputs, name='embed_op')

    # Construct the variables for the NCE loss
    name_scope = 'nce_loss'
    with tf.name_scope(name_scope):
      nce_weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size), name='nce_weights'))
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name='nce_biases')
      tf.summary.histogram('nce_biases', nce_biases)
      tf.summary.histogram('nce_weights', nce_weights)
      nce_loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_outputs,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size))
    tf.summary.scalar('nce_loss', nce_loss)
    name_scope = 'gradient_desc'
    with tf.name_scope(name_scope):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(nce_loss)

    # Compute the cosine similarity between minibatch examples and all
    # embeddings.
    name_scope = 'cos_sim'
    with tf.name_scope(name_scope):
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm

      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)

      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
  with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')
    num_steps = 10000
    average_loss = 0

    # Tensorboard
    merged_summary = tf.summary.merge_all()
    LOG_DIR = './tmp/' + model_name
    writer = tf.summary.FileWriter(LOG_DIR)
    writer.add_graph(session.graph)

    # Generate mini batch
    for step in range(num_steps):
      batch_inputs, batch_context = generate_batch(data,
                                                   batch_size, num_skips, skip_window)
      feed_dict = {train_inputs: batch_inputs, train_outputs: batch_context}

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      _, loss_val = session.run(
          [optimizer, nce_loss], feed_dict=feed_dict)
      average_loss += loss_val
      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000
        # batches.
        print('================================================')
        print('Average loss at step ', step, ': ', average_loss)
        print(loss_val)
        average_loss = 0
        # tf.summary.scalar('average_loss', average_loss)
        # tf.summary.scalar('loss_val', loss_val)
        s = session.run(merged_summary, feed_dict=feed_dict)
        writer.add_summary(s, step)

        saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)
      average_loss += loss_val
      # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 1000 == 0:
        sim = similarity.eval()
        for i in range(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word
          for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
          print(log_str)
    final_embeddings = normalized_embeddings.eval()

    # Save the model:
    file_loc = './saved_models/' + model_name + '/trained_w2v'
    print("Training complete. Exporting the model to " + file_loc + "...")
    saver.save(session, file_loc, global_step=num_steps)
    print('Done exporting!')

data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # input word at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
      # these are the context words
      context[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, context


def init(*args):
  # Number of words
  n_words = 3600
  # How many times to reuse an input to generate a context.
  num_skips = 2
  if args:
    learning_rate = args[0]
    # How many words to consider left and right.
    skip_window = args[1]
    # Size of the batch
    batch_size = args[2]
    # directory name for the model in tmp/
    model_name = args[3]
    print("running model: " + model_name)
    print("**********************************")
    print("learning rate:")
    print(learning_rate)
    print("skip window:")
    print(skip_window)
    print("batch_size:")
    print(batch_size)
    print("**********************************")
  else:
    learning_rate = 0.9
    # How many words to consider left and right.
    skip_window = 1
    # directory name for the model in tmp/
    model_name = '1'
    # size of the batch
    batch_size = 128
  
  ##############################################################################
  # This script requires that you create and load a list with all your strings
  # and a dictionary with the index number as the key and the word as the value
  # 
  # data, reverse_dictionary = your_preprocessing_script
  ##############################################################################

  tf_model(data, reverse_dictionary, n_words, skip_window, learning_rate, num_skips, batch_size, model_name)

if __name__ == '__main__':
  init()
