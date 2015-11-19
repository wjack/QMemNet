import math
import tensorflow as tf

#memory network for language modelling


#reads previous N words into memory, individually
#each memory cell holds a single word
#temporal encoding is used
#query vector is fixed to constant .1 without embedding.
#cross entropy loss is used


import tensorflow as tf
import numpy as np
import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(A_embed_weights, B_embed_weights, C_embed_weights, W_output_weights, T_C_encoding_weights, T_M_encoding_weights, input_vectors, query):
    #model performs k = 1 memory hops

    #encode input vectors into m and c memory
    m_memory = []
    c_memory = []

    vector_index = 0
    for input_vector in input_vectors:
        #T_C and T_M are learned matricies that account for time encoding 
        embedded_m_vector = tf.add(tf.matmul(input_vector, A_embed_weights, a_is_sparse = True), T_C_encoding_weights[:, vector_index])
        embedded_c_vector  = tf.matmul(input_vector, C_embed_weights, a_is_sparse = True), T_M_encoding_weights[:, vector_index]
        m_memory.append(embedded_m_vector)
        c_memory.append(embedded_c_vector)

        vector_index += 1
    m_memory = tf.convert_to_tensor(m_memory, dtype = tf.float32)
    c_memory = tf.convert_to_tensor(c_memory, dtype = tf.float32)

    #compute match between embedded memories in m_memory, and embedded query, p is a probability vector over the inputs.

    embedded_query = tf.matmul(query, B_embed_weights)
    memory_matches = tf.matmul(embedded_query, m_memory, transpose_a=True)

    p = tf.softmax(memory_matches)

    #compute output vector o by weighting c_memory by p

    o = tf.matmul(p, c_memory, transpose_a = True)

    #in a single memory hop, sum o with embedded_query, pass through output weights W, and then softmax

    a = tf.softmax(tf.matmul(tf.add(o, embedded_query), W_output_weights))

    #a is a probability distribution over the vocabulary
    return a
