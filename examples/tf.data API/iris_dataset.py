#coding=utf-8

'''
Use the tf.data API to read iris data from raw text file
'''

import tensorflow as tf
import numpy as np

iris_data_path = "../../data/iris/iris.data"

# two dict to store the relation between label id(string) and label index(int)
label_id_index_dict = {}
label_index_id_dict = {}
last_label_index = -1

# get or create label id by label index
def get_or_create_label_index(label_id):
    if label_id in label_id_index_dict:
        return label_id_index_dict[label_id]
    global last_label_index
    last_label_index += 1
    label_id_index_dict[label_id] = last_label_index
    label_index_id_dict[last_label_index] = label_id
    return last_label_index


# filter while line
# will be used with py_func to read data
def is_not_empty_line(line):
    return len(line) > 0

# parse each line
# will be used with py_func to read data
def parse_iris_line(line):
    items = line.split(b",")
    features = np.array([float(item) for item in items[:-1]]).astype(np.float32)
    return [features, get_or_create_label_index(items[-1])]


epochs = 10
batch_size = 20
num_classes = 3

# tf.data API
# looks like map and filter operation in Apache Spark
dataset = tf.data.TextLineDataset(iris_data_path)\
    .filter(lambda line: tf.py_func(is_not_empty_line, [line], tf.bool))\
    .map(lambda line: tf.py_func(parse_iris_line, [line], [tf.float32, tf.int32]))\
    .shuffle(500)\
    .batch(batch_size)


# make a iterator for dataset
ite = dataset.make_initializable_iterator()
# next_element works like an operation in the graph
next_element = ite.get_next()
# each line is parsed to [features, label]
batch_features = next_element[0]
batch_labels = tf.one_hot(next_element[1], num_classes)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        # initialize the dataset iterator
        sess.run(ite.initializer)
        while True:
            try:
                batch_features_val, batch_labels_val = sess.run([batch_features, batch_labels])
                print("features:")
                print(batch_features_val)
                print("one hot labels:")
                print(batch_labels_val)
            except tf.errors.OutOfRangeError:
                # when reading to the end, break the while loop
                break