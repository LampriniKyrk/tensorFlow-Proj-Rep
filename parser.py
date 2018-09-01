import tensorflow as tf
import numpy as np

filename = ["/home/nana/ukdale/house_1/channel_1.dat"]
record_defaults = [tf.float32] * 2
dataset = tf.contrib.data.CsvDataset(filename, record_defaults, field_delim=' ')

next_element = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
  i = 0
  while i < 20:
    try:
      print(sess.run(next_element))
      i+=1
    except tf.errors.OutOfRangeError:
      break
