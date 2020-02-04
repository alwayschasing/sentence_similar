#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def inference():
    export_path="saved_model/1580742289/"
    imported = tf.saved_model.load(export_path)
    def predict(input_ids_a,input_ids_b,labels):
        # print(imported.signatures.keys())
        return imported.signatures["serving_default"](
           input_ids_a=tf.constant(input_ids_a,dtype=tf.int64),
            input_ids_b=tf.constant(input_ids_b,dtype=tf.int64),
            labels=tf.constant(labels,dtype=tf.int64)
           )

    input_ids_a = np.random.randint(10000,size=64)
    input_ids_b = np.random.randint(10000,size=64)
    labels = [1]
    print("### predict output ###")
    print(predict(input_ids_a,input_ids_b,labels))

def inference_test():
    export_path="saved_model/1580695091/"
    input_ids_a_ = np.random.randint(10000,size=64)
    input_ids_b_ = np.random.randint(10000,size=64)
    with tf.Session() as sess:
        imported = tf.saved_model.load(sess,export_path)
        input_ids_a = sess.graph.get_tensor_by_name('input_ids_a:0')
        input_ids_b = sess.graph.get_tensor_by_name('input_ids_b:0')
        # op = sess.graph.get_tensor_by_name('op_to_store:0')
        sess.run(tf.global_variables_initializer())
        ret = sess.run(op, feed_dict={input_ids_a:input_ids_a_,input_ids_b:input_ids_b_})
        print(ret)

if __name__ == "__main__":
    inference()
