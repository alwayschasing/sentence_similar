#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import random
import jieba
import tokenization

flags=tf.flags
FLAGS=flags.FLAGS

flags.DEFINE_string("raw_data_file",None,
                    "raw train data file")
flags.DEFINE_string("train_data_file",None,
                    "train data file")
flags.DEFINE_string("recorddata_file",None,
                    "record data file")
flags.DEFINE_string("vocab_file",None,
                    "vocab word and vector file")


def create_model_item():
    pass


def create_positive_item_from_sentence(sentence, drop_ratio,rng):
    items = jieba.cut(sentence)
    sen_a=[]; sen_b=[]
    for it in items:
        if rng.random() > drop_ratio:
            sen_a.append(it)

        if rng.random() > drop_ratio:
            sen_b.append(it)
    return sen_a,sen_b


def create_train_data(input_files, output_file, drop_ratio=0.2):
    rng = random.Random()
    wfp = open(output_file,"w")
    for input_file in input_files:
        with open(input_file,"r") as fp:
            lines = list(fp.readlines())
            total_num = len(lines)
            for line in lines:
                line = line.strip()
                sen_a, sen_b = create_positive_item_from_sentence(line,drop_ratio,rng)
                positive_item_str = " ".join(sen_a) + "\t" + " ".join(sen_b) + "\t1\n"
                wfp.write(positive_item_str)

                sen_a = line
                sen_b = lines[rng.randint(0,total_num-1)].rstrip()
                items_a = jieba.cut(sen_a)
                items_b = jieba.cut(sen_b)
                negative_item_str = " ".join(items_a) + "\t" + " ".join(items_b) + "\t0\n"
                wfp.write(negative_item_str)
    wfp.close()

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_train_recorddata(input_files,output_file,vocab_file):
    tokenizer = tokenization.Tokenizer(vocab_file)
    writer = tf.python_io.TFRecordWriter(output_file)
    for input_file in input_files:
        with open(input_file,"r") as fp:
            for line in fp:
                line = line.rstrip()
                items = line.split('\t')
                if len(items) != 3:
                    continue
                sen_a = items[0]
                sen_b = items[1]
                label = int(items[2])
                ids_a = tokenizer.translate2id(sen_a)
                ids_b = tokenizer.translate2id(sen_b)
                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    "input_a":create_int_feature(ids_a),
                    "input_b":create_int_feature(ids_b),
                    "labels":create_int_feature([label])
                }))
                writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    raw_data_file = FLAGS.raw_data_file
    train_data_file = FLAGS.train_data_file
    create_train_data([raw_data_file], train_data_file)
    tf.logging.info("Writing to train data file:%s\n", train_data_file)
    recorddata_file = FLAGS.recorddata_file
    vocab_file = FLAGS.vocab_file
    create_train_recorddata([train_data_file], recorddata_file,vocab_file)
    tf.logging.info("Writing to RecordData file:%s\n", recorddata_file)

if __name__ == "__main__":
    flags.mark_flag_as_required("raw_data_file")
    flags.mark_flag_as_required("train_data_file")
    flags.mark_flag_as_required("recorddata_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()


