#!/usr/bin/env python
# -*- coding: utf-8 -*-
import modeling
import tensorflow as tf
from tensorflow.data import Dataset
from create_data import create_model_item
from cmd_parse import cmd_parse
import tokenization
import optimization
import numpy as np
import sys
flags = tf.flags
# FLAGS = flags.FLAGS
FLAGS = cmd_parse()
def freeze_model():
    pass
def record_input_fn_train_builder(input_files,seq_length,batch_size):
    def recordfile_input_fn_train(params):
        name_to_features = {
            "input_ids_a":tf.io.FixedLenFeature([seq_length],tf.int64),
            "input_ids_b":tf.io.FixedLenFeature([seq_length],tf.int64),
            "labels":tf.io.FixedLenFeature([1],tf.int64)
        }
        def _decode_record(record,name_to_features):
            """"""
            example = tf.io.parse_single_example(record, name_to_features)
            return example
        d = tf.data.Dataset.from_tensor_slices(input_files)
        d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))
        # d = d.apply(
            # tf.data.Dataset.interleave(tf.data.TFRecordDataset,
                                       # cycle_length=4,
                                       # block_length=16)
            # )
        d = d.interleave(lambda x:tf.data.TFRecordDataset(x),
                               cycle_length=4, 
                               block_length=16)
        d = d.shuffle(buffer_size=64)
        d = d.map(lambda record:_decode_record(record,name_to_features)).batch(batch_size,drop_remainder=True)
        # d = d.map(lambda record:_decode_record(record,name_to_features))
        return d
    return recordfile_input_fn_train
def file_input_fn_predict(input_files):
    # d = tf.data.Dataset.from_tensor_slices(input_files)
    # d = dataset.interleave(lambda x:tf.data.TextLineDataset(x).map(parse_line), cycle_length=4, block_length=16)
    def generate_fn():
        for input_file in input_files:
            with open(input_file,'r') as fp:
                for line in fp:
                    # model_item format:
                    # 
                    model_item = create_model_item(line)
                    yield model_item
    dataset = Dataset.from_generator(
        generate_fn,
        output_shapes=(tf.TensorShape([seq_length])))
    return dataset
def create_model(input_ids_a,
                 input_ids_b,
                 labels,
                 embedding_table,
                 is_training,
                 config):
    with tf.variable_scope("similarmodel") as vs:
        embedding_table_tensor = tf.get_variable(
            "embedding_table",
            shape=[config.vocab_size, config.vocab_vec_size],
            trainable=config.embedding_table_trainable)
        def init_embedding_table(scaffold,sess):
            sess.run(embedding_table_tensor.initializer, {embedding_table_tensor.initial_value:embedding_table})
        scaffold = tf.train.Scaffold(init_fn=init_embedding_table)
        model_a = modeling.TransformerSimilar(
            config,
            is_training,
            input_ids_a,
            input_mask=None,
            embedding_table=embedding_table,
            embedding_table_trainable=False,
            scope=vs)
        model_b = modeling.TransformerSimilar(
            config,
            is_training,
            input_ids_b,
            input_mask=None,
            embedding_table=embedding_table,
            embedding_table_trainable=config.embedding_table_trainable,
            scope=vs)
    output_layer_a = model_a.get_output()
    output_layer_b = model_b.get_output()
    # logits = tf.matmul(output_layer_a,output_layer_b, transpose_b=True)
    tf.logging.info("layer_a shape:%s"%(output_layer_a.shape))
    tf.logging.info("layer_b shape:%s"%(output_layer_b.shape))
    logits = tf.reduce_sum(tf.math.multiply(output_layer_a,output_layer_b),axis=[-1,-2])
    logits = tf.reshape(logits,[-1,1])
    # tf.print(logits, output_stream=sys.stderr)
    # logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.math.sigmoid(logits)
    labels = tf.cast(labels,dtype=tf.float32)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name=None))
    return (loss, probabilities)

def model_fn_builder(config,
                     init_checkpoint=None,
                     embedding_table=None,
                     embedding_table_trainable=False):
    def model_fn(features,labels,mode,params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(" name = %s, shape = %s" % (name, features[name].shape))
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        input_ids_a = features["input_ids_a"]
        input_ids_b = features["input_ids_b"]
        # input_mask = features["input_ids"]
        # segment_ids = features["segment_ids"]
        labels = features["labels"]
        embedding_table_tensor = tf.get_variable(
            "embedding_table",
            shape=[config.vocab_size, config.vocab_vec_size],
            trainable=config.embedding_table_trainable)
        def init_embedding_table(scaffold,sess):
            sess.run(embedding_table_tensor.initializer, {embedding_table_tensor.initial_value:embedding_table})
        scaffold = tf.train.Scaffold(init_fn=init_embedding_table)
        loss,probabilities = create_model(
            input_ids_a,
            input_ids_b,
            labels,
            embedding_table_tensor,
            is_training,
            config)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
        tf.logging.info("*** Trainable Variables ***")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape, init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss,0.01,10,1,False)
            export_output = {"predict":tf.estimator.export.PredictOutput(probabilities)}
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold=scaffold,
                export_outputs=export_output)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn():
                pass
            output_spec = tf.estimator.EstimatorSpec()
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=probabilities
                )
        return output_spec
    return model_fn
def load_embedding_table(embedding_table_file,binary=True):
    embedding_table = []
    if binary != True:
        with open(embedding_table_file) as fp:
            for line in fp:
                nums = line.rstrip().split(' ')
                embedding_table.append(nums)
    else:
        embedding_table = np.load(embedding_table_file)
    embedding_vec_size = len(embedding_table[0])
    unk_vec = np.random.rand(embedding_vec_size)
    embedding_table = np.asarray(embedding_table)
    # first vector is the unknown token
    embedding_table = np.insert(embedding_table,0,unk_vec,axis=0)
    return embedding_table
def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # config = modeling.Config(
    #     vocab_size=636025,
    #     vocab_vec_size=300,
    #     hidden_size=1024,
    #     num_hidden_layers=4,
    #     num_attention_heads=3,
    #     intermediate_size=1024)
    config = modeling.Config.from_json_file(FLAGS.config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    output_dir=FLAGS.output_dir
    input_file = FLAGS.input_file
    tf.logging.info("Input File:%s"%(input_file))
    init_checkpoint = FLAGS.init_checkpoint
    embedding_table_file = FLAGS.embedding_table_file
    embedding_table = None
    if embedding_table_file is not None:
        embedding_table = load_embedding_table(embedding_table_file)
    run_config = tf.estimator.RunConfig(
        save_summary_steps=3,
        save_checkpoints_steps=3,
        session_config=None,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=1,
        train_distribute=None,
        device_fn=None,
        protocol=None,
        eval_distribute=None)
    model_fn = model_fn_builder(
        config,
        init_checkpoint,
        embedding_table)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=output_dir,
        config=run_config,
        params=None,
        warm_start_from=None)
    input_files = [input_file]
    batch_size = FLAGS.batch_size
    if FLAGS.do_train:
        train_steps = FLAGS.train_steps
        # train_input_fn = recordfile_input_fn_train(input_files, config.max_seq_length, batch_size)
        train_input_fn = record_input_fn_train_builder(input_files, config.max_seq_length, batch_size)
        estimator.train(
            input_fn=train_input_fn,
            hooks=None,
            steps=train_steps,
            max_steps=None,
            saving_listeners=None)
        # save model
        def serving_input_receiver_fn():
            features = {"input_ids_a":tf.placeholder(dtype=tf.int64,shape=[None,64],name="input_ids_a"),
                        "input_ids_b":tf.placeholder(dtype=tf.int64,shape=[None,64],name="input_ids_b"),
                        "labels":tf.placeholder(dtype=tf.int64,shape=[None,1],name="labels")}
            return tf.estimator.export.ServingInputReceiver(features=features,receiver_tensors=features)
        estimator.export_saved_model("saved_model",serving_input_receiver_fn)

    elif FLAGS.do_eval:
        eval_input_fn = None
        estimator.evaluate(
            input_fn=eval_input_fn,
            steps=100,
            hooks=None,
            checkpoint_path=None,
            name=None)
    else:
        pred_input_fn = None
        estimator.predict(
            input_fn=pred_input_fn,
            predict_keys=None,
            hooks=None,
            checkpoint_path=None,
            yield_single_examples=True)
if __name__ == "__main__":
    flags.mark_flag_as_required("config_file")
    tf.app.run()
    # tf.enable_eager_execution()
    # tf.logging.set_verbosity(tf.logging.INFO)
    # input_file="/search/odin/self-refresh/intelligent-chinese/data/train_data/recorddata"
    # input_fn = record_input_fn_train_builder([input_file],64,32)
    # params = None
    # d = input_fn(params)
    # iterator = d.make_one_shot_iterator()
    # # iterator = tf.compat.v1.data.make_one_shot_iterator(d)
    # tf.logging.info("lrh check")

    # for i in range(10):
        # tf.logging.info("show check")
        # next_element = iterator.get_next()
        # input_ids_a = next_element["input_ids_a"]
        # input_ids_b = next_element["input_ids_b"]
        # # tf.logging.info("input_ids_a:%s"%(input_ids_a.numpy()))
        # tf.logging.info("input_ids_a:%s"%(input_ids_a))
