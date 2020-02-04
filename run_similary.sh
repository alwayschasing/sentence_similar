#!/bin/bash

input_file="/search/odin/self-refresh/intelligent-chinese/data/train_data/recorddata"
output_dir="/search/odin/self-refresh/model/sentence_similar"
embedding_table_file="/search/odin/open_source/dataset/baidubaike_Embedding/embtable_word-ngram.npy"

python sentence_similary.py \
    --config_file=configure.json \
    --input_file=$input_file \
    --output_dir=$output_dir \
    --batch_size=32 \
    --train_steps=10 \
    --embedding_table_file=$embedding_table_file \
    --do_train=True
