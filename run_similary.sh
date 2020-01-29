#!/bin/bash

input_files="/search/odin/self-refresh/intelligent-chinese/data/train_data/recorddata"
output_dir="/search/odin/self-refresh/model/sentence_similar"
embedding_table_file=""

python sentence_similary.py \
    --config_file=configure.json \
    --input_files=$input_files \
    --output_dir=$output_dir \
    --init_checkpoint=$output_dir \
    --batch_size=32 \
    --train_steps=10 \
    --embedding_table_file=$embedding_table_file \
    --do_train=True
