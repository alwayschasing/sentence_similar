#!/bin/bash 

raw_data_file="/search/odin/self-refresh/intelligent-chinese/data/postprocess_data/idiom_explanation.txt"
train_data_file="/search/odin/self-refresh/intelligent-chinese/data/train_data/train_data"
recorddata_file="/search/odin/self-refresh/intelligent-chinese/data/train_data/recorddata"
vocab_file="/search/odin/open_source/dataset/baidubaike_Embedding/word-ngram"

python create_data.py \
    --raw_data_file=$raw_data_file \
    --train_data_file=$train_data_file \
    --recorddata_file=$recorddata_file \
    --vocab_file=$vocab_file
