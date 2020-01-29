#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def gen_embedding_table_file(ori_embedding_file,vocab_file,embedding_table_file,skip_first_line=True):
    emb_fp = open(embedding_table_file,"w")
    vocab_fp = open(vocab_file,"w")
    embedding_table = []
    vocab = []
    with open(ori_embedding_file,"r") as fp:
        first_line = True 
        for line in fp.readlines():
            if first_line == True:
                first_line = False
                continue
            else:
                items = line.rstrip().split(' ')
                word = items[0]
                embedding_table.append(items[1:])
                vocab_fp.write("%s\n"%(word))
    embedding_table = np.asarray(embedding_table)
    np.save(embedding_table_file,embedding_table)

if __name__ == "__main__":
    ori_w2v_file = "/search/odin/open_source/dataset/baidubaike_Embedding/word-ngram"
    vocab_file = "/search/odin/open_source/dataset/baidubaike_Embedding/vocab_word-ngram"
    embedding_table_file = "/search/odin/open_source/dataset/baidubaike_Embedding/embtable_word-ngram"
    gen_embedding_table_file(ori_w2v_file,vocab_file,embedding_table_file)

