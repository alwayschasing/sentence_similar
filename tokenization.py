#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
# from segmentor import segmentor
# from util import CodeUtil
import jieba

class Tokenizer(object):
    """Runs basic tokenization (punctuation spliting, lower casing, etc.)."""
    # word_start_index: the start index of words
    def __init__(self,data_file):
        """
        data_file:word2vec data file, line:word vec
        """
        self.vocab,self.vectors = load_data(data_file)
        self.vocab_size = len(self.vocab)
        self.word2id = {}
        for i in range(self.vocab_size):
            self.word2id[self.vocab[i]] = i

        self.vec_size=0
        if self.vocab_size > 0:
            self.vec_size = len(self.vectors[0])
                

    def tokenize(self, text):
        """Tokenizes a piece of text"""
        words = jieba.cut(text)
        tokens = []
        for w in words:
            if w in self.word2id:
                tokens.append(self.word2id[w])
        return tokens

    def untokenize(self, ids):
        words = []
        for id in ids:
            words.append(self.vocab[id])
        return words

    def translate2id(self,sentence,is_split=True, delimeter=' '):
        word_list = sentence.split(' ')
        ids = []
        for w in word_list:
            if w in self.word2id:
                ids.append(id)

        return ids

    def translate2word(self,ids):
        word_list = []
        for id in ids:
            word_list.append(self.vocab[id])

        return word_list




def verify_data(words,vectors):
    count = len(words)
    if count != len(vectors) or count == 0:
        return False
    
    vec_size = len(vectors[0])
    for i in range(count):
        if len(vectors[i]) != vec_size:
            return False
    return True


def load_data(data_file):
    words = []
    vectors = []
    first_line = True
    with open(data_file,'r') as rfp:
        for line in rfp:
            if first_line:
                first_line = False
                continue
            items = line.rstrip().split(' ')
            words.append(items[0])
            vectors.append(items[1:])
    if verify_data(words,vectors) == True:
        return words,vectors
    else:
        return [],[]

if __name__ == "__main__":
    # vocab_file = "/search/odin/workspace/querySemantic/data/transformerData/vocab"
    # vocab_vec_file="/search/odin/workspace/querySemantic/data/transformerData/vocab_vec.npy"
    # tokenizer = Tokenizer(vocab_file,vocab_vec_file)
    data_file = "/search/odin/open_source/dataset/baidubaike_Embedding/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5"
    tokenizer = Tokenizer(data_file)
    print("vocab size:%d,vec_size:%d"%(tokenizer.vocab_size,tokenizer.vec_size))
    
