""" Adapted from Tensorflow's seq2seq tutorial code.
https://www.tensorflow.org/tutorials/seq2seq/
"""
from __future__ import print_function

import os
import jieba
import sys
import codecs
import xlrd

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_NT = "_NT"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _NT]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
NT_ID = 4


def basic_tokenizer(sentence):
    return jieba.cut(sentence)


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with open(data_path, "r") as f:
        counter = 0
        for line in f:
            counter += 1
            line = line.split("\t")[1].strip()
            if counter % 100000 == 0:
                print("  processing line %d" % counter)
            tokens = basic_tokenizer(line)
            for word in tokens:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with codecs.open(vocabulary_path, "w", 'utf-8') as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with codecs.open(vocabulary_path, "r",'utf-8') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_tok_ids(sentence, vocabulary):
    words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_tok_ids(data_path, target_path, vocabulary_path, model="seq2seq"):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with open(data_path, "r") as data_file:
        with open(target_path, "w") as tokens_file:
            counter = 0
            for line in data_file:
                line = line.split("\t")[1].strip()
                counter += 1
                if counter % 100000 == 0:
                    print("  tokenizing line %d" % counter)
                if model == 'seq2tree':
                    tree = Tree.fromstring(line.strip())
                    seqlist = tree.get_sequences()
                    tok_seqlist = [[vocab.get(w, UNK_ID) for w in seq] for seq in seqlist]
                    tokens_file.write(
                        "\t".join([" ".join([str(tok) for tok in seq_ids]) for seq_ids in tok_seqlist]) + "\n")
                else:
                    tokdst_ids = sentence_to_tok_ids(line, vocab)
                    tokens_file.write(" ".join([str(tok) for tok in tokdst_ids]) + "\n")


def read_data(source_path, target_path, max_size=None, model="seq2seq"):
    data_set = []
    with open(source_path, "r") as source_file:
        with open(target_path, "r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                if model == "seq2seq":
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(EOS_ID)
                else:
                    seqlist = target.split("\t")
                    target_ids = []
                    for seq in seqlist:
                        seq_target_ids = [int(x) for x in seq.split()]
                        seq_target_ids.append(EOS_ID)
                        target_ids.append(seq_target_ids)
                data_set.append([source_ids, target_ids])
                source, target = source_file.readline(), target_file.readline()
    return data_set

def data_to_tok_ids_dev(data_path, source_path, target_path, vocabulary_path, model="seq2seq"):

    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    data = xlrd.open_workbook(data_path)
    table = data.sheet_by_index(0)
    nrows = table.nrows
    counter = 0
    with open(source_path, "w") as src_tokens_file:
        with open(target_path, "w") as dst_tokens_file:
            for i in range(1, nrows):
                source, target = unicode(table.cell(i,3).value), unicode(table.cell(i,5).value)
                counter += 1
                if counter % 100000 == 0:
                    print("  tokenizing line %d" % counter)
                if model == 'seq2tree':
                    """
                    tree = Tree.fromstring(line.strip())
                    seqlist = tree.get_sequences()
                    tok_seqlist = [[vocab.get(w, UNK_ID) for w in seq] for seq in seqlist]
                    tokens_file.write(
                        "\t".join([" ".join([str(tok) for tok in seq_ids]) for seq_ids in tok_seqlist]) + "\n")
                    """
                else:
                    src_tokdst_ids = sentence_to_tok_ids(source, vocab)
                    dst_tokdst_ids = sentence_to_tok_ids(target, vocab)
                    src_tokens_file.write(" ".join([str(tok) for tok in src_tokdst_ids]) + "\n")
                    dst_tokens_file.write(" ".join([str(tok) for tok in dst_tokdst_ids]) + "\n")


def prepare_data(data_dir, src_vocabulary_size, dst_vocabulary_size,
                 recreate=False, model="seq2seq"):

    # Create vocabularies of the appropriate sizes.
    src_vocab_path = os.path.join(data_dir, "vocab.q")
    dst_vocab_path = os.path.join(data_dir, "vocab.lf")
    if not os.path.exists(src_vocab_path):
        create_vocabulary(src_vocab_path, os.path.join(data_dir,"mnist-stc2-repos-id-post"), src_vocabulary_size)
    if not os.path.exists(dst_vocab_path):
        create_vocabulary(dst_vocab_path, os.path.join(data_dir,"mnist-stc2-repos-id-cmnt"), dst_vocabulary_size)

    # Create token ids for the training data.
    src_train_ids_path = os.path.join(data_dir,"train.ids.q")
    dst_train_ids_path = os.path.join(data_dir,"train.ids.lf")
    if not os.path.exists(src_train_ids_path):
        data_to_tok_ids(os.path.join(data_dir,"mnist-stc2-repos-id-post"), src_train_ids_path, src_vocab_path)
    if not os.path.exists(dst_train_ids_path):
        data_to_tok_ids(os.path.join(data_dir,"mnist-stc2-repos-id-cmnt"), dst_train_ids_path, dst_vocab_path, model)


    # Create token ids for the development data.
    src_dev_ids_path = os.path.join(data_dir,"dev.ids.q")
    dst_dev_ids_path = os.path.join(data_dir,"dev.ids.lf")
    if not os.path.exists(src_dev_ids_path) or not os.path.exists(dst_dev_ids_path):
        data_to_tok_ids_dev(os.path.join(data_dir,"mnist-STC2_Train_V1.0.xls"), src_dev_ids_path, dst_dev_ids_path, src_vocab_path)
    """
    # Create token ids for the testing data.
    src_test_ids_path = test_path + (".ids.q")
    dst_test_ids_path = test_path + (".ids.lf")
    data_to_tok_ids(test_path + ".q", src_test_ids_path, src_vocab_path)
    data_to_tok_ids(test_path + ".lf", dst_test_ids_path, dst_vocab_path, model)
       
    return read_data(src_train_ids_path, dst_train_ids_path, model=model), read_data(src_dev_ids_path, dst_dev_ids_path,
                                                                                     model=model), read_data(
        src_test_ids_path, dst_test_ids_path, model=model)
        
    """
    return read_data(src_train_ids_path, dst_train_ids_path, model=model), \
            read_data(src_dev_ids_path, dst_dev_ids_path, model=model), \
           read_data(src_dev_ids_path, dst_dev_ids_path, model=model)

