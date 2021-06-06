# coding=utf-8

import numpy as np
import re
import itertools
from collections import Counter
import tensorflow.compat.v1  as tf
import csv
import pickle
import os

tf.disable_v2_behavior()
tf.flags.DEFINE_string("valid_data", "/content/NARRE/data/music/music_valid.csv", "Data for validation")
tf.flags.DEFINE_string("test_data", "/content/NARRE/data/music/music_test.csv", "Data for testing")
tf.flags.DEFINE_string("train_data", "/content/NARRE/data/music/music_train.csv", "Data for training")
tf.flags.DEFINE_string("user_review", "/content/NARRE/data/music/user_review", "User's reviews")
tf.flags.DEFINE_string("item_review", "/content/NARRE/data/music/item_review", "Item's reviews")
tf.flags.DEFINE_string("user_review_id", "/content/NARRE/data/music/user_rid", "user_review_id")
tf.flags.DEFINE_string("item_review_id", "/content/NARRE/data/music/item_rid", "item_review_id")
tf.flags.DEFINE_string("stopwords", "/content/NARRE/data/stopwords", "stopwords")

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n'\t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def pad_sentences(u_text, u_len, u2_len, padding_word="<PAD/>"):

    """
    Pads all sentences to teh same length. The length is defined by the longest sentence.
    """
    review_num = u_len
    review_len = u2_len

    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        padded_u_train = []
        for ri in range(review_num):
            if ri < len(u_reviews):
                sentence = u_reviews[ri]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append(new_sentence)
                else:
                    new_sentence = sentence[:review_len]
                    padded_u_train.append(new_sentence)
            else:
                new_sentence = [padding_word] * review_len
                padded_u_train.append(new_sentence)
        u_text2[i] = padded_u_train

    return u_text2

def pad_reviewid(u_train, u_valid, u_len, num):

    """
    num is the padding id
    """

    pad_u_train = []
    for i in range(len(u_train)):
        x = u_train[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_train.append(x)

    pad_u_valid = []
    for i in range(len(u_valid)):
        x = u_valid[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_valid.append(x)

    return pad_u_train, pad_u_valid

def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 =  {x: i for i, x in enumerate(vocabulary_inv1)}

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]

def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentence and labels to vectors based on a vocabulary
    """
    l = len(u_text)
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([[vocabulary_u[word] for word in words] for words in u_reviews])
        u_text2[i] = u
    l = len(i_text)

    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([[vocabulary_i[word] for word in words] for words in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2

def load_data(train_data, valid_data, user_review, item_review, user_rid, item_rid, stopwords):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    # ===============================================
    print("load data")
    u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num \
        , reid_user_train, reid_item_train, reid_user_valid, reid_item_valid = \
        load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, stopwords)

    # padding
    # ===============================================
    print("pad user")
    u_text = pad_sentences(u_text, u_len, u2_len)
    reid_user_train, reid_user_valid = pad_reviewid(reid_user_train, reid_user_valid, u_len, item_num + 1)
    print("pad item")
    i_text = pad_sentences(i_text, i_len, i2_len)
    reid_item_train, reid_item_valid = pad_reviewid(reid_item_train, reid_item_valid, i_len, user_num + 1)

    # get vocabulary
    # ===============================================
    user_voc = [xx for x in u_text.values() for xx in x]
    item_voc = [xx for x in i_text.values() for xx in x]

    vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
    print(len(vocabulary_user))
    print(len(vocabulary_item))

    u_text, i_text = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)

    return [u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item,
            vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, reid_user_train,
            reid_item_train, reid_user_valid, reid_item_valid]

def load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, stopwords):

    """
    Loads data from files, splits the data into words and generate labels.
    """

    # Load data from file

    f_train = open(train_data, "r")
    f1 = open(user_review, "rb")
    f2 = open(item_review, "rb")
    f3 = open(user_rid, "rb")
    f4 = open(item_rid, "rb")

    user_reviews = pickle.load(f1)
    item_reviews = pickle.load(f2)
    user_rids = pickle.load(f3)
    item_rids = pickle.load(f4)

    reid_user_train = []
    reid_item_train = []
    uid_train = []
    iid_train = []
    y_train = []
    u_text = {}
    u_rid = {}
    i_text = {}
    i_rid = {}
    i = 0

    for line in f_train:
        i = i + 1
        line = line.split(",")
        user_id = int(line[0])
        item_id = int(line[1])
        uid_train.append(user_id)
        iid_train.append(item_id)

        # add user_id
        if user_id in u_text:
            reid_user_train.append(u_rid[user_id])
        else:
            u_text[user_id] = []
            for s in user_reviews[user_id]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                u_text[user_id].append(s1)
            u_rid[user_id] = []
            for s in user_rids[user_id]:
                u_rid[user_id].append(int(s))
            reid_user_train.append(u_rid[user_id])

        # add item_id
        if item_id in i_text:
            reid_item_train.append(i_rid[item_id])
        else:
            i_text[item_id] = []
            for s in item_reviews[item_id]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                i_text[item_id].append(s1)
            i_rid[item_id] = []
            for s in item_rids[item_id]:
                i_rid[item_id].append(int(s))
            reid_item_train.append(i_rid[item_id])
        y_train.append(float(line[2]))


    print("valid")
    reid_user_valid = []
    reid_item_valid = []

    uid_valid = []
    iid_valid = []
    y_valid = []
    f_valid = open(valid_data)
    for line in f_valid:
        line = line.split(",")
        user_id = int(line[0])
        item_id = int(line[1])
        uid_valid.append(user_id)
        iid_valid.append(item_id)

        if user_id in u_text:
            reid_user_valid.append(u_rid[user_id])
        else:
            u_text[user_id] = [['<PAD/>']]
            u_rid[user_id] = [int(0)]
            reid_user_valid.append(u_rid[user_id])

        if item_id in i_text:
            reid_item_valid.append(i_rid[item_id])
        else:
            i_text[item_id] = [['<PAD/>']]
            i_text[item_id] = [int(0)]
            reid_item_valid.append(i_rid[item_id])

        y_valid.append(float(line[2]))


    review_num_u = np.array([len(x) for x in u_text.values()])
    x = np.sort(review_num_u)
    u_len = x[int(0.9 * len(review_num_u)) - 1]
    review_len_u = np.array([len(j) for i in u_text.values() for j in i])
    x2 = np.sort(review_len_u)
    u2_len = x2[int(0.9 * len(review_len_u)) - 1]

    review_num_i = np.array([len(x) for x in i_text.values()])
    y = np.sort(review_num_i)
    i_len = y[int(0.9 * len(review_num_i)) - 1]
    review_len_i = np.array([len(j) for i in i_text.values() for j in i])
    y2 = np.sort(review_len_i)
    i2_len = y2[int(0.9 * len(review_len_i)) - 1]
    print("u_len:", u_len)
    print("i_len:", i_len)
    print("u2_len:", u2_len)
    print("i2_len:", i2_len)
    user_num = len(u_text)
    item_num = len(i_text)
    print("user_num:", user_num)
    print("item_num:", item_num)
    return [u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train,
            iid_train, uid_valid, iid_valid, user_num, item_num,
            reid_user_train, reid_item_train, reid_user_valid, reid_item_valid]



if __name__ == '__main__':
    TPS_DIR = "/content/NARRE/data/music"
    FLAGS = tf.flags.FLAGS
    #FLAGS._parse_flags()

    u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item, \
    vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid = \
        load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.user_review, FLAGS.item_review, FLAGS.user_review_id,
                  FLAGS.item_review_id, FLAGS.stopwords)

    np.random.seed(2017)

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    reid_user_train = reid_user_train[shuffle_indices]
    reid_item_train = reid_item_train[shuffle_indices]

    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    batches_train = list(
        zip(userid_train, itemid_train, reid_user_train, reid_item_train, y_train)
    )
    batches_test = list(zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid, y_valid))
    print("write begin")

    output = open(os.path.join(TPS_DIR, 'music.train'), 'wb')
    pickle.dump(batches_train, output)
    output = open(os.path.join(TPS_DIR, 'music.test'), 'wb')
    pickle.dump(batches_test, output)

    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['review_num_u'] = u_text[0].shape[0]
    para['review_num_i'] = i_text[0].shape[0]
    para['review_len_u'] = u_text[1].shape[1]
    para['review_len_i'] = i_text[1].shape[1]
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['train_length'] = len(y_train)
    para['test_length'] = len(y_valid)
    para['u_text'] = u_text
    para['i_text'] = i_text

    # print("user_num", para['user_num'])
    # print("item num", para['item_num'])
    # print("review_num_u", para["review_num_u"])
    # print("review_num_i", para["review_num_i"])
    # print("review_len_u", para["review_len_u"])
    # print("review_len_i", para["review_len_i"])
    # print("user_vocab", para["user_vocab"])
    # print("item_vocab", para["item_vocab"])
    # print("train_length", para["train_length"])
    # print("test_length", para["test_length"])
    # print("u_text", u_text[0])
    # print("i_text", i_text[0])

    output = open(os.path.join(TPS_DIR, 'music.para'), 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(para, output)


