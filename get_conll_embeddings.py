#%%
import pickle
import numpy as np 
from pprint import pprint
import gensim 

def find_max_length_sentence(file_name):
    temp_len = 0
    max_length = 0
    for line in open(file_name,'r'):
        if line in ['\n', '\r\n']:
            if temp_len > max_length:
                max_length = temp_len
            temp_len = 0
        else:
            temp_len += 1
    return max_length

def pos(tag):
    one_hot = np.zeros(5)
    if tag == 'NN' or tag == 'NNS':
        one_hot[0] = 1
    elif tag == 'FW':
        one_hot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        one_hot[2] = 1
    elif 'VB' in tag:
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot

def chunk(tag):
    one_hot = np.zeros(5)
    if 'NP' in tag:
        one_hot[0] = 1
    elif 'VP' in tag:
        one_hot[1] = 1
    elif 'PP' in tag:
        one_hot[2] = 1
    elif tag == 'O':
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot

def capital(word):
    if ord('A') <= ord(word[0]) <= ord('Z'):
        return np.array([1])
    else:
        return np.array([0])


def label(tag):
    one_hot = np.zeros(5)
    if tag.endswith('PER'):
        one_hot[0] =1
    elif tag.endswith('LOC'):
        one_hot[1] =1
    elif tag.endswith('ORG'):
        one_hot[2] =1
    elif tag.endswith('MISC'):
        one_hot[3] =1
    elif tag.endswith('O'):
        one_hot[4] =1
    return one_hot

def pickle_file_with_padding(model_file_name,input_file_name,output_file_name):
    model = gensim.models.Word2Vec.load(model_file_name)
    train_data = []
    train_label = []
    input_file = open(input_file_name)
    lines = input_file.readlines()
    input_file.close()
    max_sentence_length = find_max_length_sentence(input_file_name)

    sentence = []
    sentence_label = []
    for line in lines:
        if 'DOCSTART' in line:
            continue
        if line in ['\n', '\r\n'] :
            if len(sentence) != 0:
                for _ in range(max_sentence_length - len(sentence)):
                    embedding = np.array([0 for _ in range(311)])
                    label_embedding = np.array([0 for _ in range(5)])
                    sentence.append(embedding)
                    sentence_label.append(label_embedding)
                train_data.append(np.array(sentence))
                train_label.append(np.array(sentence_label))
                sentence = []
                sentence_label = []
            else:
                continue
            
        else:
            assert (len(line.split()) == 4)
            line = line.split()
            word = line[0]
            pos_tag = line[1]
            chunk_tag = line[2]
            label_tag = line[3]
            try:
                word_embedding = model.wv[word]
                pos_embedding = pos(pos_tag)
                chunk_embedding = chunk(chunk_tag)
                capital_embedding = capital(word)
                label_embedding = label(label_tag)
                embedding = np.append(word_embedding,pos_embedding)
                embedding = np.append(embedding,chunk_embedding)
                embedding = np.append(embedding,capital_embedding)
                sentence.append(embedding)
                sentence_label.append(label_embedding)
            except:
                print(line,input_file_name)

    assert(len(train_data) == len(train_label))
    f = open(output_file_name,'wb')
    data = {'train_data': train_data, 'train_label':train_label}
    pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    f.close()


def pickle_file_without_padding(model_file_name,input_file_name,output_file_name):
    model = gensim.models.Word2Vec.load(model_file_name)
    train_data = []
    train_label = []
    input_file = open(input_file_name)
    lines = input_file.readlines()
    input_file.close()
    max_sentence_length = find_max_length_sentence(input_file_name)

    sentence = []
    sentence_label = []
    for line in lines:
        if 'DOCSTART' in line:
            continue
        if line in ['\n', '\r\n'] :
            if len(sentence) != 0:
                train_data.append(np.array(sentence))
                train_label.append(np.array(sentence_label))
                sentence = []
                sentence_label = []
            else:
                continue
            
        else:
            assert (len(line.split()) == 4)
            line = line.split()
            word = line[0]
            pos_tag = line[1]
            chunk_tag = line[2]
            label_tag = line[3]
            try:
                word_embedding = model.wv[word]
                pos_embedding = pos(pos_tag)
                chunk_embedding = chunk(chunk_tag)
                capital_embedding = capital(word)
                label_embedding = label(label_tag)
                embedding = np.append(word_embedding,pos_embedding)
                embedding = np.append(embedding,chunk_embedding)
                embedding = np.append(embedding,capital_embedding)
                sentence.append(embedding)
                sentence_label.append(label_embedding)
            except:
                print(line,input_file_name)

    assert(len(train_data) == len(train_label))
    f = open(output_file_name,'wb')
    data = {'train_data': train_data, 'train_label':train_label}
    pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    f.close()

#%%
word2vec_model_path = 'data/models_trained/conll_article_model/conll_article_model'
pickle_file_without_padding(word2vec_model_path,'data/conll2003/en/train.txt','data/lstm_train_file/train_sentences_without_padding.pickle')
pickle_file_without_padding(word2vec_model_path,'data/conll2003/en/test.txt', 'data/lstm_train_file/test_sentences_without_padding.pickle')
pickle_file_without_padding(word2vec_model_path,'data/conll2003/en/valid.txt','data/lstm_train_file/valid_sentences_without_padding.pickle')




    