#%%
from pprint import pprint

train_file_path = 'data/conll2003/en/train.txt'
test_file_path = 'data/conll2003/en/test.txt'
valid_file_path = 'data/conll2003/en/valid.txt'

out_train_file_path = 'data/conll2003/en/train_sentences.txt'
out_test_file_path = 'data/conll2003/en/test_sentences.txt'
out_valid_file_path = 'data/conll2003/en/valid_sentences.txt'


train_file = open(train_file_path,'r')
test_file  = open(test_file_path,'r')
valid_file = open(valid_file_path,'r')

out_train_file = open(out_train_file_path,'w')
out_test_file  = open(out_test_file_path,'w')
out_valid_file = open(out_valid_file_path,'w')

inputs = [train_file,test_file,valid_file]
outs = [out_train_file,out_test_file,out_valid_file]

for idx,input_file in enumerate(inputs):
    lines = input_file.readlines()
    sentence = []
    for line in lines:
        if 'DOCSTART' in line:
            if len(sentence) > 1:
                out_file = outs[idx]
                out_file.write(' '.join(sentence)+'\n')
                sentence = []
                continue
        try:
            word = line.split()[0]
            if 'DOCSTART' not in word:
                sentence.append(word)
        except:
            pass 

for file in inputs:
    file.close()
for file in outs:
    file.close()

