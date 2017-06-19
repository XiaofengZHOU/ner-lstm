#%%
import gensim 
import logging
from pprint import pprint
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



#%%
class MySentences(object):
    def __init__(self,fname):
        self.fname = fname
    def __iter__(self):
        with open(self.fname,'r') as text_file :
            for line in text_file:
                yield line.split()
        
sentences = MySentences('data/gensim_training_file/articles_in_sentences_format.txt') # a memory-friendly iterator
#to modify


#%%
#to modify
model = gensim.models.Word2Vec(size=300,min_count=1)  # an empty model, no training yet
model.build_vocab(sentences)
model.intersect_word2vec_format('tmp/GoogleNews-vectors-negative300.bin', binary=True)

#%%
print(model)
total_words = 45658
model.train(sentences,total_words = total_words,epochs=10)

#%%
model.most_similar('Automata',topn=20)

#%%
import pickle
model.save('data/article_model/articles_model')

#%%
new_model = gensim.models.Word2Vec.load('data/models_trained/article_model/articles_model')

#%%
new_model.most_similar('Automata')