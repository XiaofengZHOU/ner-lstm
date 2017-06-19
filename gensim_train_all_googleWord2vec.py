#%%
import os
import gensim 
import logging
from pprint import pprint
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#%%
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
 
sentences = MySentences('data/gensim_training_file/') # a memory-friendly iterator


#%%
#to modify
model = gensim.models.Word2Vec(size=300,min_count=0)  # an empty model, no training yet
model.build_vocab(sentences)
model.intersect_word2vec_format('tmp/GoogleNews-vectors-negative300.bin', binary=True)
print(model)



#%%
model.train(sentences,total_examples = 3589,epochs=10)

#%%
model.most_similar('KwaZulu-Natal')

#%%
model.save('data/models_trained/conll_article_model/conll_article_model')


#%%
new_model = gensim.models.Word2Vec.load('data/models_trained/conll_article_model/conll_article_model')

#%%
new_model.most_similar('KwaZulu-Natal')

#%%
new_model.wv['U.S.'].shape