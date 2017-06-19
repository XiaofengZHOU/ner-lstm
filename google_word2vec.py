#%%
import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_google = gensim.models.KeyedVectors.load_word2vec_format('tmp/GoogleNews-vectors-negative300.bin', binary=True)


#%%
model_google.most_similar("Amazon")