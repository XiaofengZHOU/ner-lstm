#%%
import json
import spacy
from pprint import pprint

nlp = spacy.load('en')

path = 'data/data.json'
f = open(path,'r')
sites = json.load(f)
f.close()
articles_list = []

for key in sites.keys():
    articles = sites[key]
    for article in articles:
        articles_list.append(article["content"])
print('number of articles: ',len(articles_list))
#%%
text_file1 = 'data/articles_in_conll_format.txt'
text_file2 = 'data/articles_in_sentences_format.txt'
f1 = open(text_file1,'w+')
f2 = open(text_file2,'w+')
for idx,article in enumerate(articles_list):
    article = article.replace('\n',' ')
    article = article.replace('\t',' ')
    doc = nlp(article)
    sentence = []
    sentence.append('-DOCSTART- -X- -X- O')
    for word in doc:
        if  word.ent_iob_ == "O":
            entity = word.ent_iob_
        else :
            entity = word.ent_iob_ + '-' + word.ent_type_
        line = word.orth_ + ' ' + word.tag_ +  ' ' + word.pos_ +  ' ' + entity + '\n'
        sentence.append(word.orth_)
        try:
            f1.write(line)
        except:
            #print(article["url"])
            pass
    f2.write(' '.join(sentence) + '\n')
    if idx%101 == 0:
        print(idx/len(articles_list)*100)
f1.close()
f2.close()
