# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:26:41 2022

@author: Siren Wang
"""

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def clean_text(str_in):
    #regex api
    import re
    tmp = re.sub("[^A-Za-z']+", ' ', str_in).lower().strip()
    return tmp


def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in word_tokenize(line):
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
#         contents_clean.append(line_clean)
        contents_clean.append(' '.join(line_clean))  # 重新拼接
#         print(contents_clean)
    return contents_clean,all_words


# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def text_lemmatization(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tagged_sent = pos_tag(tokens)   # 获取单词词性
    lemma_text = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemma_text.append(wordnet_lemmatizer.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
    lemma_text = ' '.join(lemma_text)
    return lemma_text

