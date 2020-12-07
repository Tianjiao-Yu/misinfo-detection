
import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset.csv')

#
# test_filename = 'test.csv'
# train_filename = 'train.csv'
# train_news = pd.read_csv(train_filename)
# test_news = pd.read_csv(test_filename)
train_news, test_news = train_test_split(df, test_size=0.2)

def data_obs():
    print("training dataset size:")
    print(train_news.shape)
    print(train_news.head(2))

    #below dataset were used for testing and validation purposes
    print(test_news.shape)
    print(test_news.head(2))
#data_obs()

def create_distribution(dataFile):

    return sb.countplot(x='credibility', data=dataFile, palette='hls')
#by calling below we can see that training, test and valid data seems to be failry evenly distributed between the classes
# create_distribution(train_news)
# create_distribution(test_news)
# plt.show()




#data integrity check (missing label values)
def data_qualityCheck():

    print("Checking data qualitites...")
    train_news.isnull().sum()
    train_news.info()

    print("check finished.")

    #below datasets were used to
    test_news.isnull().sum()
    test_news.info()
#run the below function call to see the quality check results
#data_qualityCheck()


#Stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

#process the data
def process_data(data,exclude_stopword=True,stem=True):
    tokens = [w.lower() for w in data]
    tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(tokens, eng_stemmer)
    tokens_stemmed = [w for w in tokens_stemmed if w not in stopwords ]
    return tokens_stemmed

#creating ngrams
#unigram
def create_unigram(words):
    assert type(words) == list
    return words

#bigram
def create_bigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 1:
        lst = []
        for i in range(Len-1):
            for k in range(1,skip+2):
                if i+k < Len:
                    lst.append(join_str.join([words[i],words[i+k]]))
    else:
        #set it as unigram
        lst = create_unigram(words)
    return lst


porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
