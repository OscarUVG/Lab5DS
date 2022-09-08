import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string

# Librerias de Natural Language Toolkit para el trato de palabras
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

from collections import defaultdict
from collections import  Counter



train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("--- Vista inicial de los datos ---")
print(train.head())

print("--- Conteo de valores NA ---")
print(train.isnull().sum())

# Tokenizacion de un mensaje en palabras
def get_corpus(texts):
    corpus = []
    for text in texts:
        corpus.extend(word_tokenize(text))
    return corpus

# Conjunto de "stop words", i.e. palabras comunes como articulos y preposiciones.
# Estas palabras no aportan mucho al mensaje o nucleo de un texto. 
def get_stop_words():
    return set(stopwords.words('english'))
    
# Removimiento de stop words de un mensaje tokenizado por palabras
def remove_stop_words(text):
    result = ""
    stop_words = get_stop_words()
    for word in get_corpus([text]):
        if word not in stop_words:
            result += word + " "
    return result.strip()

# Conjunto de signos de puntuación.
# Estos no aportan al mensaje principal de un texto. 
def get_punctuation():
    return string.punctuation

# Removimiento de signos de puntuacion
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

# Homogenizamos al data set conviertiendo todo a minusculas
def get_lowercase(texts):
    lowercase_texts = []
    for text in texts:
        lowercase_texts.append(text.lower())
    return lowercase_texts

# Removimiento de URL's
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# Removimiento de HTML's
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# Aplicacion de la limpieza
# En resumen, tenemos:
# - Conversion a minusculas
# - Remover URL
# - Remover HTML
# - Remover signos de puntuacion
# - Remover stopwords
train["text"] = get_lowercase(train["text"])
train["text"] = train["text"].apply(lambda x : remove_URL(x))
train["text"] = train["text"].apply(lambda x : remove_html(x))
train["text"] = train["text"].apply(lambda x : remove_punct(x))
train["text"] = train["text"].apply(lambda x : remove_stop_words(x))

# --- EDA de palabras mas comunes ---
corpus_disaster = get_corpus(train[train["target"] == 1]["text"])
corpus_non_disaster = get_corpus(train[train["target"] == 0]["text"])

# Palabras comunes en texts de desastre
# Las palabras mas comunes son:
# fire, news, via, disaster, california, suicide, police, amp, people, killed

counter=Counter(corpus_disaster)
most=counter.most_common()
words=[]
freq=[]
for word,count in most:
    words.append(word)
    freq.append(count)

sns.barplot(x = freq[:10] , y = words[:10], palette="Blues_r").set_title("Palabras más comunes en tweets desastres")
#plt.show()

# Palabras comunes en texts de no desastre
# Las palabras mas comunes son:
# like, im, amp, new, get, dont, one, body, via, would

counter=Counter(corpus_non_disaster)
most=counter.most_common()
words=[]
freq=[]
for word,count in most:
        words.append(word)
        freq.append(count)

sns.barplot(x = freq[:10] , y = words[:10], palette="Blues_r").set_title("Palabras más comunes en tweets normales")
#plt.show()

# Bigramas en texts de desastre

from sklearn.feature_extraction.text import CountVectorizer

def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

top_tweet_bigrams_disaster = get_top_tweet_bigrams(train[train["target"] == 1]['text'])[:10]
top_tweet_bigrams_non_disaster = get_top_tweet_bigrams(train[train["target"] == 0]['text'])[:10]
x,y=map(list,zip(*top_tweet_bigrams_disaster))
sns.barplot(x=y,y=x, palette="Blues_r").set_title("Bigrama desastre")
plt.show()

x,y=map(list,zip(*top_tweet_bigrams_non_disaster))
sns.barplot(x=y,y=x, palette="Blues_r").set_title("Bigrama no desastre")
plt.show()

# Trigramas en texts de desastre

def get_top_tweet_trigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

top_tweet_trigrams_disaster = get_top_tweet_trigrams(train[train["target"] == 1]['text'])[:10]
top_tweet_trigrams_non_disaster = get_top_tweet_trigrams(train[train["target"] == 0]['text'])[:10]
x,y=map(list,zip(*top_tweet_trigrams_disaster))
sns.barplot(x=y,y=x, palette="Blues_r").set_title("Trigrama desastre")
plt.show()

x,y=map(list,zip(*top_tweet_trigrams_non_disaster))
sns.barplot(x=y,y=x, palette="Blues_r").set_title("Trigrama no desastre")
plt.show()


# --- Analisis de sentimientos ---
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

sentimentValue = np.zeros(len(train['text']))
for k in range(len(train['text'])):
    sentimentValue[k] = sia.polarity_scores(train['text'][k]).get('compound')
train.insert(loc=len(train.columns), column = 'sentimentValue', value = sentimentValue)


# En promedio, el data set es un poco negativo.
# La media de la columna sentimentValue es -0.14. 
print("Media de sentimiento: ", train['sentimentValue'].mean())

# Top 10 tweets mas negativos segun sentimentValue.
print(train.sort_values(by='sentimentValue').head(10)['text'])

# Top 10 tweets mas negativos segun sentimentValue.
print(train.sort_values(by='sentimentValue', ascending=False).head(10)['text'])

# Promedio de sentimentValue para tweets relacionados a desastre y no desastre.
# La media de los tweets sobre desastre es -0.26.
# La media de los tweets sobre no desastre es -0.05. 
print("Media sentimentValue desastre: ", train[train["target"] == 1]['sentimentValue'].mean())
print("Media sentimentValue no desastre: ", train[train["target"] == 0]['sentimentValue'].mean())



