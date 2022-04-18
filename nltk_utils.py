import numpy as np
import nltk

nltk.download('punkt')
#from nltk.stem.porter import PorterStemmer

from nltk.stem.snowball import SnowballStemmer

#stemmer = PorterStemmer()
spanishStemmer=SnowballStemmer("spanish", ignore_stopwords=True)

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return spanishStemmer.stem(word.lower())


"""
def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    new_words = [stem(word) for word in words]
    # initialize bag with 0 for each word
    bag = np.zeros(len(new_words), dtype=np.float32)
    for idx, w in enumerate(new_words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag
"""

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

"""
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag=[]
    for w in words:
        bag.append(sentence_words.count(w))
    return bag
"""


"""
#DEMOSTRACION DE CADA PASO (python nltk_utils.py)
#BAG OF WORDS (Utiliza los dos metodos para verificar)
sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag = bag_of_words(sentence, words)
#bag = bag_of_words(sentence, words)
print(bag)
"""








