import numpy as np
import nltk
nltk.download('punkt')
#from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
#stemmer = PorterStemmer()
spanishStemmer=SnowballStemmer("spanish", ignore_stopwords=True)

"""TOKENIZACION"""
def tokenize(sentence):
    """
    Aqui se dividira la oracion en una matriz de palabras
    o tokens ya sea una palabra, caracter, puntuacion o
    un numero
    """
    return nltk.word_tokenize(sentence)

"""STEMMING"""
def stem(word):
    """
    Aqui se encontrara la palabra raiz de cada palabra
    encontrada en la matriz
    """
    return spanishStemmer.stem(word.lower())

"""BAG OF WORDS"""
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag


"""
#TOKENIZACION
a = "¿ Hola, mucho gusto 3?"
print(a)
a = tokenize(a)
print(a)
#STEMMING
words = ["MATRICULA", "matricular", "matricularse"]
stemmed_words= [stem(w) for w in words]
print(stemmed_words)
#BAG OF WORDS
sentence = ["hola", "como", "estas"]
words = ["hola", "adios", "yo", "tu", "estas", "gracias", "como"]
bag = bag_of_words(sentence, words)
print(bag)
"""











