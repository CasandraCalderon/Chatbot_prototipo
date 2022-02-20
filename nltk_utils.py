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



def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence: 
            bag[idx] = 1.0
    return bag

#Problemas con bag_of_words, algunas palabras no las reconoce
#sentence = ["hello", "how", "are", "you"]
#words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
#bag = bag_of_words(sentence, words)
#print(bag)

    

#Separar una oracion por tokens
#a = "Hola, gracias por comunicarte conmigo"
#print(a)
#a = tokenize(a)
#print(a)

#Usar stemming y Lemmatization en español
#words = ["Organizar", "organizo", "organizamos"]
#stemmed_words= [stem(w) for w in words]
#print(stemmed_words)
#print(spanishStemmer.stem("matriculas"))

