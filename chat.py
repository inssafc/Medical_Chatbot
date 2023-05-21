import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

model = load_model(r'C:\Users\DELL\Documents\chatbotfiles\testmodel.h5')

intents = json.loads(open(r'C:\Users\DELL\Documents\chatbotfiles\intents.json').read())
words = pickle.load(open(r'C:\Users\DELL\Documents\chatbotfiles\words.pkl','rb'))
classes = pickle.load(open(r'C:\Users\DELL\Documents\chatbotfiles\classes.pkl','rb'))


#clean_up_messages
def clean_up_sentence(sentence):
  sentence_words=nltk.word_tokenize(sentence)
  sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
  return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence):
  sentence_words=clean_up_sentence(sentence)
  bag=[0]*len(words)
  for w in sentence_words:
    for i,word in enumerate(words):
      if word == w:
        bag[i]=1
  return np.array(bag)

def predict_class(sentence,model):
    # filter out predictions below a threshold
    p = bow(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])}) #r[0] E 0,...14    
        #print(return_list)
    return return_list


def get_response(intents_list,intents_json): #intents_list = return_list
  tag=intents_list[0]['intent'] #first tag (that has the highest probability)
  list_of_intents=intents_json['intents'] #ensemble of intents ? oui ig
  for i in list_of_intents:
    if i['tag']==tag:
      result=random.choice(i['responses'])
      break
  return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res

