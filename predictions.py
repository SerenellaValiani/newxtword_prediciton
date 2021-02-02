#!/usr/bin/env python
# coding: utf-8
import keras
from tensorflow.keras.models import load_model
import numpy as np
import pickle

#carico il modello e il tokenizer
model = load_model('nextword1.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):
    #text = text.split()
    #print(text)
    #print(type(text))
    #for i in range(5):
    #    sequence = tokenizer.texts_to_sequences([text])[0]
    #    sequence = np.array(sequence)
    #    
    #    preds = model.predict_classes(sequence)
    #    predicted_word = ""
    

    list_text = text.split()
    list_sequence = [0, 0, 0, 0, 0 ]
    for i in range(len(list_text)):
        list_sequence[i] = tokenizer.text_to_sequence([list_text[i]])[0]
        
    sequence = np.array(list_sequence)
    preds = model.predict_classes(sequence)
    predicted_word = ''
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
        
        print(predicted_word)
        return predicted_word

while(True):

    text = input("Enter your line, using only lowercase letters: ")
    
    if text == "stop the script":
        print("Ending The Program.....")
        break
    
    else:
        try:
            #text = text.split(" ")
            #text = text[-1]

            #text = ''.join(text)
            Predict_Next_Words(model, tokenizer, text)
            
        except:
            continue

