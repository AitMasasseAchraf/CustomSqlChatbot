

# from tqdm import tqdm
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.tokenize import RegexpTokenizer 
# from nltk.tokenize import word_tokenize
# import os, re, csv, math, codecs


# # For Training
# import tensorflow as tf




# from keras.preprocessing import sequence
# # from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

# # For array, dataset, and visualizing
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Dict, List
# import json
# app = FastAPI()

# import string
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import spacy
# nltk.download('wordnet')


# nltk.download('stopwords')
# nltk.download('punkt')

# def remove_punctuation(text):
#     translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
#     return text.translate(translation_table)


# def to_lower_case(input_string):
#     return input_string.lower()
    

# def remove_stop_words(vocab):
#     # Tokenize the text into words
#     stop_words = set(stopwords.words('french'))

#     word_to_remove = ['ne','pas','ni' , 'que' , 'non']
#     for word in word_to_remove:
#         if word in stop_words:
#             stop_words.remove(word)

#     # words = word_tokenize(text, language='french')

#     filtered_words = [word for word in vocab if word.lower() not in stop_words]

#     # Reconstruct the text without stop words
#     filtered_text = ' '.join(filtered_words)

#     return filtered_text









# # Provided data
# tier_categories = [
#     'blacklisté', 'blacklistés', 'fournisseur', 'fournisseurs', 'avocats', 'avocat', 
#     'bailleur', 'bailleurs', 'locataire', 'locataires'
# ]
# personne_categories = [
#     "agent de maintenance", "agents de maintenance", "responsable de suivi d'achat", 
#     "responsables de suivi d'achats", "acheteur", "acheteurs", "encaisseurs", "encaisseur", 
#     "responsables de maintenance", "responsable de maintenance"
# ]
# actif_category = ['actifs', 'actif', 'active', 'actives']
# inactifs_category = [
#     'inactifs', 'inactif', 'inactive', 'inactives', 'non actifs', 'non actif', 
#     'non active', 'non actives'
# ]
# cols_dict = {
#     "blacklisté": "blackliste", "blacklistés": "blackliste", "fournisseur": "supplier", 
#     "fournisseurs": "supplier", "avocat": "lawer", "avocats": "lawer", "bailleur": "lessor", 
#     "bailleurs": "lessor", "locataire": "tenant", "locataires": "tenant", 
#     "agent de maintenance": "maintenance_agent", "agents de maintenance": "maintenance_agent", 
#     "responsable de suivi d'achat": "purshase", "responsables de suivi d'achat": "purshase", 
#     "acheteur": "buyer", "acheteurs": "buyer", "encaisseur": "collector", 
#     "encaisseurs": "collector", "responsable de maintenance": "maintenance_responsable", 
#     "responsables de maintenance": "maintenance_responsable"
# }
# path_model='C:/Users/aitma/Downloads/my_LSTM_model.h5'
# model=tf.keras.models.load_model(path_model)
# path1='C:/Users/aitma/Downloads/word_index.json'
# path2='C:/Users/aitma/Downloads/tag_dict.json'
# with open(path1, 'r') as json_file:
#     word_index = json.load(json_file)

# with open(path2, 'r') as json_file:
#     tag_dict = json.load(json_file)
    
# def predict(embedded_text,tag_dict):
#     # embedded_text=np.array([embedded_text])
#     predictions=model.predict(embedded_text)
#     predicted_classes = np.argmax(predictions, axis=1)

#     name_label = [key for val in predicted_classes for key, value in tag_dict.items() if value == val]
#     return name_label
# def generate_query(input_label: str) -> str:
#     nlp = spacy.load('fr_core_news_lg')
    
#     i = 0
#     query = ''
#     condition = ''
#     input_label=remove_punctuation(input_label)
#     input_label=to_lower_case(input_label)
#     input_label=word_tokenize(input_label)
#     input_label=remove_stop_words(input_label)

#     liste=[]
#     doc=nlp(input_label)
#     for token in doc:
#         liste.append(token.lemma_)
    
#     tokenizer = Tokenizer(num_words=len(word_index), lower=True, char_level=False)
#     tokenizer.fit_on_texts(word_index.keys())  #leaky

#     word_seq_train = tokenizer.texts_to_sequences([liste])  # Correct usage

#     # Ensure sequences are properly formatted
#     word_seq_train = [item for sublist in word_seq_train for item in sublist]

#     # Pad sequences
#     max_length = 40
#     word_seq_train = pad_sequences([word_seq_train], maxlen=max_length, padding='pre')
#     input_label=predict(word_seq_train ,tag_dict)

#     if 'tiers' in input_label[0].split():
#         query = 'select * from tier '
#         condition = 'where '
#         for category in tier_categories:
#             if category in input_label[0].split():
#                 i += 1
#                 if i > 1:
#                     condition += ' and '
#                 condition += cols_dict[category] + ' = True'
#         for category in actif_category:
#             if category in input_label[0].split():
#                 if i >= 1:
#                     condition += ' and ' + ' actif = True'
#                 else:
#                     condition += ' actif = True'
#         for category in inactifs_category:
#             if category in input_label[0].split():
#                 if i >= 1:
#                     condition += ' and ' + ' actif = False'
#                 else:
#                     condition += ' actif = False'
#     else:
#         query = 'select * from personne '
#         condition = 'where '
#         for category in personne_categories:
#             if category in input_label[0].split():
#                 i += 1
#                 if i > 1:
#                     condition += ' and '
#                 condition += cols_dict[category] + ' = True'
#         for category in actif_category:
#             if category in input_label[0].split():
#                 if i >= 1:
#                     condition += ' and ' + ' active = True'
#                 else:
#                     condition += ' active = True'
#         for category in inactifs_category:
#             if category in input_label[0].split():
#                 if i >= 1:
#                     condition += ' and ' + ' active = False'
#                 else:
#                     condition += ' active = False'
    
#     return query + condition

# class InputLabel(BaseModel):
#     input_label: str

# @app.post("/generate_query/")
# async def create_query(input_data: InputLabel):
#     try:
#         query = generate_query(input_data.input_label)
#         return {"query": query}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from tqdm import tqdm
import os, re, csv, math, codecs
import string
import json

# For Training
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# For array, dataset, and visualizing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
app = FastAPI()
# NLTK and Spacy for text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load Spacy model once
nlp = spacy.load('fr_core_news_lg')

# Preprocessing functions
def remove_punctuation(text):
    translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translation_table)

def to_lower_case(input_string):
    return input_string.lower()

def remove_stop_words(vocab):
    stop_words = set(stopwords.words('french'))
    word_to_remove = ['ne', 'pas', 'ni', 'que', 'non']
    stop_words.difference_update(word_to_remove)
    return [word for word in vocab if word.lower() not in stop_words]

# Provided data
tier_categories = [
    'blacklisté', 'blacklistés', 'fournisseur', 'fournisseurs', 'avocats', 'avocat', 
    'bailleur', 'bailleurs', 'locataire', 'locataires'
]
personne_categories = [
    "agent de maintenance", "agents de maintenance", "responsable de suivi d'achat", 
    "responsables de suivi d'achats", "acheteur", "acheteurs", "encaisseurs", "encaisseur", 
    "responsables de maintenance", "responsable de maintenance"
]
actif_category = ['actifs', 'actif', 'active', 'actives']
inactifs_category = [
    'inactifs', 'inactif', 'inactive', 'inactives', 'non actifs', 'non actif', 
    'non active', 'non actives'
]
cols_dict = {
    "blacklisté": "blackliste", "blacklistés": "blackliste", "fournisseur": "supplier", 
    "fournisseurs": "supplier", "avocat": "lawer", "avocats": "lawer", "bailleur": "lessor", 
    "bailleurs": "lessor", "locataire": "tenant", "locataires": "tenant", 
    "agent de maintenance": "maintenance_agent", "agents de maintenance": "maintenance_agent", 
    "responsable de suivi d'achat": "purshase", "responsables de suivi d'achat": "purshase", 
    "acheteur": "buyer", "acheteurs": "buyer", "encaisseur": "collector", 
    "encaisseurs": "collector", "responsable de maintenance": "maintenance_responsable", 
    "responsables de maintenance": "maintenance_responsable"
}

# Load LSTM model 
path_model = 'C:/Users/aitma/Downloads/my_LSTM_model.h5'
model = tf.keras.models.load_model(path_model)
nlp_nouns = spacy.load("fr_dep_news_trf")


path1 = 'C:/Users/aitma/Downloads/word_index.json'
path2 = 'C:/Users/aitma/Downloads/tag_dict.json'
with open(path1, 'r') as json_file:
    word_index = json.load(json_file)
with open(path2, 'r') as json_file:
    tag_dict = json.load(json_file)

# Initialize Tokenizer once
tokenizer = Tokenizer(num_words=len(word_index), lower=True, char_level=False)
tokenizer.fit_on_texts(word_index.keys())
nlp_date = spacy.load("en_core_web_trf")




from datetime import datetime
def convert_date(date_str):
    # List of possible date formats to parse
    date_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y"
    ]
    
    for date_format in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, date_format)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    raise ValueError(f"Date format for {date_str} not recognized.")

def predict(embedded_text, tag_dict):
    confidence_threshold=0.7
    predictions = model.predict(embedded_text)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_label_index]
    
    # Check confidence
    if confidence < confidence_threshold:
        return  ["Désolé, je n'ai pas compris votre question. Pouvez-vous reformuler, s'il vous plaît ?"]
    else : 
        return  [key for val in predicted_classes for key, value in tag_dict.items() if value == val]
    

def generate_query(input_label: str) -> str:
    
    sent_user = input_label
    date_value = None
    country=None
    numbers=None
    nom=None
    # Extract detected dates
    input_label = remove_punctuation(input_label)
    input_label = to_lower_case(input_label)
    input_label = word_tokenize(input_label)
    input_label = remove_stop_words(input_label)

    doc = nlp(' '.join(input_label))
    liste = [token.lemma_ for token in doc]
  
    word_seq_train = tokenizer.texts_to_sequences([liste])
    word_seq_train = [item for sublist in word_seq_train for item in sublist]
    max_length = 40
    word_seq_train = pad_sequences([word_seq_train],maxlen=max_length, padding='pre')
    input_label = predict(word_seq_train, tag_dict)
    print(input_label[0])
    
    if 'date' in input_label[0]:
        doc = nlp_date(sent_user)
        date_value = [ent.text for ent in doc.ents if ent.label_ == "DATE"][0]
        date_value = convert_date(date_value)
        
    if 'pays' in input_label[0]:
        doc=nlp(sent_user)
        country = [ent.text for ent in doc.ents if ent.label_ == "LOC"][0]
        
    if 'capital' in input_label[0]:
        doc = nlp(sent_user)
        numbers = [token.text for token in doc if token.like_num][0]
        numbers=numbers.replace(',','')
        
    if 'nom' in input_label[0]:
        doc=nlp_nouns(sent_user)
        proper_nouns = [token.text for token in doc if token.pos_ == 'PROPN'][0]
        nom=proper_nouns
        
        
     

    query = ''
    condition = ''
    i = 0
    if input_label[0] in tag_dict:
   
#     input_label[0]=input_label[0].lower
    
        if 'tiers' in input_label[0].lower().split():
            if 'total' in input_label[0].lower():
                query = 'select COUNT(*) from tier '
            else:    
                query = 'select * from tier '
            matched_tiers = [category for category in tier_categories if category in input_label[0].split()]
            condition_category=[category for category in actif_category if category in input_label[0].split()]
            uncondition_category=[category for category in inactifs_category if category in input_label[0].split()]


            if matched_tiers or condition_category or uncondition_category or 'date' in input_label[0].split() or 'pays' in input_label[0].split() or 'capital' in input_label[0].split() or 'nom' in input_label[0].split():
                condition = 'where '
            if matched_tiers:
                condition += ' and '.join([cols_dict[category] + ' = True' for category in matched_tiers])
                i += len(matched_tiers)
            if any(category in input_label[0].split() for category in actif_category):
                condition += (' and ' if i else '') + 'actif = True '
                i += 1
            if any(category in input_label[0].split() for category in inactifs_category):
                condition += (' and ' if i else '') + 'actif = False '
                i += 1

            if 'date' in input_label[0]:
                if 'sup' in input_label[0] and date_value:
                    condition += (' and ' if i else '') + f"date > '{date_value}'"
                if 'inf' in input_label[0] and date_value:
                    condition += (' and ' if i else '') + f"date < '{date_value}'"
                    
            if 'pays' in input_label[0]:
                if 'pays' in country.split():
                    condition += (' and ' if i else '') + f"country = '{country.split()[1]}'"
                else:    
                    condition += (' and ' if i else '') + f"country = '{country}'"
            if 'capital' in input_label[0]:
                if 'sup' in input_label[0] and numbers:
                    condition += (' and ' if i else '') + f"capital > {numbers}"
                if 'inf' in input_label[0] and numbers:
                    condition += (' and ' if i else '') + f"capital < {numbers}"
        
            
                
                
        else:
            if 'total' in input_label[0].lower():
                query = 'select COUNT(*) from personne '   
            else:    
                query = 'select * from personne '
            if 'e-mail' in input_label[0].lower():
                query='select firstname , lastname , email from personne '    
            if 'mobile' in input_label[0].lower():
                query='select firstname , lastname , phone_number from personne '    
            condition_category=[category for category in actif_category if category in input_label[0].split()]
            uncondition_category=[category for category in inactifs_category if category in input_label[0].split()]
            matched_personne = [category for category in personne_categories if category in input_label[0].split()]
            if matched_personne or condition_category or uncondition_category or  'nom' in input_label[0].lower():
                condition = 'where '
            if matched_personne:
                condition += ' and '.join([cols_dict[category] + ' = True' for category in matched_personne])
                i += len(matched_personne)
            if any(category in input_label[0].split() for category in actif_category):
                condition += (' and ' if i else '') + 'active = True '
                i += 1
            if any(category in input_label[0].split() for category in inactifs_category):
                condition += (' and ' if i else '') + 'active = False '
                i += 1
            if 'nom' in input_label[0] and nom:
                condition += (' and ' if i else '') + f"lastname = '{nom}'"
        print(query + condition)    
    return input_label[0],query + condition
import mysql.connector

def connect_to_database(host, username, password, database):
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host=host,
            user=username,
            password=password,
            database=database
        )

        if connection.is_connected():
            print("Connected to MySQL database")
            return connection

    except mysql.connector.Error as error:
        print("Error:", error)
        return None

def execute_query(connection, query):
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        return result

    except mysql.connector.Error as error:
        print("Error executing query:", error)
        return None
import random
def close_connection(connection):
    # Close the connection
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")

def get_response(tag,data):
    choices=data[data['names']==tag]['responses']
    response=random.choice(list(choices))
    return response        
       
def convert_to_dataframe(query, query_result):
    column_names = None
    if 'tier' in query:
        column_names = [
            "id", "firstname", "lastname", "actif", "blackliste", "supplier", "lawer", 
            "lessor", "tenant", "date", "country", "capital"
        ]
    elif 'personne' in query:
        column_names = [
            "id", "firstname", "lastname", "matricule", "phone_number", 
            "active", "maintenance_agent", "purshase", "buyer", "collector", 
            "maintenance_responsable", "email", "date"
        ] 

    formatted_result = []
    for row in query_result:
        formatted_row = [int(value) if isinstance(value, np.int64) else value for value in row]
        formatted_result.append(formatted_row)

    df = pd.DataFrame(formatted_result, columns=column_names)
    return df
host = "localhost"
username = "root"
password = "Qwerty1234/"
database = "new_schema"



class InputLabel(BaseModel):
    input_label: str

@app.post("/generate_query/")
async def create_query(input_data: InputLabel):
    connection = None
    try:
        query_label, query = generate_query(input_data.input_label)
        connection = connect_to_database(host, username, password, database)
        result = execute_query(connection, query)
        data=pd.read_csv('C:/Users/aitma/Downloads/APIs_Versions/API fatstext/data-responses.csv')
        #  data['names'].unique()
        if query_label not in tag_dict:
            return {"data":query_label}
        if query_label in data['names'].unique():
            response=get_response(query_label,data)
            return {"query_label": query_label,"data": response}
        if 'COUNT(*)' in query:
            return {"query_label": query_label,"query":query, "data": result[0][0]}
        else:    
            return {"query_label": query_label,"query":query, "data": result}
        
          
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection:
            close_connection(connection)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
