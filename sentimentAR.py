from typing import Dict, Text, Any, List
from rasa.engine.graph import GraphComponent, ExecutionContext

from rasa.engine.recipes.default_recipe import DefaultV1Recipe

from rasa.engine.storage.resource import Resource

from rasa.engine.storage.storage import ModelStorage

from rasa.shared.nlu.training_data.message import Message

from rasa.shared.nlu.training_data.training_data import TrainingData
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
from sklearn.model_selection import train_test_split

import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyarabic.araby import strip_tatweel,strip_tashkeel


DATA_PATH = "dataset/data_ar.csv"
df = pd.read_csv(DATA_PATH)

with open('Model/model_ar.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data and vectorizer

data,sentiment = df['Twits'].values,df['label']
X_train, X_test, y_train, y_test = train_test_split(data, sentiment, test_size=0.2, random_state=1)
vectorizer = TfidfVectorizer(ngram_range = (1,1))
vectorizer.fit_transform(X_train)




from rasa.shared.nlu.constants import (
    TEXT
)

@DefaultV1Recipe.register(

   [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=False

)


class SentimentAnalyzer_Arabic(GraphComponent):

   name = "sentiment"
   provides = ["entities"]
   requires = []
   defaults = {}
   language_list = ["en"]

   def __init__(self, component_config: Dict[Text, Any]) -> None:

       self.component_config = component_config

       
   @classmethod
   def create(
           cls,
           config: Dict[Text, Any],
           model_storage: ModelStorage,
           resource: Resource,
           execution_context: ExecutionContext,
   ) -> GraphComponent:
       return cls(config)
       

   def train(self, training_data: TrainingData) -> Resource:

       pass

   def convert_to_rasa(self, value, confidence):
      """Convert model output into the Rasa NLU compatible output format."""


      entity = {"value": value,
                "confidence": confidence,
                "entity": "sentiment_arabic",
                "extractor": "sentiment_extractor"}


      return entity



   
   
   def preprocessing(self, txt,vectorizer) -> None:
        """Create bag-of-words representation of the training examples."""
        
        def repted(text):
        
            text=re.sub(r'(.)\1+', r'\1', text) # Replace with only one (remove repetitions)  
            return text
            
        txt = re.sub(r'http\S+', 'URL', txt)
        txt = re.sub(r'www\S+', 'URL', txt)  # Replace URLs with URL string
        txt= re.sub(r'@[^\s]+', 'USER',txt) # Replace user mentions with USER string
        txt= re.sub(r'#[^\s]+', 'HASHTAG', txt) # Replace Hashtags with HASHTAG string
        
        txt= strip_tatweel(txt) #Remove Tatweel string 
        txt= strip_tashkeel(txt) # Remove Diacritics
        txt= repted(txt)

        txt = vectorizer.transform([txt])
        return txt
        

   def process(self, messages: List[Message]) -> List[Message]:

       # TODO: This is the method which Rasa Open Source will call during inference.
    """Retrieve the tokens of the new message, pass it to the classifier
         and append prediction results to the message class."""


    for message in messages:
            
        tokens = message.get(TEXT)
        processed_tokens = self.preprocessing(tokens,vectorizer)
        prediction = model.predict(processed_tokens)
        confidence = model._predict_proba_lr(processed_tokens)[0][prediction]
        if prediction[0]==0:
            sentiment="Negative"
        
        if prediction[0]==1:
            sentiment="Neutral"
            
        if prediction[0]==2:
            sentiment="Positive"
        entity = self.convert_to_rasa(sentiment, str(confidence))
        
        message.set("entities", [entity], add_to_output=True)

        

    return messages   
       


   
