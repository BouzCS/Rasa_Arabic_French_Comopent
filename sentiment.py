from typing import Dict, Text, Any, List
from rasa.engine.graph import GraphComponent, ExecutionContext

from rasa.engine.recipes.default_recipe import DefaultV1Recipe

from rasa.engine.storage.resource import Resource

from rasa.engine.storage.storage import ModelStorage

from rasa.shared.nlu.training_data.message import Message

from rasa.shared.nlu.training_data.training_data import TrainingData
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

PICKLE_PATH = "dataset/allocine_dataset.pickle"
MAX_NB_WORDS = 20000
MAX_LENGTH_SEQ=350

with open(PICKLE_PATH, 'rb') as reader:
    data = pickle.load(reader)

# Reviews need to be tokenized
train_reviews = np.array(data["train_set"]['review'])
tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=MAX_NB_WORDS,
        oov_token="<unk>",
        )
tokenizer.fit_on_texts(train_reviews)
        
        

SENTIMENT_MODEL_FILE_NAME = "Model/Model_GRU.h5"
model = tf.keras.models.load_model(SENTIMENT_MODEL_FILE_NAME)



from rasa.shared.nlu.constants import (
    TEXT
)

@DefaultV1Recipe.register(

   [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=False

)


class SentimentAnalyzer(GraphComponent):

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
                "entity": "sentiment",
                "extractor": "sentiment_extractor"}


      return entity



   
   
   def preprocessing(self, txt,tokenizer) -> None:
        """Create bag-of-words representation of the training examples."""
        
        
        txt_tokenized = tokenizer.texts_to_sequences([txt])

        txt_padded = pad_sequences(
            txt_tokenized, 
            maxlen=MAX_LENGTH_SEQ,
            padding='post'
        )
        return txt_padded

   def process(self, messages: List[Message]) -> List[Message]:

       # TODO: This is the method which Rasa Open Source will call during inference.
    """Retrieve the tokens of the new message, pass it to the classifier
         and append prediction results to the message class."""


    for message in messages:
            
        tokens = message.get(TEXT)
        processed_tokens = self.preprocessing(tokens,tokenizer)
        prediction = model.predict(processed_tokens)
        confidence = prediction[0][0]
        if confidence>0.5:
            sentiment="Positive"
        else:
            sentiment="Negative"
        entity = self.convert_to_rasa(sentiment, str(confidence))
        
        message.set("entities", [entity], add_to_output=True)

        

    return messages   
       


   
