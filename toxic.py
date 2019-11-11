import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import load_model  
import joblib
from collections import defaultdict

class toxicity_classifier():
  '''
  Toxicity classifier using a recurrent neural network that was previously trained on a Kaggle toxicity dataset
  '''
  def __init__(self):
    self.tokenizer = joblib.load('toxic_tokenizer.pkl')
    self.rnn_model = load_model('models/my_model.h5')
    self.df = None
    self.count = None

  def tokenize(self, comments, maxlen=100):
    '''
    Tokenize comments for pre-trained RNN predictions
    '''
    self.raw = comments
    self.comments = self.tokenizer.texts_to_sequences(comments)
    self.comments = sequence.pad_sequences(self.comments, maxlen=100)

  def classify(self):
    '''
    Classify comments and return a df of probabilities and original comment
    '''
    preds = self.rnn_model.predict(self.comments)
    self.df = pd.DataFrame(preds, columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    self.df.insert(0,'comment',self.raw)
    return self.df

  def count_toxic_comments(self):
    '''
    Sum and return total counts of toxic comments
    '''
    num_severe_toxic = len(self.df[self.df.severe_toxic >= .5])
    num_threat = len(self.df[self.df.threat >= .5])
    num_insult =  len(self.df[self.df.insult >= .5])
    num_identity_hate = len(self.df[self.df.identity_hate >= .5])
    self.count = defaultdict(int)
    self.count['severe_toxic'] = num_severe_toxic
    self.count['threat'] = num_threat
    self.count['insult'] = num_insult
    self.count['identity_hate'] = num_identity_hate
    return self.count

  def top_3_severe_toxic(self):
    '''
    Return top 3 comments for severe toxic category
    '''
    idxs = np.argsort(self.df.severe_toxic.values)[:-4:-1]
    return self.df.comment[idxs].values

  def top_3_threat(self):
    '''
    Return top 3 comments for threat category
    '''
    idxs = np.argsort(self.df.threat.values)[:-4:-1]
    return self.df.comment[idxs].values

  def top_3_insult(self):
    '''
    Return top 3 comments for insult category
    '''
    idxs = np.argsort(self.df.insult.values)[:-4:-1]
    return self.df.comment[idxs].values

  def top_3_identity_hate(self):
    '''
    Return top 3 comments for identity hate category
    '''
    idxs = np.argsort(self.df.identity_hate.values)[:-4:-1]
    return self.df.comment[idxs].values