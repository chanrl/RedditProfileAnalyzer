import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim import corpora, models
from nltk.tokenize import word_tokenize
import joblib
from operator import itemgetter 
from collections import defaultdict

class LDA_predict():
  def __init__(self):
    self.stemmer = SnowballStemmer("english")
    self.model = joblib.load('models/lda_model.pkl')
    self.topic_model = joblib.load('models/topic_model.pkl')
    self.dictionary = self.model.id2word
    self.dictionary_2 = self.topic_model.id2word
    self.preprocessed_docs = None
    # Current latent topics in LDA model
    self.topic_dict = {
                      1: 'sports',
                      4: 'science/technology',
                      6: 'food',
                      7: 'conservative',
                      8: 'liberal',
                      9: 'entertainment'
                      }

  def lemmatize_stemming(self, text):
    return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

  def preprocess(self, text):
    '''
    Tokenize and lemmatize text for LDA model predictions
    '''
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(self.lemmatize_stemming(token))
    return result

  def preprocess_docs(self, documents):
    self.preprocessed_docs = []
    for doc in documents:
      self.preprocessed_docs.append(self.preprocess(doc))

  def count_trader(self):
    '''
    Predict if comment has to do with buy/sell/trading using pre-trained LDA model and sum total buy/sell/trade comments
    '''
    trader_count = 0
    for token in self.preprocessed_docs:
      predictions = self.model[self.dictionary.doc2bow(token)]
      if max(predictions, key = itemgetter(1))[0] == 0:
        trader_count += 1
    return trader_count

  def count_topics(self):
    '''
    Predict comment topics and count the total from a user's history
    '''
    count = defaultdict(int)
    for token in self.preprocessed_docs:
      predictions = self.topic_model[self.dictionary_2.doc2bow(token)]
      topic = max(predictions, key = itemgetter(1))[0]
      if topic in self.topic_dict.keys():
        count[self.topic_dict[topic]] += 1
    return count