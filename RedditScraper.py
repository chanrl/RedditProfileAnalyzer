import praw
import time
from datetime import datetime
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict, Counter
import statistics as stats
import language_check
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from LDA import LDA_predict
from toxic import toxicity_classifier

class RedditSubUsersScraper():
  def __init__(self):
    self.reddit = praw.Reddit(user_agent='Comment History Parser',
                  client_id='nkVxbwp1RsHHCA',
                  client_secret='SlzWUhAhV5nIXPy4_1PTJSOaLrA')
    self.sid = SentimentIntensityAnalyzer()
    self.tool = language_check.LanguageTool('en-US')
    self.sub = None
    # self.tokenizer = joblib.load('toxic_tokenizer.pkl')
    # self.rnn_model = load_model('models/my_model.h5')
    self.df = None

  def create_sub(self, sub_name):
    '''
    Create PRAW subreddit object
    '''
    sub_name = str(sub_name)
    self.sub = self.reddit.subreddit(sub_name)

  def sub_user_list(self, limit=1000):
    '''
    Scrape unique users from a given subreddit
    Limited to 1000 submissions id max per subreddit per PRAW

    OUTPUT
    ------
    DataFrame of unique users for a given subreddit
    '''
    #extract as much users as possible in subreddit
    #start by collecting all submission ids in subreddit
    #hot, new, controversial, rising
    submissions_id = []
    for submission in self.sub.hot(limit=limit):
      if submission not in submissions_id:
        submissions_id.append(submission)
    for submission in self.sub.rising(limit=limit):
      if submission not in submissions_id:
        submissions_id.append(submission)
    for submission in self.sub.controversial(limit=limit):
      if submission not in submissions_id:
        submissions_id.append(submission)
    for submission in self.sub.new(limit=limit):
      if submission not in submissions_id:
        submissions_id.append(submission)

    #initial scrape, authors of submissions
    users = []
    for submission in submissions_id:
      if submission.author not in users:
        users.append(submission.author)
  
    #secondary scrape, iterate through submission comment forest and extract all users inside

    for submission in submissions_id:
      submission.comments.replace_more(limit=None)
      for comment in submission.comments.list():
          if comment.author not in users:
              users.append(comment.author)

    if 'AutoModerator' in users:
      users.remove('AutoModerator') #remove reddit bot
  
    self.df = pd.DataFrame(users, columns=['users'])
  
  def fetch_comments_id(self, user, limit=1000):
      '''
      Returns user's comment history, potential max of 3000, as a list of comment objects
      '''
      all_comments = list(user.comments.controversial(limit=limit))
      for comment in user.comments.hot(limit=limit):
          if comment not in all_comments:
              all_comments.append(comment)
      for comment in user.comments.new(limit=limit):
          if comment not in all_comments:
              all_comments.append(comment)
      return all_comments

  def retrieve_text(self, user, limit=1000):
    '''
    Converts all comment objects into string text, as a list of documents
    '''
    all_text = [comment.body for comment in self.fetch_comments_id(user, limit=limit)]
    return all_text

  def get_user_details(self, user):
    '''
    Fetches all other information available on a user's profile with PRAW
    Link karma, comment karma, verified, mod status, gold status, account age
    '''
    #self-explanatory
    try:
      link_karma = user.link_karma #if user is banned and profile is unaccessible, it will raise an error when fetching link_karma.
    except:
      return -1 #set as -1 to filter out later
    comment_karma = user.comment_karma
    verified = user.has_verified_email
    mod = user.is_mod
    gold = user.is_gold
    days_old =(datetime.fromtimestamp(1571093268) - datetime.fromtimestamp(user.created_utc)).days
    return link_karma, comment_karma, verified, mod, gold, days_old

  def create_basic_df(self):
    '''
    Collect basic Reddit profile information for users
    '''
    df = self.df
    df['details'] = df['users'].map(self.get_user_details)
    df = df[df['details'] != -1] #removes banned users with unaccessible profiles
    df['link_karma'] = df['details'].map(lambda x : x[0])
    df['comment_karma'] = df['details'].map(lambda x : x[1])
    df['verified'] = df['details'].map(lambda x : x[2])
    df['mod'] = df['details'].map(lambda x : x[3])
    df['gold'] = df['details'].map(lambda x : x[4])
    df['days_old'] = df['details'].map(lambda x : x[5])
    df = df.drop(columns='details')
    self.df = df

  def create_raw_df(self, limit=1000):
    self.df['comments'] = self.df['users'].map(self.retrieve_text)

  def quick_analyze(self):
    '''
    Quick analysis of user comments
    '''
    df = self.df
    df['total_comments'] = df.comments.map(lambda x: len(x))
    df = df[df.total_comments != 0] #remove users with no comments. Why are you interacting with someone with no comments anyways?
    df['len_cs'] = df['comments'].map(lambda x: [len(comment) for comment in x])
    df['mean_comment_length'] = df['len_cs'].map(lambda x: stats.mean(x))
    df['mode_comment_length'] = df['len_cs'].map(lambda x: Counter(x).most_common()[0][0])
    df['median_comment_length'] = df['len_cs'].map(lambda x: stats.median(x))
    df['duplicate_comments'] = df['comments'].map(lambda x: len(x) - len(set(x)))
    self.df = df

  def apply_vader(self, comments):
    '''
    Adding feature for comment sentiment
    '''
    scores = defaultdict(int)
    for comment in comments:
        if self.sid.polarity_scores(comment)['compound'] >= 0.05:
            scores['positive'] += 1
        elif self.sid.polarity_scores(comment)['compound'] > -0.05 and self.sid.polarity_scores(comment)['compound'] < 0.05:
            scores['neutral'] += 1
        elif self.sid.polarity_scores(comment)['compound'] <= -0.05:
            scores['negative'] += 1
        else:
            scores['somethingwrong'] += 1
    return scores

  def vader_analyze(self):
    '''
    Organizing more information into dataframe
    '''
    df = self.df
    df['polarity'] = df['comments'].map(self.apply_vader)
    df['positive'] = df['polarity'].map(lambda x: x['positive'])/df['total_comments']
    df['neutral'] = df['polarity'].map(lambda x: x['neutral'])/df['total_comments']
    df['negative'] = df['polarity'].map(lambda x: x['negative'])/df['total_comments']
    df = df.drop(columns = 'polarity')
    self.df = df

  def cap_check(self, row):
    caps = []
    for comment in row:
      if len(comment) == 0:
        pass
      else:
        c = Counter("upper" if x.isupper() else "rest" for x in comment)
        caps.append(c['upper']/(c['rest']+c['upper']))
    return caps

  def grammar_check(self, row):
    errors = []
    for comment in row:
        errors.append(len(self.tool.check(comment.replace('\n', ' '))))
    return (np.average(errors), np.sum(errors))

  def grammar_analyze(self):
    df = self.df
    df['grammar'] = df['comments'].map(self.grammar_check)
    df['cap_freq'] = df['comments'].map(self.cap_check)
    df['avg_grammar'] = df['grammar'].map(lambda x: x[0])
    df['total_grammar'] = df['grammar'].map(lambda x: x[1])
    df['cap_freq_mean'] = df['cap_freq'].map(lambda x: np.mean(x))
    self.df = df

  def indepth_analyze(self):
    self.vader_analyze()
    self.grammar_analyze()

  def preprocess_df(self):
    '''
    Create a document of comments from list of comments to be vectorized for TFIDF - Naive Bayes 
    '''
    self.df['comments_new'] = self.df['comments'].map(lambda x: " ".join(x))

  def full_scrape(self, sub_name, limit=None, save=False):
    '''
    '''
    self.create_sub(sub_name)
    self.sub_user_list(limit=limit)
    #test line
    self.df = self.df[:1]

    self.create_basic_df()
    self.create_raw_df()
    self.quick_analyze()
    self.indepth_analyze()
    self.preprocess_df

    if save == True:
      self.df.to_csv(f'data/df_{sub_name}.csv')

class RedditSubScraper():
  '''
  Subreddit comment scrapper
  '''
  def __init__(self):
    self.reddit = praw.Reddit(user_agent='Comment History Parser',
                  client_id='nkVxbwp1RsHHCA',
                  client_secret='SlzWUhAhV5nIXPy4_1PTJSOaLrA')
  
  def create_sub(self, sub_name):
    '''
    Create PRAW subreddit object
    '''
    sub_name = str(sub_name)
    self.sub = self.reddit.subreddit(sub_name)
    self.df = None
  
  def scrape_sub(self, limit=1000):
    '''
    Scrape subreddit for comments
    '''

    comments_list = []
    for comment in self.sub.comments(limit=limit):
      comments_list.append(comment.body)
    if len(self.df) > 0:
      new_df = pd.DataFrame({'comments': comments_list, 'subreddit': f'{self.sub}'})
      self.df = pd.concat((self.df, new_df))  
    else:
      self.df = pd.DataFrame({'comments': comments_list, 'subreddit': f'{self.sub}'})

class RedditorScraper(RedditSubUsersScraper):
  def __init__(self):
    self.reddit = praw.Reddit(user_agent='Comment History Parser',
                  client_id='nkVxbwp1RsHHCA',
                  client_secret='SlzWUhAhV5nIXPy4_1PTJSOaLrA')
    self.sid = SentimentIntensityAnalyzer()
    self.tool = language_check.LanguageTool('en-US')
    self.LDA = LDA_predict()
    self.txc = toxicity_classifier()

  def scrape_user(self, username):
    username = str(username)
    user = self.reddit.redditor(username)
    self.df = pd.DataFrame([user], columns=['users'])
    self.create_basic_df()
    self.create_raw_df()
  
  def LDA_analyze(self):
    self.LDA.preprocess_docs(self.df.comments[0])
    self.trade_count = self.LDA.count_trader()
    self.topic_count = self.LDA.count_topics()
    return self.trade_count, self.topic_count
  
  def toxicity_analyze(self):
    self.txc.tokenize(self.df.comments[0])
    self.txc_df = self.txc.classify()
    self.toxic_count = self.txc.count_toxic_comments()
    self.top_3 = {'severe_toxic': None,
                  'threat': None,
                  'insult': None,
                  'identity_hate': None
    }
    self.top_3['severe_toxic'] = list(self.txc.top_3_severe_toxic())
    self.top_3['threat'] = list(self.txc.top_3_threat())
    self.top_3['insult'] = list(self.txc.top_3_insult())
    self.top_3['identity_hate'] = list(self.txc.top_3_identity_hate())

  def full_analyze(self):
    self.quick_analyze()
    self.indepth_analyze()
    self.preprocess_df()
    if self.df.verified[0] != bool:
      self.df.verified = self.df.verified.fillna(True)
    self.LDA_analyze()
    self.toxicity_analyze()

if __name__ == "__main__":
  rs = RedditorScraper()
  rs.scrape_user('tractorfactor')
  rs.toxicity_analyze()