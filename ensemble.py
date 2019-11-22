from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
#import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
#from rfpimp import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from joblib import load

class ensemble:
    def __init__(self, df):
        '''
        Initialize with dataframe to train models on

        Ex. e = ensemble(df)
        '''
        self.df = df
        self.X = df.drop(columns='is_scammer')
        self.y = df['is_scammer']
        #baseline vectorizer parameters
        tfidf = TfidfVectorizer(
        stop_words='english',
        min_df=3,  # min count for relevant vocabulary
        max_features=5000,  # maximum number of features
        strip_accents='unicode',  # replace all accented unicode char by their corresponding ASCII char
        analyzer='word',  # features made of words
        token_pattern=r'[a-zA-Z]{3,}',  # tokenize only words of 3+ chars
        ngram_range=(1, 1),  # features made of a single tokens
        use_idf=True,  # enable inverse-document-frequency reweighting
        smooth_idf=True,  # prevents zero division for unseen words
        sublinear_tf=False)
        
        #instantiate classifiers for ensemble
        self.rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, oob_score = True)
        self.xgb = XGBClassifier()
        self.lgb = LGBMClassifier()
        
        #creating ensemble classifier
        self.eclf = VotingClassifier(estimators=[('xgb', self.xgb), ('lgb', self.lgb), ('rf', self.rf)], voting='soft')

        #create pipeline for vectorizing user's comments for Naive Bayes
        self.model = make_pipeline(tfidf, MultinomialNB())

        #numerical columns to use for rf/gb models
        self.num_cols = ['link_karma', 'comment_karma', 'verified', 'mod', 'gold',
            'days_old', 'total_comments', 'positive', 'neutral',
            'negative', 'mean_comment_length', 'mode_comment_length',
            'median_comment_length', 'duplicate_comments',
            'avg_grammar', 'total_grammar',
            'cap_freq_mean']

    def split(self,random_state = None):
        '''
        Split imported dataframe into a train and test set. Use with train_fit and test_predict to tune parameters
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=None)
        self.X_train_MNB = self.X_train['comments_new']
        self.X_train = self.X_train[self.num_cols]
        self.X_test_MNB = self.X_test['comments_new']
        self.X_test = self.X_test[self.num_cols]

    def train_fit(self):
        '''
        Fits on training data
        '''
        self.eclf.fit(self.X_train, self.y_train)
        self.model.fit(self.X_train_MNB, self.y_train)

    def fit(self):
        '''
        Fits on full dataset for predicting unlabeled data
        '''
        self.eclf.fit(self.X[self.num_cols], self.y)
        self.model.fit(self.X['comments_new'], self.y)
    
    def test_predict(self):
        '''
        Returns test data prediction probability
        '''
        y_pred = self.eclf.predict_proba(self.X_test)[:,1]
        y_pred_MNB = self.model.predict_proba(self.X_test_MNB)[:,1]
        y_final_pred = (y_pred+y_pred_MNB/2)
        return y_final_pred

    def predict(self):
        '''
        Input Reddit username
        Returns prediction probability for new data
        '''
        # X = get_user_profile(str(username))
        X = None
        X_MNB = X['comments_new']
        X = X[self.num_cols]
        y_pred = self.eclf.predict_proba(X)[:,1]
        y_pred_MNB = self.model.predict_proba(X_MNB)[:,1]
        y_final_pred = (y_pred+y_pred_MNB/2)
        return y_final_pred
        
    def rf_predict(self, X):
        '''
        Fit and only return prediction probability for Random Forest Classifier
        '''
        self.rf.fit(self.X_train, self.y_train)
        # X = X[self.num_cols]
        return self.rf.predict_proba(X)[:,1]

    def xgb_predict(self, X):
        '''
        Fit and only return prediction probability for XGBoost classifier
        '''
        self.xgb.fit(self.X_train, self.y_train)
        # X = X[self.num_cols]
        return self.xgb.predict_proba(X)[:,1]

    def lgb_predict(self, X):
        '''
        Fit and only return prediction probability for LightGBM classifier
        '''
        self.lgb.fit(self.X_train, self.y_train)
        # X = X[self.num_cols]
        return self.lgb.predict_proba(X)[:,1]

    def MNB_predict(self, X):
        '''
        Fit and only return prediction probability for Multinomial Naive Bayes classifier
        '''
        self.model.fit(self.X_train_MNB, self.y_train)
        # X = X['comments_new']
        return self.model.predict_proba(X)[:,1]

    def score(self):
        '''
        Returns Area Under Receiver Operator Characteristic Curve for ensemble method
        '''
        print(f'''ROC AUC score: {roc_auc_score(self.y_test, self.test_predict())}''')
