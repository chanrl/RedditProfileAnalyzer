from flask import Flask, render_template, jsonify, request
import requests

import pandas as pd 
import numpy as np
from joblib import load

from RedditScraper import RedditorScraper
from ensemble import ensemble

import plotly
import plotly.graph_objs as go
import json

app = Flask(__name__)

test_df = pd.read_pickle('sample.pkl')
scam_df = pd.read_pickle('scammers_mean.pkl')
pd.set_option('display.max_colwidth', -1)
user = ['sample']
# rs = RedditorScraper()
# e = load('eclf.pkl')
def scrape_user(text1):
  user.append(str(text1))
  rs.scrape_user(str(text1))
  rs.full_analyze()

scammer_probability = [1]

def predict_scammer(df):
   '''
   Takes input data from the web form and passes it into the trained ensemble classifier
   '''

   X_MNB = df['comments_new']
   X = df[e.num_cols]
   y_pred = e.eclf.predict_proba(X)[:,1]
   y_pred_MNB = e.model.predict_proba(X_MNB)[:,1]
   proba = (y_pred+y_pred_MNB/2)
   
   return str(round(proba[0],3))

def create_bar_plot(d):
    '''
    Create bar plot from dictionary
    '''
    data = [
        go.Bar(
            x=list(d.keys()),
            y=list(d.values())
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
    
def create_grouped_bar_plot(d,d1):
    '''
    Create grouped bar plot from dictionaries
    '''
    data = [
        go.Bar(name = user[0],
            x=list(d.keys()),
            y=list(d.values())
        ),
        go.Bar(name = 'average scammer',
            x=list(d1.keys()),
            y=list(d1.values())
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_pie_chart(d):
    '''
    Create pie chart from dictionary
    '''
    data = [
        go.Pie(
            labels=list(d.keys()), # assign x as the dataframe column 'x'
            values=list(d.values())
        )
    ]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def preprocess_df(df):
  '''
  Setup DataFrame as in a dictionary format for plotly
  '''
  try: 
    d = {key:value for key, value in zip(df.columns, df.values[0])}
  except:
    d = {key:value for key, value in zip(df.index, df.values)}
  return d

@app.route('/about')
def welcome():
  return render_template('about.html')

@app.route('/', methods=['GET', 'POST'])
def handle_form():
    if request.method == 'POST':
        form_input = dict(request.form)
        user_name = form_input['reddit_username']
        # scrape_user(user_name)
        # scammer_probability = predict_scammer(rs.df)
        scammer_probability.append(.76)
        user.append('test')
        return render_template('test_result.html', 
                 scammer_probability=scammer_probability[0], user=user[0])

    else:
        return render_template('input_form.html')

@app.route('/basic')
def basic():
  basic_cols = ['link_karma', 'comment_karma', 'verified', 'mod', 'gold', 'days_old', 'total_comments']
  karma_cols = ['link_karma', 'comment_karma'] 
  other_cols = ['days_old', 'total_comments']
  # plot1 = karma, plot2 = days old, total comments
  # df = rs.df[basic_cols]
  df = test_df[basic_cols]
  scam = scam_df.T
  bar = create_grouped_bar_plot(preprocess_df(df[karma_cols]), preprocess_df(scam[karma_cols]))
  bar2 = create_grouped_bar_plot(preprocess_df(df[other_cols]), preprocess_df(scam[other_cols]))
  return render_template('dataframe.html', tables=[df.to_html(index=False, classes='data')], titles=['basic'],
   scammer_probability=scammer_probability[0], user=user[0], plot = bar, plot2 = bar2)

@app.route('/advanced')
def advanced():
  advanced_cols = ['positive', 'neutral', 'negative', 'mean_comment_length', 'mode_comment_length', 'median_comment_length',
                'duplicate_comments', 'avg_grammar', 'total_grammar', 'cap_freq_mean']
                # 'severe_toxic_freq', 'threat_freq', 'insult_freq', 'identity_hate_freq']
  num_cols = ['mean_comment_length', 'mode_comment_length', 'median_comment_length', 'duplicate_comments']
  grammar_errors = ['avg_grammar']
  percent_cols = ['positive','neutral','negative','cap_freq_mean']
  # df = rs.df[basic_cols]
  df = test_df
  df = df[advanced_cols]
  scam = scam_df.T
  bar = create_grouped_bar_plot(preprocess_df(df[num_cols]), preprocess_df(scam[num_cols]))
  grammar_bar = create_grouped_bar_plot(preprocess_df(df[grammar_errors]), preprocess_df(scam[grammar_errors]))
  percent_bar = create_grouped_bar_plot(preprocess_df(df[percent_cols]), preprocess_df(scam[percent_cols]))
  return render_template('dataframe.html', tables=[df.to_html(index=False, classes='data')], titles=['advanced'],
   scammer_probability=scammer_probability[0], user=user[0], plot = bar, plot2 = grammar_bar, plot3 = percent_bar)

@app.route('/toxic-count')
def toxic_count():
  # df = pd.DataFrame(rs.toxic_count, index=[0])
  test_d = {'severe_toxic': 4, 'threat': 2, 'insult': 11, 'identity_hate': 1}
  toxic_cols = ['severe_toxic', 'threat', 'insult', 'identity_hate']
  df = pd.DataFrame(test_d, index=[0])
  scam = scam_df.T
  # bar = create_bar_plot(test_d)
  bar = create_grouped_bar_plot(test_d, preprocess_df(scam[toxic_cols]))
  return render_template('dataframe.html', tables=[df.to_html(index=False, classes='data')], titles=['Toxic Count'],
   plot = bar, scammer_probability=scammer_probability[0], user=user[0])

@app.route('/top-3')
def top_3():
  # df = pd.DataFrame(rs.top_3)
  test_top3 = {'severe_toxic': ['Nigga u are 16 cut this shit and her out of her life,u will be happier',
  'fucking faggot,OP please delete this after he gets banned dont give attention to magon',
  'Fuck u'],
 'threat': ['Kill urself',
  'Nigga u are 16 cut this shit and her out of her life,u will be happier',
  'Go back to aqn you cancer'],
 'insult': ['Nigga u are 16 cut this shit and her out of her life,u will be happier',
  'vivian leave you fucking cunt and suicide for real',
  'fucking faggot,OP please delete this after he gets banned dont give attention to magon'],
 'identity_hate': ['Nigga u are 16 cut this shit and her out of her life,u will be happier',
  'The jew keep getting gold,shut it down',
  'Id jerk off but im sleeping next to 3 other guys and that would be gay...thanks anyway ']}
  df = pd.DataFrame(test_top3)

  return render_template('dataframe.html', tables=[df.to_html(index=False, classes='data')], titles=['Top 3 Toxic Comments'],
   scammer_probability=scammer_probability[0], user=user[0])

@app.route('/topic')
def topic():
  # df = pd.DataFrame(rs.topic_count, index=[0])
  # df.insert(0, 'buy/trade/sell', rs.trade_count) 
  test_d = {'conservative': 31,
             'liberal': 18,
             'science/technology': 69,
             'sports': 42,
             'food': 70,
             'entertainment': 35}
  df = pd.DataFrame(test_d, index=[0])
  pie = create_pie_chart(test_d)
  return render_template('dataframe.html', tables=[df.to_html(index=False, classes='data')], titles=['Topic Counts'],
   plot = pie, scammer_probability=scammer_probability[0], user=user[0])

if __name__ == '__main__':
   app.run(debug=True)