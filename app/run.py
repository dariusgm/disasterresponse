import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine
import json
# this hack is required to run the webserver from within the app directory.
# I would have put the run.py in the project root 
import sys
sys.path.append("..")
sys.path.append("../models")

from constant import Constant
from train_classifier import tokenize

from pathlib import Path
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

'''
Return ready to use graph to generate them once ";"

:param df: (pd.DataFrame) Data used for the entire pipeline 
:param model_meta: (dict) Meta information with f1 score for each column
:returns: (dict) ready to use dict for plotly
'''
def build_graphs(df: pd.DataFrame, model_meta: dict):
    # prepare data for visualisations once
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    metric_values = list(model_meta.values())
    metric_name = list(model_meta.keys())

    positives = []
    negatives = []
    for column in model_meta.keys():
        if column in df.columns:
            positives.append(len(df[df[column] == 1]))
            negatives.append(len(df[df[column] == 0]))

    print(positives)
    print(negatives)
    # create visuals
    graphs = [
    {
            'data': [
                Bar(
                    x=metric_name,
                    y=metric_values
                )
            ],

            'layout': {
                'title': 'Model F1 Score by Column',
                'yaxis': {
                    'title': "Score"
                },
                'xaxis': {
                    'title': "Column"
                }
            }

        },    
        {
            'data': [
                Bar(x=metric_name,y=positives, name='positives'),
                Bar(x=metric_name,y=negatives, name='negatives')
            ],

            'layout': {
                'title': 'Examples by Column',
                'xaxis': {
                    'title': "Column"
                },
                'yaxis': {
                    'title': "Count"
                },
                'bargap': 0.3, 
                'bargroupgap': 0.1 
            }

        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
        
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    return (ids, graphJSON)


def setup():
    def extract_by_extention(extension):
        result = []
        for path in Path('..').rglob(extension):
            result.append(path)
        return result
    
    databases = extract_by_extention('*.db')
    if len(databases) != 1:
        raise RuntimeError("more than one database found. Delete unused datbase.")

    models = extract_by_extention('*.pkl')
    if len(models) != 1:
         raise RuntimeError("more than one model found. Delete unused models.")
    model = joblib.load(models[0])
    
    model_metas = extract_by_extention('*.json')
    if len(model_metas) != 1:
         raise RuntimeError("more than one model meta file found. Delete unused model meta files.")

    return databases[0], model, model_metas[0] 



# as we can not pass any names here (not like in etl and ml pipeline)
# we will search the db and model
database, model, model_meta = setup()
app = Flask(__name__)

# load data
engine = create_engine(f'sqlite:///{database}')
df = pd.read_sql_table(Constant.table_name(), engine)

# load model meta
with(open(model_meta, 'rt')) as f:
    model_meta = json.loads(f.read())

ids, graphJSON = build_graphs(df, model_meta)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()