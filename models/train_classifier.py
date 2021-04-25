# import libraries
from nltk.tokenize import TweetTokenizer
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin
import math
import pickle
import os
import json
from sklearn.exceptions import ConvergenceWarning
import time
import warnings
import sys

# ignore warnings from grid search
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# tokenize function, we share it here with run.py
def tokenize(text):
    return TweetTokenizer().tokenize(text)

# Data Container for metric calculation
class Metric():
    def __init__(self, model_name):
        self.avg_f1 = []
        # using macro avg here, source: https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
        self.metric_key = 'macro avg'
        self.metric_dict = {}
        self.model_name = model_name        

    def push(self, metrics: dict, column: str) -> None:
        '''
        push a new entry of metric
        :param metrics: (dict) new metric to add
        :param column: (dict) name of the column that this metric was calculated for.
        :returns: None
        ''' 
        nested = metrics[self.metric_key]
        f1 = nested['f1-score']
        precision = nested['precision']
        recall = nested['recall']
        self.avg_f1.append(f1)
        self.metric_dict[column] = f1
        print(f"column: {column}, f1 (macro): {round(f1, 2)}, \
            precision: {round(precision, 2)}, \
            recall: {round(recall, 2)}")

    def __average_f1(self) -> float:
        '''
        calculate f1  avg on all columns to better compare different pipelines
        :returns: float
        '''         
        return np.average(self.avg_f1)

    def dump(self) -> dict:
        '''
        return al collected metric data as dict
        :returns: dict
        ''' 
        all_metrics = self.metric_dict.copy()
        all_metrics['f1_average'] = self.__average_f1()

        return all_metrics

class MLPipeline():
    def __init__(self, database_filepath: str, model_filepath: str):
        self.table = 'etl'
        self.database_filepath = database_filepath
        self.model_filepath = model_filepath
        self.labels = [
            'related',
            'request',
            'offer',
            'aid_related',
            'medical_help',
            'medical_products',
            'search_and_rescue',
            'security',
            'military',
            'child_alone',
            'water',
            'food',
            'shelter',
            'clothing',
            'money',
            'missing_people',
            'refugees',
            'death',
            'other_aid',
            'infrastructure_related',
            'transport',
            'buildings',
            'electricity',
            'tools',
            'hospitals',
            'shops',
            'aid_centers',
            'other_infrastructure',
            'weather_related',
            'floods',
            'storm',
            'fire',
            'earthquake',
            'cold',
            'other_weather',
            'direct_report'
        ]

    def __pipeline_fit(self, classifier: ClassifierMixin, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        '''
        fit a pipeline
        :param classifier: (ClassifierMixin) classifier to use for pipeline.
        :param X_train: (pd.DataFrame) matrix for train data.
        :param Y_train: (pd.DataFrame) matrix for label data.
        :returns: (ClassifierMixin) fitted pipeline.
        ''' 
        model = self.__build_pipeline(classifier)
        model.fit(X_train, Y_train)
        return model

    def __pipeline_metric(self, model: ClassifierMixin, X_test: pd.DataFrame, Y_test: pd.DataFrame, cleaned_labels: list, model_name: str) -> dict:
        '''
        calculate quality of a trained pipeline.
        :param model: (ClassifierMixin) trained pipeine to calculate quality for.
        :param X_test: (pd.DataFrame) matrix for test data.
        :param Y_test: (pd.DataFrame) matrix for test label data.
        :param cleaned_labels: (list(str)) list of labels that should be take into acount for quality check.
        :param model_name: (str) name of the model that should be checked for quality.
        :returns: (dict) quality of model
        '''        
        Y_pred = model.predict(X_test)
        Y_test_numpy = Y_test.to_numpy()

        metric = Metric(model_name)
        for i, column in enumerate(cleaned_labels):
            feature_predictions = Y_pred[:, i]
            feature_truth = Y_test_numpy[:, i]
            current_metric = classification_report(
                            y_pred=feature_predictions,
                            y_true=feature_truth,
                            output_dict=True,
                            zero_division=0)
            
            metric.push(current_metric, column) 
            
        all_metrics = metric.dump()
        all_metrics['model'] = model
        return all_metrics

    def __clean_labels(self, Y: pd.DataFrame) -> list:
        '''
        list of labels that can be used for training and quality check
        :param Y: (pd.DataFrame) trained pipeine to calculate quality for.
        :returns: (list) list of valid labels
        ''' 
        mins = Y.min()
        maxes = Y.max()
        cleaned_labels = []
        for i, column in enumerate(self.labels):
          min_value = mins[i]
          max_value = maxes[i]
          # invalid lables should have positive (1) and negative examples (0)
          # also the min should be 0 for this case and max 1, otherwise its not a binary classification problem
          if min_value != 0 or max_value != 1:
            print(f"column: {column} min: {min_value} max: {max_value} invalid, removing label")
          else:
            cleaned_labels.append(column)

        return cleaned_labels

    def __build_pipeline(self, classifier: ClassifierMixin) -> Pipeline:

        return Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(classifier, n_jobs=1))
        ])

    def run(self) -> None:
        '''
        main entry point for MLPipeline
        :returns: (None)
        '''         
        result = {}
        print("Reading data")
        engine = create_engine(f"sqlite:///{self.database_filepath}")
        df = pd.read_sql_table(self.table, engine)

        X = df['message']
        Y = df[self.labels]
        print("cleaning labels")
        cleaned_labels = self.__clean_labels(Y)
        Y = Y[cleaned_labels]

        X_test, X_train, Y_test, Y_train = train_test_split(X, Y, random_state=42)
        
        print("Run Decision Tree Pipeline")
        start = time.time()
        decision_tree_classifier = DecisionTreeClassifier()
        decision_tree_pipeline = self.__pipeline_fit(decision_tree_classifier, X_train, Y_train)
        result['DecisionTree'] = self.__pipeline_metric(decision_tree_pipeline, X_test, Y_test, cleaned_labels, 'DecisionTree')
        print(f"Decision Tree finished in {time.time() - start} sec")

        print("Run SGD Pipeline")
        start = time.time()
        sgd_classifier = SGDClassifier(max_iter=5000, tol=1e-3)
        sgd_pipeline = self.__pipeline_fit(sgd_classifier, X_train, Y_train)
        result['SGD'] = self.__pipeline_metric(sgd_pipeline, X_test, Y_test, cleaned_labels, 'SGD')
        print(f"SGD finished in {time.time() - start} sec")

        parameters = {
            'clf__estimator__loss': [
                'hinge', 
                'log', 
                'modified_huber', 
                'squared_hinge'
            ],
            'clf__estimator__max_iter': [50000],
            'clf__estimator__tol': [1e-3]
        }

        # Note: here I would prefer Hyperopt or Optuna
        # But project requires GridSearch
        print("Run SGD Grid Pipeline (takes ~45 minutes)")
        start = time.time()
        sgd_grid = GridSearchCV(sgd_pipeline, parameters, n_jobs=1).fit(X_train, Y_train)
        result['SGDGrid'] = self.__pipeline_metric(sgd_grid, X_test, Y_test, cleaned_labels, 'SGDGrid')
        print(f"SGDGrid finished in {time.time() - start} sec")
        
        # Calculate winning algorithm
        winner_data = None
        winner_f1 = None
        winner_name = None

        print("Summary (f1 avg): ")
        for model_name, v in result.items():
            f1 = v['f1_average']
            model = v['model']
            score = abs(1 - f1) # as close to 1, as better

            print(f"{model_name}: F1 Avg: {f1}")

            if winner_f1 == None:
                winner_data = v
                winner_f1 = score
                winner_name = model_name
            elif score < winner_f1:
                winner_data = model
                winner_f1 = score
                winner_name = model_name           

        print(f"Saving winning model: {winner_name}")
        filename = f"{self.model_filepath}"
        with open(filename, 'wb') as f:
            pickle.dump(obj=winner_data['model'], file=f)

        print("Saving winning model meta")
        prefix = self.model_filepath.split(".")[0]
        with(open(f"{prefix}.json", 'wt')) as f:
            del winner_data['model']
            f.write(json.dumps(winner_data))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        pipeline = MLPipeline(database_filepath, model_filepath)
        pipeline.run()
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()