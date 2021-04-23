# import libraries
from nltk.tokenize import TweetTokenizer
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from constant import Constant
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin
import math

def tokenize(text):
    return TweetTokenizer().tokenize(text)

# Data Container for metric calculation
class Metric():
    def __init__(self):
        self.avg_f1 = []
        # using macro avg here, source: https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
        self.metric_key = 'macro avg'

    def push(self, metrics: dict, column: str):
        nested = metrics[self.metric_key]
        f1 = nested['f1-score']
        precision = nested['precision']
        recall = nested['recall']
        self.avg_f1.append(f1)
        print(f"column: {column}, f1 (macro): {round(f1, 2)}, \
            precision: {round(precision, 2)}, \
            recall: {round(recall, 2)}")

    def average_f1(self):
        return np.average(self.avg_f1)



class MLPipeline():
    def __init__(self):
        self.table = Constant.table_name()
        self.tokenizer = TweetTokenizer()

    def __pipeline_fit(self, classifier, X_train, Y_train):
        model = self.__build_pipeline(classifier)
        model.fit(X_train, Y_train)
        return model

    def __pipeline_metric(self, model, X_test, Y_test, cleaned_labels):
        Y_pred = model.predict(X_test)
        Y_test_numpy = Y_test.to_numpy()

        metrics = Metric()
        for i, column in enumerate(cleaned_labels):
            feature_predictions = Y_pred[:, i]
            feature_truth = Y_test_numpy[:, i]
            metric = classification_report(
                            y_pred=feature_predictions,
                            y_true=feature_truth,
                            output_dict=True,
                            zero_division=0)
            
            metrics.push(metric, column) 
            

        return {'model': model, 'f1': metrics.average_f1()}

    def __clean_labels(self, Y):
        print("checking labels")
        mins = Y.min()
        maxes = Y.max()
        cleaned_labels = []
        for i, column in enumerate(Constant.labels()):
          min_value = mins[i]
          max_value = maxes[i]
          if min_value != 0 or max_value != 1:
            print(f"column: {column} min: {min_value} max: {max_value} invalid, removing label")
          else:
            cleaned_labels.append(column)

        return cleaned_labels

    def __build_pipeline(self, classifier: ClassifierMixin):
        return Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(classifier, n_jobs=1))
        ])

    def run(self):
        result = {}
        print("Reading data")
        engine = create_engine('sqlite:///etl.db')
        df = pd.read_sql_table(self.table, engine)

        X = df['message']
        Y = df[Constant.labels()]
        print("cleaning labels")
        cleaned_labels = self.__clean_labels(Y)
        Y = Y[cleaned_labels]

        X_test, X_train, Y_test, Y_train = train_test_split(X, Y, random_state=42)
        print("Run SGD Pipeline")
        sgd_classifier = SGDClassifier(max_iter=5000, tol=1e-3)
        sgd_pipeline = self.__pipeline_fit(sgd_classifier, X_train, Y_train)
        result['SGD'] = self.__pipeline_metric(sgd_pipeline, X_test, Y_test, cleaned_labels)

        parameters = {'clf__estimator__loss': [
            'hinge', 
            'log', 
            'modified_huber', 
            'squared_hinge'
            ],
            'clf__estimator__max_iter': [5000],
            'clf__estimator__tol': [1e-3]
        }


        # Note: here I would prefer Hyperopt or Optuna
        # But project requires GridSearch
        print("Run SGD Grid Pipeline")
        sgd_grid = GridSearchCV(sgd_pipeline, parameters, n_jobs=-1).fit(X_train, Y_train)
        result['SGD Grid'] = self.__pipeline_metric(sgd_grid, X_test, Y_test, cleaned_labels)
        
        print("Run KN Pipeline")
        k_neighbors_classifier = KNeighborsClassifier()
        k_neighbors_pipeline = self.__pipeline_fit(k_neighbors_classifier, X_train, Y_train)
        result['KN'] = self.__pipeline_metric(k_neighbors_pipeline, X_test, Y_test, cleaned_labels)

        print("Summary (f1 avg): ")
        for k,v in result.items():
            f1 = v['f1']
            print(f"{k}: {f1}")



# ### 9. Export your model as a pickle file

# In[ ]:


def main():
    ml_pipeline = MLPipeline()
    ml_pipeline.run()

if __name__ == '__main__':
    main()