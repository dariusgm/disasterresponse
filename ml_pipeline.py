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


# load data from database
engine = create_engine('sqlite:///etl.db')
df = pd.read_sql_table('etl', engine)

# Set X, y
X = df['message']
Y = df[Constant.labels()]


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

Y = Y[cleaned_labels]
print(Y.columns)

    

t = TweetTokenizer()
def tokenize(text):
    return t.tokenize(text)

X_test, X_train, y_test, y_train = train_test_split(X, Y, random_state=42)
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(SGDClassifier(max_iter=5000, tol=1e-3), n_jobs=1))
])

pipeline.fit(X_train, y_train)
print(pipeline.get_params())
y_pred = pipeline.predict(X_test)
y_test_numpy = y_test.to_numpy()

all_class_avg = {'f1': [], 'precision': [], 'recall': []}
for i, e in enumerate(cleaned_labels):
    feature_predictions = y_pred[:, i]
    feature_truth = y_test_numpy[:, i]
    metrics = classification_report(
        y_pred=feature_predictions,
        y_true=feature_truth,
        output_dict=True,
        zero_division=0
    )

    print(f"Metrics for column {e}")
    # source: https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
    f1 = metrics['macro avg']['f1-score']
    precision = metrics['macro avg']['precision']
    recall = metrics['macro avg']['recall']
    all_class_avg['f1'].append(f1)
    all_class_avg['precision'].append(precision)
    all_class_avg['recall'].append(recall)
    print(f"f1 (macro): {f1}, precision: {precision}, recall: {recall}")

print(f"Overall Metric Average")
for k,v in all_class_avg.items():
    avg = np.average(v)
    print(f"{k}: {avg}")

parameters = {'clf__estimator__loss': [
    'hinge', 
    'log', 
    'modified_huber', 
    'squared_hinge', 
    'perceptron',
    'squared_loss',
    'huber',
    'epsilon_insensitive',
    'squared_epsilon_insensitive']
    }


cv = GridSearchCV(pipeline, parameters, n_jobs=-1)
cv.fit(X_train, y_train)
print(cv.get_params())
y_pred = cv.predict(X_test)





# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:



