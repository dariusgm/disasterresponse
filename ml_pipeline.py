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
    print(f"column: {column} min: {min_value} max: {max_value} invalid, skipping label")
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
    ('clf', MultiOutputClassifier(SGDClassifier(), n_jobs=1))
    # ('clf', MultiOutputClassifier(KNeighborsClassifier(), n_jobs=1))
])

pipeline.fit(X_train, y_train)
print(pipeline.get_params())
y_pred = pipeline.predict(X_test)
print()

# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[ ]:


from sklearn.metrics import classification_report


for i in range(0, len(y_pred)):
  m = classification_report(y_test[i], y_pred[i])


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[ ]:


parameters = None

cv = None


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:





# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:



