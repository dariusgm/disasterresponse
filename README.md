# distaster response

This project contains code for my solution for the "disaster response" project at he nanodegree of udacity, data science.

# Fetching Repository

```
git clone https://github.com/dariusgm/disasterresponse 
```


# Installation Native

You can install a webserver in case you want to play around with the code and add external resources to it. When you want to use python:

Install pyenv on your platform, see: https://github.com/pyenv/pyenv

```
pyenv install 3.9.2
pyenv local 3.9.2
python3 -m pip install --upgrade pip
pip install pipenv
pyenv rehash
pipenv install --dev --python 3.9.2
pipenv shell
```

# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
	
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


You can now access http://localhost:8888 and start playing around.

Or you can use ./all.sh for doing all steps in one

## Repository Structure

app/templates/go.html - result page for predictions
app/templates/master.html - header and body part of index page
app/run.py - start flask server and allow predicions (make sure you are in the app directory before you run it)
data/disaster_categories.csv - raw category data provided by udacity
data/disaster_messages.csv   - raw message data provided by udacity
data/process_data.py         - run ETL Pipeline to prepare the data for MLPipeline
models/ train_classifer.py   - run MLPipeline, producing a model and a model meta file.
.gitignore
.python-version             - sets the python version for pyenv
.all.sh                     - run ETL Pipeline, than MLPipeline and finally start the webserver
LICENCE
Pipfile                     - set depedencies using pipenv
Pipfile.lock                - lock file for pipenv
README.md                   - this file




