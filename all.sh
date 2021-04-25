#!/bin/bash

set -x -e
rm -f data/DisasterResponse.db
rm -f models/classifier.pkl

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
cd app && python run.py