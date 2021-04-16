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

```

# Start native
```
python -m http.server 8888
```


You can now access http://localhost:8888 and start playing around.

# Installation Docker

Depending on your installation, sudo may not be required.

```
sudo docker build -t dariuspixelart .
```

# Start Docker

```
sudo docker run -p 8888:8888 dariuspixelart
```

You can now access http://localhost:8888 and start playing around. The data is not mounted - in case you do changes and what them persistant, rebuild the image using the build command.

# Blog

This Code is a reference from https://www.murawski.blog/pixelart.html
