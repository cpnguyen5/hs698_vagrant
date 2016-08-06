from flask import Flask
from flask_sqlalchemy import SQLAlchemy
# from flask.ext.sqlalchemy import SQLAlchemy


app = Flask(__name__) # Holds the Flask instance
# Initialize app with config settings
app.config.from_object('config.ProductionConfig') #import config.py into app
db = SQLAlchemy(app)

from api import views, models
