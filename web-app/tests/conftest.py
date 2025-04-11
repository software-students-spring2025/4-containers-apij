import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock environment variables
os.environ['MONGODB_URI'] = 'mongodb://mongodb:27017/'
os.environ['FLASK_APP'] = 'web_app.py'
os.environ['FLASK_ENV'] = 'testing' 