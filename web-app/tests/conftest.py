import pytest
from app import app
from pymongo import MongoClient
import os

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['MONGODB_URI'] = 'mongodb://localhost:27017/test_db'
    
    with app.test_client() as client:
        yield client
    
    # Cleanup after tests
    client = MongoClient(app.config['MONGODB_URI'])
    client.drop_database('test_db')
    client.close()

@pytest.fixture
def sample_detection():
    """Create a sample detection for testing."""
    return {
        'sign': 'A',
        'confidence': 0.95,
        'timestamp': '2024-01-01T00:00:00'
    }

@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8) 
