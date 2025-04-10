import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import json
from web_app import app, DemoDataStore
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import io
import numpy as np
import cv2
import base64
import requests

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_mongodb():
    """Mock MongoDB connection and operations."""
    with patch('web_app.MongoClient') as mock_client:
        # Create mock database and collection
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.detections = mock_collection
        mock_client.return_value.server_info.return_value = {'version': '4.4.0'}
        mock_client.return_value.asl_detections = mock_db
        
        # Patch the db directly
        with patch('web_app.db', mock_db):
            yield mock_collection

def test_index_route(client):
    """Test the index route returns the correct template."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'ASL Detection' in response.data

def test_health_route(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_stats_route_with_demo_data(client):
    """Test the stats endpoint when using demo data."""
    with patch('web_app.mongodb_connected', False):
        response = client.get('/api/stats')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check stats structure
        assert 'stats' in data
        assert 'charts' in data
        stats = data['stats']
        assert 'total_detections' in stats
        assert 'unique_signs' in stats
        assert 'avg_confidence' in stats
        assert 'last_detection' in stats
        
        # Check charts structure
        charts = data['charts']
        assert 'frequency' in charts
        assert 'timeline' in charts


def test_stats_route_with_mongodb(client, mock_mongodb):
    """Test the stats endpoint when using MongoDB data."""
    # Mock MongoDB data
    mock_detections = [
        {
            'sign': 'A',
            'confidence': 0.95,
            'timestamp': datetime.now() - timedelta(minutes=5)
        },
        {
            'sign': 'B',
            'confidence': 0.85,
            'timestamp': datetime.now() - timedelta(minutes=10)
        }
    ]
    mock_mongodb.find.return_value = mock_detections
    
    with patch('web_app.mongodb_connected', True):
        response = client.get('/api/stats')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify MongoDB was queried
        mock_mongodb.find.assert_called_once()
        
        # Check stats structure and values
        stats = data['stats']
        assert stats['total_detections'] == 2
        assert stats['unique_signs'] == 2
        assert 0.85 <= stats['avg_confidence'] <= 0.95

def test_stats_route_with_mongodb_empty(client, mock_mongodb):
    """Test the stats endpoint when MongoDB is connected but empty."""
    # Mock empty MongoDB data
    mock_mongodb.find.return_value = []
    
    with patch('web_app.mongodb_connected', True):
        response = client.get('/api/stats')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify MongoDB was queried
        mock_mongodb.find.assert_called_once()
        
        # Check that demo data was used
        stats = data['stats']
        assert stats['total_detections'] > 0

def test_stats_route_with_error(client, mock_mongodb):
    """Test the stats endpoint when an error occurs."""
    # Mock MongoDB error
    mock_mongodb.find.side_effect = Exception("Test error")
    
    with patch('web_app.mongodb_connected', True):
        response = client.get('/api/stats')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check error response
        stats = data['stats']
        assert stats['total_detections'] == 0
        assert stats['unique_signs'] == 0
        assert stats['avg_confidence'] == 0
        assert stats['last_detection'] == "Error retrieving data"

def test_process_frame_route_success(client):
    """Test successful frame processing."""
    with patch('web_app.process_with_model') as mock_process:
        mock_process.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 'A')
        
        # Create a proper file-like object for testing
        test_image = io.BytesIO(b'test image data')
        
        # Mock cv2.imdecode to return a valid frame
        with patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            # Mock cv2.imencode to return a valid encoded image
            with patch('cv2.imencode', return_value=(True, np.array([1, 2, 3], dtype=np.uint8))):
                response = client.post('/process_frame', 
                                     data={'frame': (test_image, 'test.jpg')},
                                     content_type='multipart/form-data')
                
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'processed_image' in data
                assert data['prediction'] == 'A'

def test_process_frame_route_no_frame(client):
    """Test frame processing with no frame provided."""
    response = client.post('/process_frame')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_process_frame_route_invalid_frame(client):
    """Test frame processing with invalid frame data."""
    # Create a file-like object for testing
    test_image = io.BytesIO(b'invalid image data')
    
    # Mock cv2.imdecode to return None (invalid frame)
    with patch('cv2.imdecode', return_value=None):
        response = client.post('/process_frame', 
                             data={'frame': (test_image, 'test.jpg')},
                             content_type='multipart/form-data')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

def test_process_frame_route_model_error(client):
    """Test frame processing when model returns an error."""
    with patch('web_app.process_with_model') as mock_process:
        mock_process.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 'Error')
        
        # Create a proper file-like object for testing
        test_image = io.BytesIO(b'test image data')
        
        # Mock cv2.imdecode to return a valid frame
        with patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            # Mock cv2.imencode to return a valid encoded image
            with patch('cv2.imencode', return_value=(True, np.array([1, 2, 3], dtype=np.uint8))):
                response = client.post('/process_frame',
                                     data={'frame': (test_image, 'test.jpg')},
                                     content_type='multipart/form-data')
                
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['prediction'] == 'Error'

def test_process_frame_route_exception(client):
    """Test frame processing when an exception occurs."""
    # Create a proper file-like object for testing
    test_image = io.BytesIO(b'test image data')
    
    # Mock cv2.imdecode to raise an exception
    with patch('cv2.imdecode', side_effect=Exception("Test error")):
        response = client.post('/process_frame',
                             data={'frame': (test_image, 'test.jpg')},
                             content_type='multipart/form-data')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data

def test_demo_data_store():
    """Test the DemoDataStore class."""
    store = DemoDataStore()
    
    # Test initialization
    assert len(store.demo_data) > 0
    assert all('sign' in d for d in store.demo_data)
    assert all('confidence' in d for d in store.demo_data)
    assert all('timestamp' in d for d in store.demo_data)
    
    # Test get_recent_detections
    recent = store.get_recent_detections(minutes=30)
    assert len(recent) > 0
    assert all(d['timestamp'] >= datetime.now() - timedelta(minutes=30) 
              for d in recent)
    
    # Test add_detection
    initial_count = len(store.demo_data)
    store.add_detection('X', 0.9)
    assert len(store.demo_data) == initial_count + 1
    assert store.demo_data[-1]['sign'] == 'X'
    assert store.demo_data[-1]['confidence'] == 0.9

def test_mongodb_connection_retry():
    """Test MongoDB connection retry mechanism."""
    with patch('web_app.MongoClient') as mock_client:
        # Simulate connection failure
        mock_client.side_effect = ConnectionFailure("Connection failed")
        
        # Import web_app to trigger connection attempt
        import web_app
        
        # Verify that mongodb_connected is False after retries
        assert not web_app.mongodb_connected

def test_mongodb_connection_timeout():
    """Test MongoDB connection timeout."""
    with patch('web_app.MongoClient') as mock_client:
        # Simulate timeout
        mock_client.side_effect = ServerSelectionTimeoutError("Timeout")
        
        # Import web_app to trigger connection attempt
        import web_app
        
        # Verify that mongodb_connected is False after retries
        assert not web_app.mongodb_connected

def test_stream_route(client):
    """Test the legacy stream endpoint."""
    response = client.get('/stream')
    assert response.status_code == 200
    assert b'Browser camera access now used instead' in response.data

def test_process_frame_with_mongodb_success(client, mock_mongodb):
    """Test frame processing with MongoDB storage."""
    with patch('web_app.process_with_model') as mock_process:
        mock_process.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 'A')
        
        # Create a proper file-like object for testing
        test_image = io.BytesIO(b'test image data')
        
        # Mock cv2.imdecode to return a valid frame
        with patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            # Mock cv2.imencode to return a valid encoded image
            with patch('cv2.imencode', return_value=(True, np.array([1, 2, 3], dtype=np.uint8))):
                with patch('web_app.mongodb_connected', True):
                    response = client.post('/process_frame', 
                                         data={'frame': (test_image, 'test.jpg')},
                                         content_type='multipart/form-data')
                    
                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert 'processed_image' in data
                    assert data['prediction'] == 'A'
                    
                    # Verify MongoDB insert was called
                    mock_mongodb.insert_one.assert_called_once()

def test_process_frame_with_mongodb_error(client, mock_mongodb):
    """Test frame processing with MongoDB storage error."""
    with patch('web_app.process_with_model') as mock_process:
        mock_process.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 'A')
        
        # Create a proper file-like object for testing
        test_image = io.BytesIO(b'test image data')
        
        # Mock cv2.imdecode to return a valid frame
        with patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            # Mock cv2.imencode to return a valid encoded image
            with patch('cv2.imencode', return_value=(True, np.array([1, 2, 3], dtype=np.uint8))):
                with patch('web_app.mongodb_connected', True):
                    # Mock MongoDB insert to raise an exception
                    mock_mongodb.insert_one.side_effect = Exception("MongoDB error")
                    
                    response = client.post('/process_frame', 
                                         data={'frame': (test_image, 'test.jpg')},
                                         content_type='multipart/form-data')
                    
                    assert response.status_code == 500
                    data = json.loads(response.data)
                    assert 'error' in data
                    assert 'MongoDB error' in data['error']

def test_process_frame_with_demo_data(client):
    """Test frame processing with demo data storage."""
    with patch('web_app.process_with_model') as mock_process:
        mock_process.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 'A')
        
        # Create a proper file-like object for testing
        test_image = io.BytesIO(b'test image data')
        
        # Mock cv2.imdecode to return a valid frame
        with patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            # Mock cv2.imencode to return a valid encoded image
            with patch('cv2.imencode', return_value=(True, np.array([1, 2, 3], dtype=np.uint8))):
                with patch('web_app.mongodb_connected', False):
                    with patch('web_app.demo_store.add_detection') as mock_add_detection:
                        response = client.post('/process_frame', 
                                             data={'frame': (test_image, 'test.jpg')},
                                             content_type='multipart/form-data')
                        
                        assert response.status_code == 200
                        data = json.loads(response.data)
                        assert 'processed_image' in data
                        assert data['prediction'] == 'A'
                        
                        # Verify demo store was used
                        mock_add_detection.assert_called_once_with('A', 0.95)

def test_process_frame_imencode_error(client):
    """Test frame processing when imencode fails."""
    with patch('web_app.process_with_model') as mock_process:
        mock_process.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 'A')
        
        # Create a proper file-like object for testing
        test_image = io.BytesIO(b'test image data')
        
        # Mock cv2.imdecode to return a valid frame
        with patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            # Mock cv2.imencode to return False (encoding failed)
            with patch('cv2.imencode', return_value=(False, None)):
                response = client.post('/process_frame', 
                                     data={'frame': (test_image, 'test.jpg')},
                                     content_type='multipart/form-data')
                
                assert response.status_code == 500
                data = json.loads(response.data)
                assert 'error' in data

def test_process_with_model_success():
    """Test successful model processing."""
    with patch('requests.post') as mock_post:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'processed_image': base64.b64encode(np.zeros((100, 100, 3), dtype=np.uint8).tobytes()).decode('utf-8'),
            'prediction': 'A'
        }
        mock_post.return_value = mock_response
        
        # Mock cv2.imdecode to return a valid frame
        with patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            # Mock cv2.imencode to return a valid encoded image
            with patch('cv2.imencode', return_value=(True, np.array([1, 2, 3], dtype=np.uint8))):
                from web_app import process_with_model
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                processed_frame, prediction = process_with_model(frame)
                
                assert prediction == 'A'
                assert isinstance(processed_frame, np.ndarray)

def test_process_with_model_error():
    """Test model processing with error response."""
    with patch('requests.post') as mock_post:
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        # Mock cv2.putText to avoid actual drawing
        with patch('cv2.putText', return_value=None):
            from web_app import process_with_model
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            processed_frame, prediction = process_with_model(frame)
            
            assert prediction == 'Error'
            assert isinstance(processed_frame, np.ndarray)

def test_process_with_model_connection_error():
    """Test model processing with connection error."""
    with patch('requests.post') as mock_post:
        # Mock connection error
        mock_post.side_effect = requests.RequestException("Connection error")
        
        # Mock cv2.putText to avoid actual drawing
        with patch('cv2.putText', return_value=None):
            from web_app import process_with_model
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            processed_frame, prediction = process_with_model(frame)
            
            assert prediction == 'Error'
            assert isinstance(processed_frame, np.ndarray)

def test_extract_prediction_from_frame():
    """Test the extract_prediction_from_frame function."""
    from web_app import extract_prediction_from_frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    prediction = extract_prediction_from_frame(frame)
    assert prediction == "ASL"

def test_mongodb_connection_unexpected_error():
    """Test MongoDB connection with unexpected error."""
    with patch('web_app.MongoClient') as mock_client:
        # Simulate unexpected error
        mock_client.side_effect = Exception("Unexpected error")
        
        # Import web_app to trigger connection attempt
        import web_app
        
        # Verify that mongodb_connected is False after retries
        assert not web_app.mongodb_connected

def test_demo_data_store_max_size():
    """Test the DemoDataStore class with maximum size limit."""
    store = DemoDataStore()
    
    # Add more than 100 detections
    for i in range(150):
        store.add_detection('X', 0.9)
    
    # Verify that only the last 100 detections are kept
    assert len(store.demo_data) == 100 


# Test to trigger CI badge - 2