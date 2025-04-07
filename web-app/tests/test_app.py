import pytest
import json
from datetime import datetime
import cv2
import io
import numpy as np

def test_index_page(client):
    """Test that the index page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'ASL Sign Language Detector' in response.data

def test_get_detections_empty(client):
    """Test getting detections when database is empty."""
    response = client.get('/api/detections')
    assert response.status_code == 200
    assert json.loads(response.data) == []

def test_add_detection(client, sample_detection):
    """Test adding a new detection."""
    response = client.post('/api/detections',
                          json=sample_detection,
                          content_type='application/json')
    assert response.status_code == 201
    data = json.loads(response.data)
    assert 'id' in data
    assert data['message'] == 'Detection added'

def test_get_detections_with_data(client, sample_detection):
    """Test getting detections after adding data."""
    # Add a detection
    client.post('/api/detections',
                json=sample_detection,
                content_type='application/json')
    
    # Get detections
    response = client.get('/api/detections')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 1
    assert data[0]['sign'] == sample_detection['sign']

def test_receive_frame(client, sample_frame):
    """Test receiving a frame."""
    # Convert frame to JPEG
    _, buffer = cv2.imencode('.jpg', sample_frame)
    frame_bytes = buffer.tobytes()
    
    # Send frame
    response = client.post('/api/frame',
                          data={'frame': (io.BytesIO(frame_bytes), 'frame.jpg')},
                          content_type='multipart/form-data')
    assert response.status_code == 200
    assert json.loads(response.data)['message'] == 'Frame received'

def test_get_frame_no_data(client):
    """Test getting frame when no frame is available."""
    response = client.get('/api/frame')
    assert response.status_code == 404

def test_get_stats_empty(client):
    """Test getting stats when database is empty."""
    response = client.get('/api/stats')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['total_detections'] == 0
    assert len(data['signs_count']) == 0

def test_get_stats_with_data(client, sample_detection):
    """Test getting stats after adding data."""
    # Add a detection
    client.post('/api/detections',
                json=sample_detection,
                content_type='application/json')
    
    # Get stats
    response = client.get('/api/stats')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['total_detections'] == 1
    assert len(data['signs_count']) == 1
    assert data['signs_count'][0]['_id'] == sample_detection['sign']
    assert data['signs_count'][0]['count'] == 1 
