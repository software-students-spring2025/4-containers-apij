import mongomock
import pytest
from datetime import datetime, timezone, timedelta 

# This test file is designed to test the MongoDB integration functions in the database module.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# Import functions to be tested and the module
import database.mongodb_integration as mongodb_integration
from database.mongodb_integration import insert_session, insert_detection, get_sessions, get_detections

# Fixture that mocks MongoDB connection
@pytest.fixture
def mock_db(monkeypatch):
    mock_client = mongomock.MongoClient()
    mock_db = mock_client["asl_database"]

    # Patch the get_database method to return the mock database
    monkeypatch.setattr(mongodb_integration, "get_database", lambda: mock_db)
    return mock_db

def test_get_database_real_connection():
    from database.mongodb_integration import get_database
    db = get_database()
    assert db.name == "asl_database"


# Test case for empty collections
def test_empty_collections_start_empty(mock_db):
    assert get_sessions() == []
    assert get_detections() == []



# Test case for inserting and retrieving sessions
def test_insert_and_get_session(mock_db):
    session_id = "test_session_001"
    start = datetime.now(timezone.utc)
    end = datetime.now(timezone.utc)

    inserted_id = insert_session(session_id, start, end)
    sessions = get_sessions()

    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id

    # Force Mongo value to have timezone
    db_start = sessions[0]["start_time"].replace(tzinfo=timezone.utc)
    db_end = sessions[0]["end_time"].replace(tzinfo=timezone.utc)

    assert abs(db_start - start) < timedelta(seconds=1)
    assert abs(db_end - end) < timedelta(seconds=1)

# Test case for inserting and retrieving detections
def test_insert_and_get_detection(mock_db):
    session_id = "test_session_002"
    letter = "A"
    timestamp = datetime.now(timezone.utc)
    confidence = 0.92

    inserted_id = insert_detection(session_id, letter, timestamp, confidence)
    detections = get_detections()

    assert len(detections) == 1
    assert detections[0]["session_id"] == session_id
    assert detections[0]["letter"] == letter

    # Match timestamps with tolerance
    db_timestamp = detections[0]["timestamp"].replace(tzinfo=timezone.utc)
    assert abs(db_timestamp - timestamp) < timedelta(seconds=1)

    assert detections[0]["confidence"] == confidence

# Test case for inserting detection without confidence
def test_insert_detection_without_confidence(mock_db):
    session_id = "test_session_003"
    letter = "B"
    timestamp = datetime.now(timezone.utc)

    insert_detection(session_id, letter, timestamp)
    detections = get_detections()

    assert len(detections) == 1
    assert detections[0]["session_id"] == session_id
    assert "confidence" not in detections[0]

# Test case for inserting multiple sessions
def test_multiple_sessions(mock_db):
    start = datetime.now(timezone.utc)
    end = datetime.now(timezone.utc)

    insert_session("session_1", start, end)
    insert_session("session_2", start, end)

    sessions = get_sessions()
    assert len(sessions) == 2
    session_ids = [s["session_id"] for s in sessions]
    assert "session_1" in session_ids
    assert "session_2" in session_ids

# Test case for different values 
def test_multiple_detections(mock_db):
    timestamp = datetime.now(timezone.utc)

    insert_detection("session_1", "A", timestamp, confidence=0.9)
    insert_detection("session_1", "B", timestamp, confidence=0.8)

    detections = get_detections()
    assert len(detections) == 2
    letters = [d["letter"] for d in detections]
    assert "A" in letters
    assert "B" in letters

def test_get_sessions_empty(mock_db):
    sessions = get_sessions()
    assert sessions == []

def test_get_detections_empty(mock_db):
    detections = get_detections()
    assert detections == []
