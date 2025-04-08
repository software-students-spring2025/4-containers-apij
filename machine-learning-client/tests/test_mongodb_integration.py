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


