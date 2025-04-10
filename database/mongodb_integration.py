import os
from pymongo import MongoClient
from datetime import datetime, timezone

def get_database():
    connection_string = os.getenv("DATABASE_URL", "mongodb://localhost:27017/")
    client = MongoClient(connection_string)
    return client["asl_database"]

def insert_session(session_id, start_time, end_time):
    db = get_database()
    sessions = db["sessions"]
    session_document = {
        "session_id": session_id,
        "start_time": start_time,
        "end_time": end_time
    }
    result = sessions.insert_one(session_document)
    return result.inserted_id

def insert_detection(session_id, letter, timestamp, confidence=None):
    db = get_database()
    detections = db["detections"]
    detection_document = {
        "session_id": session_id,
        "letter": letter,
        "timestamp": timestamp
    }
    if confidence is not None:
        detection_document["confidence"] = confidence
    result = detections.insert_one(detection_document)
    return result.inserted_id

def get_sessions():
    db = get_database()
    sessions = db["sessions"]
    return list(sessions.find())

def get_detections():
    db = get_database()
    detections = db["detections"]
    return list(detections.find())

if __name__ == '__main__':
    from datetime import datetime, timezone
    session_id = "test_session_001"
    start = datetime.now(timezone.utc)
    end = datetime.now(timezone.utc)
    sid = insert_session(session_id, start, end)
    print("Inserted session id:", sid)