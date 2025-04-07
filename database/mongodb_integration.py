from pymongo import MongoClient
from datetime import datetime 

def get_database():
    """
    Connect to MongoDB and return the database object.
    If using docker-compose, you can use 'mongodb' as the host in the connection string.
    For local testing, you might use "mongodb://localhost:27018/".
    """
    CONNECTION_STRING = "mongodb://localhost:27017/"
    client = MongoClient(CONNECTION_STRING)
    # Set the database name, e.g., "asl_database"
    return client["asl_database"]

def insert_session(session_id, start_time, end_time):
    """
    Insert a session record that logs the start and end times.

    :param session_id: Unique session identifier (string)
    :param start_time: Start time (datetime object)
    :param end_time: End time (datetime object)
    :return: The inserted document's ID
    """
    db = get_database()
    sessions = db["sessions"]
    session_doc = {
        "session_id": session_id,
        "start_time": start_time,
        "end_time": end_time
    }
    result = sessions.insert_one(session_doc)
    return result.inserted_id

def insert_detection(session_id, letter, timestamp, confidence=None):
    """
    Insert a detection record that logs the detected letter and its timestamp.

    :param session_id: The session ID to which this detection belongs
    :param letter: Detected letter (e.g., "A")
    :param timestamp: Detection time (datetime object)
    :param confidence: (Optional) Confidence level of the prediction
    :return: The inserted document's ID
    """
    db = get_database()
    detections = db["detections"]
    detection_doc = {
        "session_id": session_id,
        "letter": letter,
        "timestamp": timestamp
    }
    if confidence is not None:
        detection_doc["confidence"] = confidence
    result = detections.insert_one(detection_doc)
    return result.inserted_id

def get_sessions():
    """
    Retrieve all session records.
    
    :return: List of session documents
    """
    db = get_database()
    sessions = db["sessions"]
    return list(sessions.find())

def get_detections():
    """
    Retrieve all detection records.
    
    :return: List of detection documents
    """
    db = get_database()
    detections = db["detections"]
    return list(detections.find())


# Test: Insert a session record
session_id = "session_001"
start = datetime.utcnow()
end = datetime.utcnow()
session_inserted_id = insert_session(session_id, start, end)
print("Inserted session with ID:", session_inserted_id)

# Test: Insert a detection record
detection_inserted_id = insert_detection(session_id, "A", datetime.utcnow(), confidence=0.98)
print("Inserted detection with ID:", detection_inserted_id)

# Test: Print all session and detection records
print("All sessions:", get_sessions())
print("All detections:", get_detections()) 

