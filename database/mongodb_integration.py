from pymongo import MongoClient, InsertOne
from datetime import datetime, timezone
import time

# Create a persistent connection to MongoDB; this connection is used for the entire program
client = MongoClient("mongodb://localhost:27018/")
db = client["asl_database"]

# Set the batch threshold—how many detection records to accumulate before inserting them in bulk
BATCH_THRESHOLD = 10
batch_queue = []  # This list will store the pending detection records

def insert_session(session_id, start_time, end_time):
    """
    Insert a session record into the 'sessions' collection.

    Parameters:
      - session_id: A unique identifier for the session.
      - start_time: The starting time of the session (timezone-aware UTC).
      - end_time: The ending time of the session (timezone-aware UTC).
    Returns:
      - The inserted document's ID.
    """
    sessions = db["sessions"]
    session_doc = {
        "session_id": session_id,
        "start_time": start_time,
        "end_time": end_time
    }
    result = sessions.insert_one(session_doc)
    return result.inserted_id

def add_detection_to_batch(session_id, detection, timestamp, confidence=None):
    """
    Add a detection record to the batch queue.

    Instead of inserting one detection at a time, we prepare them as InsertOne operations,
    and when we reach the batch threshold, we perform a bulk insert.
    
    Parameters:
      - session_id: Identifier to group detections with a session.
      - detection: The letter or sign recognized.
      - timestamp: The time when the detection occurred.
      - confidence: (Optional) Confidence level of the recognition.
    """
    global batch_queue
    detection_doc = {
        "session_id": session_id,
        "detection": detection,
        "timestamp": timestamp
    }
    if confidence is not None:
        detection_doc["confidence"] = confidence
    
    batch_queue.append(InsertOne(detection_doc))
    
    # Once we have accumulated enough records, execute a bulk write
    if len(batch_queue) >= BATCH_THRESHOLD:
        flush_detection_batch()

def flush_detection_batch():
    """
    Execute a bulk write operation for all detection records in the batch queue.
    After the bulk write, reset the batch queue.
    """
    global batch_queue
    if batch_queue:
        detections = db["detections"]
        result = detections.bulk_write(batch_queue)
        print(f"Bulk inserted {result.inserted_count} detection records.")
        batch_queue = []

# Example testing and demonstration
if __name__ == "__main__":
    # Insert a session record as a test
    session_id = "session_001"
    start_time = datetime.now(timezone.utc)
    end_time = datetime.now(timezone.utc)
    session_inserted_id = insert_session(session_id, start_time, end_time)
    print("Inserted session with ID:", session_inserted_id)

    # Simulate detection events
    # For example, simulate 25 detection records with a small delay between each
    for i in range(25):
        detection = "A"  # For example, detection is the letter "A"
        timestamp = datetime.now(timezone.utc)
        confidence = 0.95
        
        add_detection_to_batch(session_id, detection, timestamp, confidence)
        time.sleep(0.1)  # Simulate a small delay between detections

    # Flush remaining records if the batch queue is not empty
    flush_detection_batch()

    # Query and print all session and detection records to verify the insertions
    print("All sessions:", list(db["sessions"].find()))
    print("All detections:", list(db["detections"].find()))
