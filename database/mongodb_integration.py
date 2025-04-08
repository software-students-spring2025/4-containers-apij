import os
import logging
import threading
import signal
import sys
import time
from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError
from datetime import datetime, timezone

# ---------------------------
# Configuration (using environment variables or defaults)
# ---------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27018/")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "asl_database")
BATCH_THRESHOLD = int(os.environ.get("BATCH_THRESHOLD", 10))
FLUSH_INTERVAL = float(os.environ.get("FLUSH_INTERVAL", 2))  # seconds

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Persistent MongoDB connection
# ---------------------------
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Global variables for batch insertion with thread safety
batch_queue = []  # will store pending detection records as InsertOne operations
batch_lock = threading.Lock()
stop_event = threading.Event()

# ---------------------------
# Database Operations
# ---------------------------
def insert_session(session_id, start_time, end_time):
    """
    Insert a session record into the 'sessions' collection.

    Parameters:
      - session_id: Unique identifier for the session.
      - start_time: Starting time of the session (timezone-aware UTC).
      - end_time: Ending time of the session (timezone-aware UTC).
    Returns:
      - The inserted document's ID or None if an error occurs.
    """
    sessions = db["sessions"]
    session_doc = {
        "session_id": session_id,
        "start_time": start_time,
        "end_time": end_time
    }
    try:
        result = sessions.insert_one(session_doc)
        logging.info(f"Inserted session with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        logging.error(f"Error inserting session: {e}")
        return None

def add_detection_to_batch(session_id, detection, timestamp, confidence=None):
    """
    Add a detection record to a global batch queue. Once the threshold is reached,
    perform a bulk write to the database.

    Parameters:
      - session_id: Identifier to group detections with a session.
      - detection: The letter or sign recognized.
      - timestamp: The time when the detection occurred.
      - confidence: (Optional) The confidence level of the recognition.
    """
    global batch_queue
    detection_doc = {
        "session_id": session_id,
        "detection": detection,
        "timestamp": timestamp
    }
    if confidence is not None:
        detection_doc["confidence"] = confidence
    
    with batch_lock:
        batch_queue.append(InsertOne(detection_doc))
        current_size = len(batch_queue)
    logging.info(f"Added detection. Current batch size: {current_size}")

    if current_size >= BATCH_THRESHOLD:
        flush_detection_batch()

def flush_detection_batch():
    """
    Perform a bulk write for all detection records in the batch queue.
    """
    global batch_queue
    with batch_lock:
        if not batch_queue:
            return
        operations = batch_queue[:]  # make a copy
        batch_queue = []  # clear queue
    
    try:
        detections = db["detections"]
        result = detections.bulk_write(operations)
        logging.info(f"Bulk inserted {result.inserted_count} detection records.")
    except BulkWriteError as bwe:
        logging.error(f"Bulk write error: {bwe.details}")
    except Exception as e:
        logging.error(f"Unexpected error during bulk write: {e}")

# ---------------------------
# Periodic Flush Background Thread
# ---------------------------
def periodic_flush():
    """Flush the batch queue at regular intervals."""
    while not stop_event.is_set():
        time.sleep(FLUSH_INTERVAL)
        flush_detection_batch()

def start_periodic_flush():
    thread = threading.Thread(target=periodic_flush)
    thread.daemon = True
    thread.start()
    return thread

# ---------------------------
# Graceful Shutdown
# ---------------------------
def graceful_exit(signum, frame):
    logging.info("Termination signal received; flushing remaining detection records.")
    stop_event.set()
    flush_detection_batch()
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)
