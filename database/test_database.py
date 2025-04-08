# test_database.py
from datetime import datetime, timezone
from mongodb_integration import (
    client, db,
    insert_session,
    add_detection_to_batch,
    flush_detection_batch,
    BATCH_THRESHOLD,
    batch_queue
)

def test_insert_session():
    session_id = "test_session"
    start_time = datetime.now(timezone.utc)
    end_time = datetime.now(timezone.utc)
    result = insert_session(session_id, start_time, end_time)
    assert result is not None

def test_batch_insertion():
    # Clear the batch_queue first:
    global batch_queue
    batch_queue[:] = []
    
    session_id = "test_batch"
    for i in range(BATCH_THRESHOLD):
        detection = "A"
        timestamp = datetime.now(timezone.utc)
        add_detection_to_batch(session_id, detection, timestamp, confidence=0.95)
    
    # After reaching the threshold, the batch should flush automatically,
    # and batch_queue should be empty.
    assert len(batch_queue) == 0
    
def test_flush_detection_batch():
    global batch_queue
    batch_queue[:] = []
    session_id = "test_flush"
    for i in range(3):
        detection = "B"
        timestamp = datetime.now(timezone.utc)
        add_detection_to_batch(session_id, detection, timestamp, confidence=0.9)
    flush_detection_batch()
    
    # Check that at least 3 records have been inserted for this session.
    detections = list(db["detections"].find({"session_id": session_id}))
    assert len(detections) >= 3
