import os
import sys
import threading
import time
import signal
import logging
from datetime import datetime, timezone
import pytest
from pymongo.errors import BulkWriteError

# Import functions and variables from your MongoDB integration module
from mongodb_integration import (
    client, db,
    insert_session,
    add_detection_to_batch,
    flush_detection_batch,
    graceful_exit,
    BATCH_THRESHOLD,
    batch_queue,
    FLUSH_INTERVAL
)

def test_bulk_write_error(monkeypatch, caplog):
    """
    Simulate a BulkWriteError and test that flush_detection_batch
    captures and logs the error appropriately.
    """
    # Dummy function that simulates a BulkWriteError for the bulk_write method.
    def dummy_bulk_write(operations):
        raise BulkWriteError({"error": "simulated error"})
    
    # Save original bulk_write method to restore later.
    original_bulk_write = db["detections"].bulk_write
    monkeypatch.setattr(db["detections"], "bulk_write", dummy_bulk_write)
    
    # Clear the batch_queue and add BATCH_THRESHOLD entries to trigger a flush
    global batch_queue
    batch_queue[:] = []
    session_id = "test_bulk_error"
    for i in range(BATCH_THRESHOLD):
        add_detection_to_batch(session_id, "X", datetime.now(timezone.utc), confidence=0.80)
    
    # Use caplog to verify if the BulkWriteError is logged.
    with caplog.at_level(logging.ERROR):
        flush_detection_batch()
        assert "Bulk write error" in caplog.text
    
    # Restore the original bulk_write method.
    monkeypatch.setattr(db["detections"], "bulk_write", original_bulk_write)

def test_multithreaded_detection():
    """
    Test that adding detection records from multiple threads 
    correctly handles thread safety and flushes data as expected.
    """
    global batch_queue
    batch_queue[:] = []
    session_id = "test_multithreaded"

    def add_detections(n):
        for i in range(n):
            add_detection_to_batch(session_id, "Y", datetime.now(timezone.utc), confidence=0.85)
    
    num_threads = 5
    detections_per_thread = BATCH_THRESHOLD  # Adjust number per thread as needed
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=add_detections, args=(detections_per_thread,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # Query the database to verify that the records for the session were inserted.
    inserted = list(db["detections"].find({"session_id": session_id}))
    expected_records = num_threads * detections_per_thread
    # If the auto flush mechanism works, the database should have at least the expected number of records.
    assert len(inserted) >= expected_records

def test_periodic_flush():
    """
    Test the background periodic flush functionality.
    Start the background flush thread, add some records that do not reach the threshold,
    and verify they are automatically flushed after FLUSH_INTERVAL.
    """
    global batch_queue
    batch_queue[:] = []
    session_id = "test_periodic"

    # Import the start_periodic_flush function and the stop_event from your module.
    from mongodb_integration import start_periodic_flush, stop_event

    flush_thread = start_periodic_flush()
    
    # Add a few records (less than BATCH_THRESHOLD) so they remain in the queue.
    num_records = 5
    for i in range(num_records):
        add_detection_to_batch(session_id, "Z", datetime.now(timezone.utc), confidence=0.90)
    
    # Wait enough time for the FLUSH_INTERVAL to trigger an automatic flush.
    time.sleep(FLUSH_INTERVAL + 1)
    
    # Verify that the batch_queue has been cleared.
    assert len(batch_queue) == 0
    
    # Verify that the records have been inserted in the database.
    inserted = list(db["detections"].find({"session_id": session_id}))
    assert len(inserted) >= num_records
    
    # Signal the flush thread to stop and wait for it to join.
    stop_event.set()
    flush_thread.join(timeout=2)

def test_graceful_shutdown(monkeypatch):
    """
    Test that graceful_exit flushes the remaining batch records before shutting down.
    Use monkeypatch to override sys.exit so that the process does not actually exit.
    """
    global batch_queue
    batch_queue[:] = []
    session_id = "test_shutdown"
    num_records = 4
    for i in range(num_records):
        add_detection_to_batch(session_id, "S", datetime.now(timezone.utc), confidence=0.92)
    
    # Prevent sys.exit from actually terminating the test process.
    exit_called = False
    def fake_exit(code):
        nonlocal exit_called
        exit_called = True
    monkeypatch.setattr(sys, "exit", fake_exit)
    
    # Simulate sending a termination signal.
    graceful_exit(signal.SIGTERM, None)
    
    # Confirm that sys.exit was called and that batch_queue was flushed.
    assert exit_called
    inserted = list(db["detections"].find({"session_id": session_id}))
    assert len(inserted) >= num_records
