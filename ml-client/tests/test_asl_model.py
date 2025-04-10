import os, sys
import cv2
import pytest
import pickle
import numpy as np
import requests
from unittest.mock import MagicMock
import builtins
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ml_client import asl_model

def test_get_frame_from_bytes():
    """Test decoding valid image bytes into a numpy array."""
    dummy_image = np.zeros((50, 50, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', dummy_image)
    frame_bytes = encoded.tobytes()

    decoded = asl_model.get_frame_from_bytes(frame_bytes)

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (50, 50, 3)


def test_get_frame_from_bytes_invalid_data():
    """Ensure invalid image data is handled gracefully."""
    invalid_bytes = b"not_a_real_image"
    frame = asl_model.get_frame_from_bytes(invalid_bytes)
    assert frame is None or isinstance(frame, np.ndarray)


def test_process_frame_no_hands(monkeypatch):
    """Ensure process_frame returns the correct format when no hands are detected."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class DummyResults:
        multi_hand_landmarks = None

    monkeypatch.setattr(asl_model.hands, "process", lambda _: DummyResults())

    processed, _ = asl_model.process_frame(dummy_frame)

    assert isinstance(processed, np.ndarray)


def test_process_frame_with_mocked_hand(monkeypatch):
    """Test process_frame with one mocked hand and prediction."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class DummyLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.visibility = 1.0
            self.presence = 1.0

        def HasField(self, name):
            return True

    class DummyHandLandmark:
        landmark = [DummyLandmark(0.1 * i, 0.1 * i) for i in range(21)]

    class DummyResults:
        multi_hand_landmarks = [DummyHandLandmark()]

    monkeypatch.setattr(asl_model.hands, "process", lambda _: DummyResults())
    asl_model.model.predict = MagicMock(return_value=["A"])

    processed, prediction = asl_model.process_frame(dummy_frame)

    assert isinstance(processed, np.ndarray)
    assert prediction == "A"


def test_generate_processed_frames_connection_fail(monkeypatch):
    """Ensure generate_processed_frames handles connection failure gracefully."""
    monkeypatch.setattr(asl_model.requests, "get", lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError()))

    gen = asl_model.generate_processed_frames()
    assert list(gen) == []


def test_generate_processed_frames_mocked(monkeypatch):
    """Test generate_processed_frames with mocked MJPEG stream."""
    dummy_image = np.zeros((50, 50, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', dummy_image)
    jpeg_bytes = encoded.tobytes()

    multipart_chunk = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" +
        jpeg_bytes +
        b"\r\n--frame\r\n"
    )

    class DummyResponse:
        def iter_content(self, chunk_size=1024):
            for i in range(0, len(multipart_chunk), chunk_size):
                yield multipart_chunk[i:i + chunk_size]

    monkeypatch.setattr(asl_model.requests, "get", lambda *a, **kw: DummyResponse())
    monkeypatch.setattr(asl_model, "process_frame", lambda x: (np.ones((50, 50, 3), dtype=np.uint8), None))

    gen = asl_model.generate_processed_frames()
    try:
        frame = next(gen)
        assert b"Content-Type: image/jpeg" in frame
    except StopIteration:
        pytest.fail("The generator exited too early.")


def test_model_fallback(monkeypatch):
    """Force model.p load failure to test fallback to MagicMock."""
    monkeypatch.setattr(builtins, "open", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("Simulated missing model.p")))

    importlib.invalidate_caches()
    from ml_client import asl_model as module
    importlib.reload(module)

    assert hasattr(module, "model")

def test_process_frame_prediction_error(monkeypatch):
    """Simulate a model.predict error to hit the fallback logic."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class DummyLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.visibility = 1.0
            self.presence = 1.0

        def HasField(self, name):
            return True

    class DummyHandLandmark:
        landmark = [DummyLandmark(0.1 * i, 0.1 * i) for i in range(21)]

    class DummyResults:
        multi_hand_landmarks = [DummyHandLandmark()]

    monkeypatch.setattr(asl_model.hands, "process", lambda _: DummyResults())
    
    def raise_exception(_): raise ValueError("Simulated prediction failure")
    asl_model.model.predict = raise_exception  # 👈 simulate prediction crash

    # Should still return a frame, but no prediction
    processed, prediction = asl_model.process_frame(dummy_frame)

    assert isinstance(processed, np.ndarray)
    assert prediction is None
