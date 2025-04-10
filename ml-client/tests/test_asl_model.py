import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch
import sys
import os
import requests

# Mock the model loading before importing asl_model
with patch('pickle.load') as mock_pickle:
    mock_pickle.return_value = {'model': MagicMock()}
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import asl_model


def test_get_frame_from_bytes():
    dummy_image = np.zeros((50, 50, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', dummy_image)
    frame_bytes = encoded.tobytes()

    decoded = asl_model.get_frame_from_bytes(frame_bytes)

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (50, 50, 3)


def test_get_frame_from_bytes_invalid_data():
    invalid_bytes = b"not_a_real_image"
    frame = asl_model.get_frame_from_bytes(invalid_bytes)
    assert frame is None or isinstance(frame, np.ndarray)


def test_get_frame_from_bytes_invalid_format():
    invalid_data = b"thisisnotanimage"
    result = asl_model.get_frame_from_bytes(invalid_data)
    assert result is None


def test_is_valid_landmark():
    class DummyLandmark:
        x = 0.5
        y = 0.5
        visibility = 1.0

    dummy = DummyLandmark()
    assert asl_model.is_valid_landmark(dummy) is True

    class InvalidLandmark:
        pass

    assert asl_model.is_valid_landmark(InvalidLandmark()) is False


def test_get_landmark_coords():
    class DummyLandmark:
        x = 0.42
        y = 0.87

    dummy = DummyLandmark()
    coords = asl_model.get_landmark_coords(dummy)
    assert coords == (0.42, 0.87)


def test_process_frame_no_hands(monkeypatch):
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class DummyResults:
        multi_hand_landmarks = None

    monkeypatch.setattr(asl_model.hands, "process", lambda x: DummyResults())

    processed_frame, prediction = asl_model.process_frame(dummy_frame)

    assert isinstance(processed_frame, np.ndarray)
    assert prediction is None


def test_process_frame_with_mocked_hand(monkeypatch):
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

    monkeypatch.setattr(asl_model.hands, "process", lambda x: DummyResults())
    asl_model.model.predict = MagicMock(return_value=["A"])

    processed_frame, prediction = asl_model.process_frame(dummy_frame)

    assert isinstance(processed_frame, np.ndarray)
    assert prediction == "A"


def test_generate_processed_frames_connection_fail(monkeypatch):
    def mock_requests_get(*args, **kwargs):
        raise requests.exceptions.ConnectionError("mocked error")

    monkeypatch.setattr(asl_model.requests, "get", mock_requests_get)

    gen = asl_model.generate_processed_frames()
    assert list(gen) == []


def test_generate_processed_frames_mocked(monkeypatch):
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
    monkeypatch.setattr(asl_model, "process_frame", lambda x: (x, None))

    gen = asl_model.generate_processed_frames()
    try:
        frame = next(gen)
        assert b"Content-Type: image/jpeg" in frame
    except StopIteration:
        pytest.fail("The generator exited too early.")


def test_generate_processed_frames_real_jpeg(monkeypatch):
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
            yield multipart_chunk

    monkeypatch.setattr(asl_model.requests, "get", lambda *a, **kw: DummyResponse())
    gen = asl_model.generate_processed_frames()
    frame = next(gen)
    assert b"Content-Type: image/jpeg" in frame


def test_generate_processed_frames_hits_yield(monkeypatch):
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
            yield multipart_chunk

    monkeypatch.setattr(asl_model.requests, "get", lambda *a, **kw: DummyResponse())
    monkeypatch.setattr(asl_model, "process_frame", lambda x: (x, None))

    gen = asl_model.generate_processed_frames()
    result = next(gen)
    assert isinstance(result, bytes)

def test_run_server(monkeypatch):
    monkeypatch.setattr(asl_model.app, "run", lambda *a, **kw: None)
    asl_model.run_server()  # should hit print + app.run()

def test_health_check():
    with asl_model.app.test_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert b"ASL model service is running!" in response.data
