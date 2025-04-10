# test_insert.py
from datetime import datetime, timezone
from mongodb_integration import insert_session, insert_detection

def test_insertion():
    # 使用一個測試用的 session_id
    session_id = "test_session_001"
    now = datetime.now(timezone.utc)
    
    # 插入一筆 session 資料
    session_id_inserted = insert_session(session_id, now, now)
    print(f"Inserted session id: {session_id_inserted}")
    
    # 插入一筆 detection 資料，假設辨識到字母 "A"，信心分數為 0.95
    detection_id_inserted = insert_detection(session_id, "A", now, confidence=0.95)
    print(f"Inserted detection id: {detection_id_inserted}")

if __name__ == '__main__':
    test_insertion()