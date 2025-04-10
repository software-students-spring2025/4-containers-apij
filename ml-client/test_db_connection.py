import db
import datetime

def test_mongodb_connection():
    """Test MongoDB connection and basic operations"""
    print("Testing MongoDB connection...")
    
    # Test database connection
    database = db.get_db()
    if database is None:
        print("Failed to connect to MongoDB")
        return False
    
    print("Successfully connected to MongoDB")
    
    # Test saving a prediction
    test_data = {
        'frame_data': 'test_base64_data',
        'prediction': 'A',
        'timestamp': datetime.datetime.utcnow()
    }
    
    # Get predictions collection
    predictions = database.predictions
    
    # Insert test document
    try:
        result = predictions.insert_one(test_data)
        print(f"Successfully inserted test prediction with ID: {result.inserted_id}")
    except Exception as e:
        print(f"Failed to insert test prediction: {e}")
        return False
    
    # Test retrieving predictions
    try:
        recent_predictions = db.get_recent_predictions(limit=1)
        if recent_predictions:
            print("Successfully retrieved recent predictions")
            print(f"Latest prediction: {recent_predictions[0]['prediction']}")
        else:
            print("No predictions found in database")
    except Exception as e:
        print(f"Failed to retrieve predictions: {e}")
        return False
    
    # Clean up test data
    try:
        predictions.delete_one({'_id': result.inserted_id})
        print("Successfully cleaned up test data")
    except Exception as e:
        print(f"Failed to clean up test data: {e}")
    
    return True

if __name__ == "__main__":
    test_mongodb_connection() 