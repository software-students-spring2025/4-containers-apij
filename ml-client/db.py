from pymongo import MongoClient
import os
import datetime

# MongoDB connection settings
MONGO_HOST = os.getenv('MONGO_HOST', 'mongodb')  # Use 'mongodb' as default for Docker
MONGO_PORT = int(os.getenv('MONGO_PORT', 27017))
MONGO_DB = os.getenv('MONGO_DB', 'asl_db')

def get_db():
    """Get MongoDB database connection"""
    try:
        # Create MongoDB client
        client = MongoClient(f'mongodb://{MONGO_HOST}:{MONGO_PORT}/')
        
        # Get database
        db = client[MONGO_DB]
        
        # Test connection
        client.server_info()
        
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def save_prediction(frame_data, prediction, confidence, timestamp=None):
    """Save prediction data to MongoDB"""
    db = get_db()
    if db is None:
        return False
    
    try:
        # Get predictions collection
        predictions = db.predictions
        
        # Create document
        document = {
            'frame_data': frame_data,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': timestamp or datetime.datetime.utcnow()
        }
        
        # Insert document
        result = predictions.insert_one(document)
        return bool(result.inserted_id)
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def get_recent_predictions(limit=10):
    """Get recent predictions from MongoDB"""
    db = get_db()
    if db is None:
        return []
    
    try:
        # Get predictions collection
        predictions = db.predictions
        
        # Find recent predictions
        cursor = predictions.find().sort('timestamp', -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return [] 