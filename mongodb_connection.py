import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Get MongoDB password from environment variable
DB_PASSWORD = os.environ.get("MONGO_DB_PASSWORD")

if not DB_PASSWORD:
    raise ValueError("MONGO_DB_PASSWORD environment variable not set. Please set it before running the application.")

# MongoDB connection string
MONGO_URI = f"mongodb+srv://vincejim91126_db_user:{DB_PASSWORD}@watchlist.l0q1rbv.mongodb.net/?appName=watchlist"

try:
    # Create MongoDB client with timeout
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    
    # Verify connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    print(f"Failed to connect to MongoDB: {e}")
    client = None

# Get database
db = client["stock_watchlist"] if client else None

def get_db():
    """
    Returns the MongoDB database connection.
    """
    return db

def close_connection():
    """
    Closes the MongoDB connection.
    """
    if client:
        client.close()
        print("MongoDB connection closed.")

# Example functions for common operations
def insert_document(collection_name, document):
    """Insert a single document into a collection."""
    if db:
        collection = db[collection_name]
        result = collection.insert_one(document)
        return result.inserted_id
    return None


def find_document(collection_name, query):
    """Find a single document in a collection."""
    if db:
        collection = db[collection_name]
        return collection.find_one(query)
    return None

def find_documents(collection_name, query=None):
    """Find multiple documents in a collection."""
    if db:
        collection = db[collection_name]
        if query:
            return list(collection.find(query))
        return list(collection.find())
    return []

def update_document(collection_name, query, update_data):
    """Update a document in a collection."""
    if db:
        collection = db[collection_name]
        result = collection.update_one(query, {"$set": update_data})
        return result.modified_count
    return 0

def delete_document(collection_name, query):
    """Delete a document from a collection."""
    if db:
        collection = db[collection_name]
        result = collection.delete_one(query)
        return result.deleted_count
    return 0
