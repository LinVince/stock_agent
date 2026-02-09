import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
load_dotenv()


# Get MongoDB password from environment variable
DB_PASSWORD = os.environ.get("MONGO_DB_PASSWORD")

if not DB_PASSWORD:
    raise ValueError("MONGO_DB_PASSWORD environment variable not set. Please set it before running the application.")

# MongoDB connection string
MONGO_URI = f"mongodb+srv://vincejim91126_db_user:{DB_PASSWORD}@watchlist.l0q1rbv.mongodb.net/stock_watchlist?retryWrites=true&w=majority&appName=watchlist"

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
db = client.get_database() if client else None

def check_db_connection():
    # Get MongoDB password from environment variable
    DB_PASSWORD = os.environ.get("MONGO_DB_PASSWORD")

    if not DB_PASSWORD:
        return "MONGO_DB_PASSWORD environment variable not set. Please set it before running the application."

    # MongoDB connection string
    MONGO_URI = f"mongodb+srv://vincejim91126_db_user:{DB_PASSWORD}@watchlist.l0q1rbv.mongodb.net/stock_watchlist?retryWrites=true&w=majority&appName=watchlist"

    try:
        # Create MongoDB client with timeout
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Verify connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        return "Successfully connected to MongoDB!"
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"Failed to connect to MongoDB: {e}")
        return f"Failed to connect to MongoDB: {e}"
        client = None


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

def add_collection(collection_name):
    """
    Create a new collection if it does not already exist.
    """
    if not db:
        return "Database connection not available."

    existing_collections = db.list_collection_names()
    
    if collection_name in existing_collections:
        return f"Collection '{collection_name}' already exists."

    db.create_collection(collection_name)
    return f"Collection '{collection_name}' created successfully."

def list_collections():
    """
    List all collections in the database.
    """
    if not db:
        return []

    return db.list_collection_names()


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

def fetch_all():
    """
    Fetch all documents from all collections in the database.
    Returns a dictionary: {collection_name: [documents]}
    """
    if not db:
        return {}

    all_data = {}

    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        all_data[collection_name] = list(collection.find())

    return all_data

def delete_by_stock(collection_name, stock):
    """
    Delete a single document from a collection by stock symbol.
    """
    if not db:
        return "Database connection not available."

    collection = db[collection_name]
    result = collection.delete_one({"stock": stock})

    if result.deleted_count == 1:
        return f"Stock '{stock}' deleted successfully."
    else:
        return f"No document found with stock '{stock}'."
