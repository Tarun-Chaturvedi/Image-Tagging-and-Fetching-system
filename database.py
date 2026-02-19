import sqlite3
import json
import numpy as np

DB_NAME = "images.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # This lets us access data by column name
    return conn

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # 1. Existing Images Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            hash TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 2. General Object Tags (YOLO labels)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            tag TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY (image_id) REFERENCES images (id)
        )
    ''')
    # 3. DISTINCT PROFILES (Unique People)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT DEFAULT 'Unknown',
            representative_embedding BLOB,  -- Store the 128D vector as binary
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 4. FACE DETECTIONS
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            profile_id INTEGER,
            bbox TEXT,  -- JSON string of [x1, y1, x2, y2]
            embedding BLOB,
            FOREIGN KEY (image_id) REFERENCES images (id),
            FOREIGN KEY (profile_id) REFERENCES profiles (id)
        )
    ''')
    conn.commit()
    conn.close()

def insert_image(path, img_hash):
    """Saves image path and hash, returns the new image ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO images (path, hash) VALUES (?, ?)", (path, img_hash))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # This handles the 'hash TEXT UNIQUE' part of your roadmap
        return image_exists(img_hash)
    finally:
        conn.close()

def image_exists(img_hash):
    """Checks if a hash is already in the DB"""
    conn = get_db_connection()
    cursor = conn.cursor()
    result = cursor.execute("SELECT id FROM images WHERE hash = ?", (img_hash,)).fetchone()
    conn.close()
    return result['id'] if result else None

def insert_tag(image_id, tag, confidence):
    """Saves a detected tag for a specific image"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tags (image_id, tag, confidence) VALUES (?, ?, ?)", 
                   (image_id, tag, confidence))
    conn.commit()
    conn.close()

def search_by_tag(tag_name):
    """Finds all images that have a specific tag"""
    conn = get_db_connection()
    cursor = conn.cursor()
    # This joins both tables to get the file path for the tag
    query = """
        SELECT images.path, tags.confidence 
        FROM images 
        JOIN tags ON images.id = tags.image_id 
        WHERE tags.tag = ?
    """
    results = cursor.execute(query, (tag_name,)).fetchall()
    conn.close()
    return results

def delete_image_from_db(image_id):
    """Removes an image and its tags from the database."""
    conn = sqlite3.connect("images.db")
    cursor = conn.cursor()
    try:
        # First, delete associated tags
        cursor.execute("DELETE FROM tags WHERE image_id = ?", (image_id,))
        # Then, delete the image entry
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting image: {e}")
        return False
    finally:
        conn.close()

def get_tag_stats():
    """Returns a list of tuples (tag, count) for all detected objects."""
    conn = sqlite3.connect("images.db")
    cursor = conn.cursor()
    # Groups by tag name and counts how many times each appears
    cursor.execute("SELECT tag, COUNT(*) as count FROM tags GROUP BY tag ORDER BY count DESC")
    stats = cursor.fetchall()
    conn.close()
    return stats

def get_or_create_profile(new_embedding, threshold=0.6):
    """Matches a face to an existing profile or creates a new one."""
    conn = sqlite3.connect("images.db")
    cursor = conn.cursor()
    
    # 1. Fetch all existing profiles
    cursor.execute("SELECT id, representative_embedding FROM profiles")
    profiles = cursor.fetchall()
    
    for p_id, p_emb_blob in profiles:
        # Convert blob back to a numpy array
        known_emb = np.frombuffer(p_emb_blob, dtype=np.float64)
        
        # Calculate distance (Euclidean)
        dist = np.linalg.norm(known_emb - new_embedding)
        
        # If distance is small, it's the same person
        if dist < threshold:
            conn.close()
            return p_id

    # 2. If no match found, create a new profile
    cursor.execute("INSERT INTO profiles (representative_embedding) VALUES (?)", 
                   (new_embedding.tobytes(),))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"[!] Created new profile: Profile_{new_id}")
    return new_id

def link_face_to_image(image_id, profile_id, encoding, location):
    """
    Links a specific face detection to an image and a profile.
    
    Args:
        image_id (int): ID from the images table
        profile_id (int): ID from the profiles table
        encoding (numpy.ndarray): The 128D face encoding
        location (tuple): (top, right, bottom, left) coordinates
    """
    conn = sqlite3.connect("images.db")
    cursor = conn.cursor()
    
    # 1. Convert the encoding array to binary (BLOB) for storage

    encoding_blob = encoding.tobytes()
    
    # 2. Convert coordinates to a JSON string for easy retrieval

    bbox_json = json.dumps(list(location))
    
    try:
        cursor.execute("""
            INSERT INTO face_detections (image_id, profile_id, bbox, embedding)
            VALUES (?, ?, ?, ?)
        """, (image_id, profile_id, bbox_json, encoding_blob))
        
        conn.commit()
    except Exception as e:
        print(f"Error linking face: {e}")
    finally:
        conn.close()

def rename_profile(profile_id, new_name):
    """Updates the name of a profile in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE profiles SET name = ? WHERE id = ?", (new_name, profile_id))
    conn.commit()
    conn.close()

def search_by_profile(profile_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Force the second column to be a float (1.0) so the math in HTML works
    query = """
        SELECT i.path, 1.0 
        FROM images i
        JOIN face_detections fd ON i.id = fd.image_id
        WHERE fd.profile_id = ?
    """
    results = cursor.execute(query, (profile_id,)).fetchall()
    conn.close()
    return results

def get_all_profiles():
    conn = sqlite3.connect("images.db")
    cursor = conn.cursor()
    # Get ID, Name, and count of detections for each person
    cursor.execute("""
        SELECT p.id, p.name, COUNT(fd.id) 
        FROM profiles p 
        LEFT JOIN face_detections fd ON p.id = fd.profile_id 
        GROUP BY p.id
    """)
    profiles = cursor.fetchall()
    conn.close()
    return profiles
