import streamlit as st
import cv2
import sqlite3
import numpy as np
import face_recognition
from PIL import Image
import time

DB_PATH = 'restaurant_bookings.db'
TABLE_LAYOUT_PATH = 'Table Layout.jpg'

# Function to set up the database
def setup_database():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                phone TEXT,
                pax INTEGER,
                table_number INTEGER,
                embedding BLOB
            )
        """)
        conn.commit()

# Function to allocate a table based on pax count
def allocate_table(pax):
    if pax <= 2:
        return 1  # Example table number for 2 people
    elif pax <= 4:
        return 2  # Example table number for 4 people
    else:
        return 3  # Example table number for more people

# Function to register a new user
def register_user(name, phone, pax, image):
    image = face_recognition.load_image_file(image)
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        face_embedding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
        table_number = allocate_table(pax)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO bookings (name, phone, pax, table_number, embedding) VALUES (?, ?, ?, ?, ?)",
                           (name, phone, pax, table_number, face_embedding.tobytes()))
            conn.commit()
        return table_number
    else:
        st.warning("No face detected in the uploaded image.")
        return None

# Function to run real-time face recognition
def run_face_recognition():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, phone, pax, table_number, embedding FROM bookings")
        bookings = cursor.fetchall()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]  # Convert from BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            for row in bookings:
                name, phone, pax, table_number, db_embedding = row
                db_embedding = np.frombuffer(db_embedding, dtype=np.float64)
                matches = face_recognition.compare_faces([db_embedding], face_encoding)
                if matches[0]:
                    cv2.putText(frame, f"Name: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Pax: {pax}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Table: {table_number}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break

        stframe.image(frame, channels="BGR")

        if st.button("Stop Webcam", key=f"stop_webcam_{time.time()}"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit UI
st.title("F&B Restaurant Seating System")

# User registration section
st.header("Register New Guest")
name = st.text_input("Name")
phone = st.text_input("Phone Number")
pax = st.number_input("Number of People (Pax)", min_value=1, max_value=10)
uploaded_image = st.file_uploader("Upload a Photo", type=["jpg", "jpeg", "png"])

if st.button("Register"):
    if name and phone and pax and uploaded_image:
        table_number = register_user(name, phone, pax, uploaded_image)
        if table_number is not None:
            st.success(f"Guest registered! Assigned Table: {table_number}")
    else:
        st.warning("Please fill in all fields and upload a photo.")

# Real-time recognition section
st.header("Real-Time Face Recognition")
if st.button("Start Webcam"):
    run_face_recognition()

# Setup the database
setup_database()
