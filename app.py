import streamlit as st
import cv2
import mediapipe as mp
import sqlite3
import numpy as np
from PIL import Image
import time

DB_PATH = 'restaurant_bookings.db'
TABLE_LAYOUT_PATH = 'Table Layout.jpg'

# Setup Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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
                table_number INTEGER
            )
        """)
        conn.commit()

# Function to allocate a table based on pax count
def allocate_table(pax):
    if pax <= 2:
        return 1
    elif pax <= 4:
        return 2
    else:
        return 3

# Function to register a new user
def register_user(name, phone, pax):
    table_number = allocate_table(pax)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO bookings (name, phone, pax, table_number) VALUES (?, ?, ?, ?)",
                       (name, phone, pax, table_number))
        conn.commit()
    return table_number

# Function to run real-time face detection using Mediapipe
def run_face_detection():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            # Draw face detections on the frame
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

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

if st.button("Register"):
    if name and phone and pax:
        table_number = register_user(name, phone, pax)
        st.success(f"Guest registered! Assigned Table: {table_number}")
    else:
        st.warning("Please fill in all fields.")

# Real-time detection section
st.header("Real-Time Face Detection")
if st.button("Start Webcam"):
    run_face_detection()

# Setup the database
setup_database()
