How to run?

First activate env: env\Scripts\activate

then : python run.py

# CricketAI – Cricket Player Recognition Using AI

## Overview
CricketAI is a Python-based project that uses facial recognition and computer vision to identify cricket players from images. It combines a Flask web application with player information, statistics, and achievements.

## Technologies Used
- Python
- Flask
- OpenCV
- face_recognition
- NumPy
- SQLAlchemy
- SQLite or the database configured for the application

## Key Features
- Cricket player face recognition
- Face encoding and training pipeline
- Player information and statistics
- Player achievements and profile data
- Web application built using Flask
- Database integration

## Project Files
- `run.py` – Starts the Flask application.
- `train_faces.py` – Generates face encodings from organized player images.
- `player_data.json` – Stores cricket player information.
- `requirements.txt` – Lists Python dependencies.
- `test.py` – Contains image preprocessing test code.

## Setup Instructions

1. Clone the repository:

   `git clone https://github.com/YOUR-USERNAME/CricketAI-Cricket-Player-Recognition.git`

2. Navigate to the project directory:

   `cd CricketAI-Cricket-Player-Recognition`

3. Create a virtual environment:

   `python -m venv env`

4. Activate it on Windows:

   `env\Scripts\activate`

5. Install dependencies:

   `pip install -r requirements.txt`

6. Configure your environment variables using `.env.example` as a reference.

7. Start the application:

   `python run.py`

## Face Recognition Training
Organize training images into folders for each player, then run:

`python train_faces.py`

Ensure that the dataset paths and application configuration match the project structure.

## Notes
The application may require additional setup for face-recognition dependencies, the database, and training images. Verify all required application modules are included before running it.

## Author
Developed as an AI and computer vision project.
