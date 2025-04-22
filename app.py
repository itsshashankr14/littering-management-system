from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from typing import List
from pydantic import BaseModel

app = FastAPI()

# Allow frontend calls (e.g., from Streamlit, React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Detection(BaseModel):
    id: int
    input_file: str
    car_number: str
    time: str
    output_file: str

@app.get("/detections", response_model=List[Detection])
def get_detections():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections")
    rows = cursor.fetchall()
    conn.close()

    return [
        Detection(id=row[0], input_file=row[1], car_number=row[2], time=row[3], output_file=row[4])
        for row in rows
    ]
