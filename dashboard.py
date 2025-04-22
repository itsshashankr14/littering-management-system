import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

# Connect to SQLite database
def get_connection():
    conn = sqlite3.connect("detections.db")
    return conn

# Fetch all data
def fetch_data():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM detections", conn)
    conn.close()
    return df

# Convert string datetime to actual datetime object
def convert_time_column(df):
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df

# Streamlit App
def main():
    st.set_page_config("Litter Detection Dashboard", layout="wide")
    st.title("🚗🗑️ Litter Detection Dashboard")

    # Load and prepare data
    df = fetch_data()
    df = convert_time_column(df)

    # Sidebar Filters
    st.sidebar.header("Filter Detections")

    # Filter by date range
    min_date = df['time'].min().date() if not df.empty else datetime.now().date()
    max_date = df['time'].max().date() if not df.empty else datetime.now().date()

    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)

    # Filter by license plate
    license_filter = st.sidebar.text_input("License Plate Contains")

    # Apply filters
    filtered_df = df[
        (df['time'].dt.date >= start_date) &
        (df['time'].dt.date <= end_date)
    ]

    if license_filter:
        filtered_df = filtered_df[filtered_df['car_number'].str.contains(license_filter, case=False)]

    st.subheader("Filtered Detections")
    st.dataframe(filtered_df, use_container_width=True)

    # Option to download data
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="filtered_detections.csv", mime="text/csv")

if __name__ == "__main__":
    main()
