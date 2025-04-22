import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('detections.db')
cursor = conn.cursor()

# Query to select all rows from the detections table
cursor.execute("SELECT * FROM detections")

# Fetch all rows from the query result
rows = cursor.fetchall()

# Print each row
for row in rows:
    print(row)
# Close the connection
conn.close()

