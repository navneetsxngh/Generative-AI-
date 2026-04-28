import sqlite3

## Connect to sqllite3
connection = sqlite3.connect("Chat SQL/student.db")

## Create a Cursor object to insert, record, create table
cursor  = connection.cursor()

## Create the table
table_info = """
CREATE TABLE student(
name VARCHAR(25),
class VARCHAR(25),
section VARCHAR(25),
marks INT
);
"""

cursor.execute(table_info)

## Insert some Records
cursor.execute('''INSERT INTO student VALUES ('Navneet', 'Data Science', 'A', 91)''')
cursor.execute('''INSERT INTO student VALUES ('Devendra', 'Data Science', 'A', 92)''')
cursor.execute('''INSERT INTO student VALUES ('Ubaid', 'Data Engineering', 'B', 50)''')
cursor.execute('''INSERT INTO student VALUES ('Khushi', 'MERN Stack', 'C', 80)''')
cursor.execute('''INSERT INTO student VALUES ('Aarti', 'MERN Stack', 'C', 81)''')

## Display all records
print("The Inserted Records are ....")
data = cursor.execute('''SELECT * FROM student''')
for row in data:
    print(row)

## Commit your changes in Database
connection.commit()
connection.close()