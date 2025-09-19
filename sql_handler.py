import mysql.connector
from shared import log_data

def create_db_connection():
    try:
        mydb = mysql.connector.connect( #* creating db connection
        host = 'bowvnujlck0dscagkyyt-mysql.services.clever-cloud.com',
        user = 'u6aitdvcermeo8c8',
        password = 'mnDwBMvfXR8CUeNeWieP',
        database = 'bowvnujlck0dscagkyyt'       
        )
        return mydb
    except Exception as e:
        print("Error:", str(e))

def add_user_interaction_db(user_id):
    mydb = create_db_connection()
    cursor  = mydb.cursor() #* cursor
    add_query = 'INSERT INTO users (input_dt, output_dt, file_type, no_of_files) VALUES (%s, %s, %s, %s)' #* sql query
    values = (log_data[user_id]['input_dt'], log_data[user_id]['output_dt'], log_data[user_id]['file_type'], log_data[user_id]['no_of_files']) #* getting values from log_data
    cursor.execute(add_query, values) #* executing the query (adding it to db)

    mydb.commit()
    cursor.close()
    mydb.close()

def create_table():
    mydb = create_db_connection()
    cursor  = mydb.cursor()
    cursor.execute("""
    CREATE TABLE users (
        id INT PRIMARY KEY AUTO_INCREMENT,
        input_dt DATETIME,
        output_dt DATETIME, 
        file_type VARCHAR(100), 
        no_of_files INT)
    """)
    mydb.commit()
    cursor.close()
    mydb.close()

if __name__ == "__main__":
    pass
    # create_table()
