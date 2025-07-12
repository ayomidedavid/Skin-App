import pymysql

def get_db_connection():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='skin_classification_db',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

def execute_query(query, params=None):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            connection.commit()
            return cursor.fetchall()
    finally:
        connection.close()