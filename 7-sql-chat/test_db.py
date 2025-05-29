from langchain.sql_database import SQLDatabase 
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def configure_db():
    print("DB connection string:") 
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    conn_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
    print(conn_str)
    
    return SQLDatabase(create_engine(conn_str))

db = configure_db()

print(db)
