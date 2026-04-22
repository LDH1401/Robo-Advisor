import os
import certifi
import json
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Hàm convert dữ liệu Mongo → JSON serializable
def serialize_doc(doc):
    if isinstance(doc, dict):
        return {k: serialize_doc(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [serialize_doc(v) for v in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)
    elif isinstance(doc, datetime):
        return doc.isoformat()
    else:
        return doc

mongo_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME")


ca = certifi.where()
client = MongoClient(mongo_uri, tlsCAFile=ca)
db = client[db_name]

collections = db.list_collection_names()

for col_name in collections:
    print(f"Đang export collection: {col_name}")

    data = list(db[col_name].find())

    data_serialized = [serialize_doc(doc) for doc in data]

    file_name = f"{col_name}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data_serialized, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu {len(data)} records vào {file_name}")