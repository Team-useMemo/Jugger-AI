import os
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/jugger")
client = AsyncIOMotorClient(MONGO_URL)
db = client.get_default_database()

async def get_user_categories(user_uuid: str) -> List[Dict]:
    # MongoDB에서 user_uuid에 해당하는 카테고리 목록을 조회
    categories_cursor = db["category"].find({"user_uuid": user_uuid})
    return await categories_cursor.to_list(length=None)

async def user_exists(user_uuid: str) -> bool:
    user_doc = await db["user"].find_one({"uuid" : user_uuid})
    return user_doc is not None