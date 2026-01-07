from app.database import collection
from bson import ObjectId

def user_helper(user) -> dict:
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "age": user["age"],
        "email": user["email"]
    }

async def create_user(user: dict):
    new_user = await collection.insert_one(user)
    created_user = await collection.find_one(
        {"_id": new_user.inserted_id}
    )
    return user_helper(created_user)

async def get_users():
    users = []
    async for user in collection.find():
        users.append(user_helper(user))
    return users

async def update_user(id: str, data: dict):
    await collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": data}
    )
    updated_user = await collection.find_one(
        {"_id": ObjectId(id)}
    )
    return user_helper(updated_user)

async def delete_user(id: str):
    await collection.delete_one({"_id": ObjectId(id)})
    return True