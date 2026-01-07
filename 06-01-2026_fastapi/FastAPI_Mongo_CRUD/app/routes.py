from fastapi import APIRouter
from app.schemas import UserCreate
from app.crud import create_user, get_users, update_user, delete_user

router = APIRouter()

@router.post("/users")
async def add_user(user: UserCreate):
    return await create_user(user.dict())

@router.get("/users")
async def read_users():
    return await get_users()

@router.put("/users/{id}")
async def modify_user(id: str, user: UserCreate):
    return await update_user(id, user.dict())

@router.delete("/users/{id}")
async def remove_user(id: str):
    await delete_user(id)
    return {"message": "User deleted successfully"}