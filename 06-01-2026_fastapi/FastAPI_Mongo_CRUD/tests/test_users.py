from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_user():
    response = client.post(
        "/users",
        json={
            "name": "Test User",
            "age": 25,
            "email": "testuser@gmail.com"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "id" in data
    assert data["name"] == "Test User"
    assert data["age"] == 25
    assert data["email"] == "testuser@gmail.com"


def test_get_users():
    response = client.get("/users")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_update_user():
    # First create a user
    create_response = client.post(
        "/users",
        json={
            "name": "Update User",
            "age": 20,
            "email": "update@gmail.com"
        }
    )

    user_id = create_response.json()["id"]

    # Update that user
    update_response = client.put(
        f"/users/{user_id}",
        json={
            "name": "Updated Name",
            "age": 21,
            "email": "updated@gmail.com"
        }
    )

    assert update_response.status_code == 200
    updated_data = update_response.json()
    assert updated_data["name"] == "Updated Name"


def test_delete_user():
    # First create a user
    create_response = client.post(
        "/users",
        json={
            "name": "Delete User",
            "age": 30,
            "email": "delete@gmail.com"
        }
    )

    user_id = create_response.json()["id"]

    # Delete the user
    delete_response = client.delete(f"/users/{user_id}")

    assert delete_response.status_code == 200
    assert delete_response.json()["message"] == "User deleted successfully"
