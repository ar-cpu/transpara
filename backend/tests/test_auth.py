"""
Test cases for authentication endpoints
"""
import pytest
import json
from app import create_app, db
from app.models.user import User


@pytest.fixture
def client():
    """Create test client"""
    app = create_app('testing')
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.session.remove()
            db.drop_all()


@pytest.fixture
def test_user(client):
    """Create a test user"""
    user = User(
        email='test@example.com',
        username='testuser',
        password='TestPassword123'
    )
    db.session.add(user)
    db.session.commit()
    return user


def test_register_success(client):
    """Test successful user registration"""
    response = client.post('/api/v1/auth/register',
                          json={
                              'email': 'newuser@example.com',
                              'username': 'newuser',
                              'password': 'SecurePass123'
                          })
    assert response.status_code == 201
    data = json.loads(response.data)
    assert 'user' in data
    assert data['user']['username'] == 'newuser'


def test_register_duplicate_email(client, test_user):
    """Test registration with duplicate email"""
    response = client.post('/api/v1/auth/register',
                          json={
                              'email': 'test@example.com',
                              'username': 'different',
                              'password': 'SecurePass123'
                          })
    assert response.status_code == 409


def test_register_weak_password(client):
    """Test registration with weak password"""
    response = client.post('/api/v1/auth/register',
                          json={
                              'email': 'test2@example.com',
                              'username': 'test2',
                              'password': 'weak'
                          })
    assert response.status_code == 400


def test_login_success(client, test_user):
    """Test successful login"""
    response = client.post('/api/v1/auth/login',
                          json={
                              'username': 'testuser',
                              'password': 'TestPassword123'
                          })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'access_token' in data
    assert 'refresh_token' in data


def test_login_invalid_credentials(client, test_user):
    """Test login with invalid credentials"""
    response = client.post('/api/v1/auth/login',
                          json={
                              'username': 'testuser',
                              'password': 'WrongPassword'
                          })
    assert response.status_code == 401


def test_get_current_user(client, test_user):
    """Test getting current user information"""
    # Login first
    login_response = client.post('/api/v1/auth/login',
                                 json={
                                     'username': 'testuser',
                                     'password': 'TestPassword123'
                                 })
    token = json.loads(login_response.data)['access_token']

    # Get user info
    response = client.get('/api/v1/auth/me',
                         headers={'Authorization': f'Bearer {token}'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['username'] == 'testuser'
