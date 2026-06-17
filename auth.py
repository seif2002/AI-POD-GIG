"""
AI POD Authentication System
Secure login with session management
"""

import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import re

# =========================================================
# User Database - Simple JSON file (for demo)
# In production, use real database like PostgreSQL
# =========================================================
class AuthManager:
    def __init__(self, users_file="users.json"):
        self.users_file = users_file
        self.init_users_file()
    
    def init_users_file(self):
        """Initialize users file if it doesn't exist"""
        if not os.path.exists(self.users_file):
            # Create default admin user
            default_users = {
                "admin": {
                    "password": self.hash_password("admin123"),
                    "name": "Administrator",
                    "email": "admin@gig-egypt.com",
                    "role": "admin",
                    "department": "IT",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "active": True
                },
                "hr_manager": {
                    "password": self.hash_password("hr123"),
                    "name": "HR Manager",
                    "email": "hr@gig-egypt.com",
                    "role": "hr",
                    "department": "HR",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "active": True
                },
                "employee": {
                    "password": self.hash_password("emp123"),
                    "name": "Employee User",
                    "email": "employee@gig-egypt.com",
                    "role": "employee",
                    "department": "General",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "active": True
                }
            }
            with open(self.users_file, 'w') as f:
                json.dump(default_users, f, indent=2)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user"""
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                user = users[username]
                if user['password'] == self.hash_password(password) and user['active']:
                    # Update last login
                    user['last_login'] = datetime.now().isoformat()
                    users[username] = user
                    with open(self.users_file, 'w') as f:
                        json.dump(users, f, indent=2)
                    
                    # Return user info without password
                    user_info = user.copy()
                    del user_info['password']
                    user_info['username'] = username
                    return user_info
            
            return None
        except Exception as e:
            st.error(f"Authentication error: {e}")
            return None
    
    def create_user(self, username: str, password: str, name: str, email: str, 
                   role: str = "employee", department: str = "General") -> bool:
        """Create new user"""
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                return False
            
            users[username] = {
                "password": self.hash_password(password),
                "name": name,
                "email": email,
                "role": role,
                "department": department,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "active": True
            }
            
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Error creating user: {e}")
            return False
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users and users[username]['password'] == self.hash_password(old_password):
                users[username]['password'] = self.hash_password(new_password)
                
                with open(self.users_file, 'w') as f:
                    json.dump(users, f, indent=2)
                
                return True
            return False
        except Exception as e:
            st.error(f"Error changing password: {e}")
            return False

# =========================================================
# Session Management
# =========================================================
def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None
    if 'session_timeout' not in st.session_state:
        st.session_state.session_timeout = 480  # 8 hours in minutes

def login_required(func):
    """Decorator to require login for pages"""
    def wrapper(*args, **kwargs):
        if not st.session_state.get('authenticated', False):
            show_login_page()
            return
        # Check session timeout
        if st.session_state.login_time:
            elapsed = datetime.now() - datetime.fromisoformat(st.session_state.login_time)
            if elapsed.total_seconds() > st.session_state.session_timeout * 60:
                st.session_state.authenticated = False
                st.session_state.user = None
                st.warning("Session expired. Please login again.")
                show_login_page()
                return
        return func(*args, **kwargs)
    return wrapper

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.login_time = None
    st.rerun()

# =========================================================
# Login UI
# =========================================================
def show_login_page():
    """Display login page"""
    st.markdown("""
    <style>
        /* Login page styling */
        .login-container {
            max-width: 400px;
            margin: 5rem auto;
            padding: 2.5rem;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            border: 1px solid #E5E7EB;
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .login-logo {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .login-title {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .login-subtitle {
            color: #6B7280;
            font-size: 0.95rem;
        }
        
        .login-divider {
            margin: 1.5rem 0;
            border-top: 1px solid #E5E7EB;
        }
        
        .demo-credentials {
            background: #F3F4F6;
            padding: 1rem;
            border-radius: 12px;
            font-size: 0.9rem;
            color: #4B5563;
            margin-top: 1.5rem;
        }
        
        .demo-credentials p {
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .demo-credentials ul {
            margin: 0;
            padding-left: 1.2rem;
        }
        
        .demo-credentials li {
            margin-bottom: 0.3rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <div class="login-title">AI POD</div>
                <div class="login-subtitle">GIG EGYPT LIFE TAKAFUL</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                login_button = st.form_submit_button("Login", use_container_width=True, type="primary")
        
        if login_button:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                auth = AuthManager()
                user = auth.authenticate(username, password)
                
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.login_time = datetime.now().isoformat()
                    st.success(f"Welcome back, {user['name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        # Demo credentials
        st.markdown("""
        <div class="demo-credentials">
            <p>Demo Credentials:</p>
            <ul>
                <li><strong>admin</strong> / admin123 (Administrator)</li>
                <li><strong>hr_manager</strong> / hr123 (HR Manager)</li>
                <li><strong>employee</strong> / emp123 (Employee)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# User Profile & Settings
# =========================================================
def show_user_profile():
    """Display user profile in sidebar"""
    if st.session_state.authenticated and st.session_state.user:
        user = st.session_state.user
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
            padding: 1.2rem;
            border-radius: 16px;
            margin: 1rem 0;
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="
                    background: #3B82F6;
                    width: 48px;
                    height: 48px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 1.5rem;
                    font-weight: 600;
                ">
                    {user['name'][0].upper()}
                </div>
                <div>
                    <div style="font-weight: 700; color: #1E3A8A;">{user['name']}</div>
                    <div style="font-size: 0.85rem; color: #4B5563;">@{user['username']}</div>
                    <div style="font-size: 0.8rem; color: #6B7280; margin-top: 4px;">
                        <span style="background: #3B82F6; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;">
                            {user['role'].upper()}
                        </span>
                        <span style="margin-left: 8px;">{user['department']}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Logout button
        if st.button("Logout", use_container_width=True, type="secondary"):
            logout()

def check_permission(required_roles=None):
    """Check if user has required role"""
    if not st.session_state.get('authenticated', False):
        return False
    
    if required_roles is None:
        return True
    
    user_role = st.session_state.user.get('role', 'employee')
    return user_role in required_roles
