"""
User Authentication Manager
Handles user login, permission verification and other functions
Supports frontend cache login state, automatically expires after 10 minutes of inactivity
"""

import streamlit as st
import hashlib
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

# Import logging module
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('auth')

# Import user activity logger
try:
    from .user_activity_logger import user_activity_logger
except ImportError:
    user_activity_logger = None
    logger.warning("⚠️ User activity logger import failed")

class AuthManager:
    """User Authentication Manager"""
    
    def __init__(self):
        self.users_file = Path(__file__).parent.parent / "config" / "users.json"
        self.session_timeout = 600  # 10 minutes timeout
        self._ensure_users_file()
    
    def _ensure_users_file(self):
        """Ensure user configuration file exists"""
        self.users_file.parent.mkdir(exist_ok=True)
        
        if not self.users_file.exists():
            # Create default user configuration
            default_users = {
                "admin": {
                    "password_hash": self._hash_password("admin123"),
                    "role": "admin",
                    "permissions": ["analysis", "config", "admin"],
                    "created_at": time.time()
                },
                "user": {
                    "password_hash": self._hash_password("user123"),
                    "role": "user", 
                    "permissions": ["analysis"],
                    "created_at": time.time()
                }
            }
            
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(default_users, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ User authentication system initialization completed")
            logger.info(f"📁 User configuration file: {self.users_file}")
    
    def _inject_auth_cache_js(self):
        """Inject frontend authentication cache JavaScript code"""
        js_code = """
        <script>
        // Authentication cache management
        window.AuthCache = {
            // Save login state to localStorage
            saveAuth: function(userInfo) {
                const authData = {
                    userInfo: userInfo,
                    loginTime: Date.now(),
                    lastActivity: Date.now()
                };
                localStorage.setItem('tradingagents_auth', JSON.stringify(authData));
                console.log('✅ Login state saved to frontend cache');
            },
            
            // Get login state from localStorage
            getAuth: function() {
                try {
                    const authData = localStorage.getItem('tradingagents_auth');
                    if (!authData) return null;
                    
                    const data = JSON.parse(authData);
                    const now = Date.now();
                    const timeout = 10 * 60 * 1000; // 10 minutes
                    
                    // Check if timeout
                    if (now - data.lastActivity > timeout) {
                        this.clearAuth();
                        console.log('⏰ Login state expired, automatically cleared');
                        return null;
                    }
                    
                    // Update last activity time
                    data.lastActivity = now;
                    localStorage.setItem('tradingagents_auth', JSON.stringify(data));
                    
                    return data.userInfo;
                } catch (e) {
                    console.error('❌ Failed to read login state:', e);
                    this.clearAuth();
                    return null;
                }
            },
            
            // Clear login state
            clearAuth: function() {
                localStorage.removeItem('tradingagents_auth');
                console.log('🧹 Login state cleared');
            },
            
            // Update activity time
            updateActivity: function() {
                const authData = localStorage.getItem('tradingagents_auth');
                if (authData) {
                    try {
                        const data = JSON.parse(authData);
                        data.lastActivity = Date.now();
                        localStorage.setItem('tradingagents_auth', JSON.stringify(data));
                    } catch (e) {
                        console.error('❌ Failed to update activity time:', e);
                    }
                }
            }
        };
        
        // Listen to user activity, update last activity time
        ['click', 'keypress', 'scroll', 'mousemove'].forEach(event => {
            document.addEventListener(event, function() {
                window.AuthCache.updateActivity();
            }, { passive: true });
        });
        
        // Check login state when page loads
        document.addEventListener('DOMContentLoaded', function() {
            const authInfo = window.AuthCache.getAuth();
            if (authInfo) {
                console.log('🔄 Restore login state from frontend cache:', authInfo.username);
                // Notify Streamlit to restore login state
                window.parent.postMessage({
                    type: 'restore_auth',
                    userInfo: authInfo
                }, '*');
            }
        });
        </script>
        """
        st.components.v1.html(js_code, height=0)
    
    def _hash_password(self, password: str) -> str:
        """Password hash"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users(self) -> Dict:
        """Load user configuration"""
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load user configuration: {e}")
            return {}
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """
        User authentication
        
        Args:
            username: Username
            password: Password
            
        Returns:
            (Authentication success, User info)
        """
        users = self._load_users()
        
        if username not in users:
            logger.warning(f"⚠️ User does not exist: {username}")
            # Record login failure
            if user_activity_logger:
                user_activity_logger.log_login(username, False, "User does not exist")
            return False, None
        
        user_info = users[username]
        password_hash = self._hash_password(password)
        
        if password_hash == user_info["password_hash"]:
            logger.info(f"✅ User login successful: {username}")
            # Record login success
            if user_activity_logger:
                user_activity_logger.log_login(username, True)
            return True, {
                "username": username,
                "role": user_info["role"],
                "permissions": user_info["permissions"]
            }
        else:
            logger.warning(f"⚠️ Password incorrect: {username}")
            # Record login failure
            if user_activity_logger:
                user_activity_logger.log_login(username, False, "Password incorrect")
            return False, None
    
    def check_permission(self, permission: str) -> bool:
        """
        Check current user permissions
        
        Args:
            permission: Permission name
            
        Returns:
            Whether has permission
        """
        if not self.is_authenticated():
            return False
        
        user_info = st.session_state.get('user_info', {})
        permissions = user_info.get('permissions', [])
        
        return permission in permissions
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        # First check authentication status in session_state
        authenticated = st.session_state.get('authenticated', False)
        login_time = st.session_state.get('login_time', 0)
        current_time = time.time()
        
        logger.debug(f"🔍 [Auth Check] authenticated: {authenticated}, login_time: {login_time}, current_time: {current_time}")
        
        if authenticated:
            # Check session timeout
            time_elapsed = current_time - login_time
            logger.debug(f"🔍 [Auth Check] Session duration: {time_elapsed:.1f}s, timeout limit: {self.session_timeout}s")
            
            if time_elapsed > self.session_timeout:
                logger.info(f"⏰ Session timeout, auto logout (elapsed time: {time_elapsed:.1f}s)")
                self.logout()
                return False
            
            logger.debug(f"✅ [Auth Check] User authenticated and not timed out")
            return True
        
        logger.debug(f"❌ [Auth Check] User not authenticated")
        return False
    
    def login(self, username: str, password: str) -> bool:
        """
        User login
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Whether login is successful
        """
        success, user_info = self.authenticate(username, password)
        
        if success:
            st.session_state.authenticated = True
            st.session_state.user_info = user_info
            st.session_state.login_time = time.time()
            
            # Save to frontend cache - use format compatible with frontend JavaScript
            current_time_ms = int(time.time() * 1000)  # Convert to milliseconds
            auth_data = {
                "userInfo": user_info,  # Use userInfo instead of user_info
                "loginTime": time.time(),
                "lastActivity": current_time_ms,  # Add lastActivity field
                "authenticated": True
            }
            
            save_to_cache_js = f"""
            <script>
            console.log('🔐 Saving authentication data to localStorage');
            try {{
                const authData = {json.dumps(auth_data)};
                localStorage.setItem('tradingagents_auth', JSON.stringify(authData));
                console.log('✅ Authentication data saved to localStorage:', authData);
            }} catch (e) {{
                console.error('❌ Failed to save authentication data:', e);
            }}
            </script>
            """
            st.components.v1.html(save_to_cache_js, height=0)
            
            logger.info(f"✅ User {username} login successful, saved to frontend cache")
            return True
        else:
            st.session_state.authenticated = False
            st.session_state.user_info = None
            return False
    
    def logout(self):
        """User logout"""
        username = st.session_state.get('user_info', {}).get('username', 'unknown')
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.session_state.login_time = None
        
        # Clear frontend cache
        clear_cache_js = """
        <script>
        console.log('🚪 Clear authentication data');
        try {
            localStorage.removeItem('tradingagents_auth');
            localStorage.removeItem('tradingagents_last_activity');
            console.log('✅ Authentication data cleared');
        } catch (e) {
            console.error('❌ Failed to clear authentication data:', e);
        }
        </script>
        """
        st.components.v1.html(clear_cache_js, height=0)
        
        logger.info(f"✅ User {username} logged out, frontend cache cleared")
        
        # Record logout activity
        if user_activity_logger:
            user_activity_logger.log_logout(username)
    
    def restore_from_cache(self, user_info: Dict, login_time: float = None) -> bool:
        """
        Restore login state from frontend cache
        
        Args:
            user_info: User information
            login_time: Original login time, use current time if None
            
        Returns:
            Whether restoration is successful
        """
        try:
            # Validate user information validity
            username = user_info.get('username')
            if not username:
                logger.warning(f"⚠️ Restoration failed: No username in user info")
                return False
            
            # Check if user still exists
            users = self._load_users()
            if username not in users:
                logger.warning(f"⚠️ Attempting to restore non-existent user: {username}")
                return False
            
            # Restore login state, use original login time or current time
            restore_time = login_time if login_time is not None else time.time()
            
            st.session_state.authenticated = True
            st.session_state.user_info = user_info
            st.session_state.login_time = restore_time
            
            logger.info(f"✅ Restored login state for user {username} from frontend cache")
            logger.debug(f"🔍 [Restore State] login_time: {restore_time}, current_time: {time.time()}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restore login state from frontend cache: {e}")
            return False
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current user information"""
        if self.is_authenticated():
            return st.session_state.get('user_info')
        return None
    
    def require_permission(self, permission: str) -> bool:
        """
        Require specific permission, show error message if no permission
        
        Args:
            permission: Permission name
            
        Returns:
            Whether has permission
        """
        if not self.check_permission(permission):
            st.error(f"❌ You do not have '{permission}' permission, please contact administrator")
            return False
        return True

# Global authentication manager instance
auth_manager = AuthManager()