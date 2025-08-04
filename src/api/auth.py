"""
Authentication and authorization for the API.
"""

import os
import hashlib
import secrets
import logging
from typing import Optional, Dict, List
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException, status
import jwt

logger = logging.getLogger(__name__)

# Default API key for development (should be changed in production)
DEFAULT_API_KEY = "synthetic-data-api-key-12345"

class APIKeyManager:
    """Manages API keys and authentication."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.jwt_secret = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.token_expire_hours = 24
        
        # Load API keys from environment or use default
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment variables."""
        # Try to load from environment
        api_keys_env = os.getenv("API_KEYS")
        
        if api_keys_env:
            try:
                # Format: "key1:name1:permissions,key2:name2:permissions"
                for key_data in api_keys_env.split(","):
                    parts = key_data.strip().split(":")
                    if len(parts) >= 2:
                        key, name = parts[0], parts[1]
                        permissions = parts[2].split(";") if len(parts) > 2 else ["read", "write"]
                        self.api_keys[key] = {
                            "name": name,
                            "permissions": permissions,
                            "created_at": datetime.now(timezone.utc),
                            "last_used": None,
                            "usage_count": 0
                        }
                logger.info(f"Loaded {len(self.api_keys)} API keys from environment")
            except Exception as e:
                logger.error(f"Error parsing API keys from environment: {e}")
                self._create_default_key()
        else:
            self._create_default_key()
    
    def _create_default_key(self):
        """Create default API key for development."""
        self.api_keys[DEFAULT_API_KEY] = {
            "name": "default-dev-key",
            "permissions": ["read", "write", "admin"],
            "created_at": datetime.now(timezone.utc),
            "last_used": None,
            "usage_count": 0
        }
        logger.warning("Using default API key for development. Change in production!")
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key."""
        if api_key in self.api_keys:
            # Update usage statistics
            self.api_keys[api_key]["last_used"] = datetime.now(timezone.utc)
            self.api_keys[api_key]["usage_count"] += 1
            return True
        return False
    
    def get_key_info(self, api_key: str) -> Optional[Dict]:
        """Get information about an API key."""
        return self.api_keys.get(api_key)
    
    def has_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission."""
        key_info = self.get_key_info(api_key)
        if not key_info:
            return False
        return permission in key_info.get("permissions", [])
    
    def create_api_key(self, name: str, permissions: List[str]) -> str:
        """Create a new API key."""
        api_key = f"sdk_{secrets.token_urlsafe(32)}"
        self.api_keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "created_at": datetime.now(timezone.utc),
            "last_used": None,
            "usage_count": 0
        }
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False
    
    def list_api_keys(self) -> Dict[str, Dict]:
        """List all API keys (without the actual keys)."""
        return {
            key[:8] + "..." + key[-4:]: {
                "name": info["name"],
                "permissions": info["permissions"],
                "created_at": info["created_at"],
                "last_used": info["last_used"],
                "usage_count": info["usage_count"]
            }
            for key, info in self.api_keys.items()
        }
    
    def create_jwt_token(self, api_key: str, additional_claims: Optional[Dict] = None) -> str:
        """Create a JWT token for an API key."""
        key_info = self.get_key_info(api_key)
        if not key_info:
            raise ValueError("Invalid API key")
        
        payload = {
            "api_key_hash": hashlib.sha256(api_key.encode()).hexdigest()[:16],
            "name": key_info["name"],
            "permissions": key_info["permissions"],
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=self.token_expire_hours)
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Dict:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

# Global API key manager
api_key_manager = APIKeyManager()

async def verify_api_key(credentials: str) -> str:
    """Verify API key from request."""
    # Remove 'Bearer ' prefix if present
    api_key = credentials.replace("Bearer ", "").strip()
    
    # Try JWT token first
    if api_key.startswith("eyJ"):  # JWT tokens typically start with this
        try:
            payload = api_key_manager.verify_jwt_token(api_key)
            return payload["name"]
        except HTTPException:
            pass  # Fall through to API key validation
    
    # Validate as API key
    if not api_key_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    key_info = api_key_manager.get_key_info(api_key)
    return key_info["name"]

async def require_permission(credentials: str, permission: str) -> str:
    """Verify API key and check for specific permission."""
    api_key = credentials.replace("Bearer ", "").strip()
    
    # JWT token handling
    if api_key.startswith("eyJ"):
        payload = api_key_manager.verify_jwt_token(api_key)
        if permission not in payload.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return payload["name"]
    
    # API key handling
    if not api_key_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    if not api_key_manager.has_permission(api_key, permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required: {permission}"
        )
    
    key_info = api_key_manager.get_key_info(api_key)
    return key_info["name"]

def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager."""
    return api_key_manager

class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 60):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, api_key: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=self.window_minutes)
        
        # Clean old requests
        if api_key in self.requests:
            self.requests[api_key] = [
                req_time for req_time in self.requests[api_key]
                if req_time > window_start
            ]
        else:
            self.requests[api_key] = []
        
        # Check limit
        if len(self.requests[api_key]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[api_key].append(now)
        return True
    
    def get_remaining_requests(self, api_key: str) -> int:
        """Get remaining requests in current window."""
        if api_key not in self.requests:
            return self.max_requests
        
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=self.window_minutes)
        
        current_requests = [
            req_time for req_time in self.requests[api_key]
            if req_time > window_start
        ]
        
        return max(0, self.max_requests - len(current_requests))

# Global rate limiter
rate_limiter = RateLimiter()

async def check_rate_limit(credentials: str):
    """Check rate limit for API key."""
    api_key = credentials.replace("Bearer ", "").strip()
    
    if not rate_limiter.is_allowed(api_key):
        remaining = rate_limiter.get_remaining_requests(api_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again later. Remaining: {remaining}",
            headers={"X-RateLimit-Remaining": str(remaining)}
        )

def create_admin_key() -> str:
    """Create an admin API key for setup."""
    return api_key_manager.create_api_key(
        name="admin-setup",
        permissions=["read", "write", "admin"]
    )