"""
User Activity Logger
Records various user operations in the system and saves them to independent log files
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import streamlit as st
from dataclasses import dataclass, asdict
import threading
import os

# Import logging module
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('user_activity')

@dataclass
class UserActivity:
    """User activity record"""
    timestamp: float
    username: str
    user_role: str
    action_type: str
    action_name: str
    details: Dict[str, Any]
    session_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    page_url: Optional[str] = None
    duration_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None

class UserActivityLogger:
    """User activity logger"""
    
    def __init__(self):
        self.activity_dir = Path(__file__).parent.parent / "data" / "user_activities"
        self.activity_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread lock to ensure safe file writing
        self._lock = threading.Lock()
        
        # Activity type definitions
        self.activity_types = {
            "auth": "Authentication related",
            "analysis": "Stock analysis",
            "config": "Configuration management", 
            "navigation": "Page navigation",
            "data_export": "Data export",
            "user_management": "User management",
            "system": "System operations"
        }
        
        logger.info(f"✅ User activity logger initialized successfully")
        logger.info(f"📁 Activity log directory: {self.activity_dir}")
    
    def _get_activity_file_path(self, date: str = None) -> Path:
        """Get activity log file path"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return self.activity_dir / f"user_activities_{date}.jsonl"
    
    def _get_session_id(self) -> str:
        """Get session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"session_{int(time.time())}_{id(st.session_state)}"
        return st.session_state.session_id
    
    def _get_user_info(self) -> Dict[str, str]:
        """Get current user information"""
        user_info = st.session_state.get('user_info')
        if user_info is None:
            user_info = {}
        return {
            "username": user_info.get('username', 'anonymous'),
            "role": user_info.get('role', 'guest')
        }
    
    def _get_request_info(self) -> Dict[str, Optional[str]]:
        """Get request information"""
        try:
            # Try to get request information (may be limited in Streamlit)
            headers = st.context.headers if hasattr(st.context, 'headers') else {}
            return {
                "ip_address": headers.get('x-forwarded-for', headers.get('remote-addr')),
                "user_agent": headers.get('user-agent'),
                "page_url": st.session_state.get('current_page', 'unknown')
            }
        except:
            return {
                "ip_address": None,
                "user_agent": None, 
                "page_url": None
            }
    
    def log_activity(self, 
                    action_type: str,
                    action_name: str,
                    details: Dict[str, Any] = None,
                    success: bool = True,
                    error_message: str = None,
                    duration_ms: int = None) -> None:
        """
        Log user activity
        
        Args:
            action_type: Activity type (auth, analysis, config, navigation, etc.)
            action_name: Activity name
            details: Activity details
            success: Whether the operation was successful
            error_message: Error message (if any)
            duration_ms: Operation duration (milliseconds)
        """
        try:
            user_info = self._get_user_info()
            request_info = self._get_request_info()
            
            activity = UserActivity(
                timestamp=time.time(),
                username=user_info["username"],
                user_role=user_info["role"],
                action_type=action_type,
                action_name=action_name,
                details=details or {},
                session_id=self._get_session_id(),
                ip_address=request_info["ip_address"],
                user_agent=request_info["user_agent"],
                page_url=request_info["page_url"],
                duration_ms=duration_ms,
                success=success,
                error_message=error_message
            )
            
            self._write_activity(activity)
            
        except Exception as e:
            logger.error(f"❌ Failed to log user activity: {e}")
    
    def _write_activity(self, activity: UserActivity) -> None:
        """Write activity record to file"""
        with self._lock:
            try:
                activity_file = self._get_activity_file_path()
                
                # Convert to JSON format
                activity_dict = asdict(activity)
                activity_dict['datetime'] = datetime.fromtimestamp(activity.timestamp).isoformat()
                
                # Append write in JSONL format
                with open(activity_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(activity_dict, ensure_ascii=False) + '\n')
                
            except Exception as e:
                logger.error(f"❌ Failed to write activity record: {e}")
    
    def log_login(self, username: str, success: bool, error_message: str = None) -> None:
        """Log login activity"""
        self.log_activity(
            action_type="auth",
            action_name="user_login",
            details={"username": username},
            success=success,
            error_message=error_message
        )
    
    def log_logout(self, username: str) -> None:
        """Log logout activity"""
        self.log_activity(
            action_type="auth",
            action_name="user_logout",
            details={"username": username}
        )
    
    def log_analysis_request(self, stock_code: str, analysis_type: str, success: bool = True, 
                           duration_ms: int = None, error_message: str = None) -> None:
        """Log stock analysis request"""
        self.log_activity(
            action_type="analysis",
            action_name="stock_analysis",
            details={
                "stock_code": stock_code,
                "analysis_type": analysis_type
            },
            success=success,
            duration_ms=duration_ms,
            error_message=error_message
        )
    
    def log_page_visit(self, page_name: str, page_params: Dict[str, Any] = None) -> None:
        """Log page visit"""
        self.log_activity(
            action_type="navigation",
            action_name="page_visit",
            details={
                "page_name": page_name,
                "page_params": page_params or {}
            }
        )
    
    def log_config_change(self, config_type: str, changes: Dict[str, Any]) -> None:
        """Log configuration change"""
        self.log_activity(
            action_type="config",
            action_name="config_update",
            details={
                "config_type": config_type,
                "changes": changes
            }
        )
    
    def log_data_export(self, export_type: str, data_info: Dict[str, Any], 
                       success: bool = True, error_message: str = None) -> None:
        """Log data export"""
        self.log_activity(
            action_type="data_export",
            action_name="export_data",
            details={
                "export_type": export_type,
                "data_info": data_info
            },
            success=success,
            error_message=error_message
        )
    
    def log_user_management(self, operation: str, target_user: str, 
                          success: bool = True, error_message: str = None) -> None:
        """Log user management operation"""
        self.log_activity(
            action_type="user_management",
            action_name=operation,
            details={"target_user": target_user},
            success=success,
            error_message=error_message
        )
    
    def get_user_activities(self, username: str = None, 
                          start_date: datetime = None,
                          end_date: datetime = None,
                          action_type: str = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get user activity records
        
        Args:
            username: Username filter
            start_date: Start date
            end_date: End date  
            action_type: Activity type filter
            limit: Limit on number of records returned
            
        Returns:
            List of activity records
        """
        activities = []
        
        try:
            # Determine the date range to query
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)  # Default to last 7 days
            if end_date is None:
                end_date = datetime.now()
            
            # Iterate through all files in the date range
            current_date = start_date.date()
            end_date_only = end_date.date()
            
            while current_date <= end_date_only:
                date_str = current_date.strftime("%Y-%m-%d")
                activity_file = self._get_activity_file_path(date_str)
                
                if activity_file.exists():
                    activities.extend(self._read_activities_from_file(
                        activity_file, username, action_type, start_date, end_date
                    ))
                
                current_date += timedelta(days=1)
            
            # Sort by timestamp in descending order
            activities.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Apply limit
            return activities[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to get user activity records: {e}")
            return []
    
    def _read_activities_from_file(self, file_path: Path, username: str = None,
                                 action_type: str = None, start_date: datetime = None,
                                 end_date: datetime = None) -> List[Dict[str, Any]]:
        """Read activity records from file"""
        activities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        activity = json.loads(line.strip())
                        
                        # Apply filter conditions
                        if username and activity.get('username') != username:
                            continue
                        
                        if action_type and activity.get('action_type') != action_type:
                            continue
                        
                        activity_time = datetime.fromtimestamp(activity['timestamp'])
                        if start_date and activity_time < start_date:
                            continue
                        if end_date and activity_time > end_date:
                            continue
                        
                        activities.append(activity)
                        
        except Exception as e:
            logger.error(f"❌ Failed to read activity file {file_path}: {e}")
        
        return activities
    
    def get_activity_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get activity statistics
        
        Args:
            days: Number of days for statistics
            
        Returns:
            Statistics dictionary
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        activities = self.get_user_activities(
            start_date=start_date,
            end_date=end_date,
            limit=10000  # Get more records for statistics
        )
        
        # Statistical analysis
        stats = {
            "total_activities": len(activities),
            "unique_users": len(set(a['username'] for a in activities)),
            "activity_types": {},
            "daily_activities": {},
            "user_activities": {},
            "success_rate": 0,
            "average_duration": 0
        }
        
        # Statistics by type
        for activity in activities:
            action_type = activity.get('action_type', 'unknown')
            stats["activity_types"][action_type] = stats["activity_types"].get(action_type, 0) + 1
            
            # Statistics by user
            username = activity.get('username', 'unknown')
            stats["user_activities"][username] = stats["user_activities"].get(username, 0) + 1
            
            # Statistics by date
            date_str = datetime.fromtimestamp(activity['timestamp']).strftime('%Y-%m-%d')
            stats["daily_activities"][date_str] = stats["daily_activities"].get(date_str, 0) + 1
        
        # Success rate statistics
        successful_activities = sum(1 for a in activities if a.get('success', True))
        if activities:
            stats["success_rate"] = successful_activities / len(activities) * 100
        
        # Average duration statistics
        durations = [a.get('duration_ms', 0) for a in activities if a.get('duration_ms')]
        if durations:
            stats["average_duration"] = sum(durations) / len(durations)
        
        return stats
    
    def cleanup_old_activities(self, days_to_keep: int = 90) -> int:
        """
        Clean up old activity records
        
        Args:
            days_to_keep: Number of days to keep
            
        Returns:
            Number of deleted files
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        
        try:
            for activity_file in self.activity_dir.glob("user_activities_*.jsonl"):
                # Extract date from filename
                try:
                    date_str = activity_file.stem.replace("user_activities_", "")
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        activity_file.unlink()
                        deleted_count += 1
                        logger.info(f"🗑️ Deleted old activity record: {activity_file.name}")
                        
                except ValueError:
                    # Incorrect filename format, skip
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Failed to clean up old activity records: {e}")
        
        return deleted_count

# Global user activity logger instance
user_activity_logger = UserActivityLogger()