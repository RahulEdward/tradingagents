"""
Progress log handler
Forwards module completion messages from the logging system to progress trackers
"""


import logging
import threading
from typing import Dict, Optional

class ProgressLogHandler(logging.Handler):
    """
    Custom log handler that forwards module start/completion messages to progress trackers
    """
    
    # Class-level tracker registry
    _trackers: Dict[str, 'AsyncProgressTracker'] = {}
    _lock = threading.Lock()
    
    @classmethod
    def register_tracker(cls, analysis_id: str, tracker):
        """Register progress tracker"""
        try:
            with cls._lock:
                cls._trackers[analysis_id] = tracker
            # Print outside the lock to avoid deadlock
            print(f"📊 [Progress Integration] Registered tracker: {analysis_id}")
        except Exception as e:
            print(f"❌ [Progress Integration] Failed to register tracker: {e}")

    @classmethod
    def unregister_tracker(cls, analysis_id: str):
        """Unregister progress tracker"""
        try:
            removed = False
            with cls._lock:
                if analysis_id in cls._trackers:
                    del cls._trackers[analysis_id]
                    removed = True
            # Print outside the lock to avoid deadlock
            if removed:
                print(f"📊 [Progress Integration] Unregistered tracker: {analysis_id}")
        except Exception as e:
            print(f"❌ [Progress Integration] Failed to unregister tracker: {e}")
    
    def emit(self, record):
        """Handle log record"""
        try:
            message = record.getMessage()
            
            # Only handle module start and completion messages
            if "[Module Start]" in message or "[Module Completed]" in message:
                # Try to extract stock symbol from message to match analysis
                stock_symbol = self._extract_stock_symbol(message)
                
                # Find matching tracker (reduce lock holding time)
                trackers_copy = {}
                with self._lock:
                    trackers_copy = self._trackers.copy()

                # Handle tracker updates outside the lock
                for analysis_id, tracker in trackers_copy.items():
                    # Simple matching: if tracker exists and status is running, update it
                    if hasattr(tracker, 'progress_data') and tracker.progress_data.get('status') == 'running':
                        try:
                            tracker.update_progress(message)
                            print(f"📊 [Progress Integration] Forwarded message to {analysis_id}: {message[:50]}...")
                            break  # Only update the first matching tracker
                        except Exception as e:
                            print(f"❌ [Progress Integration] Update failed: {e}")
                        
        except Exception as e:
            # Don't let log handler errors affect the main program
            print(f"❌ [Progress Integration] Log handling error: {e}")
    
    def _extract_stock_symbol(self, message: str) -> Optional[str]:
        """Extract stock symbol from message"""
        import re
        
        # Try to match "Stock: XXXXX" format
        match = re.search(r'Stock:\s*([A-Za-z0-9]+)', message)
        if match:
            return match.group(1)
        
        return None

# Global log handler instance
_progress_handler = None

def setup_progress_log_integration():
    """Setup progress log integration"""
    global _progress_handler
    
    if _progress_handler is None:
        _progress_handler = ProgressLogHandler()
        _progress_handler.setLevel(logging.INFO)
        
        # Add to tools logger (module completion messages come from here)
        tools_logger = logging.getLogger('tools')
        tools_logger.addHandler(_progress_handler)
        
        print("✅ [Progress Integration] Log handler setup complete")
    
    return _progress_handler

def register_analysis_tracker(analysis_id: str, tracker):
    """Register analysis tracker"""
    handler = setup_progress_log_integration()
    ProgressLogHandler.register_tracker(analysis_id, tracker)

def unregister_analysis_tracker(analysis_id: str):
    """Unregister analysis tracker"""
    ProgressLogHandler.unregister_tracker(analysis_id)
