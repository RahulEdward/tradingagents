#!/usr/bin/env python3
"""
Async Progress Tracker
Supports both Redis and file storage methods, frontend polls for progress updates
"""

import json
import time
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
from pathlib import Path

# Import logging module
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('async_progress')

def safe_serialize(obj):
    """Safely serialize objects, handling non-serializable types"""
    # Special handling for LangChain message objects
    if hasattr(obj, '__class__') and 'Message' in obj.__class__.__name__:
        try:
            # Try using LangChain's serialization methods
            if hasattr(obj, 'dict'):
                return obj.dict()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            else:
                # Manually extract message content
                return {
                    'type': obj.__class__.__name__,
                    'content': getattr(obj, 'content', str(obj)),
                    'additional_kwargs': getattr(obj, 'additional_kwargs', {}),
                    'response_metadata': getattr(obj, 'response_metadata', {})
                }
        except Exception:
            # If all methods fail, return string representation
            return {
                'type': obj.__class__.__name__,
                'content': str(obj)
            }
    
    if hasattr(obj, 'dict'):
        # Pydantic objects
        try:
            return obj.dict()
        except Exception:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        # Regular objects, convert to dictionary
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                try:
                    json.dumps(value)  # Test if serializable
                    result[key] = value
                except (TypeError, ValueError):
                    result[key] = safe_serialize(value)  # Recursive handling
        return result
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_serialize(value) for key, value in obj.items()}
    else:
        try:
            json.dumps(obj)  # Test if serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)  # Convert to string

class AsyncProgressTracker:
    """Async Progress Tracker"""
    
    def __init__(self, analysis_id: str, analysts: List[str], research_depth: int, llm_provider: str):
        self.analysis_id = analysis_id
        self.analysts = analysts
        self.research_depth = research_depth
        self.llm_provider = llm_provider
        self.start_time = time.time()
        
        # Generate analysis steps
        self.analysis_steps = self._generate_dynamic_steps()
        self.estimated_duration = self._estimate_total_duration()
        
        # Initialize state
        self.current_step = 0
        self.progress_data = {
            'analysis_id': analysis_id,
            'status': 'running',
            'current_step': 0,
            'total_steps': len(self.analysis_steps),
            'progress_percentage': 0.0,
            'current_step_name': self.analysis_steps[0]['name'],
            'current_step_description': self.analysis_steps[0]['description'],
            'elapsed_time': 0.0,
            'estimated_total_time': self.estimated_duration,
            'remaining_time': self.estimated_duration,
            'last_message': 'Preparing to start analysis...',
            'last_update': time.time(),
            'start_time': self.start_time,
            'steps': self.analysis_steps
        }
        
        # Try to initialize Redis, fallback to file if failed
        self.redis_client = None
        self.use_redis = self._init_redis()
        
        if not self.use_redis:
            # Use file storage
            self.progress_file = f"./data/progress_{analysis_id}.json"
            os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        
        # Save initial state
        self._save_progress()
        
        logger.info(f"📊 [Async Progress] Initialization complete: {analysis_id}, Storage method: {'Redis' if self.use_redis else 'File'}")

        # Register to logging system for automatic progress updates
        try:
            from .progress_log_handler import register_analysis_tracker
            import threading

            # Use timeout mechanism to avoid deadlock
            def register_with_timeout():
                try:
                    register_analysis_tracker(self.analysis_id, self)
                    print(f"✅ [Progress Integration] Tracker registered successfully: {self.analysis_id}")
                except Exception as e:
                    print(f"❌ [Progress Integration] Tracker registration failed: {e}")

            # Register in separate thread to avoid blocking main thread
            register_thread = threading.Thread(target=register_with_timeout, daemon=True)
            register_thread.start()
            register_thread.join(timeout=2.0)  # 2 second timeout

            if register_thread.is_alive():
                print(f"⚠️ [Progress Integration] Tracker registration timeout, continuing execution: {self.analysis_id}")

        except ImportError:
            logger.debug("📊 [Async Progress] Log integration not available")
        except Exception as e:
            print(f"❌ [Progress Integration] Tracker registration exception: {e}")
    
    def _init_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            # First check REDIS_ENABLED environment variable
            redis_enabled_raw = os.getenv('REDIS_ENABLED', 'false')
            redis_enabled = redis_enabled_raw.lower()
            logger.info(f"🔍 [Redis Check] REDIS_ENABLED original value='{redis_enabled_raw}' -> processed='{redis_enabled}'")

            if redis_enabled != 'true':
                logger.info(f"📊 [Async Progress] Redis disabled, using file storage")
                return False

            import redis

            # Get Redis configuration from environment variables
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            redis_password = os.getenv('REDIS_PASSWORD', None)
            redis_db = int(os.getenv('REDIS_DB', 0))

            # Create Redis connection
            if redis_password:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    db=redis_db,
                    decode_responses=True
                )
            else:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True
                )

            # Test connection
            self.redis_client.ping()
            logger.info(f"📊 [Async Progress] Redis connection successful: {redis_host}:{redis_port}")
            return True
        except Exception as e:
            logger.warning(f"📊 [Async Progress] Redis connection failed, using file storage: {e}")
            return False
    
    def _generate_dynamic_steps(self) -> List[Dict]:
        """Dynamically generate analysis steps based on number of analysts and research depth"""
        steps = [
            {"name": "📋 Preparation Phase", "description": "Validate stock symbol, check data source availability", "weight": 0.05},
            {"name": "🔧 Environment Check", "description": "Check API key configuration, ensure data retrieval is normal", "weight": 0.02},
            {"name": "💰 Cost Estimation", "description": "Estimate API call costs based on analysis depth", "weight": 0.01},
            {"name": "⚙️ Parameter Setup", "description": "Configure analysis parameters and AI model selection", "weight": 0.02},
            {"name": "🚀 Engine Startup", "description": "Initialize AI analysis engine, prepare to start analysis", "weight": 0.05},
        ]

        # Add dedicated steps for each analyst
        analyst_base_weight = 0.6 / len(self.analysts)  # 60% of time for analyst work
        for analyst in self.analysts:
            analyst_info = self._get_analyst_step_info(analyst)
            steps.append({
                "name": analyst_info["name"],
                "description": analyst_info["description"],
                "weight": analyst_base_weight
            })

        # Add subsequent steps based on research depth
        if self.research_depth >= 2:
            # Standard and deep analysis include analyst debates
            steps.extend([
                {"name": "📈 Bull Perspective", "description": "Analyze investment opportunities and upside potential from optimistic angle", "weight": 0.06},
                {"name": "📉 Bear Perspective", "description": "Analyze investment risks and downside possibilities from cautious angle", "weight": 0.06},
                {"name": "🤝 View Integration", "description": "Integrate bull and bear views to form balanced investment advice", "weight": 0.05},
            ])

        # All depths include trading decisions
        steps.append({"name": "💡 Investment Advice", "description": "Formulate specific buy/sell recommendations based on analysis results", "weight": 0.06})

        if self.research_depth >= 3:
            # Deep analysis includes detailed risk assessment
            steps.extend([
                {"name": "🔥 Aggressive Strategy", "description": "Evaluate high-risk high-return investment strategies", "weight": 0.03},
                {"name": "🛡️ Conservative Strategy", "description": "Evaluate low-risk stable investment strategies", "weight": 0.03},
                {"name": "⚖️ Balanced Strategy", "description": "Evaluate risk-return balanced investment strategies", "weight": 0.03},
                {"name": "🎯 Risk Control", "description": "Develop risk control measures and stop-loss strategies", "weight": 0.04},
            ])
        else:
            # Simplified risk assessment for quick and standard analysis
            steps.append({"name": "⚠️ Risk Warning", "description": "Identify major investment risks and provide risk warnings", "weight": 0.05})

        # Final organization step
        steps.append({"name": "📊 Generate Report", "description": "Organize all analysis results and generate final investment report", "weight": 0.04})

        # Rebalance weights to ensure total equals 1.0
        total_weight = sum(step["weight"] for step in steps)
        for step in steps:
            step["weight"] = step["weight"] / total_weight

        return steps
    
    def _get_analyst_display_name(self, analyst: str) -> str:
        """Get analyst display name (maintain compatibility)"""
        name_map = {
            'market': 'Market Analyst',
            'fundamentals': 'Fundamental Analyst',
            'technical': 'Technical Analyst',
            'sentiment': 'Sentiment Analyst',
            'risk': 'Risk Analyst'
        }
        return name_map.get(analyst, f'{analyst} Analyst')

    def _get_analyst_step_info(self, analyst: str) -> Dict[str, str]:
        """Get analyst step information (name and description)"""
        analyst_info = {
            'market': {
                "name": "📊 Market Analysis",
                "description": "Analyze stock price trends, trading volume, market heat and other market performance"
            },
            'fundamentals': {
                "name": "💼 Fundamental Analysis",
                "description": "Analyze company financial status, profitability, growth and other fundamentals"
            },
            'technical': {
                "name": "📈 Technical Analysis",
                "description": "Analyze candlestick patterns, technical indicators, support/resistance and other technicals"
            },
            'sentiment': {
                "name": "💭 Sentiment Analysis",
                "description": "Analyze market sentiment, investor psychology, public opinion trends"
            },
            'news': {
                "name": "📰 News Analysis",
                "description": "Analyze impact of related news, announcements, industry dynamics on stock price"
            },
            'social_media': {
                "name": "🌐 Social Media",
                "description": "Analyze social media discussions, online heat, retail investor sentiment"
            },
            'risk': {
                "name": "⚠️ Risk Analysis",
                "description": "Identify investment risks, assess risk levels, develop risk control measures"
            }
        }

        return analyst_info.get(analyst, {
            "name": f"🔍 {analyst} Analysis",
            "description": f"Conduct professional analysis related to {analyst}"
        })
    
    def _estimate_total_duration(self) -> float:
        """Estimate total duration based on number of analysts, research depth, model type (seconds)"""
        # Base time (seconds) - environment preparation, configuration, etc.
        base_time = 60
        
        # Actual time per analyst (based on real test data)
        analyst_base_time = {
            1: 120,  # Quick analysis: about 2 minutes per analyst
            2: 180,  # Basic analysis: about 3 minutes per analyst  
            3: 240   # Standard analysis: about 4 minutes per analyst
        }.get(self.research_depth, 180)
        
        analyst_time = len(self.analysts) * analyst_base_time
        
        # Model speed impact (based on actual testing)
        model_multiplier = {
            'dashscope': 1.0,  # Alibaba DashScope moderate speed
            'deepseek': 0.7,   # DeepSeek faster
            'google': 1.3      # Google slower
        }.get(self.llm_provider, 1.0)
        
        # Research depth additional impact (tool call complexity)
        depth_multiplier = {
            1: 0.8,  # Quick analysis, fewer tool calls
            2: 1.0,  # Basic analysis, standard tool calls
            3: 1.3   # Standard analysis, more tool calls and reasoning
        }.get(self.research_depth, 1.0)
        
        total_time = (base_time + analyst_time) * model_multiplier * depth_multiplier
        return total_time
    
    def update_progress(self, message: str, step: Optional[int] = None):
        """Update progress status"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Auto-detect step
        if step is None:
            step = self._detect_step_from_message(message)

        # Update step (prevent regression)
        if step is not None and step >= self.current_step:
            self.current_step = step
            logger.debug(f"📊 [Async Progress] Step advanced to {self.current_step + 1}/{len(self.analysis_steps)}")

        # If completion message, ensure progress is 100%
        if "Analysis Completed" in message or "Analysis Successful" in message or "✅ Analysis Completed" in message:
            self.current_step = len(self.analysis_steps) - 1
            logger.info(f"📊 [Async Progress] Analysis complete, set to final step")

        # Calculate progress
        progress_percentage = self._calculate_weighted_progress() * 100
        remaining_time = self._estimate_remaining_time(progress_percentage / 100, elapsed_time)

        # Update progress data
        current_step_info = self.analysis_steps[self.current_step] if self.current_step < len(self.analysis_steps) else self.analysis_steps[-1]

        # Special handling for tool call messages, update step description but don't change step
        step_description = current_step_info['description']
        if "Tool Call" in message:
            # Extract tool name and update description
            if "get_stock_market_data_unified" in message:
                step_description = "Retrieving market data and technical indicators..."
            elif "get_stock_fundamentals_unified" in message:
                step_description = "Retrieving fundamental data and financial indicators..."
            elif "get_china_stock_data" in message:
                step_description = "Retrieving A-share market data..."
            elif "get_china_fundamentals" in message:
                step_description = "Retrieving A-share fundamental data..."
            else:
                step_description = "Calling analysis tools..."
        elif "Module Start" in message:
            step_description = f"Starting {current_step_info['name']}..."
        elif "Module Completed" in message:
            step_description = f"{current_step_info['name']} completed"

        self.progress_data.update({
            'current_step': self.current_step,
            'progress_percentage': progress_percentage,
            'current_step_name': current_step_info['name'],
            'current_step_description': step_description,
            'elapsed_time': elapsed_time,
            'remaining_time': remaining_time,
            'last_message': message,
            'last_update': current_time,
            'status': 'completed' if progress_percentage >= 100 else 'running'
        })

        # Save to storage
        self._save_progress()

        # Detailed update log
        step_name = current_step_info.get('name', 'Unknown')
        logger.info(f"📊 [Progress Update] {self.analysis_id}: {message[:50]}...")
        logger.debug(f"📊 [Progress Details] Step {self.current_step + 1}/{len(self.analysis_steps)} ({step_name}), Progress {progress_percentage:.1f}%, Time {elapsed_time:.1f}s")
    
    def _detect_step_from_message(self, message: str) -> Optional[int]:
        """Intelligently detect current step based on message content"""
        message_lower = message.lower()

        # Start analysis phase - only match initial start message
        if "🚀 Starting stock analysis" in message:
            return 0
        # Data validation phase
        elif "Validating" in message or "pre-fetching" in message or "data preparation" in message:
            return 0
        # Environment preparation phase
        elif "environment" in message or "api" in message_lower or "key" in message:
            return 1
        # Cost estimation phase
        elif "cost" in message or "Estimated" in message:
            return 2
        # Parameter configuration phase
        elif "Configuring" in message or "parameters" in message:
            return 3
        # Engine initialization phase
        elif "Initializing" in message or "engine" in message:
            return 4
        # Module start log - only advance step on first start
        elif "Module started" in message:
            # Extract analyst type from log, match new step names
            if "market_analyst" in message or "market" in message:
                return self._find_step_by_keyword(["Market Analysis", "Market"])
            elif "fundamentals_analyst" in message or "fundamentals" in message:
                return self._find_step_by_keyword(["Fundamental Analysis", "Fundamental"])
            elif "technical_analyst" in message or "technical" in message:
                return self._find_step_by_keyword(["Technical Analysis", "Technical"])
            elif "sentiment_analyst" in message or "sentiment" in message:
                return self._find_step_by_keyword(["Sentiment Analysis", "Sentiment"])
            elif "news_analyst" in message or "news" in message:
                return self._find_step_by_keyword(["News Analysis", "News"])
            elif "social_media_analyst" in message or "social" in message:
                return self._find_step_by_keyword(["Social Media", "Social"])
            elif "risk_analyst" in message or "risk" in message:
                return self._find_step_by_keyword(["Risk Analysis", "Risk"])
            elif "bull_researcher" in message or "bull" in message:
                return self._find_step_by_keyword(["Bull Perspective", "Bull", "Bullish"])
            elif "bear_researcher" in message or "bear" in message:
                return self._find_step_by_keyword(["Bear Perspective", "Bear", "Bearish"])
            elif "research_manager" in message:
                return self._find_step_by_keyword(["View Integration", "Integration"])
            elif "trader" in message:
                return self._find_step_by_keyword(["Investment Advice", "Advice"])
            elif "risk_manager" in message:
                return self._find_step_by_keyword(["Risk Control", "Control"])
            elif "graph_signal_processing" in message or "signal" in message:
                return self._find_step_by_keyword(["Generate Report", "Report"])
        # Tool call log - don't advance step, only update description
        elif "Tool call" in message:
            # Keep current step, don't advance
            return None
        # Module completion log - advance to next step
        elif "Module completed" in message:
            # When module completes, advance from current step to next step
            # No longer rely on module name, advance based on current progress
            next_step = min(self.current_step + 1, len(self.analysis_steps) - 1)
            logger.debug(f"📊 [Step Advance] Module completed, advancing from step {self.current_step} to step {next_step}")
            return next_step

        return None

    def _find_step_by_keyword(self, keywords) -> Optional[int]:
        """Find step index by keywords"""
        if isinstance(keywords, str):
            keywords = [keywords]

        for i, step in enumerate(self.analysis_steps):
            for keyword in keywords:
                if keyword in step["name"]:
                    return i
        return None

    def _get_next_step(self, keyword: str) -> Optional[int]:
        """Get the next step for the specified step"""
        current_step_index = self._find_step_by_keyword(keyword)
        if current_step_index is not None:
            return min(current_step_index + 1, len(self.analysis_steps) - 1)
        return None

    def _calculate_weighted_progress(self) -> float:
        """Calculate progress based on step weights"""
        if self.current_step >= len(self.analysis_steps):
            return 1.0

        # If it's the last step, return 100%
        if self.current_step == len(self.analysis_steps) - 1:
            return 1.0

        completed_weight = sum(step["weight"] for step in self.analysis_steps[:self.current_step])
        total_weight = sum(step["weight"] for step in self.analysis_steps)

        return min(completed_weight / total_weight, 1.0)
    
    def _estimate_remaining_time(self, progress: float, elapsed_time: float) -> float:
        """Calculate remaining time based on total estimated time"""
        # If progress is complete, remaining time is 0
        if progress >= 1.0:
            return 0.0

        # Use simple and accurate method: total estimated time - elapsed time
        remaining = max(self.estimated_duration - elapsed_time, 0)

        # If already exceeded estimated time, dynamically adjust based on current progress
        if remaining <= 0 and progress > 0:
            # Re-estimate total time based on current progress, then calculate remaining
            estimated_total = elapsed_time / progress
            remaining = max(estimated_total - elapsed_time, 0)

        return remaining
    
    def _save_progress(self):
        """Save progress to storage"""
        try:
            current_step_name = self.progress_data.get('current_step_name', 'Unknown')
            progress_pct = self.progress_data.get('progress_percentage', 0)
            status = self.progress_data.get('status', 'running')

            if self.use_redis:
                # Save to Redis (safe serialization)
                key = f"progress:{self.analysis_id}"
                safe_data = safe_serialize(self.progress_data)
                data_json = json.dumps(safe_data, ensure_ascii=False)
                self.redis_client.setex(key, 3600, data_json)  # 1 hour expiration

                logger.info(f"📊 [Redis Write] {self.analysis_id} -> {status} | {current_step_name} | {progress_pct:.1f}%")
                logger.debug(f"📊 [Redis Details] Key: {key}, Data size: {len(data_json)} bytes")
            else:
                # Save to file (safe serialization)
                safe_data = safe_serialize(self.progress_data)
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(safe_data, f, ensure_ascii=False, indent=2)

                logger.info(f"📊 [File Write] {self.analysis_id} -> {status} | {current_step_name} | {progress_pct:.1f}%")
                logger.debug(f"📊 [File Details] Path: {self.progress_file}")

        except Exception as e:
            logger.error(f"📊 [Async Progress] Save failed: {e}")
            # Try backup storage method
            try:
                if self.use_redis:
                    # Redis failed, try file storage
                    logger.warning(f"📊 [Async Progress] Redis save failed, trying file storage")
                    backup_file = f"./data/progress_{self.analysis_id}.json"
                    os.makedirs(os.path.dirname(backup_file), exist_ok=True)
                    safe_data = safe_serialize(self.progress_data)
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(safe_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"📊 [Backup Storage] File save successful: {backup_file}")
                else:
                    # File storage failed, try simplified data
                    logger.warning(f"📊 [Async Progress] File save failed, trying simplified data")
                    simplified_data = {
                        'analysis_id': self.analysis_id,
                        'status': self.progress_data.get('status', 'unknown'),
                        'progress_percentage': self.progress_data.get('progress_percentage', 0),
                        'last_message': str(self.progress_data.get('last_message', '')),
                        'last_update': self.progress_data.get('last_update', time.time())
                    }
                    backup_file = f"./data/progress_{self.analysis_id}.json"
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(simplified_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"📊 [Backup Storage] Simplified data save successful: {backup_file}")
            except Exception as backup_e:
                logger.error(f"📊 [Async Progress] Backup storage also failed: {backup_e}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress"""
        return self.progress_data.copy()
    
    def mark_completed(self, message: str = "Analysis completed", results: Any = None):
        """Mark analysis as completed"""
        self.update_progress(message)
        self.progress_data['status'] = 'completed'
        self.progress_data['progress_percentage'] = 100.0
        self.progress_data['remaining_time'] = 0.0

        # Save analysis results (safe serialization)
        if results is not None:
            try:
                self.progress_data['raw_results'] = safe_serialize(results)
                logger.info(f"📊 [Async Progress] Save analysis results: {self.analysis_id}")
            except Exception as e:
                logger.warning(f"📊 [Async Progress] Result serialization failed: {e}")
                self.progress_data['raw_results'] = str(results)  # Final fallback

        self._save_progress()
        logger.info(f"📊 [Async Progress] Analysis completed: {self.analysis_id}")

        # Unregister from logging system
        try:
            from .progress_log_handler import unregister_analysis_tracker
            unregister_analysis_tracker(self.analysis_id)
        except ImportError:
            pass
    
    def mark_failed(self, error_message: str):
        """Mark analysis as failed"""
        self.progress_data['status'] = 'failed'
        self.progress_data['last_message'] = f"Analysis failed: {error_message}"
        self.progress_data['last_update'] = time.time()
        self._save_progress()
        logger.error(f"📊 [Async Progress] Analysis failed: {self.analysis_id}, Error: {error_message}")

        # Unregister from logging system
        try:
            from .progress_log_handler import unregister_analysis_tracker
            unregister_analysis_tracker(self.analysis_id)
        except ImportError:
            pass

def get_progress_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Get progress by analysis ID"""
    try:
        # Check REDIS_ENABLED environment variable
        redis_enabled = os.getenv('REDIS_ENABLED', 'false').lower() == 'true'

        # If Redis is enabled, try Redis first
        if redis_enabled:
            try:
                import redis

                # Get Redis configuration from environment variables
                redis_host = os.getenv('REDIS_HOST', 'localhost')
                redis_port = int(os.getenv('REDIS_PORT', 6379))
                redis_password = os.getenv('REDIS_PASSWORD', None)
                redis_db = int(os.getenv('REDIS_DB', 0))

                # Create Redis connection
                if redis_password:
                    redis_client = redis.Redis(
                        host=redis_host,
                        port=redis_port,
                        password=redis_password,
                        db=redis_db,
                        decode_responses=True
                    )
                else:
                    redis_client = redis.Redis(
                        host=redis_host,
                        port=redis_port,
                        db=redis_db,
                        decode_responses=True
                    )

                key = f"progress:{analysis_id}"
                data = redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.debug(f"📊 [Async Progress] Redis read failed: {e}")

        # Try file
        progress_file = f"./data/progress_{analysis_id}.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        return None
    except Exception as e:
        logger.error(f"📊 [Async Progress] Get progress failed: {analysis_id}, Error: {e}")
        return None

def format_time(seconds: float) -> str:
    """Format time display"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_latest_analysis_id() -> Optional[str]:
    """Get the latest analysis ID"""
    try:
        # Check REDIS_ENABLED environment variable
        redis_enabled = os.getenv('REDIS_ENABLED', 'false').lower() == 'true'

        # If Redis is enabled, try to get from Redis first
        if redis_enabled:
            try:
                import redis

                # Get Redis configuration from environment variables
                redis_host = os.getenv('REDIS_HOST', 'localhost')
                redis_port = int(os.getenv('REDIS_PORT', 6379))
                redis_password = os.getenv('REDIS_PASSWORD', None)
                redis_db = int(os.getenv('REDIS_DB', 0))

                # Create Redis connection
                if redis_password:
                    redis_client = redis.Redis(
                        host=redis_host,
                        port=redis_port,
                        password=redis_password,
                        db=redis_db,
                        decode_responses=True
                    )
                else:
                    redis_client = redis.Redis(
                        host=redis_host,
                        port=redis_port,
                        db=redis_db,
                        decode_responses=True
                    )

                # Get all progress keys
                keys = redis_client.keys("progress:*")
                if not keys:
                    return None

                # Get data for each key, find the latest
                latest_time = 0
                latest_id = None

                for key in keys:
                    try:
                        data = redis_client.get(key)
                        if data:
                            progress_data = json.loads(data)
                            last_update = progress_data.get('last_update', 0)
                            if last_update > latest_time:
                                latest_time = last_update
                                # Extract analysis_id from key name (remove "progress:" prefix)
                                latest_id = key.replace('progress:', '')
                    except Exception:
                        continue

                if latest_id:
                    logger.info(f"📊 [Restore Analysis] Found latest analysis ID: {latest_id}")
                    return latest_id

            except Exception as e:
                logger.debug(f"📊 [Restore Analysis] Redis search failed: {e}")

        # If Redis failed or not enabled, try to find from files
        data_dir = Path("data")
        if data_dir.exists():
            progress_files = list(data_dir.glob("progress_*.json"))
            if progress_files:
                # Sort by modification time, get the latest
                latest_file = max(progress_files, key=lambda f: f.stat().st_mtime)
                # Extract analysis_id from filename
                filename = latest_file.name
                if filename.startswith("progress_") and filename.endswith(".json"):
                    analysis_id = filename[9:-5]  # Remove prefix and suffix
                    logger.debug(f"📊 [Restore Analysis] Found latest analysis ID from file: {analysis_id}")
                    return analysis_id

        return None
    except Exception as e:
        logger.error(f"📊 [Restore Analysis] Get latest analysis ID failed: {e}")
        return None
