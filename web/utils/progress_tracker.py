"""
Smart Progress Tracker
Dynamically calculates progress and time estimation based on number of analysts and research depth
"""

import time
from typing import Optional, Callable, Dict, List
import streamlit as st

# Import logging module
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('progress')

class SmartAnalysisProgressTracker:
    """Smart Analysis Progress Tracker"""

    def __init__(self, analysts: List[str], research_depth: int, llm_provider: str, callback: Optional[Callable] = None):
        self.callback = callback
        self.analysts = analysts
        self.research_depth = research_depth
        self.llm_provider = llm_provider
        self.steps = []
        self.current_step = 0
        self.start_time = time.time()

        # Dynamically generate steps based on number of analysts and research depth
        self.analysis_steps = self._generate_dynamic_steps()
        self.estimated_duration = self._estimate_total_duration()

    def _generate_dynamic_steps(self) -> List[Dict]:
        """Dynamically generate analysis steps based on number of analysts"""
        steps = [
            {"name": "Data Validation", "description": "Validate stock code and pre-fetch data", "weight": 0.05},
            {"name": "Environment Setup", "description": "Check API keys and environment configuration", "weight": 0.02},
            {"name": "Cost Estimation", "description": "Estimate analysis cost", "weight": 0.01},
            {"name": "Parameter Configuration", "description": "Configure analysis parameters and models", "weight": 0.02},
            {"name": "Engine Initialization", "description": "Initialize AI analysis engine", "weight": 0.05},
        ]

        # Add dedicated steps for each analyst
        analyst_weight = 0.8 / len(self.analysts)  # 80% of time for analyst work
        for analyst in self.analysts:
            analyst_name = self._get_analyst_display_name(analyst)
            steps.append({
                "name": f"{analyst_name} Analysis",
                "description": f"{analyst_name} is conducting professional analysis",
                "weight": analyst_weight
            })

        # Final consolidation step
        steps.append({"name": "Result Compilation", "description": "Compile analysis results and generate report", "weight": 0.05})

        return steps

    def _get_analyst_display_name(self, analyst: str) -> str:
        """Get analyst display name"""
        name_map = {
            'market': 'Market Analyst',
            'fundamentals': 'Fundamental Analyst',
            'technical': 'Technical Analyst',
            'sentiment': 'Sentiment Analyst',
            'risk': 'Risk Analyst'
        }
        return name_map.get(analyst, analyst)

    def _estimate_total_duration(self) -> float:
        """Estimate total duration based on number of analysts, research depth, and model type (seconds)"""
        # Base time (seconds) - environment setup, configuration, etc.
        base_time = 60

        # Actual time consumption per analyst (based on real test data)
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
    
    def update(self, message: str, step: Optional[int] = None, total_steps: Optional[int] = None):
        """Update progress"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Record step
        self.steps.append({
            'message': message,
            'timestamp': current_time,
            'elapsed': elapsed_time
        })

        # Automatically determine current step based on message content
        if step is None:
            step = self._detect_step_from_message(message)

        if step is not None:
            # Special handling: if "module completed" is detected, advance to next step
            if "Module Completed" in message and step == self.current_step:
                # Analyst completed, advance to next step
                next_step = min(step + 1, len(self.analysis_steps) - 1)
                self.current_step = next_step
                logger.info(f"📊 [Progress Update] Analyst completed, advancing to step {self.current_step + 1}/{len(self.analysis_steps)}")
            # Prevent step regression: only update when detected step is greater than or equal to current step
            elif step >= self.current_step:
                self.current_step = step
                logger.debug(f"📊 [Progress Update] Step advanced to {self.current_step + 1}/{len(self.analysis_steps)}")
            else:
                logger.debug(f"📊 [Progress Update] Ignoring backward step: detected step {step + 1}, current step {self.current_step + 1}")

        # If it's a completion message, ensure progress is 100%
        if "Analysis Completed" in message or "Analysis Successful" in message or "✅ Analysis Completed" in message:
            self.current_step = len(self.analysis_steps) - 1
            logger.info(f"📊 [Progress Update] Analysis completed, set to final step {self.current_step + 1}/{len(self.analysis_steps)}")

        # Call callback function
        if self.callback:
            progress = self._calculate_weighted_progress()
            remaining_time = self._estimate_remaining_time(progress, elapsed_time)
            self.callback(message, self.current_step, len(self.analysis_steps), progress, elapsed_time, remaining_time)

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
        """Smart estimation of remaining time"""
        if progress <= 0:
            return self.estimated_duration

        # If progress exceeds 20%, use actual progress for estimation
        if progress > 0.2:
            estimated_total = elapsed_time / progress
            return max(estimated_total - elapsed_time, 0)
        else:
            # Use estimated time in early stages
            return max(self.estimated_duration - elapsed_time, 0)
    
    def _detect_step_from_message(self, message: str) -> Optional[int]:
        """Intelligently detect current step based on message content"""
        message_lower = message.lower()

        # Check specific stage keywords
        if any(keyword in message for keyword in ["Start Analysis Phase", "Data Validation Phase", "Environment Setup Phase", "Cost Estimation Phase", "Parameter Configuration Phase", "Engine Initialization Phase"]):
            for i, step in enumerate(self.analysis_steps):
                if "Data Validation" in message and "Data Validation" in step["name"]:
                    return i
                elif "Environment Setup" in message and "Environment Setup" in step["name"]:
                    return i
                elif "Cost Estimation" in message and "Cost Estimation" in step["name"]:
                    return i
                elif "Parameter Configuration" in message and "Parameter Configuration" in step["name"]:
                    return i
                elif "Engine Initialization" in message and "Engine Initialization" in step["name"]:
                    return i
        # Start Analysis Phase - only match initial start messages
        elif "🚀 Start Stock Analysis" in message:
            return 0
        # Data Validation Phase
        elif "Validation" in message or "Pre-fetch" in message or "Data Preparation" in message:
            return 0
        # Environment Setup Phase
        elif "Environment" in message or "api" in message_lower or "Key" in message:
            return 1
        # Cost Estimation Phase
        elif "Cost" in message or "Estimation" in message:
            return 2
        # Parameter Configuration Phase
        elif "Configuration" in message or "Parameters" in message:
            return 3
        # Engine Initialization Phase
        elif "Initialization" in message or "Engine" in message:
            return 4
        # Check analyst work stages
        elif any(analyst in message for analyst in ["Market Analyst", "Fundamentals Analyst", "Technical Analyst", "Sentiment Analyst", "Risk Analyst"]):
            for i, step in enumerate(self.analysis_steps):
                if "Market Analyst" in message and "Market" in step["name"]:
                    return i
                elif "Fundamentals Analyst" in message and "Fundamentals" in step["name"]:
                    return i
                elif "Technical Analyst" in message and "Technical" in step["name"]:
                    return i
                elif "Sentiment Analyst" in message and "Sentiment" in step["name"]:
                    return i
                elif "Risk Analyst" in message and "Risk" in step["name"]:
                    return i
                # Generic matching
                for analyst_key in ["Market", "Fundamentals", "Technical", "Sentiment", "Risk"]:
                    if analyst_key in message and analyst_key in step["name"]:
                        return i
                    elif "Risk" in message and "Risk" in step["name"]:
                        return i
        # Tool calling stage - detect analyst using tools
        elif "Tool Call" in message or "Calling" in message or "tool" in message.lower():
            # If current step is analyst step, maintain current step
            if self.current_step < len(self.analysis_steps) and "Analyst" in self.analysis_steps[self.current_step]["name"]:
                return self.current_step
        # Module start/completion logs
        elif "Module Start" in message or "Module Completed" in message:
            # Extract analyst type from logs
            if "market_analyst" in message or "market" in message or "Market" in message:
                for i, step in enumerate(self.analysis_steps):
                    if "Market" in step["name"]:
                        return i
            elif "fundamentals_analyst" in message or "fundamentals" in message or "Fundamentals" in message:
                for i, step in enumerate(self.analysis_steps):
                    if "Fundamentals" in step["name"]:
                        return i
            elif "technical_analyst" in message or "technical" in message or "Technical" in message:
                for i, step in enumerate(self.analysis_steps):
                    if "Technical" in step["name"]:
                        return i
            elif "sentiment_analyst" in message or "sentiment" in message or "Sentiment" in message:
                for i, step in enumerate(self.analysis_steps):
                    if "Sentiment" in step["name"]:
                        return i
            elif "risk_analyst" in message or "risk" in message or "Risk" in message:
                for i, step in enumerate(self.analysis_steps):
                    if "Risk" in step["name"]:
                        return i
            elif "graph_signal_processing" in message or "signal" in message or "Signal" in message:
                for i, step in enumerate(self.analysis_steps):
                    if "Signal" in step["name"] or "Compilation" in step["name"]:
                        return i
        # Result compilation stage
        elif "Organizing" in message or "Results" in message:
            return len(self.analysis_steps) - 1
        # Completion stage
        elif "Completed" in message or "Successful" in message:
            return len(self.analysis_steps) - 1

        return None
    
    def get_current_step_info(self) -> Dict:
        """Get current step information"""
        if self.current_step < len(self.analysis_steps):
            return self.analysis_steps[self.current_step]
        return {"name": "Completed", "description": "Analysis completed", "weight": 0}

    def get_progress_percentage(self) -> float:
        """Get progress percentage"""
        return self._calculate_weighted_progress() * 100

    def get_elapsed_time(self) -> float:
        """Get elapsed time"""
        return time.time() - self.start_time

    def get_estimated_total_time(self) -> float:
        """Get estimated total time"""
        return self.estimated_duration

    def format_time(self, seconds: float) -> str:
        """Format time display"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

class SmartStreamlitProgressDisplay:
    """Smart Streamlit progress display component"""

    def __init__(self, container):
        self.container = container
        self.progress_bar = None
        self.status_text = None
        self.step_info = None
        self.time_info = None
        self.setup_display()

    def setup_display(self):
        """Setup display components"""
        with self.container:
            st.markdown("### 📊 Analysis Progress")
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
            self.step_info = st.empty()
            self.time_info = st.empty()

    def update(self, message: str, current_step: int, total_steps: int, progress: float, elapsed_time: float, remaining_time: float):
        """Update display"""
        # Update progress bar
        self.progress_bar.progress(progress)

        # Update status text
        self.status_text.markdown(f"**Current Status:** 📋 {message}")

        # Update step information
        step_text = f"**Progress:** Step {current_step + 1} of {total_steps} ({progress:.1%})"
        self.step_info.markdown(step_text)

        # Update time information
        time_text = f"**Elapsed Time:** {self._format_time(elapsed_time)}"
        if remaining_time > 0:
            time_text += f" | **Estimated Remaining:** {self._format_time(remaining_time)}"

        self.time_info.markdown(time_text)
    
    def _format_time(self, seconds: float) -> str:
        """Format time display"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def clear(self):
        """Clear display"""
        self.container.empty()

def create_smart_progress_callback(display: SmartStreamlitProgressDisplay, analysts: List[str], research_depth: int, llm_provider: str) -> Callable:
    """Create smart progress callback function"""
    tracker = SmartAnalysisProgressTracker(analysts, research_depth, llm_provider)

    def callback(message: str, step: Optional[int] = None, total_steps: Optional[int] = None):
        # If step and total_steps are explicitly specified, use old fixed mode (compatibility)
        if step is not None and total_steps is not None and total_steps == 10:
            # Compatible with old 10-step mode, but use smart time estimation
            progress = step / max(total_steps - 1, 1) if total_steps > 1 else 1.0
            progress = min(progress, 1.0)
            elapsed_time = tracker.get_elapsed_time()
            remaining_time = tracker._estimate_remaining_time(progress, elapsed_time)
            display.update(message, step, total_steps, progress, elapsed_time, remaining_time)
        else:
            # Use new smart tracking mode
            tracker.update(message, step, total_steps)
            current_step = tracker.current_step
            total_steps_count = len(tracker.analysis_steps)
            progress = tracker.get_progress_percentage() / 100
            elapsed_time = tracker.get_elapsed_time()
            remaining_time = tracker._estimate_remaining_time(progress, elapsed_time)
            display.update(message, current_step, total_steps_count, progress, elapsed_time, remaining_time)

    return callback

# Backward compatible function
def create_progress_callback(display, analysts=None, research_depth=2, llm_provider="dashscope") -> Callable:
    """Create progress callback function (backward compatible)"""
    if hasattr(display, '__class__') and 'Smart' in display.__class__.__name__:
        return create_smart_progress_callback(display, analysts or ['market', 'fundamentals'], research_depth, llm_provider)
    else:
        # Old version compatibility
        tracker = SmartAnalysisProgressTracker(analysts or ['market', 'fundamentals'], research_depth, llm_provider)

        def callback(message: str, step: Optional[int] = None, total_steps: Optional[int] = None):
            if step is not None and total_steps is not None:
                progress = step / max(total_steps - 1, 1) if total_steps > 1 else 1.0
                progress = min(progress, 1.0)
                elapsed_time = tracker.get_elapsed_time()
                display.update(message, step, total_steps, progress, elapsed_time)
            else:
                tracker.update(message, step, total_steps)
                current_step = tracker.current_step
                total_steps_count = len(tracker.analysis_steps)
                progress = tracker.get_progress_percentage() / 100
                elapsed_time = tracker.get_elapsed_time()
                display.update(message, current_step, total_steps_count, progress, elapsed_time)

        return callback
