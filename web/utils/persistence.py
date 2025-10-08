"""
Persistence utilities
Uses URL parameters combined with session state to persist user selections
"""

import streamlit as st
import logging
from urllib.parse import urlencode, parse_qs
import json

logger = logging.getLogger(__name__)

class ModelPersistence:
    """Model selection persistence manager"""
    
    def __init__(self):
        self.storage_key = "model_config"
    
    def save_config(self, provider, category, model):
        """Save configuration to session state and URL"""
        config = {
            'provider': provider,
            'category': category,
            'model': model
        }
        
        # Save to session state
        st.session_state[self.storage_key] = config
        
        # Save to URL parameters (via query_params)
        try:
            st.query_params.update({
                'provider': provider,
                'category': category,
                'model': model
            })
            logger.debug(f"💾 [Persistence] Configuration saved: {config}")
        except Exception as e:
            logger.warning(f"⚠️ [Persistence] Failed to save URL parameters: {e}")
    
    def load_config(self):
        """Load configuration from session state or URL"""
        # First try to load from URL parameters
        try:
            query_params = st.query_params
            if 'provider' in query_params:
                config = {
                    'provider': query_params.get('provider', 'dashscope'),
                    'category': query_params.get('category', 'openai'),
                    'model': query_params.get('model', '')
                }
                logger.debug(f"📥 [Persistence] Configuration loaded from URL: {config}")
                return config
        except Exception as e:
            logger.warning(f"⚠️ [Persistence] Failed to load URL parameters: {e}")
        
        # Then try to load from session state
        if self.storage_key in st.session_state:
            config = st.session_state[self.storage_key]
            logger.debug(f"📥 [Persistence] Configuration loaded from Session State: {config}")
            return config
        
        # Return default configuration
        default_config = {
            'provider': 'dashscope',
            'category': 'openai',
            'model': ''
        }
        logger.debug(f"📥 [Persistence] Using default configuration: {default_config}")
        return default_config
    
    def clear_config(self):
        """Clear configuration"""
        if self.storage_key in st.session_state:
            del st.session_state[self.storage_key]
        
        try:
            st.query_params.clear()
            logger.info("🗑️ [Persistence] Configuration cleared")
        except Exception as e:
            logger.warning(f"⚠️ [Persistence] Failed to clear: {e}")

# Global instance
persistence = ModelPersistence()

def save_model_selection(provider, category="", model=""):
    """Save model selection"""
    persistence.save_config(provider, category, model)

def load_model_selection():
    """Load model selection"""
    return persistence.load_config()

def clear_model_selection():
    """Clear model selection"""
    persistence.clear_config()
