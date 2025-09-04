"""
Demonology Configuration Management - Handles application settings and user preferences.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class APIConfig:
    """Configuration for API connection."""
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "Qwen-3-Coder-30B"
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 2.0
    context_length: int = 26768  # Match your server's context length
    context_buffer: int = 2048   # Reserved tokens for response generation
    auto_trim_threshold: float = 0.85  # Auto-trim when 85% full
    smart_trimming: bool = True  # Enable intelligent message preservation


@dataclass
class UIConfig:
    """Configuration for UI preferences."""
    theme: str = "amethyst"
    permissive_mode: bool = False
    auto_save_conversations: bool = True
    max_history_length: int = 1000


@dataclass
class ToolsConfig:
    """Configuration for tool integration."""
    enabled: bool = True
    allowed_tools: list = None
    working_directory: Optional[str] = None
    
    def __post_init__(self):
        if self.allowed_tools is None:
            self.allowed_tools = ["file_operations", "code_execution", "web_search", "project_planning", "reddit_search"]


class Config:
    """Main configuration manager for Demonology."""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = Path(config_path)
        self.api = APIConfig()
        self.ui = UIConfig()
        self.tools = ToolsConfig()
        
        # Load configuration if file exists
        if self.config_path.exists():
            self.load()
        else:
            # Create default config
            self.save()
    
    @staticmethod
    def _get_default_config_path() -> Path:
        """Get the default configuration file path."""
        # Use XDG config directory or fallback to home
        if os.name == "posix":
            config_dir = os.environ.get("XDG_CONFIG_HOME")
            if not config_dir:
                config_dir = os.path.expanduser("~/.config")
        else:
            config_dir = os.environ.get("APPDATA", os.path.expanduser("~"))
        
        config_dir = Path(config_dir) / "demonology"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.yaml"
    
    def load(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            # Update API config
            if 'api' in data:
                for key, value in data['api'].items():
                    if hasattr(self.api, key):
                        setattr(self.api, key, value)
            
            # Update UI config
            if 'ui' in data:
                for key, value in data['ui'].items():
                    if hasattr(self.ui, key):
                        setattr(self.ui, key, value)
            
            # Update tools config
            if 'tools' in data:
                for key, value in data['tools'].items():
                    if hasattr(self.tools, key):
                        setattr(self.tools, key, value)
        
        except (yaml.YAMLError, IOError) as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            print("Using default configuration.")
    
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'api': asdict(self.api),
                'ui': asdict(self.ui),
                'tools': asdict(self.tools)
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        
        except (yaml.YAMLError, IOError) as e:
            print(f"Error: Failed to save config to {self.config_path}: {e}")
    
    def update_api_config(self, **kwargs) -> None:
        """Update API configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.api, key):
                setattr(self.api, key, value)
    
    def update_ui_config(self, **kwargs) -> None:
        """Update UI configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.ui, key):
                setattr(self.ui, key, value)
    
    def update_tools_config(self, **kwargs) -> None:
        """Update tools configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.tools, key):
                setattr(self.tools, key, value)
    
    def get_conversations_dir(self) -> Path:
        """Get the directory for storing conversations."""
        conv_dir = self.config_path.parent / "conversations"
        conv_dir.mkdir(exist_ok=True)
        return conv_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'api': asdict(self.api),
            'ui': asdict(self.ui),
            'tools': asdict(self.tools)
        }
    
    def __repr__(self) -> str:
        return f"Config(api={self.api}, ui={self.ui}, tools={self.tools})"


# Mystical error messages for configuration issues
CONFIG_ERROR_MESSAGES = [
    "The ancient scrolls are corrupted... 📜",
    "Configuration sigils have been disturbed... 𖤍",
    "The binding circle is incomplete... ○",
    "Mystic parameters refuse alignment... ⚡",
    "The demonology's pages are torn... 📖",
]


def get_mystical_error_message() -> str:
    """Get a random mystical error message."""
    import random
    return random.choice(CONFIG_ERROR_MESSAGES)