
# demonology/config.py (patched, safe + feature-aligned)
from __future__ import annotations

import os
import time
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field

__CONFIG_VERSION__ = 2  # bump when schema changes


# ---------------- Dataclasses ----------------

@dataclass
class APIConfig:
    """Configuration for API connection and streaming behavior."""
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen3-coder-30b-tools"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 2.0
    context_length: int = 26768
    context_buffer: int = 2048
    auto_trim_threshold: float = 0.85
    smart_trimming: bool = True
    # New: align with client_patched.py
    sse_heartbeat_timeout: float = 120.0
    allow_server_restart: bool = False

    def validate(self) -> List[str]:
        errs: List[str] = []
        if self.max_tokens <= 0:
            errs.append("max_tokens must be > 0; clamping to 256")
            self.max_tokens = 256
        if not (0.0 <= self.temperature <= 2.0):
            errs.append("temperature must be within [0,2]; clamping")
            self.temperature = min(max(self.temperature, 0.0), 2.0)
        if not (0.0 < self.top_p <= 1.0):
            errs.append("top_p must be (0,1]; clamping")
            self.top_p = min(max(self.top_p, 1e-6), 1.0)
        if self.timeout <= 0:
            errs.append("timeout must be > 0; clamping")
            self.timeout = 10.0
        if self.max_retries < 0:
            errs.append("max_retries must be >= 0; clamping")
            self.max_retries = 0
        if self.retry_delay < 0:
            errs.append("retry_delay must be >= 0; clamping")
            self.retry_delay = 0.0
        if self.context_length <= 1024:
            errs.append("context_length suspiciously small; raising to 4096")
            self.context_length = 4096
        if not (0 < self.auto_trim_threshold <= 1):
            errs.append("auto_trim_threshold must be in (0,1]; clamping")
            self.auto_trim_threshold = 0.85
        if self.sse_heartbeat_timeout <= 0:
            errs.append("sse_heartbeat_timeout must be > 0; clamping")
            self.sse_heartbeat_timeout = 120.0
        return errs


@dataclass
class UIConfig:
    """Configuration for UI preferences."""
    theme: str = "amethyst"
    permissive_mode: bool = False
    auto_save_conversations: bool = True
    max_history_length: int = 1000
    auto_continue_enabled: bool = True
    auto_continue_timeout: float = 60.0

    def validate(self) -> List[str]:
        errs: List[str] = []
        if self.max_history_length < 1:
            errs.append("max_history_length must be >=1; clamping")
            self.max_history_length = 100
        if self.auto_continue_timeout <= 0:
            errs.append("auto_continue_timeout must be > 0; clamping")
            self.auto_continue_timeout = 60.0
        return errs


@dataclass
class ToolsConfig:
    """Configuration for tool integration."""
    enabled: bool = True
    allowed_tools: list = field(default_factory=lambda: [
        "file_operations", "code_execution", "web_search",
        "project_planning", "reddit_search", "wikipedia_search",
        "hackernews_search", "stackoverflow_search", "open_web_search"
    ])
    working_directory: Optional[str] = None  # relative to workspace root

    def validate(self, workspace_root: Path) -> List[str]:
        errs: List[str] = []
        if self.working_directory:
            wd = Path(self.working_directory)
            # force relative to workspace
            wd_abs = (workspace_root / wd).resolve() if not wd.is_absolute() else wd.resolve()
            try:
                wd_abs.relative_to(workspace_root)
            except Exception:
                errs.append(f"working_directory '{wd_abs}' escapes workspace; resetting to workspace root")
                self.working_directory = None
        return errs


# ---------------- Config Manager ----------------

class Config:
    """Main configuration manager for Demonology with atomic save & validation."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = self._get_default_config_path()

        self.config_path = Path(config_path)
        self.api = APIConfig()
        self.ui = UIConfig()
        self.tools = ToolsConfig()
        self.version = __CONFIG_VERSION__

        # Load configuration if file exists, else create default
        if self.config_path.exists():
            self.load()
        else:
            self.save()

    # ---------- Paths & workspace ----------

    @staticmethod
    def _get_default_config_path() -> Path:
        """Get the default configuration file path."""
        if os.name == "posix":
            config_dir = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        else:
            config_dir = os.environ.get("APPDATA", os.path.expanduser("~"))
        config_dir = Path(config_dir) / "demonology"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.yaml"

    def get_workspace_root(self) -> Path:
        """Resolve workspace root from env or next to the config directory."""
        root = os.environ.get("DEMONOLOGY_ROOT")
        if root:
            p = Path(root).expanduser().resolve()
        else:
            p = (self.config_path.parent / "workspace").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------- Load / Save ----------

    def load(self) -> None:
        """Load configuration from file, apply env overrides, validate, and migrate."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            print("Using default configuration.")
            data = {}

        # Apply stored values
        api_data = data.get("api", {})
        ui_data = data.get("ui", {})
        tools_data = data.get("tools", {})
        self._assign(self.api, api_data)
        self._assign(self.ui, ui_data)
        self._assign(self.tools, tools_data)

        # Migration hook
        old_ver = data.get("version", 1)
        if old_ver != __CONFIG_VERSION__:
            self._migrate(old_ver, __CONFIG_VERSION__)

        # Env overrides (optional, minimal set)
        self.api.base_url = os.environ.get("DEMONOLOGY_API_BASE_URL", self.api.base_url)
        self.api.model = os.environ.get("DEMONOLOGY_API_MODEL", self.api.model)

        # Validate + clamp
        errs = []
        errs += self.api.validate()
        errs += self.ui.validate()
        errs += self.tools.validate(self.get_workspace_root())
        if errs:
            for e in errs:
                print(f"[config] {e}")
            # Persist clamped values
            self.save(backup=False)

    def save(self, backup: bool = True) -> None:
        """Save current configuration to file atomically with optional .bak backup."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.config_path.with_suffix(".tmp")
            data = {
                "version": __CONFIG_VERSION__,
                "api": asdict(self.api),
                "ui": asdict(self.ui),
                "tools": asdict(self.tools),
            }
            # Backup
            if backup and self.config_path.exists():
                ts = time.strftime("%Y%m%d-%H%M%S")
                bak = self.config_path.with_suffix(f".{ts}.bak")
                try:
                    shutil.copy2(self.config_path, bak)
                except Exception:
                    pass

            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, indent=2)
            os.replace(tmp, self.config_path)
        except (yaml.YAMLError, OSError) as e:
            print(f"Error: Failed to save config to {self.config_path}: {e}")
            # Cleanup temp if present
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass

    # ---------- Updates ----------

    def update_api_config(self, **kwargs) -> None:
        self._assign(self.api, kwargs, strict=False)
        self.save()

    def update_ui_config(self, **kwargs) -> None:
        self._assign(self.ui, kwargs, strict=False)
        self.save()

    def update_tools_config(self, **kwargs) -> None:
        self._assign(self.tools, kwargs, strict=False)
        self.save()

    # ---------- Utilities ----------

    @staticmethod
    def _assign(obj: Any, data: Dict[str, Any], strict: bool = True) -> None:
        for k, v in (data or {}).items():
            if hasattr(obj, k):
                setattr(obj, k, v)
            elif strict:
                print(f"[config] Unknown key ignored: {k}")

    def _migrate(self, old: int, new: int) -> None:
        """Migrate fields between versions (lightweight placeholder)."""
        if old < 2:
            # v1 -> v2: ensure new API fields exist with defaults
            if not hasattr(self.api, "sse_heartbeat_timeout"):
                self.api.sse_heartbeat_timeout = 120.0
            if not hasattr(self.api, "allow_server_restart"):
                self.api.allow_server_restart = False
        self.version = new

    # ---------- Conversation dir ----------

    def get_conversations_dir(self) -> Path:
        conv_dir = self.config_path.parent / "conversations"
        conv_dir.mkdir(exist_ok=True)
        return conv_dir

    # ---------- Serialization ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "api": asdict(self.api),
            "ui": asdict(self.ui),
            "tools": asdict(self.tools),
        }

    def __repr__(self) -> str:
        return f"Config(version={self.version}, api={self.api}, ui={self.ui}, tools={self.tools})"


# Mystical error messages for configuration issues
CONFIG_ERROR_MESSAGES = [
    "The ancient scrolls are corrupted... \U0001F4DC",
    "Configuration sigils have been disturbed... \u16CD",
    "The binding circle is incomplete... \u25CB",
    "Mystic parameters refuse alignment... \u26A1",
    "The demonology's pages are torn... \U0001F4D6",
]

def get_mystical_error_message() -> str:
    """Get a random mystical error message."""
    import random
    return random.choice(CONFIG_ERROR_MESSAGES)
