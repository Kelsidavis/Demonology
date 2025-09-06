
# demonology/utils.py (patched: atomic IO, safer timestamps, pagination, edits)
from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# --------- Small helpers ---------

def _iso_now() -> str:
    """UTC ISO8601 with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically to avoid partial/corrupt files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def _truncate(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."

# --------- Conversation Manager ---------

class ConversationManager:
    """Manages conversation history and persistence (now atomic + editable)."""

    def __init__(self, conversations_dir: Path):
        self.conversations_dir = conversations_dir
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        # Files
        self.metadata_file = self.conversations_dir / "metadata.json"
        if not self.metadata_file.exists():
            self._save_metadata({})

    # ---- Metadata IO ----

    def _load_metadata(self) -> Dict[str, Any]:
        """Load conversations metadata (tolerant to corruption)."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Metadata load issue ({e}); using empty index.")
            return {}

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save conversations metadata atomically with a lightweight backup."""
        try:
            # rotate single backup
            bak = self.metadata_file.with_suffix(".json.bak")
            if self.metadata_file.exists():
                try:
                    # best-effort small backup
                    self.metadata_file.replace(bak)
                except Exception:
                    pass
            _atomic_write_json(self.metadata_file, metadata)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            # attempt to restore backup
            try:
                if bak.exists():
                    bak.replace(self.metadata_file)
            except Exception:
                pass

    # ---- Create / Read ----

    def save_conversation(
        self,
        messages: List[Dict[str, str]],
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save a new conversation with optional name and tags.
        Returns the conversation ID.
        """
        if not messages:
            raise ValueError("Cannot save empty conversation")

        # Generate conversation ID
        conversation_id = self._generate_conversation_id(messages)

        # Derive name from first user message if not provided
        if not name:
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                name = _truncate(user_messages[0].get("content", ""), 50) + ("..." if len(user_messages[0].get("content","")) > 50 else "")
            else:
                name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        now_iso = _iso_now()
        conversation_data = {
            "id": conversation_id,
            "name": name,
            "messages": messages,
            "created_at": now_iso,
            "updated_at": now_iso,
            "tags": tags or [],
            "message_count": len(messages)
        }

        # Persist conversation atomically
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        try:
            _atomic_write_json(conversation_file, conversation_data)
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise

        # Update metadata
        metadata = self._load_metadata()
        preview = ""
        for m in reversed(messages):
            if m.get("content"):
                preview = _truncate(m["content"], 120)
                break

        metadata[conversation_id] = {
            "name": name,
            "created_at": now_iso,
            "updated_at": now_iso,
            "tags": tags or [],
            "message_count": len(messages),
            "file": f"{conversation_id}.json",
            "preview": preview
        }
        self._save_metadata(metadata)

        return conversation_id

    def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation by ID."""
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        if not conversation_file.exists():
            return None
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return None

    # ---- Update / Edit ----

    def append_messages(self, conversation_id: str, new_messages: List[Dict[str, str]]) -> bool:
        """Append messages to an existing conversation and update metadata."""
        if not new_messages:
            return True
        convo = self.load_conversation(conversation_id)
        if not convo:
            return False
        convo["messages"].extend(new_messages)
        convo["updated_at"] = _iso_now()
        convo["message_count"] = len(convo["messages"])

        try:
            _atomic_write_json(self.conversations_dir / f"{conversation_id}.json", convo)
        except Exception as e:
            logger.error(f"Failed to append messages: {e}")
            return False

        md = self._load_metadata()
        if conversation_id in md:
            md[conversation_id]["updated_at"] = convo["updated_at"]
            md[conversation_id]["message_count"] = convo["message_count"]
            # update preview with last non-empty message
            for m in reversed(new_messages):
                if m.get("content"):
                    md[conversation_id]["preview"] = _truncate(m["content"], 120)
                    break
            self._save_metadata(md)
        return True

    def rename_or_retag(
        self,
        conversation_id: str,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None
    ) -> bool:
        """Rename and/or update tags for a conversation."""
        convo = self.load_conversation(conversation_id)
        if not convo:
            return False

        changed = False
        if name:
            convo["name"] = name
            changed = True
        # update tags
        tagset = set(convo.get("tags", []))
        if tags is not None:
            tagset = set(tags)
            changed = True
        if add_tags:
            tagset.update(add_tags); changed = True
        if remove_tags:
            tagset.difference_update(remove_tags); changed = True
        if changed:
            convo["tags"] = sorted(tagset)
            convo["updated_at"] = _iso_now()

        if changed:
            try:
                _atomic_write_json(self.conversations_dir / f"{conversation_id}.json", convo)
            except Exception as e:
                logger.error(f"Failed to update conversation: {e}")
                return False

            md = self._load_metadata()
            md.setdefault(conversation_id, {})
            md[conversation_id]["name"] = convo["name"]
            md[conversation_id]["tags"] = convo["tags"]
            md[conversation_id]["updated_at"] = convo["updated_at"]
            md[conversation_id]["message_count"] = convo.get("message_count", len(convo.get("messages", [])))
            md[conversation_id]["file"] = f"{conversation_id}.json"
            self._save_metadata(md)
        return True

    # ---- Listing / Search ----

    def list_conversations(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """List conversations, newest first, with optional pagination."""
        metadata = self._load_metadata()
        conversations = [ { "id": cid, **info } for cid, info in metadata.items() ]
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        if offset < 0: offset = 0
        if limit is not None and limit >= 0:
            conversations = conversations[offset : offset + limit]
        else:
            conversations = conversations[offset:]
        return conversations

    def search_conversations(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search by name/content/tags with optional pagination."""
        conversations = self.list_conversations()  # already sorted
        results: List[Dict[str, Any]] = []
        q = (query or "").lower()

        for conv_info in conversations:
            # Tag filter
            if tags:
                existing = set(conv_info.get("tags", []))
                if not existing.intersection(tags):
                    continue

            # Name/preview quick check first
            if q and (q in (conv_info.get("name","").lower()) or q in (conv_info.get("preview","").lower())):
                results.append(conv_info); continue

            # Load full convo for deep search
            convo = self.load_conversation(conv_info["id"])
            if not convo:
                continue
            if not q:
                results.append(conv_info); continue

            for message in convo.get("messages", []):
                content = (message.get("content") or "").lower()
                if q in content:
                    results.append(conv_info)
                    break

        # Pagination
        if offset < 0: offset = 0
        if limit is not None and limit >= 0:
            return results[offset : offset + limit]
        return results[offset:]

    # ---- Delete ----

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation by ID (file + metadata)."""
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        ok = True
        try:
            if conversation_file.exists():
                conversation_file.unlink()
            md = self._load_metadata()
            if conversation_id in md:
                del md[conversation_id]
                self._save_metadata(md)
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            ok = False
        return ok

    # ---- Export / Import ----

    def export_conversation(
        self,
        conversation_id: str,
        format: str = "json",
        output_path: Optional[Path] = None
    ) -> Path:
        """Export conversation to json/txt/md/html (atomic write)."""
        convo = self.load_conversation(conversation_id)
        if not convo:
            raise ValueError(f"Conversation {conversation_id} not found")

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{conversation_id}_{timestamp}.{format}"
            output_path = self.conversations_dir / "exports" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            _atomic_write_json(output_path, convo)

        elif format == "txt":
            tmp = output_path.with_suffix(output_path.suffix + ".tmp")
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(f"Conversation: {convo['name']}\n")
                f.write(f"Created: {convo['created_at']}\n")
                f.write("=" * 50 + "\n\n")
                for message in convo['messages']:
                    role = message.get('role', 'unknown').title()
                    content = message.get('content', '')
                    f.write(f"{role}:\n{content}\n\n")
            os.replace(tmp, output_path)

        elif format == "md":
            tmp = output_path.with_suffix(output_path.suffix + ".tmp")
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(f"# {convo['name']}\n\n")
                f.write(f"**Created:** {convo['created_at']}\n")
                f.write(f"**Messages:** {len(convo['messages'])}\n\n")
                for message in convo['messages']:
                    role = message.get('role', 'unknown').title()
                    content = message.get('content', '')
                    f.write(f"## {role}\n\n{content}\n\n")
            os.replace(tmp, output_path)

        elif format == "html":
            tmp = output_path.with_suffix(output_path.suffix + ".tmp")
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write("<!doctype html><meta charset='utf-8'><title>Conversation</title>\n")
                f.write(f"<h1>{convo['name']}</h1>\n")
                f.write(f"<p><b>Created:</b> {convo['created_at']} &nbsp; <b>Messages:</b> {len(convo['messages'])}</p>")
                for message in convo['messages']:
                    role = message.get('role', 'unknown').title()
                    content = message.get('content', '')
                    f.write(f"<h3>{role}</h3><pre>{content}</pre>\n")
            os.replace(tmp, output_path)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        return output_path

    def import_conversation(self, path: Path, name: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        """Import a conversation from a JSON file matching the expected schema."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not validate_conversation_format(data):
            raise ValueError(f"File {path} does not match conversation schema")
        return self.save_conversation(data["messages"], name=name or data.get("name"), tags=tags or data.get("tags"))

    # ---- ID ----

    def _generate_conversation_id(self, messages: List[Dict[str, str]]) -> str:
        """Generate a unique ID (12 hex chars) for a conversation."""
        content = ""
        for msg in messages[:3]:
            content += f"{msg.get('role','')}:{(msg.get('content') or '')[:100]}"
        content += _iso_now()
        return hashlib.sha256(content.encode()).hexdigest()[:12]


# --------- Shared utilities (unchanged APIs) ---------

def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str

def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text to specified length with ellipsis."""
    return text if len(text) <= max_length else text[: max_length - 3] + "..."

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    import re
    sanitized = re.sub(r'[<>:"/\\\\|?*]', '_', filename)
    sanitized = sanitized.strip(' .')
    return sanitized[:100]

def count_tokens_estimate(text: str) -> int:
    """Rough estimate of token count for text (~4 chars/token)."""
    return max(1, len(text) // 4)

def validate_conversation_format(data: Any) -> bool:
    """Validate that data matches expected conversation format."""
    if not isinstance(data, dict):
        return False
    required_fields = ["messages", "created_at"]
    for field in required_fields:
        if field not in data:
            return False
    if not isinstance(data["messages"], list):
        return False
    for message in data["messages"]:
        if not isinstance(message, dict):
            return False
        if "role" not in message or "content" not in message:
            return False
        if message["role"] not in ["user", "assistant", "system"]:
            return False
    return True
