"""
Demonology Utilities - Helper functions and conversation management.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history and persistence."""
    
    def __init__(self, conversations_dir: Path):
        self.conversations_dir = conversations_dir
        self.conversations_dir.mkdir(exist_ok=True)
        
        # Create metadata file if it doesn't exist
        self.metadata_file = self.conversations_dir / "metadata.json"
        if not self.metadata_file.exists():
            self._save_metadata({})
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load conversations metadata."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save conversations metadata."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def save_conversation(
        self, 
        messages: List[Dict[str, str]], 
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save a conversation with optional name and tags.
        Returns the conversation ID.
        """
        if not messages:
            raise ValueError("Cannot save empty conversation")
        
        # Generate conversation ID
        conversation_id = self._generate_conversation_id(messages)
        
        # Use provided name or generate from first user message
        if not name:
            user_messages = [msg for msg in messages if msg['role'] == 'user']
            if user_messages:
                name = user_messages[0]['content'][:50] + "..."
            else:
                name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Create conversation data
        conversation_data = {
            "id": conversation_id,
            "name": name,
            "messages": messages,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "tags": tags or [],
            "message_count": len(messages)
        }
        
        # Save conversation file
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        try:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise
        
        # Update metadata
        metadata = self._load_metadata()
        metadata[conversation_id] = {
            "name": name,
            "created_at": conversation_data["created_at"],
            "updated_at": conversation_data["updated_at"],
            "tags": tags or [],
            "message_count": len(messages),
            "file": f"{conversation_id}.json"
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
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all saved conversations."""
        metadata = self._load_metadata()
        conversations = []
        
        for conv_id, conv_info in metadata.items():
            conversations.append({
                "id": conv_id,
                **conv_info
            })
        
        # Sort by updated_at descending
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation by ID."""
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        
        try:
            if conversation_file.exists():
                conversation_file.unlink()
            
            # Remove from metadata
            metadata = self._load_metadata()
            if conversation_id in metadata:
                del metadata[conversation_id]
                self._save_metadata(metadata)
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False
    
    def search_conversations(
        self, 
        query: str, 
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search conversations by content or tags."""
        conversations = self.list_conversations()
        results = []
        
        query_lower = query.lower()
        
        for conv_info in conversations:
            # Check tags if specified
            if tags:
                if not any(tag in conv_info.get("tags", []) for tag in tags):
                    continue
            
            # Load full conversation to search content
            conversation = self.load_conversation(conv_info["id"])
            if not conversation:
                continue
            
            # Search in conversation name
            if query_lower in conv_info["name"].lower():
                results.append(conv_info)
                continue
            
            # Search in message content
            for message in conversation["messages"]:
                if query_lower in message["content"].lower():
                    results.append(conv_info)
                    break
        
        return results
    
    def _generate_conversation_id(self, messages: List[Dict[str, str]]) -> str:
        """Generate a unique ID for a conversation."""
        # Create a hash based on the first few messages and timestamp
        content = ""
        for msg in messages[:3]:  # Use first 3 messages
            content += f"{msg['role']}:{msg['content'][:100]}"
        
        content += datetime.now().isoformat()
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def export_conversation(
        self, 
        conversation_id: str, 
        format: str = "json",
        output_path: Optional[Path] = None
    ) -> Path:
        """Export conversation to different formats."""
        conversation = self.load_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{conversation_id}_{timestamp}.{format}"
            output_path = self.conversations_dir / "exports" / filename
            output_path.parent.mkdir(exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
        
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Conversation: {conversation['name']}\n")
                f.write(f"Created: {conversation['created_at']}\n")
                f.write("=" * 50 + "\n\n")
                
                for message in conversation['messages']:
                    role = message['role'].title()
                    content = message['content']
                    f.write(f"{role}:\n{content}\n\n")
        
        elif format == "md":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# {conversation['name']}\n\n")
                f.write(f"**Created:** {conversation['created_at']}\n")
                f.write(f"**Messages:** {len(conversation['messages'])}\n\n")
                
                for message in conversation['messages']:
                    role = message['role'].title()
                    content = message['content']
                    f.write(f"## {role}\n\n{content}\n\n")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return output_path


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    import re
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    # Limit length
    return sanitized[:100]


def count_tokens_estimate(text: str) -> int:
    """Rough estimate of token count for text."""
    # Very rough approximation: ~4 characters per token
    return len(text) // 4


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