# demonology/tools/web.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import Tool

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo or similar search engines."""
    
    def __init__(self):
        super().__init__("web_search", "Search the web for information")
    
    def is_available(self) -> bool:
        try:
            import requests
            return True
        except ImportError:
            return False
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5)", "default": 5}
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, query: str, num_results: int = 5, **kwargs) -> Dict[str, Any]:
        try:
            import requests
            from urllib.parse import quote_plus
            import re
            
            # Use DuckDuckGo instant answer API
            search_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_redirect=1"
            
            response = requests.get(search_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Get instant answer
            if data.get("Abstract"):
                results.append({
                    "title": data.get("AbstractText", "")[:100] + "..." if len(data.get("AbstractText", "")) > 100 else data.get("AbstractText", ""),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", "")
                })
            
            # Get related topics
            for topic in data.get("RelatedTopics", [])[:num_results-1]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100] + "..." if len(topic.get("Text", "")) > 100 else topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results[:num_results],
                "total_results": len(results)
            }
            
        except Exception as e:
            logger.exception("WebSearchTool error")
            return {"success": False, "error": str(e)}


class RedditSearchTool(Tool):
    """Search Reddit for posts and comments using the Reddit API."""
    
    def __init__(self):
        super().__init__("reddit_search", "Search Reddit posts and discussions")
    
    def is_available(self) -> bool:
        try:
            # Check if we have requests at minimum for public API
            import requests
            return True
        except ImportError:
            return False
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "subreddit": {"type": "string", "description": "Subreddit to search in (optional, searches all if not specified)"},
                    "sort": {"type": "string", "enum": ["relevance", "hot", "top", "new", "comments"], "description": "Sort method", "default": "relevance"},
                    "time_filter": {"type": "string", "enum": ["all", "day", "week", "month", "year"], "description": "Time filter", "default": "all"},
                    "limit": {"type": "integer", "description": "Number of results (1-25)", "default": 5, "minimum": 1, "maximum": 25}
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, query: str, subreddit: str = None, sort: str = "relevance", 
                     time_filter: str = "all", limit: int = 5, **kwargs) -> Dict[str, Any]:
        try:
            import os
            
            # Check for Reddit API credentials in environment variables
            client_id = os.environ.get('REDDIT_CLIENT_ID')
            client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
            user_agent = os.environ.get('REDDIT_USER_AGENT', f'Demonology:v1.0 (by /u/demonology_user)')
            
            # Try to use PRAW if available and credentials exist
            if client_id and client_secret:
                try:
                    import praw
                    from datetime import datetime
                    
                    # Initialize Reddit API client
                    reddit = praw.Reddit(
                        client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent
                    )
                    
                    # Perform search
                    if subreddit:
                        search_results = reddit.subreddit(subreddit).search(query, limit=limit, sort=sort, time_filter=time_filter)
                    else:
                        search_results = reddit.subreddit('all').search(query, limit=limit, sort=sort, time_filter=time_filter)
                    
                    results = []
                    for submission in search_results:
                        results.append({
                            "title": submission.title,
                            "url": f"https://reddit.com{submission.permalink}",
                            "subreddit": submission.subreddit.display_name,
                            "author": str(submission.author) if submission.author else "[deleted]",
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "created": datetime.fromtimestamp(submission.created_utc).isoformat(),
                            "selftext": submission.selftext[:500] + "..." if len(submission.selftext) > 500 else submission.selftext
                        })
                    
                    return {
                        "success": True,
                        "query": query,
                        "subreddit": subreddit,
                        "results": results,
                        "total_results": len(results),
                        "method": "praw"
                    }
                    
                except ImportError:
                    logger.warning("PRAW not available, falling back to public API")
                except Exception as e:
                    logger.warning(f"PRAW failed: {e}, falling back to public API")
            
            # Fallback to public Reddit JSON API
            import requests
            from urllib.parse import quote_plus
            import json
            
            # Construct search URL
            base_url = f"https://www.reddit.com/r/{subreddit}" if subreddit else "https://www.reddit.com"
            search_url = f"{base_url}/search.json"
            
            params = {
                'q': query,
                'limit': min(limit, 25),
                'sort': sort if sort != 'relevance' else 'hot',
                't': time_filter,
                'raw_json': 1
            }
            
            if subreddit:
                params['restrict_sr'] = 1
            
            # Set headers to mimic browser request
            headers = {
                'User-Agent': user_agent,
                'Accept': 'application/json'
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if 'data' in data and 'children' in data['data']:
                for post in data['data']['children']:
                    post_data = post['data']
                    results.append({
                        "title": post_data.get('title', ''),
                        "url": f"https://reddit.com{post_data.get('permalink', '')}",
                        "subreddit": post_data.get('subreddit', ''),
                        "author": post_data.get('author', '[deleted]'),
                        "score": post_data.get('score', 0),
                        "num_comments": post_data.get('num_comments', 0),
                        "created": post_data.get('created_utc', 0),
                        "selftext": (post_data.get('selftext', '')[:500] + "..." 
                                   if len(post_data.get('selftext', '')) > 500 
                                   else post_data.get('selftext', ''))
                    })
            
            return {
                "success": True,
                "query": query,
                "subreddit": subreddit,
                "results": results,
                "total_results": len(results),
                "method": "public_api"
            }
            
        except Exception as e:
            logger.exception("RedditSearchTool error")
            return {"success": False, "error": str(e)}