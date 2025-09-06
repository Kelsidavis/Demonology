
# demonology/tools/web_free_extras.py
from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import Tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.environ.get("DEMONOLOGY_HTTP_TIMEOUT", "15"))
MAX_RETRIES = int(os.environ.get("DEMONOLOGY_HTTP_RETRIES", "2"))
BACKOFF_BASE = float(os.environ.get("DEMONOLOGY_HTTP_BACKOFF", "0.6"))  # seconds

def _http_get(url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Tuple[Optional[requests.Response], Optional[str]]:
    headers = headers or {}
    timeout = timeout or DEFAULT_TIMEOUT
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code >= 500 or r.status_code == 429:
                last_err = f"HTTP {r.status_code}"
                raise RuntimeError(last_err)
            return r, None
        except Exception as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE * (2 ** attempt))
            else:
                break
    return None, last_err

# ============================================================
# Wikipedia (MediaWiki) Search
# ============================================================
class WikipediaSearchTool(Tool):
    """
    Free search over Wikipedia via MediaWiki APIs.
    - Primary: opensearch (suggests titles/urls)
    - Enrichment: page summary for the top result
    """

    def __init__(self):
        super().__init__("wikipedia_search", "Search Wikipedia (free MediaWiki API) and return titles/urls/snippets.")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "lang": {"type": "string", "description": "Language code", "default": "en"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10}
                },
                "required": ["query"]
            }
        }

    async def execute(self, query: str, lang: str = "en", limit: int = 10, **_) -> Dict[str, Any]:
        base = f"https://{lang}.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": query, "limit": max(1, int(limit)), "namespace": 0, "format": "json"}
        resp, err = _http_get(base, params=params, headers={"Accept": "application/json"})
        if err or resp is None:
            return {"success": False, "error": err or "request failed", "api": "wikipedia", "status_code": None}
        status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            return {"success": False, "error": "invalid JSON", "api": "wikipedia", "status_code": status}

        # data is [search, titles[], descriptions[], urls[]]
        results: List[Dict[str, Any]] = []
        try:
            titles = data[1] if len(data) > 1 else []
            descs  = data[2] if len(data) > 2 else []
            urls   = data[3] if len(data) > 3 else []
            for i, title in enumerate(titles[:limit]):
                results.append({
                    "title": title,
                    "url": urls[i] if i < len(urls) else "",
                    "snippet": descs[i] if i < len(descs) else ""
                })
        except Exception as e:
            return {"success": False, "error": f"parse error: {e}", "api": "wikipedia", "status_code": status}

        # Optional: enrich top result with /page/summary
        summary = None
        if results:
            t0 = results[0]["title"]
            sum_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{t0}"
            sresp, _ = _http_get(sum_url, headers={"Accept": "application/json"})
            if sresp and sresp.ok:
                try:
                    sdata = sresp.json()
                    summary = sdata.get("extract")
                    if summary:
                        results[0]["snippet"] = summary
                except Exception:
                    pass

        return {"success": True, "operation": "search", "api": "wikipedia", "lang": lang, "status_code": status, "count": len(results), "results": results}

# ============================================================
# Hacker News (Algolia) Search
# ============================================================
class HackerNewsSearchTool(Tool):
    """
    Free search of Hacker News via Algolia's public API (no key required).
    """

    def __init__(self):
        super().__init__("hackernews_search", "Search Hacker News via Algolia API (free, no key).")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "tags": {"type": "string", "description": "Filter tags: story, comment, poll, ask_hn, show_hn", "default": "story"},
                    "page": {"type": "integer", "description": "Page (0-indexed)", "default": 0},
                    "hits_per_page": {"type": "integer", "description": "Max items per page", "default": 10}
                },
                "required": ["query"]
            }
        }

    async def execute(self, query: str, tags: str = "story", page: int = 0, hits_per_page: int = 10, **_) -> Dict[str, Any]:
        url = "https://hn.algolia.com/api/v1/search"
        params = {"query": query, "tags": tags, "page": max(0, int(page)), "hitsPerPage": max(1, int(hits_per_page))}
        resp, err = _http_get(url, params=params, headers={"Accept": "application/json"})
        if err or resp is None:
            return {"success": False, "error": err or "request failed", "api": "hn", "status_code": None}
        status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            return {"success": False, "error": "invalid JSON", "api": "hn", "status_code": status}

        results: List[Dict[str, Any]] = []
        for h in (data.get("hits") or [])[: hits_per_page]:
            results.append({
                "title": h.get("title") or h.get("story_title") or "",
                "url": h.get("url") or h.get("story_url") or "",
                "points": h.get("points"),
                "author": h.get("author"),
                "created_at": h.get("created_at"),
                "objectID": h.get("objectID"),
                "snippet": (h.get("_highlightResult", {}).get("title", {}).get("value") or "")[:400]
            })

        return {"success": True, "operation": "search", "api": "hn", "status_code": status, "count": len(results), "results": results}

# ============================================================
# Stack Exchange (StackOverflow) Search
# ============================================================
class StackOverflowSearchTool(Tool):
    """
    Free search of StackOverflow using the Stack Exchange API (keyless with rate limits).
    """

    def __init__(self):
        super().__init__("stackoverflow_search", "Search StackOverflow (Stack Exchange API, free, no key needed).")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms"},
                    "sort": {"type": "string", "enum": ["relevance","votes","creation","activity"], "default": "relevance"},
                    "order": {"type": "string", "enum": ["desc","asc"], "default": "desc"},
                    "tagged": {"type": "string", "description": "Comma-separated tags", "default": ""},
                    "pagesize": {"type": "integer", "description": "Max results", "default": 10}
                },
                "required": ["query"]
            }
        }

    async def execute(self, query: str, sort: str = "relevance", order: str = "desc", tagged: str = "", pagesize: int = 10, **_) -> Dict[str, Any]:
        url = "https://api.stackexchange.com/2.3/search/advanced"
        params = {
            "q": query,
            "site": "stackoverflow",
            "sort": sort,
            "order": order,
            "pagesize": max(1, min(int(pagesize), 50)),
            "tagged": tagged or None,
            "filter": "!nKzQUR693x"  # includes title, link, score, is_answered, tags, creation_date, owner
        }
        resp, err = _http_get(url, params=params, headers={"Accept": "application/json"})
        if err or resp is None:
            return {"success": False, "error": err or "request failed", "api": "stackoverflow", "status_code": None}
        status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            return {"success": False, "error": "invalid JSON", "api": "stackoverflow", "status_code": status}

        items = data.get("items") or []
        results: List[Dict[str, Any]] = []
        for it in items[: params["pagesize"]]:
            results.append({
                "title": it.get("title",""),
                "url": it.get("link",""),
                "score": it.get("score", 0),
                "is_answered": bool(it.get("is_answered")),
                "creation_date": it.get("creation_date"),
                "tags": it.get("tags", []),
                "owner": (it.get("owner") or {}).get("display_name")
            })

        return {"success": True, "operation": "search", "api": "stackoverflow", "status_code": status, "count": len(results), "results": results}

# ============================================================
# Aggregate: OpenWebSearchTool
# ============================================================
class OpenWebSearchTool(Tool):
    """
    Fan-out search across free sources: Wikipedia, Hacker News, StackOverflow.
    Optionally includes DuckDuckGo IA if your `web.WebSearchTool` is installed.
    """

    def __init__(self):
        super().__init__("open_web_search", "Aggregate free web search (Wikipedia, HN, StackOverflow, optional DDG IA).")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit_per_source": {"type": "integer", "description": "Max results per source", "default": 5},
                    "include_ddg": {"type": "boolean", "description": "Also query DuckDuckGo IA if available", "default": True}
                },
                "required": ["query"]
            }
        }

    async def execute(self, query: str, limit_per_source: int = 5, include_ddg: bool = True, **_) -> Dict[str, Any]:
        all_results: List[Dict[str, Any]] = []
        sources_used: List[str] = []
        errors: List[Dict[str, Any]] = []

        # Wikipedia
        try:
            wiki = WikipediaSearchTool()
            res = await wiki.execute(query=query, limit=limit_per_source)
            if res.get("success"):
                for r in res.get("results", [])[:limit_per_source]:
                    r2 = dict(r); r2["source"] = "wikipedia"
                    all_results.append(r2)
                sources_used.append("wikipedia")
            else:
                errors.append({"source": "wikipedia", "error": res.get("error")})
        except Exception as e:
            errors.append({"source": "wikipedia", "error": str(e)})

        # Hacker News
        try:
            hn = HackerNewsSearchTool()
            res = await hn.execute(query=query, hits_per_page=limit_per_source)
            if res.get("success"):
                for r in res.get("results", [])[:limit_per_source]:
                    r2 = dict(r); r2["source"] = "hn"
                    all_results.append(r2)
                sources_used.append("hn")
            else:
                errors.append({"source": "hn", "error": res.get("error")})
        except Exception as e:
            errors.append({"source": "hn", "error": str(e)})

        # StackOverflow
        try:
            so = StackOverflowSearchTool()
            res = await so.execute(query=query, pagesize=limit_per_source)
            if res.get("success"):
                for r in res.get("results", [])[:limit_per_source]:
                    r2 = dict(r); r2["source"] = "stackoverflow"
                    all_results.append(r2)
                sources_used.append("stackoverflow")
            else:
                errors.append({"source": "stackoverflow", "error": res.get("error")})
        except Exception as e:
            errors.append({"source": "stackoverflow", "error": str(e)})

        # Optional: DuckDuckGo (from your existing web.py) for breadth
        if include_ddg:
            try:
                from .web import WebSearchTool  # type: ignore
                ddg = WebSearchTool()
                res = await ddg.execute(query=query, max_results=limit_per_source)
                if res.get("success"):
                    for r in res.get("results", [])[:limit_per_source]:
                        r2 = dict(r); r2["source"] = "duckduckgo"
                        all_results.append(r2)
                    sources_used.append("duckduckgo")
                else:
                    errors.append({"source": "duckduckgo", "error": res.get("error")})
            except Exception as e:
                errors.append({"source": "duckduckgo", "error": str(e)})

        # Simple relevance sort heuristic: prefer items with more metadata / points / score
        def score(item: Dict[str, Any]) -> float:
            s = 0.0
            if item.get("source") == "hn": s += float(item.get("points") or 0) * 0.5
            if item.get("source") == "stackoverflow":
                s += float(item.get("score") or 0) * 0.5
                if item.get("is_answered"): s += 2.0
            if item.get("snippet"): s += 0.2
            if item.get("url"): s += 0.2
            return s

        all_results.sort(key=score, reverse=True)
        return {
            "success": True,
            "operation": "aggregate_search",
            "query": query,
            "sources": sources_used,
            "count": len(all_results),
            "results": all_results[: max(1, 3*limit_per_source)],  # cap the merged list a bit
            "errors": errors
        }
