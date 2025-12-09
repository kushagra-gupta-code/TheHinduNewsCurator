
import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

class CacheManager:
    """
    Manages filesystem-based caching for news articles and their analysis.
    Stores data in a 'cache' directory with filenames based on date and edition.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def _get_cache_path(self, date: str, edition: str) -> str:
        """Generate cache filename: cache/YYYY-MM-DD_edition.json"""
        safe_edition = "".join(c for c in edition if c.isalnum() or c in ('-', '_'))
        filename = f"{date}_{safe_edition}.json"
        return os.path.join(self.cache_dir, filename)
        
    def load(self, date: str, edition: str) -> Optional[Dict[str, Any]]:
        """
        Load data from cache if it exists.
        
        Args:
            date: Date string (YYYY-MM-DD)
            edition: Edition ID
            
        Returns:
            Dict containing 'articles', 'top_20', 'timestamp' if found, else None
        """
        path = self._get_cache_path(date, edition)
        if not os.path.exists(path):
            return None
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ Failed to load cache from {path}: {e}")
            return None
            
    def save(self, date: str, edition: str, articles: list, top_20: list, provider_status: dict) -> bool:
        """
        Save analysis results to cache.
        
        Args:
            date: Date string
            edition: Edition ID
            articles: List of NewsArticle objects (will be serialized)
            top_20: List of top 20 NewsArticle objects (will be serialized)
            provider_status: Dict of LLM provider status
            
        Returns:
            True if successful
        """
        path = self._get_cache_path(date, edition)
        
        # Serialize articles
        serialized_articles = [self._serialize_article(a) for a in articles]
        serialized_top_20 = [self._serialize_article(a) for a in top_20]
        
        data = {
            "timestamp": time.time(),
            "date": date,
            "edition": edition,
            "articles": serialized_articles,
            "top_20": serialized_top_20,
            "provider_status": provider_status,
            "count": len(articles)
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved cache to {path}")
            return True
        except OSError as e:
            print(f"❌ Failed to save cache: {e}")
            return False
            
    def clear(self, date: str, edition: str) -> bool:
        """Delete specific cache file"""
        path = self._get_cache_path(date, edition)
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except OSError:
                return False
        return True

    def _serialize_article(self, article) -> Dict:
        """Convert NewsArticle object to dictionary"""
        return {
            "title": article.title,
            "url": article.url,
            "page": article.page,
            "section": article.section,
            "teaser": article.teaser,
            "impact_score": article.impact_score,
            "analysis": article.analysis,
            "content": article.content
        }
