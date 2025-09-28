"""
Data caching utilities for BackT

Provides caching functionality to avoid re-downloading data
and improve backtesting performance.
"""

import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import pandas as pd

from ..utils.types import TimeSeriesData
from ..utils.constants import DEFAULT_CACHE_SIZE, CACHE_EXPIRY_HOURS
from ..utils.logging_config import LoggerMixin


class DataCache(LoggerMixin):
    """Simple file-based cache for market data"""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size: int = DEFAULT_CACHE_SIZE,
        expiry_hours: int = CACHE_EXPIRY_HOURS,
        enabled: bool = True
    ):
        """
        Initialize data cache

        Args:
            cache_dir: Directory to store cache files (default: ./cache)
            max_size: Maximum number of cache entries
            expiry_hours: Hours before cache entries expire
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.max_size = max_size
        self.expiry_hours = expiry_hours

        if cache_dir is None:
            cache_dir = Path.cwd() / "cache"
        self.cache_dir = Path(cache_dir)

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_expired()

    def get(
        self,
        key: str,
        symbols: Union[str, list],
        start_date: str,
        end_date: str
    ) -> Optional[Union[TimeSeriesData, Dict[str, TimeSeriesData]]]:
        """
        Get data from cache

        Args:
            key: Cache key (usually data source identifier)
            symbols: Symbol(s) requested
            start_date: Start date
            end_date: End date

        Returns:
            Cached data if available and valid, None otherwise
        """
        if not self.enabled:
            return None

        cache_key = self._generate_cache_key(key, symbols, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_meta.json"

        if not cache_file.exists() or not metadata_file.exists():
            return None

        try:
            # Check if cache is expired
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            cache_time = datetime.fromisoformat(metadata['timestamp'])
            expiry_time = cache_time + timedelta(hours=self.expiry_hours)

            if datetime.now() > expiry_time:
                self.logger.debug(f"Cache expired for key: {cache_key}")
                self._remove_cache_entry(cache_key)
                return None

            # Load cached data
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            self.logger.debug(f"Cache hit for key: {cache_key}")
            return data

        except Exception as e:
            self.logger.warning(f"Error reading cache for {cache_key}: {e}")
            self._remove_cache_entry(cache_key)
            return None

    def put(
        self,
        key: str,
        symbols: Union[str, list],
        start_date: str,
        end_date: str,
        data: Union[TimeSeriesData, Dict[str, TimeSeriesData]]
    ) -> None:
        """
        Store data in cache

        Args:
            key: Cache key (usually data source identifier)
            symbols: Symbol(s) stored
            start_date: Start date
            end_date: End date
            data: Data to cache
        """
        if not self.enabled:
            return

        cache_key = self._generate_cache_key(key, symbols, start_date, end_date)

        try:
            # Ensure cache size limit
            self._enforce_size_limit()

            # Store data
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

            # Store metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'key': key,
                'symbols': symbols if isinstance(symbols, list) else [symbols],
                'start_date': start_date,
                'end_date': end_date,
                'data_type': type(data).__name__
            }

            metadata_file = self.cache_dir / f"{cache_key}_meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)

            self.logger.debug(f"Cached data for key: {cache_key}")

        except Exception as e:
            self.logger.warning(f"Error caching data for {cache_key}: {e}")

    def clear(self) -> None:
        """Clear all cache entries"""
        if not self.enabled:
            return

        try:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            for file in self.cache_dir.glob("*_meta.json"):
                file.unlink()

            self.logger.info("Cache cleared")

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def _generate_cache_key(
        self,
        key: str,
        symbols: Union[str, list],
        start_date: str,
        end_date: str
    ) -> str:
        """Generate unique cache key"""
        symbols_str = ','.join(sorted(symbols if isinstance(symbols, list) else [symbols]))
        cache_input = f"{key}_{symbols_str}_{start_date}_{end_date}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _cleanup_expired(self) -> None:
        """Remove expired cache entries"""
        if not self.enabled:
            return

        try:
            current_time = datetime.now()
            expired_count = 0

            for metadata_file in self.cache_dir.glob("*_meta.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    cache_time = datetime.fromisoformat(metadata['timestamp'])
                    expiry_time = cache_time + timedelta(hours=self.expiry_hours)

                    if current_time > expiry_time:
                        cache_key = metadata_file.stem.replace('_meta', '')
                        self._remove_cache_entry(cache_key)
                        expired_count += 1

                except Exception as e:
                    self.logger.warning(f"Error checking expiry for {metadata_file}: {e}")

            if expired_count > 0:
                self.logger.info(f"Removed {expired_count} expired cache entries")

        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")

    def _enforce_size_limit(self) -> None:
        """Enforce maximum cache size by removing oldest entries"""
        if not self.enabled:
            return

        try:
            metadata_files = list(self.cache_dir.glob("*_meta.json"))

            if len(metadata_files) >= self.max_size:
                # Sort by creation time and remove oldest
                file_times = []
                for file in metadata_files:
                    try:
                        with open(file, 'r') as f:
                            metadata = json.load(f)
                        timestamp = datetime.fromisoformat(metadata['timestamp'])
                        file_times.append((timestamp, file))
                    except Exception:
                        file_times.append((datetime.min, file))

                file_times.sort(key=lambda x: x[0])

                # Remove oldest entries
                to_remove = len(file_times) - self.max_size + 1
                for _, metadata_file in file_times[:to_remove]:
                    cache_key = metadata_file.stem.replace('_meta', '')
                    self._remove_cache_entry(cache_key)

                self.logger.info(f"Removed {to_remove} old cache entries to enforce size limit")

        except Exception as e:
            self.logger.error(f"Error enforcing cache size limit: {e}")

    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a specific cache entry"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            metadata_file = self.cache_dir / f"{cache_key}_meta.json"

            if cache_file.exists():
                cache_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()

        except Exception as e:
            self.logger.warning(f"Error removing cache entry {cache_key}: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache"""
        if not self.enabled:
            return {"enabled": False}

        try:
            metadata_files = list(self.cache_dir.glob("*_meta.json"))
            total_size = sum(
                (self.cache_dir / f"{f.stem.replace('_meta', '')}.pkl").stat().st_size
                for f in metadata_files
                if (self.cache_dir / f"{f.stem.replace('_meta', '')}.pkl").exists()
            )

            return {
                "enabled": True,
                "cache_dir": str(self.cache_dir),
                "entries": len(metadata_files),
                "max_size": self.max_size,
                "total_size_mb": total_size / (1024 * 1024),
                "expiry_hours": self.expiry_hours
            }

        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}")
            return {"enabled": True, "error": str(e)}