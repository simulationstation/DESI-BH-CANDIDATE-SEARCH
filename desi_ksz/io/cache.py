"""
Caching utilities for kSZ analysis pipeline.

This module provides HDF5-based caching for intermediate products
to avoid redundant computation.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
import logging
import json
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    H5PY_AVAILABLE = False
    logger.warning("h5py not available - caching will use numpy format")


class CacheManager:
    """
    Manager for cached intermediate products.

    Provides a simple key-value store backed by HDF5 files, with
    automatic cache invalidation based on input hashes.

    Parameters
    ----------
    cache_dir : Path
        Directory for cache files
    max_age_days : int
        Maximum age of cache entries before expiration

    Examples
    --------
    >>> cache = CacheManager(cache_dir='data/ksz/cache/')
    >>> cache.save('temperatures_LRG_z0.4-0.6', temperatures, metadata={'n': 10000})
    >>> T = cache.load('temperatures_LRG_z0.4-0.6')
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = Path("data/ksz/cache/"),
        max_age_days: int = 30,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self._index_file = self.cache_dir / "cache_index.json"
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Sanitize key for use as filename
        safe_key = key.replace("/", "_").replace("\\", "_")
        if H5PY_AVAILABLE:
            return self.cache_dir / f"{safe_key}.h5"
        else:
            return self.cache_dir / f"{safe_key}.npz"

    def exists(self, key: str, input_hash: Optional[str] = None) -> bool:
        """
        Check if cache entry exists and is valid.

        Parameters
        ----------
        key : str
            Cache key
        input_hash : str, optional
            Hash of inputs to verify cache validity

        Returns
        -------
        bool
            True if valid cache entry exists
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return False

        # Check age
        if key in self._index:
            created = datetime.fromisoformat(self._index[key].get("created", "1970-01-01"))
            age = (datetime.now() - created).days
            if age > self.max_age_days:
                logger.info(f"Cache entry expired: {key}")
                return False

            # Check input hash if provided
            if input_hash is not None:
                stored_hash = self._index[key].get("input_hash")
                if stored_hash != input_hash:
                    logger.info(f"Cache entry invalid (hash mismatch): {key}")
                    return False

        return True

    def save(
        self,
        key: str,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        metadata: Optional[Dict[str, Any]] = None,
        input_hash: Optional[str] = None,
    ) -> None:
        """
        Save data to cache.

        Parameters
        ----------
        key : str
            Cache key
        data : np.ndarray or dict of np.ndarray
            Data to cache
        metadata : dict, optional
            Additional metadata to store
        input_hash : str, optional
            Hash of inputs for cache invalidation
        """
        cache_path = self._get_cache_path(key)

        if H5PY_AVAILABLE:
            with h5py.File(cache_path, "w") as f:
                if isinstance(data, dict):
                    for name, arr in data.items():
                        f.create_dataset(name, data=arr, compression="gzip")
                else:
                    f.create_dataset("data", data=data, compression="gzip")

                if metadata:
                    f.attrs["metadata"] = json.dumps(metadata)
        else:
            if isinstance(data, dict):
                np.savez_compressed(cache_path, **data)
            else:
                np.savez_compressed(cache_path, data=data)

        # Update index
        self._index[key] = {
            "created": datetime.now().isoformat(),
            "path": str(cache_path),
            "input_hash": input_hash,
            "metadata": metadata,
        }
        self._save_index()

        logger.info(f"Cached: {key}")

    def load(
        self,
        key: str,
        input_hash: Optional[str] = None,
    ) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Load data from cache.

        Parameters
        ----------
        key : str
            Cache key
        input_hash : str, optional
            Expected input hash for validation

        Returns
        -------
        np.ndarray or dict or None
            Cached data, or None if not found/invalid
        """
        if not self.exists(key, input_hash):
            return None

        cache_path = self._get_cache_path(key)

        try:
            if H5PY_AVAILABLE:
                with h5py.File(cache_path, "r") as f:
                    if "data" in f:
                        return f["data"][:]
                    else:
                        return {name: f[name][:] for name in f.keys()}
            else:
                loaded = np.load(cache_path)
                if "data" in loaded:
                    return loaded["data"]
                else:
                    return dict(loaded)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for cache entry."""
        if key in self._index:
            return self._index[key].get("metadata")
        return None

    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Parameters
        ----------
        key : str, optional
            Specific key to clear. If None, clears all cache.
        """
        if key is not None:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            if key in self._index:
                del self._index[key]
        else:
            # Clear all
            for cached_key in list(self._index.keys()):
                cache_path = self._get_cache_path(cached_key)
                if cache_path.exists():
                    cache_path.unlink()
            self._index = {}

        self._save_index()

    def list_entries(self) -> Dict[str, Dict[str, Any]]:
        """List all cache entries with metadata."""
        return self._index.copy()


def compute_input_hash(*args, **kwargs) -> str:
    """
    Compute hash of inputs for cache invalidation.

    Parameters
    ----------
    *args, **kwargs
        Inputs to hash

    Returns
    -------
    str
        MD5 hash of inputs
    """
    hasher = hashlib.md5()

    for arg in args:
        if isinstance(arg, np.ndarray):
            hasher.update(arg.tobytes())
        elif isinstance(arg, (str, int, float)):
            hasher.update(str(arg).encode())
        elif isinstance(arg, dict):
            hasher.update(json.dumps(arg, sort_keys=True, default=str).encode())
        elif hasattr(arg, "__dict__"):
            hasher.update(json.dumps(arg.__dict__, sort_keys=True, default=str).encode())

    for key in sorted(kwargs.keys()):
        hasher.update(key.encode())
        val = kwargs[key]
        if isinstance(val, np.ndarray):
            hasher.update(val.tobytes())
        else:
            hasher.update(str(val).encode())

    return hasher.hexdigest()


def cache_to_hdf5(
    data: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = "gzip",
) -> None:
    """
    Save dictionary of arrays to HDF5 file.

    Parameters
    ----------
    data : dict
        Dictionary mapping names to numpy arrays
    output_path : Path
        Output file path
    metadata : dict, optional
        Metadata to store as attributes
    compression : str
        Compression algorithm
    """
    if not H5PY_AVAILABLE:
        # Fallback to numpy
        np.savez_compressed(output_path, **data)
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        for name, arr in data.items():
            f.create_dataset(name, data=arr, compression=compression)

        if metadata:
            f.attrs["metadata"] = json.dumps(metadata, default=str)
            for key, val in metadata.items():
                if isinstance(val, (str, int, float)):
                    f.attrs[key] = val


def load_from_hdf5(
    input_path: Union[str, Path],
    keys: Optional[list] = None,
) -> Dict[str, np.ndarray]:
    """
    Load arrays from HDF5 file.

    Parameters
    ----------
    input_path : Path
        Input file path
    keys : list, optional
        Specific keys to load. If None, loads all.

    Returns
    -------
    dict
        Dictionary mapping names to numpy arrays
    """
    input_path = Path(input_path)

    if not H5PY_AVAILABLE:
        # Fallback to numpy
        loaded = np.load(input_path)
        if keys is None:
            return dict(loaded)
        return {k: loaded[k] for k in keys if k in loaded}

    data = {}
    with h5py.File(input_path, "r") as f:
        load_keys = keys if keys is not None else list(f.keys())
        for key in load_keys:
            if key in f:
                data[key] = f[key][:]

    return data
