from typing import Any, Dict, Hashable, Tuple
from pathlib import Path
import numpy as np
import pickle


class Cache:
    def __init__(self, filename: str):
        self._data_dir = Path(__file__).resolve().parent / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            raise ValueError("filename must be a non-empty string")
        filename = f"{filename}.pkl"
        self.path = self._data_dir / filename
        self._store: Dict[Any, np.ndarray] = {}

        if not self.path.exists():
            self._save()

        with self.path.open("rb") as f:
            obj = pickle.load(f)
        for k, v in obj.items():
            nk = self._normalize_key(k)
            self._validate_value(v)
            self._store[nk] = v

    def _normalize_key(self, *keys: Any) -> Tuple[Hashable, ...]:
        if len(keys) == 1:
            k0 = keys[0]
            if isinstance(k0, tuple):
                normalized = k0
            elif isinstance(k0, list):
                normalized = tuple(k0)
            else:
                normalized = (k0,)
        else:
            normalized = tuple(keys)
        for part in normalized:
            try:
                hash(part)
            except TypeError as e:
                raise TypeError(
                    f"Cache key elements must be hashable. Offending element: {part!r}"
                ) from e
        return normalized

    def _validate_value(self, value: Any) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Cache value must be a numpy.ndarray")

    def _save(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(self._store, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.path)

    def get(self, *keys: Any) -> np.ndarray:
        nk = self._normalize_key(*keys)
        return self._store[nk]

    def set(self, *keys_and_value: Any) -> None:
        if len(keys_and_value) < 2:
            raise ValueError("set requires at least a key and a value")
        *keys, value = keys_and_value
        self._validate_value(value)
        nk = self._normalize_key(*keys)
        self._store[nk] = value
        self._save()

    def has(self, *keys: Any) -> bool:
        nk = self._normalize_key(*keys)
        return nk in self._store

    def __contains__(self, keys: Any) -> bool:
        try:
            nk = self._normalize_key(keys)
        except Exception:
            return False
        return nk in self._store

    def __len__(self) -> int:
        return len(self._store)
