from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class DictCache(Generic[T]):
    values: Dict[str, T] = field(default_factory=dict)

    def get(self, key: str) -> Optional[T]:
        return self.values.get(key)

    def put(self, key: str, value: T) -> T:
        self.values[key] = value
        return value

    def get_or_put(self, key: str, factory) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.put(key, value)
        return value
