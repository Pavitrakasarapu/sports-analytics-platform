from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

class FeatureStatus(str, Enum):
    AVAILABLE = "Available"
    EXPERIMENTAL = "Experimental"
    NOT_SUPPORTED = "Not supported"

@dataclass(frozen=True)
class Feature:
    name: str
    status: FeatureStatus
    note: str = ""

@dataclass(frozen=True)
class SportProfile(ABC):
    key: str
    name: str
    object_types: list[str] = field(default_factory=list)
    statistics: list[str] = field(default_factory=list)
    visualizations: list[str] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    analytics: Mapping[str, Feature] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "object_types": list(self.object_types),
            "statistics": list(self.statistics),
            "visualizations": list(self.visualizations),
            "event_types": list(self.event_types),
            "analytics": {
                key: {"name": feature.name, "status": feature.status.value, "note": feature.note}
                for key, feature in self.analytics.items()
            },
        }

    @abstractmethod
    def build_capability_summary(self) -> dict:
        raise NotImplementedError
