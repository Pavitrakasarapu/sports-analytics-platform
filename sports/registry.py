from .base import SportProfile
from .profiles import SoccerProfile, BasketballProfile, TennisProfile, CricketProfile, VolleyballProfile

class SportRegistry:
    def __init__(self):
        self._profiles = {}
        self.register(SoccerProfile(), "football", "soccer / football", "soccer/football")
        self.register(BasketballProfile())
        self.register(TennisProfile())
        self.register(CricketProfile())
        self.register(VolleyballProfile())

    @staticmethod
    def normalize_key(key):
        value = str(key or "").strip().lower()
        return {"football": "soccer", "soccer / football": "soccer", "soccer/football": "soccer"}.get(value, value)

    def register(self, profile: SportProfile, *aliases):
        for key in (profile.key, *aliases):
            self._profiles[self.normalize_key(key)] = profile

    def get(self, key):
        normalized = self.normalize_key(key)
        if normalized not in self._profiles:
            raise KeyError(f"Unsupported sport: {key}")
        return self._profiles[normalized]

    def all(self):
        seen, result = set(), []
        for profile in self._profiles.values():
            if profile.key not in seen:
                seen.add(profile.key)
                result.append(profile)
        return sorted(result, key=lambda p: p.name)

    def as_dict(self):
        return [profile.to_dict() for profile in self.all()]

registry = SportRegistry()
