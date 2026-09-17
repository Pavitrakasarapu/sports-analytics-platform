from .base import Feature, FeatureStatus, SportProfile

def feature(name, status, note=""):
    return Feature(name, status, note)

class SoccerProfile(SportProfile):
    def __init__(self):
        super().__init__(
            key="soccer", name="Soccer / Football",
            object_types=["player", "ball"],
            statistics=["player_count", "distance", "speed", "possession", "player_movement", "player_spacing", "passing"],
            visualizations=["player_tracks", "heatmap", "formation", "spacing"],
            event_types=["pass", "possession_change", "transition", "set_piece"],
            analytics={
                "player_detection": feature("Player detection", FeatureStatus.AVAILABLE),
                "ball_detection": feature("Ball detection", FeatureStatus.AVAILABLE),
                "player_tracking": feature("Player tracking", FeatureStatus.AVAILABLE),
                "ball_tracking": feature("Ball tracking", FeatureStatus.AVAILABLE),
                "possession_estimation": feature("Possession estimation", FeatureStatus.AVAILABLE),
                "player_movement": feature("Player movement", FeatureStatus.AVAILABLE),
                "distance": feature("Distance", FeatureStatus.AVAILABLE),
                "speed": feature("Speed", FeatureStatus.AVAILABLE),
                "heatmaps": feature("Heatmaps", FeatureStatus.AVAILABLE),
                "team_formations": feature("Team formations", FeatureStatus.AVAILABLE),
                "player_spacing": feature("Player spacing", FeatureStatus.AVAILABLE),
                "passing_analysis": feature("Passing analysis", FeatureStatus.EXPERIMENTAL, "Requires reliable ball/player evidence."),
                "tactical_insights": feature("Tactical insights", FeatureStatus.AVAILABLE),
            },
        )
    def build_capability_summary(self): return self.to_dict()

class BasketballProfile(SportProfile):
    def __init__(self):
        super().__init__(
            key="basketball", name="Basketball",
            object_types=["player", "ball", "basketball_court"],
            statistics=["player_count", "distance", "speed", "movement", "court_position", "team_spacing", "possession"],
            visualizations=["player_tracks", "heatmap", "court_positions", "spacing"],
            event_types=["shot", "possession_change", "rebound"],
            analytics={
                "player_detection": feature("Player detection", FeatureStatus.EXPERIMENTAL, "Current detector is generic person detection."),
                "ball_detection": feature("Ball detection", FeatureStatus.EXPERIMENTAL, "Generic ball detection; basketball-specific validation is pending."),
                "player_tracking": feature("Player tracking", FeatureStatus.EXPERIMENTAL),
                "court_positioning": feature("Court positioning", FeatureStatus.EXPERIMENTAL, "Requires court calibration."),
                "movement": feature("Movement", FeatureStatus.EXPERIMENTAL),
                "distance": feature("Distance", FeatureStatus.EXPERIMENTAL),
                "speed": feature("Speed", FeatureStatus.EXPERIMENTAL),
                "shot_zone_analysis": feature("Shot-zone analysis", FeatureStatus.NOT_SUPPORTED, "Requires reliable shot events and court zones."),
                "possession_estimation": feature("Possession estimation", FeatureStatus.EXPERIMENTAL),
                "team_spacing": feature("Team spacing", FeatureStatus.EXPERIMENTAL),
            },
        )
    def build_capability_summary(self): return self.to_dict()

class TennisProfile(SportProfile):
    def __init__(self):
        super().__init__(
            key="tennis", name="Tennis",
            object_types=["player", "ball", "tennis_court"],
            statistics=["player_count", "distance", "speed", "court_position", "rally_length"],
            visualizations=["player_tracks", "heatmap", "court_positions", "rally_timeline"],
            event_types=["serve", "rally", "point", "bounce"],
            analytics={
                "player_tracking": feature("Player tracking", FeatureStatus.EXPERIMENTAL),
                "ball_tracking": feature("Ball tracking", FeatureStatus.EXPERIMENTAL),
                "court_positioning": feature("Court positioning", FeatureStatus.EXPERIMENTAL, "Requires court calibration."),
                "movement": feature("Movement", FeatureStatus.EXPERIMENTAL),
                "rally_tracking": feature("Rally tracking", FeatureStatus.EXPERIMENTAL, "Requires reliable trajectory and point segmentation."),
                "serve_event_tracking": feature("Serve/event tracking", FeatureStatus.EXPERIMENTAL),
                "distance": feature("Distance", FeatureStatus.EXPERIMENTAL),
                "speed": feature("Speed", FeatureStatus.EXPERIMENTAL),
            },
        )
    def build_capability_summary(self): return self.to_dict()

class CricketProfile(SportProfile):
    def __init__(self):
        super().__init__(
            key="cricket", name="Cricket",
            object_types=["player", "ball", "pitch"],
            statistics=["player_count", "distance", "speed", "movement"],
            visualizations=["player_tracks", "heatmap", "pitch_positions", "ball_trajectory"],
            event_types=["delivery", "shot", "wicket", "run"],
            analytics={
                "player_detection": feature("Player detection", FeatureStatus.EXPERIMENTAL),
                "ball_tracking": feature("Ball tracking", FeatureStatus.EXPERIMENTAL, "Requires cricket-ball-specific trajectory detection."),
                "player_movement": feature("Player movement", FeatureStatus.EXPERIMENTAL),
                "pitch_area_analysis": feature("Pitch-area analysis", FeatureStatus.EXPERIMENTAL, "Requires pitch geometry."),
                "shot_direction": feature("Shot direction", FeatureStatus.EXPERIMENTAL, "Requires bat/ball event detection."),
                "bowling_events": feature("Bowling events", FeatureStatus.EXPERIMENTAL),
                "batting_events": feature("Batting events", FeatureStatus.EXPERIMENTAL),
                "distance": feature("Distance", FeatureStatus.EXPERIMENTAL),
                "speed": feature("Speed", FeatureStatus.EXPERIMENTAL),
            },
        )
    def build_capability_summary(self): return self.to_dict()

class VolleyballProfile(SportProfile):
    def __init__(self):
        super().__init__(
            key="volleyball", name="Volleyball",
            object_types=["player", "ball", "volleyball_court"],
            statistics=["player_count", "distance", "speed", "movement", "team_positioning"],
            visualizations=["player_tracks", "heatmap", "formation", "court_positions"],
            event_types=["serve", "rally", "attack", "block"],
            analytics={
                "player_detection": feature("Player detection", FeatureStatus.EXPERIMENTAL),
                "ball_tracking": feature("Ball tracking", FeatureStatus.EXPERIMENTAL),
                "player_positioning": feature("Player positioning", FeatureStatus.EXPERIMENTAL),
                "team_formations": feature("Team formations", FeatureStatus.EXPERIMENTAL, "Requires court calibration and role inference."),
                "movement": feature("Movement", FeatureStatus.EXPERIMENTAL),
                "rally_statistics": feature("Rally-related statistics", FeatureStatus.EXPERIMENTAL, "Requires rally/event segmentation."),
                "distance": feature("Distance", FeatureStatus.EXPERIMENTAL),
                "speed": feature("Speed", FeatureStatus.EXPERIMENTAL),
            },
        )
    def build_capability_summary(self): return self.to_dict()
