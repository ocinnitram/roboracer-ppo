"""Pydantic schema for the training configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from sim.track import TRACKS

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "ppo" / "configs"
RUNS = REPO / "runs"
SEED = 17
# Selection score: success first, pace as the tie-breaker.
SCORE_PACE_WEIGHT = 0.1


class PPOConfig(BaseModel):
    """PPO hyperparameters, reward weights, and CAPS smoothness weights."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    batch_size: int = Field(gt=0)
    rollout: int = Field(gt=0)
    clip: float = Field(gt=0.0, le=0.5)
    c2_i: float = Field(ge=0.0, le=1.0)
    c2_f: float = Field(ge=0.0, le=1.0)
    gae_lambda: float = Field(ge=0.0, le=1.0)
    gamma: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(gt=0.0, lt=0.1)
    log_std_i: float = Field(ge=-5.0, le=2.0)
    n_epochs: int = Field(ge=1, le=20)
    normalise_advantages: bool
    target_kl: float = Field(gt=0.0, lt=1.0)
    c1: float = Field(ge=0.0, le=1.0)
    hidden_size: int = Field(default=256, ge=32, le=512)
    r_progress: float = Field(gt=0.0)
    r_collision: float = Field(lt=0.0)
    r_lap: float = Field(gt=0.0)
    caps_lambda_t: float = Field(default=1.0, ge=0.0)
    caps_lambda_s: float = Field(default=0.5, ge=0.0)
    caps_lambda_t_v: float = Field(default=0.25, ge=0.0)
    caps_lambda_s_v: float = Field(default=0.5, ge=0.0)
    caps_sigma_s: float = Field(default=0.05, gt=0.0)

    @field_validator("c2_f")
    @classmethod
    def final_c2_should_be_less_than_initial(cls, v: float, info: ValidationInfo) -> float:
        """Reject an entropy coefficient that grows over the run."""
        if "c2_i" in info.data and v > info.data["c2_i"]:
            raise ValueError(f"c2_f ({v}) should be <= c2_i ({info.data['c2_i']})")
        return v


class TrackConfig(BaseModel):
    """Map name plus the numbers merged from tracks/<map>/track.yaml at load."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    map: str
    opt_time: float = Field(gt=0.0)
    # Plant speed cap for this map; None keeps the vehicle's own cap.
    max_speed_mps: Optional[float] = Field(default=None, gt=0.0)


class TrainingConfig(BaseModel):
    """Run length, checkpoints, evaluation, and plant noise."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_name: str
    steps: int = Field(gt=0)
    ckpt_interval: int = Field(default=250_000, ge=1000)
    eval_interval: int = Field(default=100_000, ge=1000)
    patience_evals: int = Field(default=40, ge=1)
    # "episode" redraws an env's layout on every reset; "complete" keeps it
    # through crashes until num_laps flying laps are done.
    layout_switch: Literal["episode", "complete"] = "episode"
    sim2real: bool
    num_laps: int = Field(default=2, ge=1)


class ObstaclesConfig(BaseModel):
    """Count and sizes of the obstacles sampled for training; 0 = empty track."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n_obstacles: int = Field(default=0, ge=0)
    min_s_distance: float = Field(default=5.0, gt=0.0)
    obstacle_width: float = Field(default=0.25, gt=0.0)
    obstacle_length: float = Field(default=0.35, gt=0.0)
    centre_deadzone: float = Field(default=0.0, ge=0.0)
    start_exclusion: float = Field(default=5.0, ge=0.0)


class Config(BaseModel):
    """Root training configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ppo: PPOConfig
    track: TrackConfig
    training: TrainingConfig
    obstacles: ObstaclesConfig = Field(default_factory=ObstaclesConfig)


def resolve(name_or_path: str) -> Path:
    """Turn a config name, file path, or run name into a config path."""
    given = Path(name_or_path)
    if given.is_file():
        return given.resolve()
    for candidate in (CONFIGS / f"{name_or_path}.yaml", RUNS / name_or_path / "config.yaml"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no config for {name_or_path!r}: tried a path, ppo/configs/{name_or_path}.yaml, "
        f"and runs/{name_or_path}/config.yaml"
    )


def _merge_track(raw: dict[str, Any]) -> None:
    """Replace the track section with the numbers from the map's track.yaml."""
    track = raw.get("track")
    if not isinstance(track, dict) or "map" not in track:
        raise KeyError("track.map")
    map_name = str(track["map"])
    path = TRACKS / map_name / "track.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"track file not found: {path}")
    with path.open() as handle:
        data = yaml.safe_load(handle)
    merged = {"map": map_name, "opt_time": data["opt_time"]}
    if data.get("max_speed_mps") is not None:
        merged["max_speed_mps"] = data["max_speed_mps"]
    raw["track"] = merged


def _set_model(cfg: BaseModel, dotted: str, value: Any) -> None:
    """Assign a dotted field on a config object, raising KeyError if any part is missing."""
    *parents, leaf = dotted.split(".")
    obj: Any = cfg
    for name in parents:
        if not isinstance(obj, BaseModel) or name not in type(obj).model_fields:
            raise KeyError(dotted)
        obj = getattr(obj, name)
    if not isinstance(obj, BaseModel) or leaf not in type(obj).model_fields:
        raise KeyError(dotted)
    setattr(obj, leaf, value)


def load(path: Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a yaml config, merge the map's track.yaml, then apply dotted overrides."""
    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise KeyError(f"config {path} is not a mapping")
    overrides = dict(overrides or {})
    if "track.map" in overrides:
        raw.setdefault("track", {})["map"] = str(overrides.pop("track.map"))
    _merge_track(raw)
    try:
        cfg = Config(**raw)
    except ValidationError as exc:
        for err in exc.errors():
            if err["type"] == "extra_forbidden":
                raise KeyError(".".join(str(part) for part in err["loc"])) from exc
        raise
    for dotted, value in overrides.items():
        _set_model(cfg, dotted, value)
    return cfg


def score(success_rate: float, pace: float) -> float:
    """Return the selection score of one evaluation."""
    return float(success_rate) + SCORE_PACE_WEIGHT * float(pace)


def dump(cfg: Config) -> str:
    """Serialize the effective config to yaml."""
    return yaml.safe_dump(cfg.model_dump(), sort_keys=False)
