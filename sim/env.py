"""ctypes interface to the RoboRacer C simulator."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
LIBRARY = HERE / "build" / "libroboracer.so"
VEHICLE_YAML = HERE / "vehicle.yaml"

OBS_SIZE = 26
COLLISION = 1
COMPLETE = 2
MAX_OBSTACLES = 16

# Policy yaw-rate normalisation [rad/s]: obs yaw rate = r / YAW_RATE_SCALE_RADPS.
YAW_RATE_SCALE_RADPS = 2.618

# Kwargs callers add to car_kwargs(): rewards and the map's lap-time target.
RUN_KEYS = (
    "progress_reward",
    "collision_reward",
    "lap_completion_reward",
    "optimal_lap_time",
)

_OBSTACLE_FIELDS = ("s", "n", "half_length", "half_width")
_VEHICLE_STATE_SIZE = 11

# Plant kwarg -> vehicle.yaml key, copied as is.
_DIRECT_KEYS = {
    "m": "mass_kg",
    "i": "inertia_kgm2",
    "lf": "length_front_m",
    "lr": "length_rear_m",
    "h": "cog_height_m",
    "ego_l": "ego_length_m",
    "ego_w": "ego_width_m",
    "mu": "tyre_mu",
    "c_sf": "cornering_stiffness_front",
    "c_sr": "cornering_stiffness_rear",
    "v_switch": "v_switch_mps",
    "tyre_s": "tyre_s",
    "alpha_pk": "alpha_peak_rad",
    "tau_v": "tau_speed_plant_s",
    "tau_a": "tau_accel_s",
    "a_max": "max_accel_mps2",
    "t_vd": "speed_transport_s",
    "t_sd": "steer_transport_s",
    "tau_delta": "tau_delta_s",
    "tau_delta_dot": "tau_delta_dot_s",
    "steer_gamma": "steer_gamma",
    "delta_th": "steer_delta_th_rad",
    "v_max": "max_speed_mps",
    "lidar_num_scans": "lidar_num_beams",
    "lidar_fov": "lidar_fov_rad",
    "lidar_max_range": "lidar_max_range_m",
    "lidar_mount_offset": "lidar_dist_m",
    "gravity": "gravity_mps2",
    "sigma_fy": "sigma_fy_m",
    "sigma_v_floor": "kinematic_floor_mps",
    "tau_beta_kin": "tau_beta_kin_s",
    "obs_noise_s_norm": "obs_noise_s_norm",
    "obs_noise_yaw_rate": "obs_noise_yaw_rate",
    "obs_noise_vel": "obs_noise_vel",
    "obs_noise_yaw": "obs_noise_yaw",
    "lidar_bias": "lidar_bias_m",
    "lidar_stochastic": "lidar_stochastic_m",
}

_lib: ctypes.CDLL | None = None

C_VOID_P = ctypes.c_void_p
C_INT = ctypes.c_int
C_FLOAT = ctypes.c_float
C_FLOAT_P = ctypes.POINTER(C_FLOAT)


def car_kwargs() -> dict[str, float]:
    """Every plant kwarg derived from vehicle.yaml, sim2real noise terms included."""
    params = yaml.safe_load(VEHICLE_YAML.read_text())
    ego_l = float(params["ego_length_m"])
    ego_w = float(params["ego_width_m"])
    lock = float(params["steer_lock_cmd_rad"])
    rate = float(params["steer_rate_max_radps"])
    kwargs = {key: float(params[source]) for key, source in _DIRECT_KEYS.items()}
    kwargs.update({
        "r_inscribed": ego_w / 2.0,
        "r_circumscribed": math.hypot(ego_l, ego_w) / 2.0,
        "s_min": -lock,
        "s_max": lock,
        "sv_min": -rate,
        "sv_max": rate,
        "v_min": 0.0,
        "r_max": YAW_RATE_SCALE_RADPS,
        "yaw_max": math.pi,
        # Periods become integer 1 ms plant ticks.
        "control_period": float(round(float(params["control_period_s"]) * 1000.0)),
        "compute_latency": float(round(float(params["compute_latency_s"]) * 1000.0)),
        "compute_latency_sd": float(params["compute_latency_sd_s"]) * 1000.0,
    })
    return kwargs


def _declare(lib: ctypes.CDLL, names: tuple[str, ...], arguments: list, result) -> None:
    """Give C functions one ctypes signature."""
    for name in names:
        function = getattr(lib, name)
        function.argtypes = arguments
        function.restype = result


def load_library() -> ctypes.CDLL:
    """Load and bind build/libroboracer.so once."""
    global _lib
    if _lib is not None:
        return _lib
    if not LIBRARY.is_file():
        raise FileNotFoundError(f"{LIBRARY} not found; build it with `make -C sim`")
    lib = ctypes.CDLL(str(LIBRARY))
    _declare(lib, ("rr_create",), [C_INT, ctypes.c_char_p, C_INT, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_double)], C_VOID_P)
    _declare(lib, ("rr_reset", "rr_step", "rr_close"), [C_VOID_P], None)
    _declare(lib, ("rr_observations", "rr_actions", "rr_rewards", "rr_terminals"), [C_VOID_P], C_FLOAT_P)
    _declare(lib, ("rr_obs_size", "rr_num_atns"), [], C_INT)
    _declare(lib, ("rr_current_lap", "rr_done_laps", "rr_done_kind"), [C_VOID_P, C_INT], C_INT)
    _declare(lib, ("rr_last_lap_time", "rr_done_lap_time"), [C_VOID_P, C_INT], C_FLOAT)
    _declare(lib, ("rr_reset_env",), [C_VOID_P, C_INT], None)
    _declare(lib, ("rr_env_ptr",), [C_VOID_P, C_INT], C_VOID_P)
    _declare(lib, ("rr_vehicle_state",), [C_VOID_P, C_INT, C_FLOAT_P], None)
    _declare(lib, ("rr_seed",), [C_VOID_P, C_INT, ctypes.c_uint], None)
    _declare(lib, ("rr_set_lap_cap", "rr_set_tick_cap"), [C_VOID_P, C_INT, C_INT], None)
    _declare(lib, ("rr_set_obstacles",), [C_VOID_P, C_INT, C_FLOAT_P, C_INT], None)
    if lib.rr_obs_size() != OBS_SIZE:
        raise RuntimeError(f"{LIBRARY} has obs size {lib.rr_obs_size()}, expected {OBS_SIZE}; rebuild with `make -C sim`")
    _lib = lib
    return lib


class VecEnv:
    """Batched C plant with zero-copy views of obs, actions, rewards, and terminals."""

    def __init__(self, num_envs: int, blob_path: str | Path, seed: int, sim2real: bool, **kwargs: float) -> None:
        self.num_envs = int(num_envs)
        track_blob = Path(blob_path)
        if not track_blob.is_file():
            raise FileNotFoundError(f"track blob not found: {track_blob}")
        missing = sorted((set(car_kwargs()) | set(RUN_KEYS)) - set(kwargs))
        if missing:
            raise ValueError(f"VecEnv kwargs missing {missing}; start from car_kwargs()")

        self._lib = load_library()
        kwargs["sim2real"] = float(sim2real)
        keys = (ctypes.c_char_p * len(kwargs))(*[key.encode() for key in kwargs])
        values = (ctypes.c_double * len(kwargs))(*[float(value) for value in kwargs.values()])
        self._vec = self._lib.rr_create(self.num_envs, str(track_blob).encode(), len(kwargs), keys, values)
        if not self._vec:
            raise RuntimeError(f"could not load track blob {track_blob}; rebuild it with `python -m sim.track <map>`")

        n_atns = int(self._lib.rr_num_atns())
        self.obs = np.ctypeslib.as_array(self._lib.rr_observations(self._vec), shape=(self.num_envs, OBS_SIZE))
        self.actions = np.ctypeslib.as_array(self._lib.rr_actions(self._vec), shape=(self.num_envs, n_atns))
        self.rewards = np.ctypeslib.as_array(self._lib.rr_rewards(self._vec), shape=(self.num_envs,))
        self.terminals = np.ctypeslib.as_array(self._lib.rr_terminals(self._vec), shape=(self.num_envs,))
        for i in range(self.num_envs):
            self.seed(i, seed * 1000 + i)

    def seed(self, i: int, seed: int) -> None:
        """Set env i's RNG; takes effect from its next reset."""
        self._lib.rr_seed(self._vec, i, seed)

    def reset(self) -> np.ndarray:
        """Reset every env and return a copy of the observations."""
        self._lib.rr_reset(self._vec)
        return self.obs.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply actions for one control period; return copies of obs, rewards, and terminals."""
        self.actions[:] = np.asarray(actions, dtype=np.float32)
        self._lib.rr_step(self._vec)
        return self.obs.copy(), self.rewards.copy(), self.terminals.copy()

    def close(self) -> None:
        """Free the C vector and drop the views that alias its buffers."""
        if getattr(self, "_vec", None) is None:
            return
        del self.obs, self.actions, self.rewards, self.terminals
        self._lib.rr_close(self._vec)
        self._vec = None

    def env_ptr(self, i: int) -> int:
        """Address of env i's C struct, for sim.render."""
        return int(self._lib.rr_env_ptr(self._vec, i))

    def reset_env(self, i: int) -> None:
        """Reset env i without latching a terminal."""
        self._lib.rr_reset_env(self._vec, i)

    def set_lap_cap(self, i: int, laps: int) -> None:
        """Laps env i must finish before COMPLETE."""
        self._lib.rr_set_lap_cap(self._vec, i, int(laps))

    def set_tick_cap(self, i: int, ticks: int) -> None:
        """Cap env i's episodes at `ticks` 1 ms plant ticks (0 = the built-in cap)."""
        self._lib.rr_set_tick_cap(self._vec, i, int(ticks))

    def set_obstacles(self, i: int, obstacles: list[dict[str, float]]) -> None:
        """Install static track-frame rectangles (s, n, half_length, half_width) on env i after reset."""
        n = len(obstacles)
        if n > MAX_OBSTACLES:
            raise ValueError(f"{n} obstacles exceeds MAX_OBSTACLES={MAX_OBSTACLES}")
        buf = (C_FLOAT * (MAX_OBSTACLES * len(_OBSTACLE_FIELDS)))(
            *[float(item[key]) for item in obstacles for key in _OBSTACLE_FIELDS]
        )
        self._lib.rr_set_obstacles(self._vec, i, buf, n)

    def current_lap(self, i: int) -> int:
        """Laps completed so far in env i's running episode."""
        return int(self._lib.rr_current_lap(self._vec, i))

    def last_lap_time(self, i: int) -> float:
        """Seconds of env i's most recent lap in the running episode (0 if none)."""
        return float(self._lib.rr_last_lap_time(self._vec, i))

    def done_laps(self, i: int) -> int:
        """Laps of env i's last finished episode."""
        return int(self._lib.rr_done_laps(self._vec, i))

    def done_lap_time(self, i: int) -> float:
        """Last lap time of env i's last finished episode."""
        return float(self._lib.rr_done_lap_time(self._vec, i))

    def done_kind(self, i: int) -> int:
        """Outcome of env i's last finished episode: 0 tick cap, COLLISION, or COMPLETE."""
        return int(self._lib.rr_done_kind(self._vec, i))

    def vehicle_state(self, i: int) -> np.ndarray:
        """True state of env i: x, y, yaw, delta, v, r, beta, accel, delta_dot, fy_f, fy_r."""
        out = np.empty(_VEHICLE_STATE_SIZE, dtype=np.float32)
        self._lib.rr_vehicle_state(self._vec, i, out.ctypes.data_as(C_FLOAT_P))
        return out
