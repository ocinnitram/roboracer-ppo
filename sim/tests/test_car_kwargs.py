"""car_kwargs maps sim/vehicle.yaml onto the C my_init keys."""

from __future__ import annotations

import math

import pytest

from sim.env import car_kwargs

# vehicle.yaml values after unit conversion
EXPECTED = {
    "m": 3.74,
    "i": 0.04712,
    "lf": 0.15875,
    "lr": 0.17145,
    "h": 0.074,
    "ego_w": 0.29,
    "ego_l": 0.5,
    "r_inscribed": 0.145,
    "r_circumscribed": 0.2890069203323685,
    "mu": 0.42,
    "c_sf": 4.718,
    "c_sr": 5.4562,
    "v_switch": 7.319,
    "tyre_s": 1.20,
    "alpha_pk": 0.16,
    "tau_v": 0.118,
    "tau_a": 0.061,
    "a_max": 3.48,
    "t_vd": 0.020,
    "t_sd": 0.060,
    "tau_delta": 0.075,
    "tau_delta_dot": 0.027,
    "steer_gamma": 0.622,
    "delta_th": 0.033,
    "s_min": -0.4189,
    "s_max": 0.4189,
    "sv_min": -3.2,
    "sv_max": 3.2,
    "v_max": 4.0,
    "v_min": 0.0,
    "r_max": 2.618,
    "yaw_max": math.pi,
    "lidar_num_scans": 811.0,
    "lidar_fov": 4.711932,  # 810 x 0.0058172, the span the scan header reports
    "lidar_max_range": 10.0,
    "lidar_mount_offset": 0.10355,
    "control_period": 67.0,
    "compute_latency": 20.0,
    "compute_latency_sd": 0.0,
    "gravity": 9.81,
    "sigma_fy": 0.03,
    "sigma_v_floor": 0.5,
    "tau_beta_kin": 0.15,
    "obs_noise_s_norm": 0.0006,
    "obs_noise_yaw_rate": 0.015,
    "obs_noise_vel": 0.0052,
    "obs_noise_yaw": 0.0042,
    "lidar_bias": 0.06,
    "lidar_stochastic": 0.02,
}


def test_car_kwargs_keys_and_values() -> None:
    got = car_kwargs()
    assert set(got) == set(EXPECTED)
    for key, expected in EXPECTED.items():
        assert got[key] == pytest.approx(expected, abs=1e-6, rel=0.0), key
