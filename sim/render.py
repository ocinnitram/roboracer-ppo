"""ctypes interface to the optional raylib window, build/librender.so (`make -C sim render`)."""

from __future__ import annotations

import ctypes

from sim.env import HERE, VecEnv

LIBRARY = HERE / "build" / "librender.so"
GL_HINT = "install libGL (Ubuntu: apt install libgl1 libglx-mesa0) and run with a display"

RUNNING = -1
TIMEOUT = 0
CRASHED = 1
COMPLETED = 2


class Window:
    """One raylib window showing one env of a VecEnv."""

    def __init__(self) -> None:
        if not LIBRARY.is_file():
            raise FileNotFoundError(f"{LIBRARY} not found; build it with `make -C sim render`")
        # raylib aborts inside InitWindow when OpenGL is missing, so check first.
        try:
            ctypes.CDLL("libGL.so.1")
        except OSError:
            raise SystemExit(f"--render needs OpenGL: {GL_HINT}") from None
        self._lib = ctypes.CDLL(str(LIBRARY))
        self._lib.render_frame.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.render_frame.restype = ctypes.c_int
        self._lib.render_close.argtypes = []
        self._lib.render_close.restype = None

    def draw(self, venv: VecEnv, i: int, label: str, outcome: int = RUNNING) -> bool:
        """Draw env i (its last running frame once outcome is set); False once the window is closed."""
        status = self._lib.render_frame(venv.env_ptr(i), label.encode(), outcome)
        if status < 0:
            raise SystemExit(f"could not open the render window: {GL_HINT}")
        return status > 0

    def close(self) -> None:
        """Close the window."""
        self._lib.render_close()
