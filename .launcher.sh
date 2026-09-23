# Sourced by the launchers: run from the repo root and require uv.
cd "$(dirname "${BASH_SOURCE[1]}")" || exit 1
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
# A sourced ROS workspace leaks site-packages through PYTHONPATH into the venv.
unset PYTHONPATH
