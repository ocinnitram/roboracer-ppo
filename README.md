# roboracer-ppo

PPO racing policy for a 1:10 RoboRacer (F1TENTH) car, trained on the modified C simulator which is taken from the F1Tenth Gym. Contains actuator response modelling, tyre dynamics and sim2real noise for transferring the policy to the physical platform.

## Setup

Linux with `gcc` (OpenMP), `make` and [uv](https://docs.astral.sh/uv/). uv installs the pinned
Python and dependencies.

```bash
./setup    # Python env, sim/build/libroboracer.so, track blobs
./test     # C simulator tests + pytest
```

## Pretrained policies

`pretrained/empty` and `pretrained/obstacles` hold the trained actors (seeds 17, 76, 693) of the two
default configs on `lab-20260906`:

```bash
./eval pretrained/empty       # success 1.00, lap 7.08 s (mean of 3 seeds)
./eval pretrained/obstacles   # success 0.94 on 500 layouts, lap 7.38 s
./eval pretrained/obstacles --render   # watch it drive the layouts live
```

## Train and evaluate

```bash
./train                 # empty track, 5M steps       -> runs/empty/
./eval                  # evaluate runs/empty on the empty track
./train obstacles       # 2 random o per episode, 10M steps -> runs/obstacles/
./eval obstacles        # evaluate on the track's 500 fixed obstacle layouts
./eval obstacles --render   # watch the policy drive the layouts live
```

`--render` opens a raylib window and drives one seed (`--seed N`, default the first) through the
layouts in real time instead of scoring them, with the same outcomes as the scored eval; `--layout K`
starts at obstacle layout K (also for an empty-track run). Mouse wheel zooms, drag pans, closing the window stops. The first `--render`
downloads raylib 5.5 and builds `sim/build/librender.so` (`make -C sim render`); it needs OpenGL
(Ubuntu: `apt install libgl1 libglx-mesa0`).

Options: `--seed N` (default 17), `--steps N`, `--name NAME`, `--track MAP`, and
`--layout-switch complete` to keep a be layout until the car finishes its laps on it instead of
redrawing it every episode (run `obstacles-complete`). A one-minute check:
`./train --steps 40000 --name test-policy && ./eval test-policy --render`. `./eval RUN_NAME --empty` evaluates an obstacle
policy on the empty track, `./eval RUN_NAME --layout x` a policy on the obstacle layout x. Hyper-parameters live in `ppo/configs/{empty,obstacles}.yaml`.

Each run writes `runs/<name>/`: `config.yaml`, checkpoints, `progress.csv`, and `eval/` results.
`--wandb` also logs to Weights & Biases (after `uv run wandb login`; project `roboracer-ppo`, or set
`WANDB_PROJECT`/`WANDB_ENTITY`); `WANDB_MODE=offline` needs no account and can be synced later.

## New track

Put a ROS map_server occupancy map in `tracks/<map>/` (`<map>.yaml` naming a `.pgm`), then:

```bash
./track <map> --direction cw     # centreline, raceline, lap time, obstacle zones, eval layouts
./train --track <map>            # -> runs/empty-<map>
./train obstacles --track <map>  # -> runs/obstacles-<map>
```

`./track` shows each stage in a plot window; in the zones window, drag the dials to where obstacles may
be placed and press Save. Headless: `--no-plot --zones a:b,c:d` (station ranges in metres along the
raceline). `--from STAGE` restarts at a stage; `--vmax MPS` caps the car's speed on that map.

## Layout

```
sim/          C simulator (csrc/), Python binding (env.py), track compiler (track.py), car (vehicle.yaml)
ppo/          PPO agent, training loop, configs
evaluation/   ./eval and the fixed obstacle layouts (layouts/<map>/)
trackgen/     occupancy map -> tracks/<map>/
tracks/       track data per map
pretrained/   trained policies for the two default configs
```

## License

MIT (`LICENSE`), except `trackgen/stages/raceline.py` and `mintime.py`, which are modified LGPL-3.0
code from TUM; see `THIRD_PARTY_NOTICES.md`.
