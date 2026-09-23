# Third-party notices

## TUM raceline optimisation (LGPL-3.0)

`trackgen/stages/raceline.py` and `trackgen/stages/mintime.py` contain modified code from the
Institute of Automotive Technology, Technical University of Munich:

- [trajectory_planning_helpers](https://github.com/TUMFTM/trajectory_planning_helpers)
  (Alexander Heilmeier, Tim Stahl et al.): spline, curvature, minimum-curvature and raceline helpers.
- [global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization)
  (Alexander Heilmeier, Fabian Christ, Thomas Herrmann, Francesco Passigato et al.): track
  preparation and the minimum-time optimal control problem.

Both are licensed under the GNU Lesser General Public License v3.0, and so are these two modified
files: see `LICENSES/LGPL-3.0.txt` and `LICENSES/GPL-3.0.txt`. The rest of the repository uses them
through their Python interface and is MIT licensed.

## PufferLib (MIT)

The vectorised environment harness in `sim/csrc/` (`vecenv.h`, the kwargs and `my_init`/`c_step`
environment interface) follows [PufferLib](https://github.com/PufferAI/PufferLib)'s environment
template.

```
MIT License

Copyright (c) 2022 PufferAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
