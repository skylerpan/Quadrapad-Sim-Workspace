English | [繁體中文](README.zh-TW.md)

# quadsim: quadruped ROS 2 navigation sim MVP (MuJoCo)

**quadsim** turns a merged LiDAR SLAM point cloud (PCD) into a navigation test environment for quadruped robots.
- **Scene builder (`pcd2sim`)**: voxelizes the map, merges voxels into collision boxes, and flattens noisy floors into planar slabs. Builds in seconds.
- **Simulator (`quadsim`)**: runs an ideal-base quadruped in MuJoCo. A Mid-360-style rolling LiDAR with per-point timestamps is ray-cast with Embree. Also provides an IMU and ground truth.
- **ROS 2 interface**: publishes `livox_ros_driver2/CustomMsg`, `PointCloud2`, IMU, `/clock`, ground-truth odometry and events; takes `/cmd_vel` and `/sim/set_pose`.
- Designed to plug straight into Point-LIO / FAST-LIO and a Nav2-based 3D navigation stack. Runs in Docker on ROS 2 Humble.

The robot is an **ideal base** (no legs are simulated): it follows `/cmd_vel` and hugs the terrain.

**Usage guide** with every command, flag and environment variable, organised by task: [`docs/usage.md`](docs/usage.md) (Traditional Chinese). Design background: [`docs/pcd-to-sim-architecture.md`](docs/pcd-to-sim-architecture.md) (Traditional Chinese, §12–§13). Navigation goal-loop test record on dp_1008 with the failure analysis: [`docs/nav-goal-loop-test-2026-10-13.md`](docs/nav-goal-loop-test-2026-10-13.md) (Traditional Chinese).

## Quick start
```bash
# 1. Build the Docker image (based on syncai-robot-base, adds mujoco and the livox CustomMsg)
docker build -t quadsim:latest docker/

# 2. Maps/*.pcd -> scenes/<name>/ (about 2-5 s per map)
scripts/build_scenes.sh

# 3. Start the sim (defaults to scenes/dp_1008, real time)
scripts/run_sim.sh
scripts/run_sim.sh --scene scenes/dp_1008_fillC --no-gt-tf   # no ground-truth TF: leave it to Point-LIO and global localization

# 4. Check
docker/run.sh "python3 -m pytest -q"                    # unit tests
docker/run.sh "python3 scripts/bench.py --scene scenes/dp_1008 physics.rtf=0"   # headless benchmark
scripts/smoke_test.sh scenes/dp_1008                    # ROS 2 end-to-end test (rates, TF, driving, collision, set_pose)

# GUI (needs an X display; run from a desktop terminal, DISPLAY and XAUTHORITY are passed into the container)
QUADSIM_GUI=1 scripts/run_sim.sh --viewer                # MuJoCo viewer (camera follows the body; the map is visual only)
QUADSIM_GUI=1 docker/run.sh "rviz2 -d rviz/quadsim.rviz --ros-args -p use_sim_time:=true"
```
`QUADSIM_GL` picks the GL path for the GUI: `nvidia` (NVIDIA GPU through the nvidia container runtime; the default when the host has `nvidia-smi` or a docker `nvidia` runtime, e.g. a Jetson), `hw` (Mesa with `/dev/dri`, Intel/AMD) or `sw` (Mesa llvmpipe, `LIBGL_ALWAYS_SOFTWARE=1`; the default otherwise, since the plain GL 2.1 fallback draws no RViz point clouds or grid). The MuJoCo viewer draws the whole map as one mesh (dp_1008: 3.1 M triangles), so the GL path decides whether it is usable; measured at 1280x720 on a Jetson Thor:

| GL path | shadows on | shadows off (default) |
|---|---|---|
| `sw` (llvmpipe) | 815 ms/frame | 391 ms/frame |
| `nvidia` (Thor GPU) | 43 ms/frame | 16 ms/frame |

The `viewer:` section of the config sets `shadows` (the map casting shadows on itself costs a second pass over the mesh) and `sync_hz` (the maximum wall-clock rate at which the sim pushes its state to the window; each push takes the render lock). The sim logs `viewer: gl=... map mesh ... faces` at start-up so the path in use is visible. In the MuJoCo viewer, press `3` to show the active collision proxy boxes (red).
The container uses `--net=host` and the same DDS settings as WebRvizLite: CycloneDDS (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`), `ROS_DOMAIN_ID=1`, and `conf/cyclonedds.xml` (unicast on loopback, multicast off), so the WebRvizLite viewer connects without extra setup. A navigation stack on the host or in another container connects directly as long as it uses the same RMW and domain (override with `ROS_DOMAIN_ID` / `CYCLONEDDS_CONFIG=<file under conf/>` when running `docker/run.sh`). **Every other node must set `use_sim_time:=true`**.

## ROS 2 interface
| Direction | Topic | Type | Rate |
|---|---|---|---|
| Pub | `/clock` | rosgraph_msgs/Clock | 200 Hz |
| Pub | `/livox/lidar` | livox_ros_driver2/CustomMsg (`offset_time` = per-point time, ns) | 10 Hz |
| Pub | `/livox/points` | PointCloud2 (livox layout: x y z intensity tag line timestamp; switchable to velodyne layout: ring, time) | 10 Hz |
| Pub | `/livox/imu` | sensor_msgs/Imu (acceleration in **g**, like livox_ros_driver2; no orientation) | 200 Hz |
| Pub | `/sim/ground_truth/odom` | nav_msgs/Odometry (map frame, **for evaluation only**) | 50 Hz |
| Pub | `/sim/events` | std_msgs/String (JSON: `spawn` / `set_pose` / `collision` / `stuck`; transient_local, so a late subscriber can still read the start-up `spawn` event) | on event |
| Pub | `/sim/map_cloud` | PointCloud2 (latched, voxelized map) | once |
| Sub | `/cmd_vel` | geometry_msgs/Twist (vx, vy, wz) | — |
| Sub | `/sim/set_pose` | geometry_msgs/PoseStamped (x, y, yaw; **z is a floor-height hint**) | — |

**TF**: the static `base_link → livox_frame` is always published. With `publish_gt_tf: true` (default), `map → odom` (identity) and `odom → base_link` (ground truth) are added. After `--no-gt-tf`, your localization stack publishes those two.

All parameters (topics, frames, noise, body size, velocity limits, …) live in [`config/sim.yaml`](config/sim.yaml). They can be overridden on the command line, e.g. `scripts/run_sim.sh lidar.range_noise=0.0 robot.vel_limits=[1.0,0.5,1.5]`.

## Navigation goal-loop test
With the sim and a navigation stack running (the `robot01` container: Point-LIO localization, `octo_planner_rviz_node`, `syncai_controller_3d`), `scripts/nav_goal_loop.py` sends map-frame waypoints one after another and verifies every leg against the sim's ground truth. It runs inside the sim container so it joins the same DDS domain:
```bash
scripts/nav_goal_loop.sh --dry-run                           # plan only: which legs the planner can solve (robot does not move)
scripts/nav_goal_loop.sh --dry-run --dry-run-from matrix --select wp01,wp03,wp05   # reachability table between points
scripts/nav_goal_loop.sh --select wp29,wp30 --loops 1        # drive to a few named points
scripts/nav_goal_loop.sh --loops 3 --order shuffle --seed 1  # soak: 3 passes in a different order each time
scripts/nav_goal_loop.sh --help                              # all tolerances, timeouts and topic names
```
Per leg it follows the same protocol as `syncai_waypoint_sequencer`: publish the current `/robot01/tracked_pose` to `/initialpose` (the planner does not read the robot pose itself), publish the waypoint to `/goal_pose`, accept the first `/robot01/planned_path` received after the goal that is non-empty and ends near the waypoint, send it to the `/robot01/follow_path` action and wait for the result. The planner's `auto_follow_path` parameter is set to false for the run (otherwise the planner would send the path too) and restored on exit, including on Ctrl-C.

A leg passes when the action reports SUCCEEDED, the ground-truth position is within `--xy-tol` (0.35 m) of the path endpoint, the base has stopped, no `collision`/`stuck` event was reported on `/sim/events` during the leg, and the localization error (`tracked_pose` vs ground truth) stayed below `--loc-fail` (1 m). Failure reasons: `NO_PATH`, `ACTION_REJECTED`, `ABORTED`, `TIMEOUT`, `COLLISION`, `STUCK`, `STILL_MOVING`, `GT_NOT_AT_GOAL`, `LOC_ERROR`. The run stops after 3 consecutive failures (`--max-consecutive-fail`). Exit code 0 = all legs passed, 1 = a leg failed, 2 = preflight failed, 130 = interrupted.

Results go to `results/nav_goal_loop/<timestamp>/` (`report.json`, `legs.csv`, `run.log`), one line per leg:
```
LEG 1 loop1 robot->wp29  path 7.2 m plan 0.05 s  exec 13.9 s SUCCEEDED  gt-dist 0.18 (wp 0.28) loc 0.05/0.26  PASS
```
Waypoints live in `config/nav_waypoints_dp_1008.yaml` (same `name, x, y, z, yaw` keys as the sequencer's YAML). `scripts/make_waypoints_from_poses.py Maps/<map>/poses.txt --spacing 10 --z-min -1 --z-max 0.5 -o <yaml>` samples them from the SLAM keyframe trajectory, which is the surest source of reachable points (one storey selected by z); run `--dry-run` afterwards and drop the points the planner cannot reach. The test never teleports the robot (`/sim/set_pose` would break localization): each run starts from wherever the robot is, at the nearest waypoint.

## Architecture
```
PCD ──build_scene──► sparse voxels (10 cm) ─┬─ ground ──► flattened slabs (sloped triangle mesh) ──┬─► Embree (open3d): LiDAR + terrain rays under the feet
                     + per-voxel mean       └─ rest ────► run-merged boxes ────────────────────────┤
                       intensity / z                                                                 └─► collision proxy pool: ~128 mocap boxes moved every
                                                                                                         control tick onto the map boxes near the body (no ground)
MuJoCo: free-body base ← soft weld ← mocap target (integrated cmd_vel + terrain following); IMU = MuJoCo sensors
```
- **The MuJoCo model is tiny** (body plus 128 proxy boxes, 130 geoms) regardless of map size. The full map geometry lives only in Embree
- LiDAR and collision use **the same boxes**, so what the LiDAR sees is what the robot hits
- **Ground flattening** (`pcd2sim/ground.py`):
  - **Regions**: walkable ground cells whose neighbours differ by less than 6 cm form one region. Stairs and platforms split into separate regions, so their edges stay sharp
  - **Plane fit**: each region (or each 4 m tile of it) is fitted with a plane that may be sloped, absorbing SLAM tilt
  - **Replacing floor voxels**: floor voxels around the plane are replaced by 0.2 m thick slabs. Slab sides form stair risers; seen from the floor below, slab bottoms form the ceiling
  - **Noise cleanup**:
    - noise spikes next to the floor, small holes, and small clusters floating above the floor are removed
    - voxels in the first layer above the plane with only 1–2 points and nothing on top (the upper tail of the floor noise) are removed too
  - **Gaps**: a Mid-360 cannot see the floor right around the robot, so SLAM floors have columns with no points at all, and the flattening seams leave more. Columns with at least 3 floor neighbours and nothing standing on them are filled as floor (`--ground-fill 12 --ground-fill-nb 3`; dp_1008 1F went from 0.37% to 0.17% no-ground samples along the mapping path). The base additionally tolerates one foot over a remaining gap (`robot.max_feet_over_gap: 1`, that foot takes the height of the plane through the other three) instead of refusing to move with a `stuck/no_ground` event
  - Turn it off with `build_scene --no-flatten` to keep the raw voxel floor
- **Spawn and `/sim/set_pose` check whether the body overlaps the map**: if it does, the base moves to the nearest free pose within 1 m (up to 90° turned) and the TF/odom follow the *moved* pose. The start-up log prints `spawn requested [x, y, yaw, z_floor] -> base at [...]`, a ROS warning is logged when the base was moved, and a `spawn` event (`requested`, `pose`, `floor_z`, `moved`) goes out on `/sim/events`; `set_pose` events carry the same fields
- **Spawn is set in `config/sim.yaml`** (`spawn: [x, y, yaw, z_floor]`); `null` takes the point `build_scene` wrote to `scenes/<name>/meta.json`. Do not edit `meta.json` by hand: it is a build cache that the next build overwrites. `z_floor` must be the floor height of the storey you want (the ground is searched from 0.5 m above it down to 1 m below it): a z between floors stands the base on clutter or a wall edge, which then triggers the relocation above
- **Rolling LiDAR**: every ray is cast from the sensor pose nearest its emission time (at most half a physics step off), so motion distortion appears naturally. Points are in the sensor frame at emission time, like the real driver
- Scan pattern: by default a synthetic Mid-360-like non-repetitive pattern (one azimuth revolution per frame, elevation −7°…52° from a low-discrepancy sequence). `lidar.pattern: <csv>` loads a measured pattern table instead (columns `t,azimuth_deg,elevation_deg[,line]`)
- Reflectivity comes from the mean intensity of the original PCD points in each voxel
- Ideal base: first-order lag, velocity and acceleration limits. Each of the 4 feet samples the ground with 3×3 rays to get height, roll and pitch (robust plane fit). Moves are rejected with an event when a single step is too high, the slope is too steep, there is no ground, or the body would collide

## Benchmarks (aarch64, 6 cores, no GPU, single Python process)
| Scene | Points | Voxels / boxes | Build | Sim RTF (headless) | Points/frame | Static frame vs map NN P95 |
|---|---|---|---|---|---|---|
| dp2f (single floor) | 1.25 M | 485 k / 154 k (ground 94% planar, 4.8 k slabs) | 2.4 s, 179 MB | **9.1×** | ~19.7 k | 9.3 cm |
| dpmap (multi-floor) | 3.15 M | 1.03 M / 300 k (ground 98% planar, 11 k slabs) | 5.7 s, 376 MB | **8.7×** | ~17.0 k | 9.6 cm |

At real time (rtf=1) the sim uses about 10% of one core. Time split: LiDAR 72–75%, control 19–21%, physics 6–7%.
5-minute soak test (rtf=1, with subscribers): the sim's RSS stays at 527 MB with no growth. A slow subscriber (1 s per message) does not slow down `/clock` or the RTF.

Body roll / pitch standard deviation while walking, before → after ground flattening:
- dp2f: 0.8° / 0.7° → 0.04° / 0.22°
- dpmap: 0.7° / 1.8° → 0°

## Known limitations (MVP)
- **No legs**: no gait dynamics and no foot contacts. Testing gait stability needs your own locomotion policy (mode A in the design doc)
- **10 cm voxels**: both PCDs are already downsampled to ~5 cm point spacing. 5 cm voxels leave too many holes in surfaces, so 10 cm is the default. Thin objects get thicker, and doorways can lose up to 10 cm of width
- **Ground flattening trade-offs**:
  - real height changes below ~6 cm (thin thresholds, cables) are merged into the floor plane
  - the bottoms of low objects within 10 cm of the floor are replaced by the slab
  - curved or cluttered areas (about 2–6% of ground cells) are median-smoothed rather than perfectly planar
  - the planes differ from the original points by ~2 cm RMS, the same order as the map's own noise
- The synthetic scan pattern only approximates the Mid-360. Recording a rosbag from the real sensor and deriving a pattern table to load as CSV is recommended
- No dynamic obstacles (can be added later with an Embree dynamic scene or mocap bodies)
- The z of `/sim/set_pose` should be close to the floor height; it selects the floor in multi-floor maps

## Files
| Path | Contents |
|---|---|
| `pcd2sim/` | PCD I/O, voxelization, denoising, box merging, ground flattening (`ground.py`), floor and spawn analysis, `build_scene` CLI |
| `quadsim/` | `sim.py` core (no ROS), `lidar.py`, `base.py`, `proxies.py`, `raycast.py`, `model.py`, `ros_io.py`, `sim_node.py` |
| `config/sim.yaml` | all parameters |
| `config/nav_waypoints_dp_1008.yaml` | waypoints for the navigation goal-loop test on dp_1008 (1F) |
| `scripts/` | `build_scenes.sh`, `run_sim.sh`, `smoke_test.sh`, `bench.py`, `ros_smoke_test.py`, `nav_goal_loop.sh` / `nav_goal_loop.py` (navigation goal-loop test), `make_waypoints_from_poses.py` |
| `docker/` | Dockerfile, `run.sh`, minimal `livox_ros_driver2` message package |
| `docs/` | `usage.md` (usage guide by task), `pcd-to-sim-architecture.md` (design), `nav-goal-loop-test-2026-10-13.md` (test record) |
| `tests/` | unit tests (PCD, voxels and box merging, ground flattening, proxies, sensor timing, collision check, CustomMsg/PointCloud2 serialization) |
