English | [繁體中文](README.zh-TW.md)

# quadsim: quadruped ROS 2 navigation sim MVP (MuJoCo)

**quadsim** turns a merged LiDAR SLAM point cloud (PCD) into a navigation test environment for quadruped robots.
- **Scene builder (`pcd2sim`)**: voxelizes the map, merges voxels into collision boxes, and flattens noisy floors into planar slabs. Builds in seconds.
- **Simulator (`quadsim`)**: runs an ideal-base quadruped in MuJoCo. A Mid-360-style rolling LiDAR with per-point timestamps is ray-cast with Embree. Also provides an IMU and ground truth.
- **ROS 2 interface**: publishes `livox_ros_driver2/CustomMsg`, `PointCloud2`, IMU, `/clock`, ground-truth odometry and events; takes `/cmd_vel` and `/sim/set_pose`.
- Designed to plug straight into Point-LIO / FAST-LIO and a Nav2-based 3D navigation stack. Runs in Docker on ROS 2 Humble.

The robot is an **ideal base** (no legs are simulated): it follows `/cmd_vel` and hugs the terrain.

Design background: [`docs/pcd-to-sim-architecture.md`](docs/pcd-to-sim-architecture.md) (Traditional Chinese, §12–§13).

## Quick start
```bash
# 1. Build the Docker image (based on syncai-robot-base, adds mujoco and the livox CustomMsg)
docker build -t quadsim:latest docker/

# 2. Maps/*.pcd -> scenes/<name>/ (about 2-5 s per map)
scripts/build_scenes.sh

# 3. Start the sim (defaults to scenes/dp2f, real time)
scripts/run_sim.sh
scripts/run_sim.sh --scene scenes/dpmap --no-gt-tf      # no ground-truth TF: leave it to Point-LIO and global localization

# 4. Check
docker/run.sh "python3 -m pytest -q"                    # unit tests
docker/run.sh "python3 scripts/bench.py --scene scenes/dp2f physics.rtf=0"   # headless benchmark
scripts/smoke_test.sh scenes/dp2f                       # ROS 2 end-to-end test (rates, TF, driving, collision, set_pose)

# GUI (needs an X display; run from a desktop terminal, DISPLAY and XAUTHORITY are passed into the container)
QUADSIM_GUI=1 scripts/run_sim.sh --viewer                # MuJoCo viewer (camera follows the body; the map is visual only)
QUADSIM_GUI=1 docker/run.sh "rviz2 -d rviz/quadsim.rviz --ros-args -p use_sim_time:=true"
```
The GUI uses Mesa software rendering by default (`LIBGL_ALWAYS_SOFTWARE=1`). Without it the container falls back to GL 2.1, and RViz draws no point clouds or grid: the whole view is black. On a machine with a GPU, set `QUADSIM_GL=hw` to pass `/dev/dri` into the container. In the MuJoCo viewer, press `3` to show the active collision proxy boxes (red).
The container uses `--net=host` and the same DDS settings as WebRvizLite: CycloneDDS (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`), `ROS_DOMAIN_ID=1`, and `conf/cyclonedds.xml` (unicast on loopback, multicast off), so the WebRvizLite viewer connects without extra setup. A navigation stack on the host or in another container connects directly as long as it uses the same RMW and domain (override with `ROS_DOMAIN_ID` / `CYCLONEDDS_CONFIG=<file under conf/>` when running `docker/run.sh`). **Every other node must set `use_sim_time:=true`**.

## ROS 2 interface
| Direction | Topic | Type | Rate |
|---|---|---|---|
| Pub | `/clock` | rosgraph_msgs/Clock | 200 Hz |
| Pub | `/livox/lidar` | livox_ros_driver2/CustomMsg (`offset_time` = per-point time, ns) | 10 Hz |
| Pub | `/livox/points` | PointCloud2 (livox layout: x y z intensity tag line timestamp; switchable to velodyne layout: ring, time) | 10 Hz |
| Pub | `/livox/imu` | sensor_msgs/Imu (acceleration in **g**, like livox_ros_driver2; no orientation) | 200 Hz |
| Pub | `/sim/ground_truth/odom` | nav_msgs/Odometry (map frame, **for evaluation only**) | 50 Hz |
| Pub | `/sim/events` | std_msgs/String (JSON: `collision` / `stuck` / `set_pose`) | on event |
| Pub | `/sim/map_cloud` | PointCloud2 (latched, voxelized map) | once |
| Sub | `/cmd_vel` | geometry_msgs/Twist (vx, vy, wz) | — |
| Sub | `/sim/set_pose` | geometry_msgs/PoseStamped (x, y, yaw; **z is a floor-height hint**) | — |

**TF**: the static `base_link → livox_frame` is always published. With `publish_gt_tf: true` (default), `map → odom` (identity) and `odom → base_link` (ground truth) are added. After `--no-gt-tf`, your localization stack publishes those two.

All parameters (topics, frames, noise, body size, velocity limits, …) live in [`config/sim.yaml`](config/sim.yaml). They can be overridden on the command line, e.g. `scripts/run_sim.sh lidar.range_noise=0.0 robot.vel_limits=[1.0,0.5,1.5]`.

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
  - Turn it off with `build_scene --no-flatten` to keep the raw voxel floor
- **Spawn and `/sim/set_pose` check whether the body overlaps the map**: if it does, the base moves to the nearest free pose within 1 m and a warning is issued
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
| `scripts/` | `build_scenes.sh`, `run_sim.sh`, `smoke_test.sh`, `bench.py`, `ros_smoke_test.py` |
| `docker/` | Dockerfile, `run.sh`, minimal `livox_ros_driver2` message package |
| `tests/` | unit tests (PCD, voxels and box merging, ground flattening, proxies, sensor timing, collision check, CustomMsg/PointCloud2 serialization) |
