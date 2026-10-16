# quadsim 使用手冊

依「要做什麼」分章，每章給指令、常用參數和注意事項。完整參數清單一律以程式的 `--help` 為準。
專案概觀、架構與實測數據見 [README.zh-TW.md](../README.zh-TW.md)；設計背景見 [pcd-to-sim-architecture.md](pcd-to-sim-architecture.md) §12–§13；導航測試的實測紀錄見 [nav-goal-loop-test-2026-10-13.md](nav-goal-loop-test-2026-10-13.md)。

## 0. 總覽

主機上沒有 ROS 2，**所有 Python 程式都在 quadsim 容器內執行**；`scripts/*.sh` 和 `docker/run.sh` 在主機上跑，會自己把指令送進容器。

| 我要… | 指令 | 在哪裡跑 | 章節 |
|---|---|---|---|
| 建 Docker 映像檔 | `docker build -t quadsim:latest docker/` | 主機 | [1](#1-環境與容器) |
| 進容器開 shell | `docker/run.sh` | 主機 | [1](#1-環境與容器) |
| 把 PCD 地圖轉成場景 | `scripts/build_scenes.sh` | 主機 | [2](#2-建場景pcd--scenes) |
| 啟動模擬器 | `scripts/run_sim.sh [參數]` | 主機 | [3](#3-跑模擬) |
| 開 MuJoCo viewer / RViz | `QUADSIM_GUI=1 scripts/run_sim.sh --viewer` | 主機（桌面終端機） | [4](#4-畫面gui) |
| 看 topic / frame 名稱 | 讀 `config/sim.yaml` 的 `ros:` 區塊 | — | [5](#5-ros-2-介面) |
| 跑單元測試 | `docker/run.sh "python3 -m pytest -q"` | 主機 | [6.1](#61-單元測試) |
| 量模擬器效能（不用 ROS） | `docker/run.sh "python3 scripts/bench.py ..."` | 主機 | [6.2](#62-headless-效能測試) |
| ROS 2 端到端冒煙測試 | `scripts/smoke_test.sh [scene]` | 主機 | [6.3](#63-ros-2-冒煙測試) |
| 自動連續下導航點 | `scripts/nav_goal_loop.sh [參數]` | 主機 | [7](#7-導航下點迴圈測試) |
| 從 SLAM 軌跡產生 waypoint | `scripts/make_waypoints_from_poses.py` | 主機或容器 | [8](#8-產生-waypoint) |
| 改參數 | 編 `config/sim.yaml` 或指令列 `key=value` | — | [9](#9-設定檔與參數覆寫) |
| 出問題 | — | — | [10](#10-常見問題) |

## 1. 環境與容器

```bash
docker build -t quadsim:latest docker/      # 以 syncai-robot-base 為基底，加裝 mujoco、scipy、open3d、pytest 和 livox CustomMsg
docker/run.sh                               # 互動 shell（repo 掛在容器的 /ws）
docker/run.sh "python3 -m pytest -q"        # 在容器內執行一條指令後退出
```

`docker/run.sh` 用 `--rm --net=host --ipc=host` 啟動，repo 目錄掛成 `/ws`，進去前會 source ROS Humble 和 livox 訊息套件。環境變數：

| 變數 | 預設 | 用途 |
|---|---|---|
| `QUADSIM_IMAGE` | `quadsim:latest` | 要用的映像檔 |
| `QUADSIM_GUI` | `0` | `1` = 轉送 X11（需要 `DISPLAY`；`XAUTHORITY` 存在時一併帶入） |
| `QUADSIM_GL` | 主機有 `nvidia-smi` 或 docker `nvidia` runtime → `nvidia`，否則 `sw` | GUI 的 GL 路徑：`nvidia`（NVIDIA GPU）、`hw`（Mesa + `/dev/dri`，Intel/AMD）、`sw`（llvmpipe 軟體繪圖） |
| `ROS_DOMAIN_ID` | `1` | DDS domain，和 WebRvizLite 相同 |
| `CYCLONEDDS_CONFIG` | `cyclonedds.xml` | `conf/` 底下的 CycloneDDS 設定檔名 |

DDS 條件：CycloneDDS（`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`）、domain 1、`conf/cyclonedds.xml`（loopback 單播、關閉 multicast）。導航 stack 或其他工具只要用相同 RMW 與 domain 就能直接連上。**sim 擁有 `/clock`，其他所有節點都要設 `use_sim_time:=true`。**

要在同一台機器上另外起一個不互相干擾的 sim（例如跑測試時），給不同的 domain：`ROS_DOMAIN_ID=77 docker/run.sh ...`。

## 2. 建場景（PCD → scenes/）

把 `Maps/<name>.pcd` 轉成模擬用的場景目錄 `scenes/<name>/`。`Maps/*.pcd` 和 `scenes/` 都在 `.gitignore` 裡，要自己放。

```bash
scripts/build_scenes.sh                       # Maps/*.pcd 全部建，額外參數會傳給 build_scene
scripts/build_scenes.sh --voxel 0.1 --ground-fill 12
docker/run.sh "python3 -m pcd2sim.build_scene --pcd Maps/dp_1008.pcd --out scenes/dp_1008"   # 只建一張
docker/run.sh "python3 -m pcd2sim.build_scene --help"
```

常用參數（`pcd2sim/build_scene.py`）：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--pcd`, `--out` | 必填 | 輸入 PCD、輸出目錄 |
| `--voxel` | 0.1 | 體素邊長（m）。5 cm 會讓表面破洞太多 |
| `--min-pts`, `--min-neighbors` | 1, 2 | 每個體素最少點數、最少 26 鄰居數（去雜訊） |
| `--crop-pct` | 0.05 | 每軸百分位裁切（0 = 不裁） |
| `--z-min`, `--z-max` | 無 | 只保留這個高度範圍 |
| `--no-flatten` | — | 不做地面平面化，保留原始體素地板 |
| `--ground-tol` | 0.06 | 同一地面區內允許的高度跳變（m） |
| `--ground-max-rms` | 0.035 | 平面擬合 RMS 超過就不壓平（m） |
| `--ground-thickness` | 0.2 | 地板平板厚度（m） |
| `--ground-fill` | 12 | 補地板破洞的輪數（SLAM 地板沒點的格子） |
| `--ground-fill-nb` | 3 | 8 個鄰居至少幾個是地板才補 |
| `--ground-max-above` | 2 | 一欄要被當成地板補起來，上方 0.8 m 內最多幾個體素 |
| `--ground-min-cells` | 30 | 小於這個格數的地面區不壓平（當成雜物頂面）。樓梯踏板很小，有樓梯的地圖用 8–12 |
| `--ground-layers` | 2 | 每格的表面高度取最上層體素往下幾層一起算 |
| `--ground-grow` | 3 | 地板往旁邊只因雜訊沒被判成可站立的欄位長幾圈 |
| `--ground-floor-tol` | 0.15 | 長過去的欄位要在平面高度 ± 這個範圍內有體素（m） |
| `--ground-fill-guard` | 0 | 欄位上方 0.1 m 到這個高度之間有實心體素（> 3 點）就不補、不長（m；0 = 關）。防止平台把旁邊的踏板埋掉 |
| `--ground-surface` | mean | 每格表面高度用點數加權平均（`mean`）或加權中位數（`median`，不受頂上一顆雜訊體素和踏板下方豎板點的影響） |
| `--ground-peel` | 0 | 不是單一平面的地面區先「剝」出主平面，剩下的格子再分成連通塊，最多剝幾層（0 = 關）。把平台和沿豎板欄位連上來的踏板分開 |
| `--ground-small-flat` | 0 | 格數不超過這個數的區直接壓成水平面（高度取中位數），不做斜面擬合（0 = 關）。踏板只有 2–3 格深、原始表面本來就斜幾公分，自由擬合會變成小斜坡；60（0.6 m²）不會影響真正的坡道 |

**樓梯配方**（dp_1012 掃描出來的最佳組合，見 [scene-tuning-dp_1012-2026-10-16.md](scene-tuning-dp_1012-2026-10-16.md)）：`--ground-min-cells 8 --ground-fill-guard 0.3 --ground-surface median --ground-peel 3 --ground-small-flat 100`。預設值完全不動，所以舊場景重建結果不變。

輸出（都在 `--out`）：

| 檔案 | 內容 |
|---|---|
| `boxes.npy` | float32 [M,6]（cx, cy, cz, hx, hy, hz）碰撞 / LiDAR box，map frame |
| `ground.npz` | 平面化後的地板平板（LiDAR、腳下地形射線、viewer 用） |
| `voxels.npz` | 體素 key + 平均 intensity（反射率）+ 平均 z |
| `map_ds.pcd` | 體素中心 + intensity（RViz、定位測試用） |
| `meta.json` | 統計、地板傾角、重力向量、出生點。**是 build cache，不要手改**，下次建置會蓋掉 |

dp_1008（1,420 萬點）在 Jetson 上約 22 s、峰值記憶體 1.5 GB。`meta.json` 的 `build_args` 記著這個場景用了哪些參數。

### 2.1 PCD 裡的合成地面（`scripts/fix_pcd_plane.py`）

有些匯出的 PCD 帶一層產生出來的地面（整張圖、完全平、密度極高的一層，`Maps/dp_1012.pcd` 在 z = −0.95，佔 25% 的點），比真正掃到的地板低 0.4–0.6 m。直接建場景的話，真地板每個掃描破洞都會變成掉到那層的坑（`step_down`）。這個 script 只在真地板沒點的欄位保留那層的點、並把它們抬到附近真地板的高度（補洞），其他地方的合成點全部丟掉：

```bash
python3 scripts/fix_pcd_plane.py Maps/dp_1012.pcd --dry-run                    # 只報告：平面在哪、多少點會抬 / 丟
python3 scripts/fix_pcd_plane.py Maps/dp_1012.pcd -o Maps/dp_1012_fixed.pcd    # 寫新檔（原檔不動，planner 的 octomap 還是用原檔）
docker/run.sh "python3 -m pcd2sim.build_scene --pcd Maps/dp_1012_fixed.pcd --out scenes/dp_1012 <樓梯配方>"
```

平面會自動偵測（最擠的 2 cm 高度層，佔 5% 以上的點而且很平）；`--plane-z` 可以指定。純 numpy/scipy，主機上 3 秒。

### 2.2 評估場景、掃參數（`scripts/eval_scene.py`、`scripts/sweep_scene.py`）

`eval_scene.py` 在 container 裡跑（要 Embree 和 MuJoCo），對一個建好的場景報告：

| 區塊 | 內容 |
|---|---|
| `ground` | `meta.json` 的地面統計（`planar_fraction`、`plane_rms_m`、`cells_filled`…） |
| `stairs` | 每個 `--stairs-box xmin,xmax,ymin,ymax[,zmin,zmax]`：平板有幾個高度層、相鄰層的階高（平均 / 標準差）、不在任何層上的格子比例、斜坡格子比例（踏板被壓成斜面）、原始可站立欄位有平板蓋到的比例（coverage）、平板和原始頂面的差（residual）、平板低於原始頂面 8 cm 以上的比例（buried）。`<scene>/eval_profile.txt` 有沿樓梯方向的 ASCII 側面圖 |
| `walk` | `--auto`：從出生點沿壓平的地板格做 BFS（8 鄰、高度差 ≤ 0.2 m，樓梯也算），每 0.3 m 一格從 BFS 的上一格「踩進去」（`IdealBase.reset` + `_check`）。機身在那格本來就放不下的（牆邊、家具）不算；有腳踩在沒平板的欄位的算 `edge_fraction`（地板邊緣或破洞）；其餘是有效取樣，`no_ground / step_up / step_down / slope / collision` 的數量就是地板缺陷，另外報有腳下方 3×3 射線打不到東西的比例 |
| `stairs_walk` | 每個 stairs box 再走一次：用設定檔的機器人限制，以及 `--climb`（預設 `robot.max_step_up=0.22,robot.max_slope_deg=45`、不檢查碰撞）—— 分開「幾何有問題」和「機器人限制不夠」 |
| `spawn` | 設定檔出生點：有沒有找到地板、有沒有被移動、地板高度 |

```bash
docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1012 --pcd Maps/dp_1012_fixed.pcd --find-stairs"   # 掃樓梯候選，印出可以直接用的 --stairs-box
docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1012 --pcd Maps/dp_1012_fixed.pcd --auto --stairs-box=-6,-3,-18,-14,-0.5,6.3"
docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1008 --poses Maps/dp_1008/poses.txt"              # 沿 SLAM keyframe 軌跡重播
```

`sweep_scene.py` 用一組參數把同一張 PCD 建成 `scenes/sweep/<prefix>_<tag>/`、每個都跑 `eval_scene.py`，把參數和所有指標寫成一張 CSV（`results/sweep_<prefix>/stage<N>.csv`，重跑會合併；`meta.json` 的 `build_args` 相同就不重建）：

```bash
docker/run.sh "python3 scripts/sweep_scene.py --pcd Maps/dp_1012_fixed.pcd --prefix dp_1012 --stage 1 --stairs-box=-6,-3,-18,-14,-0.5,6.3 --stairs-box=-42,-40,7,10,-1,5.9"
docker/run.sh "python3 scripts/sweep_scene.py ... --stage 2 --variant 'tol05:--ground-tol 0.05' --variant 'fill16:--ground-fill 16'"   # 自訂變體
```

stage 1 內建 10 個變體（預設、`min-cells 12`、加 `fill-guard`、加 `surface median`、加 `peel`、median + peel、再 ± `layers 1` / `min-cells 8` / `max-above 1` / `tol 0.05`），stage 2 內建 6 個 `small-flat` 變體；其他用 `--variant` 加。

## 3. 跑模擬

```bash
scripts/run_sim.sh                                   # 用 config/sim.yaml（預設 scene: scenes/dp_1008、rtf 1.0）
scripts/run_sim.sh --scene scenes/dp_1008_fillC      # 換場景
scripts/run_sim.sh --rtf 0 --duration 60             # 全速跑 60 秒 sim time 後結束
scripts/run_sim.sh --no-gt-tf                        # 不發真值 map→odom→base_link TF（交給 Point-LIO / 定位 stack）
scripts/run_sim.sh lidar.range_noise=0.0 robot.vel_limits=[1.0,0.5,1.5]
docker/run.sh "python3 -m quadsim.sim_node --help"
```

`scripts/run_sim.sh` 把所有參數原樣交給 `python3 -m quadsim.sim_node`：

| 參數 | 說明 |
|---|---|
| `--config <yaml>` | 換整份設定檔（預設 `config/sim.yaml`） |
| `--scene <dir>` | 場景目錄，覆寫設定檔的 `scene:` |
| `--rtf <f>` | 即時倍率；0 = 能多快跑多快 |
| `--no-gt-tf` | 不發真值 TF（`map → odom`、`odom → base_link`），交給定位 stack；等同 `ros.publish_gt_tf=false` |
| `--duration <s>` | 跑這麼多 sim 秒後自動結束（0 = 不停） |
| `--viewer` | 開 MuJoCo viewer（需要 `QUADSIM_GUI=1`，見第 4 章） |
| `key=value ...` | 任何設定檔欄位的覆寫（見第 9 章） |

出生點：`config/sim.yaml` 的 `spawn: [x, y, yaw, z_floor]`（目前 `[0, 0, 0, 0]`）；設成 `null` 會用 `meta.json` 裡 build_scene 算出的點。`z_floor` 要給**目標樓層的地板高度**（會從它上方 0.5 m 往下找到下方 1 m）；給到兩層之間會站在雜物或牆緣上。出生點或 `set_pose` 和地圖重疊時，機身會自動移到 1 m 內最近的空位（最多轉 90°）。

啟動 log 要看的幾行：

```
scene scenes/dp_1008 (231371 boxes), spawn requested [x, y, yaw, z_floor] [0.0, 0.0, 0.0, 0.0] -> base at [0.0, 0.0, -0.15] yaw 0.00 (floor z -0.45, moved False), rtf 1.0, gt_tf True, custom_msg True
viewer: gl=nvidia (docker/run.sh QUADSIM_GL), map mesh 3100000 faces, shadows False, sync <= 30 Hz   # 只有 --viewer 時出現
event: {'type': 'spawn', ...}                   # 同一筆也會發到 /sim/events
sim t=10.0s pose=[...]                          # 每 10 sim 秒一行
```

結束用 Ctrl-C（再按一次強制）。

## 4. 畫面（GUI）

要在**有 X display 的桌面終端機**執行，`DISPLAY` 和 `XAUTHORITY` 會自動帶進容器。

```bash
QUADSIM_GUI=1 scripts/run_sim.sh --viewer                                            # MuJoCo viewer，相機跟隨機身
QUADSIM_GUI=1 docker/run.sh "rviz2 -d rviz/quadsim.rviz --ros-args -p use_sim_time:=true"   # RViz
QUADSIM_GL=sw QUADSIM_GUI=1 scripts/run_sim.sh --viewer                              # 強制軟體繪圖
```

MuJoCo viewer 把整張地圖當一個 mesh 畫（dp_1008 約 310 萬個三角形），所以 GL 路徑決定它能不能用。Jetson Thor、1280x720 實測：

| `QUADSIM_GL` | 有陰影 | 無陰影（預設） |
|---|---|---|
| `sw`（llvmpipe） | 815 ms/幀 | 391 ms/幀 |
| `nvidia`（Thor GPU） | 43 ms/幀 | 16 ms/幀 |

- 設定檔 `viewer:` 區塊：`shadows`（預設 false，開了要多畫一遍 mesh）、`sync_hz`（預設 30，sim 推狀態給視窗的最高頻率）。
- viewer 裡按 `3` 顯示目前生效的碰撞代理 box（紅色）。地圖在 viewer 裡只是顯示用，碰撞走 Embree。
- RViz 要畫點雲，GL 路徑不能退到 GL 2.1 fallback；沒有 NVIDIA GPU 時預設就是 `sw` 的原因在此。

## 5. ROS 2 介面

名稱全部來自 `config/sim.yaml` 的 `ros:` 區塊，任何一個都可以用 `ros.topics.<key>=<name>` / `ros.frames.<key>=<name>` 覆寫，不用改檔。

| 方向 | Topic | 設定 key | 型別 | 頻率 |
|---|---|---|---|---|
| 發布 | `/clock` | `clock` | rosgraph_msgs/Clock | 200 Hz |
| 發布 | `/livox/lidar` | `lidar_custom` | livox_ros_driver2/CustomMsg（`offset_time` 逐點時間，ns） | 10 Hz |
| 發布 | `/livox/points` | `lidar_points` | PointCloud2（`pointcloud2_format`：livox 或 velodyne 欄位） | 10 Hz |
| 發布 | `/livox/imu` | `imu` | sensor_msgs/Imu（加速度單位 g，無 orientation） | 200 Hz |
| 發布 | `/sim/ground_truth/odom` | `gt_odom` | nav_msgs/Odometry，map frame，**只用於評估** | 50 Hz |
| 發布 | `/sim/events` | `events` | std_msgs/String（JSON），transient_local | 事件時 |
| 發布 | `/sim/map_cloud` | `map_cloud` | PointCloud2，latched 的體素化地圖 | 一次 |
| 訂閱 | `/cmd_vel` | `cmd_vel` | geometry_msgs/Twist（vx, vy, wz）；超過 `robot.cmd_timeout`（0.5 s）沒收到就歸零 | — |
| 訂閱 | `/sim/set_pose` | `set_pose` | geometry_msgs/PoseStamped（x, y, yaw；**z 是地板高度提示**，多樓層時用來選層） | — |

Frame（`ros.frames`）：`map`、`odom`、`base_link`、`livox_frame`。靜態 `base_link → livox_frame` 一律發布；`publish_gt_tf: true`（預設）時再加 `map → odom`（單位矩陣）和 `odom → base_link`（真值）。用 `--no-gt-tf` 關掉後，這兩段由 Point-LIO / 定位 stack 發。

`/sim/events` 的 `type`：

| type | 什麼時候 | 欄位 |
|---|---|---|
| `spawn` | 啟動時（transient_local，晚訂閱也收得到） | `requested`、`pose`、`z`、`floor_z`、`ok`、`moved` |
| `set_pose` | 收到 `/sim/set_pose` 後 | 同上 |
| `collision` | 機身會和地圖重疊（`reason: collision`）或已接觸（`reason: contact`） | `reason`、`t`、`pose` |
| `stuck` | 底盤拒絕移動：`no_ground`、單步高差、坡度超限等 | `reason`、`t`、`pose` |

同一個原因 1 秒內只發一次。

## 6. 驗證與測試

### 6.1 單元測試

```bash
docker/run.sh "python3 -m pytest -q"               # tests/ 下 7 個檔、22 個測試，不用 ROS 也不用場景
docker/run.sh "python3 -m pytest -q tests/test_ground.py -k fill"
```

涵蓋：PCD 讀寫、體素化與 box 合併、地面平面化、碰撞代理、感測器時間、碰撞檢查、CustomMsg / PointCloud2 序列化。

### 6.2 Headless 效能測試

```bash
docker/run.sh "python3 scripts/bench.py --scene scenes/dp_1008 --seconds 30 physics.rtf=0"
```

直接跑 `quadsim.sim.Simulator`，不經過 ROS。參數：`--config`、`--scene`、`--seconds`（預設 30）、`key=value` 覆寫。前 1.5 秒靜止，之後直走加緩慢蛇行。輸出並檢查：

- RTF 與時間分配（physics / lidar / control）
- 每幀點數；`offset_time` 單調且在 [0, 幀週期) 內
- 靜止時 IMU ≈ 1 g、沿地板法向
- 靜態一幀對 `map_ds.pcd` 的最近鄰距離 P95（應 < 2 個體素）
- 行走期間的 stuck / collision 事件

### 6.3 ROS 2 冒煙測試

```bash
scripts/smoke_test.sh                      # 預設 scenes/dp2f
scripts/smoke_test.sh scenes/dp_1008       # 指定場景
```

在同一個容器裡 headless 起 sim（`--rtf 1.0`），再跑 `scripts/ros_smoke_test.py`，結束後印 sim log 的最後幾行，以測試程式的 exit code 結束。測試依序檢查：topic 頻率（以 sim time 計：`/clock`、`/livox/imu`、`/livox/points`、`/livox/lidar`、真值 odom）→ TF `map → livox_frame` → `/cmd_vel` 直走（`--drive-seconds`，預設 6）→ 直走到 sim 回報 collision 或 stuck 事件 → `/sim/set_pose` 回到起點。約 20–30 秒跑完。

測試程式用的是第 5 章的預設 topic / frame 名，而且需要 PointCloud2 與真值 TF 都開著；如果你的 `ros:` 區塊改過名稱，請先改回預設或在容器內自己起 sim 加覆寫後再跑 `ros_smoke_test.py`。跑的時候若機器上已有另一個 sim，用 `ROS_DOMAIN_ID=77 scripts/smoke_test.sh ...` 隔開。

## 7. 導航下點迴圈測試

把 map frame 的 waypoint 一個接一個下給導航 stack，並用 sim 真值驗證每一段。`scripts/nav_goal_loop.sh` 在 sim 容器內執行 `scripts/nav_goal_loop.py`，所以會在同一個 DDS domain。

前置條件：

1. sim 用 `scripts/run_sim.sh --no-gt-tf` 啟動（TF 交給定位），`ros.topics` 的 LiDAR / IMU / cmd_vel 名稱要和導航 stack 用的一致（見第 5、9 章）。
2. 導航 stack 起好：定位（Point-LIO）、全域規劃（`octo_planner_rviz_node`）、控制（`syncai_controller_3d`），且 planner 的地圖和 sim 場景來自同一份 PCD。測試程式的 `--*-topic` 預設是 `robot01` 導航 stack 的名稱，接別的 stack 用這些參數改。
3. 測試期間 **不要按 teleop**（鍵盤節點也會發 cmd_vel）。
4. 定位已收斂：preflight 會比對 tracked pose 和真值，差超過 `--loc-preflight`（0.5 m）或 pose 太舊就以 exit 2 結束。

```bash
scripts/nav_goal_loop.sh --dry-run                                   # 只規劃不走：沿 waypoint 順序一段接一段問 planner
scripts/nav_goal_loop.sh --dry-run --dry-run-from matrix --select p01,p02,p03   # 任兩點之間的可達表
scripts/nav_goal_loop.sh --dry-run --dry-run-from robot              # 每個點都從機器人現在位置規劃
scripts/nav_goal_loop.sh --select wp29,wp30 --loops 1                # 只走指定的點
scripts/nav_goal_loop.sh --skip wp40                                 # 跳過規不到的點
scripts/nav_goal_loop.sh --loops 3 --order shuffle --seed 1          # 3 圈，每圈順序不同
scripts/nav_goal_loop.sh --waypoints config/nav_waypoints_dp_1008_2pt.yaml --loops 0 --duration 7200   # 兩點來回跑 2 小時 sim time
scripts/nav_goal_loop.sh --waypoints config/nav_waypoints_dp_1008_2pt.yaml --loops 1500 --no-start-nearest   # 固定從檔案第一點開始，不產生 0 m 空段
scripts/nav_goal_loop.sh --xy-tol 0.5 --max-consecutive-fail 5 --stop-on-fail
scripts/nav_goal_loop.sh --help
```

選點與順序：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--waypoints` | `config/nav_waypoints_dp_1008.yaml` | waypoint 檔 |
| `--select` / `--skip` | — | 逗號分隔的名稱，只用 / 排除 |
| `--order` | `sequential` | `shuffle` 每圈換一次排列；`random` 每段獨立抽 |
| `--start-nearest` / `--no-start-nearest` | 開 | sequential 時從離機器人最近的點開始 |
| `--start-at` | — | sequential 時把清單旋轉到從這個點開始 |
| `--loops` | 1 | 圈數；0 = 跑到 `--duration` |
| `--duration` | 0 | 跑這麼多 sim 秒後（在段落結束時）停 |
| `--dry-run` | — | 只規劃不送 FollowPath；`--dry-run-from robot|chain|matrix` 決定起點 |
| `--max-consecutive-fail` | 3 | 連續失敗幾段就停；`--stop-on-fail` 第一次失敗就停 |
| `--stuck-is-fail` | true | sim 的 `stuck` 事件算不算失敗 |
| `--results-dir` | `results/nav_goal_loop` | 結果目錄 |

容差與逾時（單位 m、m/s、sim 秒）：`--xy-tol 0.35`（真值到 path 終點）、`--loc-warn 0.3` / `--loc-fail 1.0` / `--loc-preflight 0.5`（定位誤差）、`--endpoint-xy-tol 1.5` / `--endpoint-z-tol 1.0`（接受 planner 把終點挪多遠）、`--plan-timeout 10`、`--empty-grace 2`、`--expected-speed 0.5` + `--leg-timeout-min 30` + `--leg-timeout-margin 15`（段落逾時 = 路長 / 速度 + margin，至少 30 s）、`--settle 1` / `--speed-eps 0.05`（停穩判定）、`--dwell 1` / `--dry-dwell 0.2`（段落間隔）、`--pose-timeout 1`、`--preflight-timeout 60`（wall 秒）。

ROS 名稱（預設值即 `robot01` 導航 stack）：`--tracked-pose-topic /robot01/tracked_pose`、`--initial-pose-topic /initialpose`、`--goal-pose-topic /goal_pose`、`--planned-path-topic /robot01/planned_path`、`--follow-path-action /robot01/follow_path`、`--planner-node /octo_planner_rviz_node`、`--controller-id FollowPath`、`--goal-checker-id general_goal_checker`、`--gt-topic /sim/ground_truth/odom`、`--events-topic /sim/events`。

每段流程（與 `syncai_waypoint_sequencer` 相同）：initial pose ← 目前的 tracked pose → goal pose ← waypoint → 等發 goal 之後收到、非空、終點在 waypoint 附近的 planned path → 送 follow_path action 等結果 → 停穩後比對真值、定位誤差和事件。執行期間 planner 的 `auto_follow_path` 會被設成 false，結束（含 Ctrl-C）時還原。測試**不會瞬移機器人**（`set_pose` 會讓定位壞掉），每次從機器人當下位置開始。

一段 PASS 的條件：action SUCCEEDED、真值距 path 終點 ≤ `--xy-tol`、機身已停、期間沒有 `collision` / `stuck` 事件、定位誤差 ≤ `--loc-fail`。失敗原因：`NO_PATH`、`ACTION_REJECTED`、`ABORTED`、`TIMEOUT`、`COLLISION`、`STUCK`、`STILL_MOVING`、`GT_NOT_AT_GOAL`、`LOC_ERROR`。exit code：0 全過、1 有失敗、2 preflight 失敗、130 被中斷。

結果在 `results/nav_goal_loop/<yyyymmdd-HHMMSS>/`（dry-run 加 `-dry` 後綴；目錄已 gitignore）：

| 檔案 | 內容 |
|---|---|
| `run.log` | 每段一行，例如 `LEG 1 loop1 robot->wp29  path 7.2 m plan 0.05 s  exec 13.9 s SUCCEEDED  gt-dist 0.18 (wp 0.28) loc 0.05/0.26  PASS` |
| `legs.csv` | 欄位：`idx, loop, from, to, result, reason, path_len_m, plan_time_s, action_status, exec_time_s, dist_to_endpoint_m, dist_to_waypoint_m, loc_err_arrival_m, loc_err_max_m, max_speed_mps, path_max_dz, last_distance_to_goal, t_start_sim, t_end_sim, events, warnings` |
| `report.json` | `run`（時間、模式、全部參數）、`waypoints`、`legs`、`summary`（pass / fail、`fail_by_reason`、總路長、平均速度、`per_waypoint`） |

Waypoint 檔（欄位 `name, x, y, z, yaw` 與 sequencer 的 YAML 相同，map frame，z 為 LiDAR 高度）：

| 檔案 | 內容 |
|---|---|
| `config/nav_waypoints_dp_1008.yaml` | 從 1F 建圖軌跡每 10 m 取樣的 40 點（wp40 規不到，已註解） |
| `config/nav_waypoints_dp_1008_manual.yaml` | 手動 teleop 到位後讀定位記下的 p01–p05，排成最長路線 |
| `config/nav_waypoints_dp_1008_3pt.yaml` | p03 → p01 → p02 三角形（避開 p04 / p05 旁的地板破洞） |
| `config/nav_waypoints_dp_1008_2pt.yaml` | p01 ↔ p02 來回，跑得最乾淨的一條路線 |

各檔的來龍去脈與當天的失敗分析見 [nav-goal-loop-test-2026-10-13.md](nav-goal-loop-test-2026-10-13.md)。

## 8. 產生 waypoint

從 SLAM keyframe 軌跡 `Maps/<map>/poses.txt`（每行 `<name>.pcd x y z qw qx qy qz`）取樣。建圖時真的走過的點是最可靠的可達點，而且和場景、planner octomap 同一個 map frame。純 Python 無相依，主機有 `python3` 也能直接跑。

```bash
docker/run.sh "python3 scripts/make_waypoints_from_poses.py Maps/dp_1008/poses.txt --spacing 10 --z-min -1 --z-max 0.5 -o config/nav_waypoints_dp_1008.yaml"
python3 scripts/make_waypoints_from_poses.py Maps/dp_1008/poses.txt --spacing 8      # 印到 stdout
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `poses` | 必填 | `Maps/<map>/poses.txt` |
| `--spacing` | 8 | 沿軌跡弧長每幾 m 取一點 |
| `--min-sep` | 4 | 和已取的點距離小於這個值就跳過（軌跡重複經過的走廊不產生重複點） |
| `--z-min` / `--z-max` | −1 / 1 | 只取 keyframe z 在這個範圍內的點，用來挑樓層 |
| `-o` | stdout | 輸出 YAML |

產生後先跑 `scripts/nav_goal_loop.sh --dry-run`，把 planner 規不到的點用 `--skip` 排除或在檔案裡註解掉。

## 9. 設定檔與參數覆寫

所有參數在 `config/sim.yaml`：

| 區塊 | 內容 |
|---|---|
| `scene`、`spawn` | 預設場景目錄、出生點 `[x, y, yaw, z_floor]` |
| `physics` | `timestep`（400 Hz）、`control_hz`（50）、`rtf` |
| `robot` | 理想底盤：機身尺寸、站高、腳位、`max_step_up/down`、`max_feet_over_gap`、`max_slope_deg`、`vel_limits` / `acc_limits`、`cmd_tau`、`cmd_timeout`、碰撞代理池大小 |
| `lidar` | Mid-360 參數：安裝位置、頻率、點數、FOV、距離、雜訊、`pattern`（`synthetic` 或 CSV 路徑） |
| `imu` | 頻率、`acc_unit`（`g` / `mps2`）、雜訊、bias random walk |
| `viewer` | `shadows`、`sync_hz` |
| `ros` | frame 名、`publish_gt_tf`、`publish_custom_msg`、`publish_pointcloud2`、`pointcloud2_format`、`publish_map_cloud`、各 topic 名 |

指令列覆寫：`區塊.欄位=值`，可以多個，`run_sim.sh`、`bench.py` 都接受。值用 YAML 解析，所以：

```bash
scripts/run_sim.sh physics.rtf=0 lidar.range_noise=0.0        # 數字
scripts/run_sim.sh robot.vel_limits=[1.0,0.5,1.5]              # list 不要有空白
scripts/run_sim.sh ros.topics.cmd_vel=/dog/cmd_vel ros.frames.base=base_footprint   # 對齊別的 stack 的名稱
scripts/run_sim.sh ros.publish_gt_tf=false ros.publish_pointcloud2=false          # 布林值
scripts/run_sim.sh lidar.pattern=config/mid360_pattern.csv     # 實測掃描樣式表（欄位 t,azimuth_deg,elevation_deg[,line]）
scripts/run_sim.sh --config config/sim_other.yaml              # 整份換掉
```

## 10. 常見問題

| 現象 | 原因 / 處理 |
|---|---|
| `QUADSIM_GUI=1 but DISPLAY is not set` | 不是在桌面終端機執行。改在有 X 的終端機跑，或不開 GUI |
| viewer 一幀要幾百 ms | 走到 `sw` 路徑。看啟動 log 的 `viewer: gl=`，有 NVIDIA GPU 就設 `QUADSIM_GL=nvidia`；確認 `viewer.shadows: false` |
| RViz 點雲、格線畫不出來 | GL 退到 2.1 fallback。用 `QUADSIM_GL=nvidia` 或 `sw`，不要用沒 `/dev/dri` 的 `hw` |
| 一起步就 `stuck` / `no_ground` | 場景地板有破洞。用目前預設（`--ground-fill 12 --ground-fill-nb 3`）重建場景，並確認 `robot.max_feet_over_gap: 1`。用 `scripts/eval_scene.py --auto` 找出缺陷在哪（[2.2](#22-評估場景掃參數scriptseval_scenepyscriptssweep_scenepy)） |
| 到處 `step_down`，地板像有很多坑 | PCD 帶合成地面層（例如 dp_1012 的 z = −0.95）。先跑 `scripts/fix_pcd_plane.py` 再建場景（[2.1](#21-pcd-裡的合成地面scriptsfix_pcd_planepy)） |
| 樓梯變斜坡、或踏板留成體素 | 用樓梯配方重建（[第 2 章](#2-建場景pcd--scenes)），用 `eval_scene.py --stairs-box` 看側面圖 |
| 啟動 log 有 `moved True` 或 ROS warning 說機身被移動 | 出生點和地圖重疊，多半是 `spawn` 的 `z_floor` 給到別層。改成目標樓層的地板高度 |
| 其他節點時間不對、TF 過期 | 沒設 `use_sim_time:=true` |
| 導航 stack 看不到 sim 的 topic | RMW 或 domain 不同。sim 用 CycloneDDS、domain 1、`conf/cyclonedds.xml`；跑 `docker/run.sh "ros2 topic list"` 比對 |
| nav 測試 exit 2（preflight failed） | tracked pose 沒在發、太舊或和真值差超過 0.5 m：定位還沒收斂，或 Point-LIO 掛了（曾經 segfault） |
| nav 測試每段 0.3 s 就 `ABORTED` | controller 判起點不安全，sim 不會自己脫困。見測試紀錄 §4.2；用 teleop 推開後再跑 |
| `scripts/smoke_test.sh` 全部 topic 收不到 | `ros:` 區塊的 topic / frame 名和第 5 章的預設不同，或 PointCloud2 / 真值 TF 被關掉。見 [6.3](#63-ros-2-冒煙測試) |
| `scenes/`、`results/`、`Maps/*.pcd` 不在 git 裡 | 本來就 gitignore；場景用第 2 章重建 |
