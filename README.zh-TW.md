[English](README.md) | 繁體中文

# quadsim：機器狗 ROS 2 導航模擬 MVP（MuJoCo）

**quadsim** 把 LiDAR SLAM 合併後的點雲地圖（PCD）轉成機器狗的導航測試環境。
- **場景建置（`pcd2sim`）**：把地圖體素化、合併成碰撞 box，並把有雜訊的地板平面化成平板；幾秒內建好
- **模擬器（`quadsim`）**：在 MuJoCo 裡跑理想底盤的機器狗。類 Mid-360 的滾動式 LiDAR（含逐點時間）用 Embree 投射射線，另外提供 IMU 和真值
- **ROS 2 介面**：發布 `livox_ros_driver2/CustomMsg`、`PointCloud2`、IMU、`/clock`、真值里程計和事件；訂閱 `/cmd_vel` 和 `/sim/set_pose`
- 可以直接接上 Point-LIO / FAST-LIO 和以 Nav2 為基礎的 3D 導航堆疊。在 Docker 裡跑 ROS 2 Humble

機器狗用**理想底盤**表示（不模擬腿）：底盤跟隨 `/cmd_vel` 移動，並貼合地形。

**使用手冊**（依任務分章，所有指令、參數和環境變數）見 [`docs/usage.md`](docs/usage.md)。設計背景見 [`docs/pcd-to-sim-architecture.md`](docs/pcd-to-sim-architecture.md)（§12–§13）。 dp_1008 導航下點迴圈測試紀錄與失敗分析見 [`docs/archive/nav-goal-loop-test-2026-10-13.md`](docs/archive/nav-goal-loop-test-2026-10-13.md)。

## 快速開始
```bash
# 1. 建 Docker 映像檔（以 syncai-robot-base 為基底，加裝 mujoco 和 livox CustomMsg）
docker build -t quadsim:latest docker/

# 2. 建預設場景：手修 OctoMap 的純 1:1 轉換（約 2 秒）
docker/run.sh "python3 -m pcd2sim.build_scene --map Maps/dp_1012_handmade.bt --out scenes/dp_1012 --no-flatten --min-neighbors 0"
#    PCD 地圖：scripts/build_scenes.sh 會把每個 Maps/*.pcd 建成帶地面平面化的場景（docs/usage.md §2）

# 3. 啟動模擬（預設 scenes/dp_1012，即時速度）
scripts/run_sim.sh
scripts/run_sim.sh --scene scenes/dp ros.publish_gt_tf=false   # 換場景；不發布真值 TF，交給 Point-LIO 和全域定位

# 4. 驗證
docker/run.sh "python3 -m pytest -q"                    # 單元測試
docker/run.sh "python3 scripts/bench.py physics.rtf=0"   # 預設場景的 headless 效能測試

# 畫面（需要 X display；請在桌面的終端機執行，DISPLAY 和 XAUTHORITY 會自動帶進容器）
QUADSIM_GUI=1 scripts/run_sim.sh --viewer                # MuJoCo viewer（相機跟隨機身，地圖只用於顯示）
QUADSIM_GUI=1 docker/run.sh "rviz2 -d rviz/quadsim.rviz --ros-args -p use_sim_time:=true"
```
`QUADSIM_GL` 決定 GUI 走哪條 GL 路徑：`nvidia`（透過 nvidia container runtime 用 NVIDIA GPU；主機有 `nvidia-smi` 或 docker `nvidia` runtime 時的預設，例如 Jetson）、`hw`（Mesa 加 `/dev/dri`，Intel/AMD）或 `sw`（Mesa llvmpipe，`LIBGL_ALWAYS_SOFTWARE=1`；其他情況的預設，因為退回 GL 2.1 時 RViz 的點雲和格線都畫不出來）。MuJoCo viewer 會把整張地圖當成一個 mesh 畫（dp_1008：310 萬個三角形），所以 GL 路徑決定它能不能用；在 Jetson Thor 上以 1280x720 實測：

| GL 路徑 | 有陰影 | 無陰影（預設） |
|---|---|---|
| `sw`（llvmpipe） | 815 ms/幀 | 391 ms/幀 |
| `nvidia`（Thor GPU） | 43 ms/幀 | 16 ms/幀 |

設定檔的 `viewer:` 區塊可以設 `shadows`（地圖對自己投影要多畫一遍整個 mesh）和 `sync_hz`（sim 把狀態推給視窗的最高 wall-clock 頻率；每次推送都會取算繪鎖）。sim 啟動時會記錄 `viewer: gl=... map mesh ... faces`，一眼就能看出用的是哪條路徑。MuJoCo viewer 裡，按 `3` 可以顯示目前生效的碰撞代理 box（紅色）。
容器使用 `--net=host`，DDS 設定與 WebRvizLite 相同：CycloneDDS（`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`）、`ROS_DOMAIN_ID=1`、`conf/cyclonedds.xml`（loopback 單播、關閉 multicast），所以 WebRvizLite 檢視器不需額外設定即可連上。主機或其他容器裡的導航堆疊，只要使用相同的 RMW 與 domain 就能直接連上（執行 `docker/run.sh` 時可用 `ROS_DOMAIN_ID` / `CYCLONEDDS_CONFIG=<conf/ 下的檔名>` 覆寫）。**其他所有節點都要設 `use_sim_time:=true`**。

## ROS 2 介面
| 方向 | Topic | 型別 | 頻率 |
|---|---|---|---|
| 發布 | `/clock` | rosgraph_msgs/Clock | 200 Hz |
| 發布 | `/livox/lidar` | livox_ros_driver2/CustomMsg（`offset_time` 為逐點時間，單位 ns） | 10 Hz |
| 發布 | `/livox/points` | PointCloud2（livox 格式：x y z intensity tag line timestamp；可切換成 velodyne 格式：ring、time） | 10 Hz |
| 發布 | `/livox/imu` | sensor_msgs/Imu（加速度單位 **g**，和 livox_ros_driver2 相同；沒有 orientation） | 200 Hz |
| 發布 | `/sim/ground_truth/odom` | nav_msgs/Odometry（map 座標系，**只用於評估**） | 50 Hz |
| 發布 | `/sim/events` | std_msgs/String（JSON：`spawn` / `set_pose` / `collision` / `stuck`；transient_local，晚訂閱也能收到啟動時的 `spawn` 事件） | 事件發生時 |
| 發布 | `/sim/map_cloud` | PointCloud2（latched，體素化後的地圖） | 一次 |
| 訂閱 | `/cmd_vel` | geometry_msgs/Twist（vx、vy、wz） | — |
| 訂閱 | `/sim/set_pose` | geometry_msgs/PoseStamped（x、y、yaw；**z 是地板高度的提示值**） | — |

**TF**：一律發布靜態的 `base_link → livox_frame`。`publish_gt_tf: true`（預設）時，再加上 `map → odom`（單位矩陣）和 `odom → base_link`（真值）。用 `ros.publish_gt_tf=false` 關掉之後，這兩段交給你們的定位堆疊發布。

所有參數（topic、frame、雜訊、機身尺寸、速度上限等）都在 [`config/sim.yaml`](config/sim.yaml)。也可以在指令列覆寫，例如 `scripts/run_sim.sh lidar.range_noise=0.0 robot.vel_limits=[1.0,0.5,1.5]`。

## 導航下點迴圈測試
sim 與導航 stack（`robot01` container：Point-LIO 定位、`octo_planner_rviz_node`、`syncai_controller_3d`）都在跑時，`scripts/nav_goal_loop.py` 會把 map frame 的 waypoint 一個接一個下給導航，並用 sim 的真值驗證每一段。它在 sim container 裡執行，才會在同一個 DDS domain：
```bash
scripts/nav_goal_loop.sh --dry-run                           # 只規劃不走：看 planner 規得出哪些段（機器人不動）
scripts/nav_goal_loop.sh --dry-run --dry-run-from matrix --select wp01,wp03,wp05   # 幾個點之間的可達表
scripts/nav_goal_loop.sh --select wp29,wp30 --loops 1        # 只走指定的幾個點
scripts/nav_goal_loop.sh --loops 3 --order shuffle --seed 1  # 久跑：3 圈，每圈順序不同
scripts/nav_goal_loop.sh --help                              # 所有容差、逾時、topic 名
```
每一段的流程跟 `syncai_waypoint_sequencer` 一樣：把目前的 `/robot01/tracked_pose` 發到 `/initialpose`（planner 不會自己讀機器人位置）、把 waypoint 發到 `/goal_pose`、只接受發 goal 之後收到、非空、終點在 waypoint 附近的 `/robot01/planned_path`，再送給 `/robot01/follow_path` action 等結果。執行期間 planner 的 `auto_follow_path` 參數會被設成 false（否則 planner 也會自己送一次 path），結束時還原，Ctrl-C 也會還原。

一段算 PASS 的條件：action 回報 SUCCEEDED、真值位置距 path 終點在 `--xy-tol`（0.35 m）內、機身已停、這段期間 `/sim/events` 沒有 `collision` / `stuck`、定位誤差（`tracked_pose` 對真值）沒超過 `--loc-fail`（1 m）。失敗原因：`NO_PATH`、`ACTION_REJECTED`、`ABORTED`、`TIMEOUT`、`COLLISION`、`STUCK`、`STILL_MOVING`、`GT_NOT_AT_GOAL`、`LOC_ERROR`。連續 3 次失敗會停（`--max-consecutive-fail`）。exit code：0 全過、1 有失敗、2 preflight 失敗、130 被中斷。

結果寫在 `results/nav_goal_loop/<timestamp>/`（`report.json`、`legs.csv`、`run.log`），每段一行：
```
LEG 1 loop1 robot->wp29  path 7.2 m plan 0.05 s  exec 13.9 s SUCCEEDED  gt-dist 0.18 (wp 0.28) loc 0.05/0.26  PASS
```
Waypoint 放在 `config/nav_waypoints_dp_1008.yaml`（欄位 `name, x, y, z, yaw` 跟 sequencer 的 YAML 相同）。`scripts/make_waypoints_from_poses.py Maps/<map>/poses.txt --spacing 10 --z-min -1 --z-max 0.5 -o <yaml>` 會從 SLAM keyframe 軌跡取樣（建圖時真的走過，是最可靠的可達點；用 z 範圍挑樓層），之後跑 `--dry-run` 把 planner 規不到的點拿掉。測試不會瞬移機器人（`/sim/set_pose` 會讓定位壞掉）：每次從機器人當下的位置、離它最近的 waypoint 開始。

## 架構
```
PCD ──build_scene──► 稀疏體素(10 cm) ─┬─ 地面 ──► 平面化厚板（斜面三角網格）──┬─► Embree（open3d）：LiDAR + 腳下地形射線
                     + 每體素平均       └─ 其他 ──► run-merge boxes ──────────────┤
                       intensity / z                                              └─► 碰撞代理池：~128 個 mocap box，每個控制週期
                                                                                       搬到機身附近的地圖 box 上（不含地面）
MuJoCo：自由剛體機身 ← 柔性 weld ← mocap 目標（cmd_vel 積分 + 地形貼合）；IMU = MuJoCo 感測器
```
- **MuJoCo 模型很小**（機身加上 128 個代理 box，共 130 個 geom），大小和地圖無關。全圖幾何只存在 Embree 裡
- LiDAR 和碰撞使用**同一組 box**，所以 LiDAR 看到的就是機器人會撞到的
- **地面平面化**（`pcd2sim/ground.py`）：
  - **分區**：可站立的地面格中，相鄰格高度差小於 6 cm 就分在同一區。樓梯、平台的高低差會切成不同區，邊緣保持銳利
  - **擬合平面**：每區（或區內每 4 m 一塊）擬合一個可以傾斜的平面，吸收 SLAM 的傾斜誤差
  - **取代地板體素**：平面上下的地板體素換成 0.2 m 厚的平板。平板側面就是樓梯的豎板；從下一層往上看，平板底面就是天花板
  - **清雜訊**：
    - 地板旁邊的雜訊突起、小破洞、浮在地面上的小團塊都會清掉
    - 平面上方第一層、只有 1–2 個點、上面也沒有東西的體素（地板雜訊的上緣）也會清掉
  - **縫隙**：Mid-360 看不到機器人腳邊的地板，所以 SLAM 地圖的地板本來就有整欄沒點的格子，平面化的接縫又會多出一些。有至少 3 個地板鄰居、上面沒有東西的格子會被補成地板（`--ground-fill 12 --ground-fill-nb 3`；dp_1008 1F 沿建圖路徑的無地面取樣從 0.37% 降到 0.17%）。機身另外允許一隻腳踩在剩下的縫隙上（`robot.max_feet_over_gap: 1`，那隻腳的高度取其他三隻腳的平面），不再發 `stuck/no_ground` 拒絕移動
  - **樓梯**：一片踏板只有 24–36 格，預設的 `--ground-min-cells 30` 會把它丟掉，平台的長 / 補又把它埋起來；豎板的點和頂上的雜訊體素也會讓踏板格子的高度變成一道斜坡、沿著豎板的欄位和平台連成一區。樓梯配方 `--ground-min-cells 8 --ground-fill-guard 0.3 --ground-surface median --ground-peel 3 --ground-small-flat 100`：保留小區、上方 0.1–0.3 m 有實心體素的欄位不補、格高用加權中位數而不是平均、不是單一平面的區先剝出主平面再分剩下的、100 格以下的區直接壓成水平面（自由擬合會把 2–3 格深的踏板變成斜坡）。全部預設關閉（舊場景重建結果不變）；`scripts/eval_scene.py --stairs-box` 量結果（層數、階高、覆蓋率、ASCII 側面圖）。目前在用的場景（`scenes/dp_1012`、`scenes/dp`）完全不走這套：它們是手修 OctoMap 的純 1:1 建置（`--no-flatten --min-neighbors 0`）。掃描紀錄見  [`docs/archive/scene-tuning-dp_1012-2026-10-16.md`](docs/archive/scene-tuning-dp_1012-2026-10-16.md)
  - **合成地面層**：匯出的 PCD 可能帶一層產生出來、完全平的地面在真地板下面（dp_1012：z = −0.95，佔 25% 的點），真地板的每個掃描破洞都會變成坑。`scripts/fix_pcd_plane.py` 只在真地板沒點的地方保留那層、抬到旁邊地板的高度，其他全部丟掉
  - 用 `build_scene --no-flatten` 可以關掉，保留原始的體素地板
- **出生點和 `/sim/set_pose` 會檢查機身有沒有和地圖重疊**：重疊時，會在 1 m 內找最近的空位（最多轉 90°），TF/odom 跟著*移動後*的位姿走。啟動 log 會印 `spawn requested [x, y, yaw, z_floor] -> base at [...]`，有被移動時另外印一行 ROS warning，並在 `/sim/events` 發一筆 `spawn` 事件（`requested`、`pose`、`floor_z`、`moved`）；`set_pose` 事件也帶同樣欄位
- **出生點在 `config/sim.yaml` 設定**（`spawn: [x, y, yaw, z_floor]`）；`null` 則用 `build_scene` 寫進 `scenes/<name>/meta.json` 的點。不要手改 `meta.json`：它是 build cache，下次 build 會被蓋掉。`z_floor` 必須是你要的那一層的地板高度（會從它上方 0.5 m 往下找到下方 1 m）：給一個夾在兩層之間的 z，機身會站在雜物或牆緣上，接著就觸發上面的移位
- **滾動式 LiDAR**：每條射線用最接近它發射時刻的感測器位姿來投射（誤差不超過半個物理步），會自然產生運動畸變。座標是發射當下的感測器座標系，和實機驅動一致
- 掃描樣式：合成的類 Mid-360 非重複樣式（每幀方位角掃一圈，仰角 −7°~52° 用低差異序列）
- 反射率來自原始 PCD 每個體素的平均 intensity
- 理想底盤：一階延遲、速度和加速度上限；4 隻腳各用 3×3 射線取地面（每腳取中位數），高度、roll、pitch 取四隻腳構成的平面，機身中心沿該平面法線抬 `stand_height`，所以在樓梯上機身底面和腳平面平行；碰撞盒跟著機身一起傾斜；遇到單步高差超限、坡度超限、沒有地面或機身會撞到時，拒絕移動並發出事件

## 實測數據（aarch64、6 核、無 GPU，單一 Python 行程）
| 場景 | 點數 | 體素 / box | 建置 | 模擬 RTF（headless） | 點數/幀 | 靜態幀對地圖 NN P95 |
|---|---|---|---|---|---|---|
| dp2f（單層） | 125 萬 | 48.5 萬 / 15.4 萬（地面 94% 平面化，4.8k 平板） | 2.4 s，179 MB | **9.1×** | 約 19.7k | 9.3 cm |
| dpmap（多層） | 315 萬 | 103 萬 / 30 萬（地面 98% 平面化，11k 平板） | 5.7 s，376 MB | **8.7×** | 約 17.0k | 9.6 cm |

即時（rtf=1）下大約只用掉一顆核心的 10%。時間分配：LiDAR 72–75%、控制 19–21%、物理 6–7%。
5 分鐘浸泡測試（rtf=1，有訂閱者）：sim 的 RSS 穩定在 527 MB，沒有成長。慢速訂閱者（每則訊息處理 1 s）不會拖慢 `/clock` 或 RTF。

地面平面化前後，機身行走時的 roll / pitch 標準差：
- dp2f：0.8° / 0.7° → 0.04° / 0.22°
- dpmap：0.7° / 1.8° → 0°

## 已知限制（MVP）
- **不模擬腿**：沒有步態動力學，也沒有腳底接觸。要測步態穩定性，需要接上你們的策略（文件中的模式 A）
- **10 cm 體素**：兩份 PCD 都已經降採樣到約 5 cm 點距。用 5 cm 體素時表面破洞太多，所以預設 10 cm。細小物體會被加粗，門寬最多會少 10 cm
- **地面平面化的取捨**：
  - 小於約 6 cm 的真實起伏（細門檻、電線）會被併進地板平面
  - 地面以上 10 cm 內的低矮物體底部會被平板取代
  - 曲面或雜物區（約 2–6% 的地面格）改用中位數平滑，不是完美平面
  - 平面和原始點雲的 RMS 差距約 2 cm，和地圖本身的雜訊同一個量級
- 合成掃描樣式只是近似 Mid-360（不支援載入實測樣式表）
- 沒有動態障礙物（之後可以用 Embree 動態場景或 mocap 物體加入）
- `/sim/set_pose` 的 z 要給地板高度附近的值，用來在多樓層地圖中選擇樓層

## 檔案
| 路徑 | 內容 |
|---|---|
| `pcd2sim/` | PCD 讀寫、OctoMap `.bt` / `.ot` 解析（`octomap_io.py`）、體素化、去雜訊、box 合併、地面平面化（`ground.py`）、地板與出生點、`build_scene` CLI |
| `quadsim/` | `sim.py` 核心（無 ROS）、`lidar.py`、`base.py`、`proxies.py`、`raycast.py`、`model.py`、`ros_io.py`、`sim_node.py` |
| `config/sim.yaml` | 所有參數 |
| `config/stairs_dp_full.yaml`、`config/stairs_dp_bt.yaml` | dp_full（0.03 m PCD）與 dp_bt（0.1 m 1:1 octree）場景的人工樓梯 spec（`scripts/make_stairs.py`，docs/usage.md §2.5） |
| `config/nav_waypoints_*.yaml` | 導航下點迴圈測試的 waypoint 集（`dp_1008*`：dp_1008 的 1F；`dp_2f`：`scenes/dp` 的 2F）；一次性的探點集在 `config/archive/` |
| `scripts/` | `build_scenes.sh`、`run_sim.sh`、`bench.py`、`nav_goal_loop.sh` / `nav_goal_loop.py`（導航下點迴圈測試）、`make_waypoints_from_poses.py`、`fix_pcd_plane.py`（合成地面層）、`eval_scene.py`（場景品質）、`make_stairs.py`（把掃描樓梯換成乾淨的 16 cm 人工階梯）、`bt_to_pcd.py` / `deploy_octomap_surface.py`（手修 OctoMap → 表面 PCD 部署給 planner） |
| `docker/` | Dockerfile、`run.sh`、最小化的 `livox_ros_driver2` 訊息套件 |
| `docs/` | `usage.md`（依任務分章的使用手冊）、`pcd-to-sim-architecture.md`（設計）；`archive/`：有日期的測試紀錄和 dp_1012 建場景參數掃描 |
| `tests/` | 單元測試（PCD、體素和 box 合併、地面平面化、碰撞代理、感測器時間、碰撞檢查、CustomMsg/PointCloud2 序列化、人工樓梯） |
