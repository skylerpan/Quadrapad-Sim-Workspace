[English](README.md) | 繁體中文

# quadsim：機器狗 ROS 2 導航模擬 MVP（MuJoCo）

**quadsim** 把 LiDAR SLAM 合併後的點雲地圖（PCD）轉成機器狗的導航測試環境。
- **場景建置（`pcd2sim`）**：把地圖體素化、合併成碰撞 box，並把有雜訊的地板平面化成平板；幾秒內建好
- **模擬器（`quadsim`）**：在 MuJoCo 裡跑理想底盤的機器狗。類 Mid-360 的滾動式 LiDAR（含逐點時間）用 Embree 投射射線，另外提供 IMU 和真值
- **ROS 2 介面**：發布 `livox_ros_driver2/CustomMsg`、`PointCloud2`、IMU、`/clock`、真值里程計和事件；訂閱 `/cmd_vel` 和 `/sim/set_pose`
- 可以直接接上 Point-LIO / FAST-LIO 和以 Nav2 為基礎的 3D 導航堆疊。在 Docker 裡跑 ROS 2 Humble

機器狗用**理想底盤**表示（不模擬腿）：底盤跟隨 `/cmd_vel` 移動，並貼合地形。

設計背景見 [`docs/pcd-to-sim-architecture.md`](docs/pcd-to-sim-architecture.md)（§12–§13）。

## 快速開始
```bash
# 1. 建 Docker 映像檔（以 syncai-robot-base 為基底，加裝 mujoco 和 livox CustomMsg）
docker build -t quadsim:latest docker/

# 2. Maps/*.pcd → scenes/<name>/（每份地圖約 2–5 秒）
scripts/build_scenes.sh

# 3. 啟動模擬（預設使用 scenes/dp2f，即時速度）
scripts/run_sim.sh
scripts/run_sim.sh --scene scenes/dpmap --no-gt-tf      # 不發布真值 TF，交給 Point-LIO 和全域定位

# 4. 驗證
docker/run.sh "python3 -m pytest -q"                    # 單元測試
docker/run.sh "python3 scripts/bench.py --scene scenes/dp2f physics.rtf=0"   # headless 效能測試
scripts/smoke_test.sh scenes/dp2f                       # ROS 2 端到端測試（頻率、TF、行走、撞擊、set_pose）

# 畫面（需要 X display；請在桌面的終端機執行，DISPLAY 和 XAUTHORITY 會自動帶進容器）
QUADSIM_GUI=1 scripts/run_sim.sh --viewer                # MuJoCo viewer（相機跟隨機身，地圖只用於顯示）
QUADSIM_GUI=1 docker/run.sh "rviz2 -d rviz/quadsim.rviz --ros-args -p use_sim_time:=true"
```
GUI 預設使用 Mesa 軟體算繪（`LIBGL_ALWAYS_SOFTWARE=1`）。不加的話，容器會退回 GL 2.1，RViz 的點雲和格線都畫不出來，整個畫面是黑的。有 GPU 的機器可以設 `QUADSIM_GL=hw`，把 `/dev/dri` 傳進容器。MuJoCo viewer 裡，按 `3` 可以顯示目前生效的碰撞代理 box（紅色）。
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

**TF**：一律發布靜態的 `base_link → livox_frame`。`publish_gt_tf: true`（預設）時，再加上 `map → odom`（單位矩陣）和 `odom → base_link`（真值）。用 `--no-gt-tf` 關掉之後，這兩段交給你們的定位堆疊發布。

所有參數（topic、frame、雜訊、機身尺寸、速度上限等）都在 [`config/sim.yaml`](config/sim.yaml)。也可以在指令列覆寫，例如 `scripts/run_sim.sh lidar.range_noise=0.0 robot.vel_limits=[1.0,0.5,1.5]`。

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
  - 用 `build_scene --no-flatten` 可以關掉，保留原始的體素地板
- **出生點和 `/sim/set_pose` 會檢查機身有沒有和地圖重疊**：重疊時，會在 1 m 內找最近的空位（最多轉 90°），TF/odom 跟著*移動後*的位姿走。啟動 log 會印 `spawn requested [x, y, yaw, z_floor] -> base at [...]`，有被移動時另外印一行 ROS warning，並在 `/sim/events` 發一筆 `spawn` 事件（`requested`、`pose`、`floor_z`、`moved`）；`set_pose` 事件也帶同樣欄位
- **出生點在 `config/sim.yaml` 設定**（`spawn: [x, y, yaw, z_floor]`）；`null` 則用 `build_scene` 寫進 `scenes/<name>/meta.json` 的點。不要手改 `meta.json`：它是 build cache，下次 build 會被蓋掉。`z_floor` 必須是你要的那一層的地板高度（會從它上方 0.5 m 往下找到下方 1 m）：給一個夾在兩層之間的 z，機身會站在雜物或牆緣上，接著就觸發上面的移位
- **滾動式 LiDAR**：每條射線用最接近它發射時刻的感測器位姿來投射（誤差不超過半個物理步），會自然產生運動畸變。座標是發射當下的感測器座標系，和實機驅動一致
- 掃描樣式：預設是合成的類 Mid-360 非重複樣式（每幀方位角掃一圈，仰角 −7°~52° 用低差異序列）。用 `lidar.pattern: <csv>` 可以改成實測樣式表（欄位 `t,azimuth_deg,elevation_deg[,line]`）
- 反射率來自原始 PCD 每個體素的平均 intensity
- 理想底盤：一階延遲、速度和加速度上限；4 隻腳各用 3×3 射線取地面，計算高度、roll、pitch（穩健平面擬合）；遇到單步高差超限、坡度超限、沒有地面或機身會撞到時，拒絕移動並發出事件

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
- 合成掃描樣式只是近似 Mid-360。建議錄一段實機 rosbag，反推出樣式表後用 CSV 載入
- 沒有動態障礙物（之後可以用 Embree 動態場景或 mocap 物體加入）
- `/sim/set_pose` 的 z 要給地板高度附近的值，用來在多樓層地圖中選擇樓層

## 檔案
| 路徑 | 內容 |
|---|---|
| `pcd2sim/` | PCD 讀寫、體素化、去雜訊、box 合併、地面平面化（`ground.py`）、地板與出生點、`build_scene` CLI |
| `quadsim/` | `sim.py` 核心（無 ROS）、`lidar.py`、`base.py`、`proxies.py`、`raycast.py`、`model.py`、`ros_io.py`、`sim_node.py` |
| `config/sim.yaml` | 所有參數 |
| `scripts/` | `build_scenes.sh`、`run_sim.sh`、`smoke_test.sh`、`bench.py`、`ros_smoke_test.py` |
| `docker/` | Dockerfile、`run.sh`、最小化的 `livox_ros_driver2` 訊息套件 |
| `tests/` | 單元測試（PCD、體素和 box 合併、地面平面化、碰撞代理、感測器時間、碰撞檢查、CustomMsg/PointCloud2 序列化） |
