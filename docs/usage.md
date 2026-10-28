# quadsim 使用手冊

依「要做什麼」分章，每章給指令、常用參數和注意事項。完整參數清單一律以程式的 `--help` 為準。
專案概觀、架構與實測數據見 [README.zh-TW.md](../README.zh-TW.md)；設計背景見 [pcd-to-sim-architecture.md](pcd-to-sim-architecture.md) §12–§13；導航測試的實測紀錄見 [archive/nav-goal-loop-test-2026-10-13.md](archive/nav-goal-loop-test-2026-10-13.md)。

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

## 2. 建場景（PCD / OctoMap → scenes/）

把 `Maps/<name>.pcd`（或 OctoMap `.bt` / `.ot`，見 §2.3）轉成模擬用的場景目錄 `scenes/<name>/`。`Maps/*.pcd` 和 `scenes/` 都在 `.gitignore` 裡，要自己放。

```bash
scripts/build_scenes.sh                       # Maps/*.pcd 全部建，額外參數會傳給 build_scene
scripts/build_scenes.sh --voxel 0.1 --ground-fill 12
docker/run.sh "python3 -m pcd2sim.build_scene --pcd Maps/dp_1008.pcd --out scenes/dp_1008"   # 只建一張
docker/run.sh "python3 -m pcd2sim.build_scene --help"
```

常用參數（`pcd2sim/build_scene.py`）：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--map`（別名 `--pcd`）, `--out` | 必填 | 輸入地圖（`.pcd` / `.bt` / `.ot`，看副檔名）、輸出目錄 |
| `--voxel` | 0.1 | 體素邊長（m）。5 cm 以下表面會破很多洞，要配 `--close-rounds`（見 [2.4](#24-003-m-高解析度建置dp_full_original)）。OctoMap 輸入固定用樹的解析度，這個參數會被忽略 |
| `--min-neighbors` | 2 | 26 鄰居中 occupied 少於這個數的體素丟掉（去雜訊）。對 OctoMap 輸入也套用，0 = 關 |
| `--min-cluster` | 0 | 26 連通的叢集少於這個體素數就整叢丟掉（0 = 關）。octree 的雜訊多半是 1–4 格的小碎點，給 4 很合適；PCD 建置通常不開 |
| `--z-min` | 無 | 丟掉這個高度以下的點（例如 PCD 的合成地面層） |
| `--no-flatten` | — | 不做地面平面化，保留原始體素地板（octree 輸入：純 1:1 box，不補洞） |
| `--ground-tol` | 0.06 | 同一地面區內允許的高度跳變（m） |
| `--ground-fill` | 12 | 補地板破洞的輪數（SLAM 地板沒點的格子；8 個鄰居至少 3 個是地板才補） |
| `--ground-max-above` | 2 | 一欄要被當成地板補起來，上方 0.8 m 內最多幾個體素 |
| `--ground-min-cells` | 30 | 小於這個格數的地面區不壓平（當成雜物頂面）。樓梯踏板很小，有樓梯的地圖用 8–12 |
| `--ground-layers` | 2 | 每格的表面高度取最上層體素往下幾層一起算 |
| `--ground-grow` | 3 | 地板往旁邊只因雜訊沒被判成可站立的欄位長幾圈 |
| `--ground-fill-guard` | 0 | 欄位上方 0.1 m（或一格，取大者）到這個高度之間有實心體素（點數 > `--ground-spike-pts`）就不補、不長（m；0 = 關）。防止平台把旁邊的踏板埋掉 |
| `--ground-surface` | mean | 每格表面高度用點數加權平均（`mean`）或加權中位數（`median`，不受頂上一顆雜訊體素和踏板下方豎板點的影響） |
| `--ground-peel` | 0 | 不是單一平面的地面區先「剝」出主平面，剩下的格子再分成連通塊，最多剝幾層（0 = 關）。把平台和沿豎板欄位連上來的踏板分開 |
| `--ground-small-flat` | 0 | 格數不超過這個數的區直接壓成水平面（高度取中位數），不做斜面擬合（0 = 關）。踏板只有 2–3 格深、原始表面本來就斜幾公分，自由擬合會變成小斜坡；60（0.6 m²）不會影響真正的坡道 |
| `--ground-spike-pts` | 3 | 長 / 補出來的地板欄位上方，點數 ≤ 這個數的體素當 SLAM 雜訊刪掉；`--ground-fill-guard` 也要點數 > 這個數才算實心。以 10 cm 體素設計；2–3 cm 體素幾乎每格只有 1 點，要給 0（每格都算真的） |
| `--ground-sheet-pts` | 2 | 地板帶上方一層、點數 ≤ 這個數、頂上沒東西的體素是地板雜訊的上緣，刪掉。2–3 cm 體素給 0 |
| `--ground-float-max` | 4 | 浮在地板上方 0.5 m 內、沒碰到別的東西、體素數 ≤ 這個數的小叢當雜訊。隨 1/體素² 放大：10 cm 用 4，3 cm 約 40 |
| `--ground-roof-cell` | 一格 | 可站立體素的「上方 1.5 m 內要有結構（天花板）」檢查看多寬的 xy 欄位（m）。天花板掃得稀，3 cm 的欄位大多沒有天花板點，整片地板會被判成不可站立；0.03 m 用 0.15 |
| `--ground-column-gap` | 0 | 多樓層補地板：長 / 補時，只有「同一欄位在提議高度 ± 這個範圍（到 2 倍）內已經有地面格」才跳過（m）。0 = 欄位裡任何樓層有地面格就不碰（舊行為；2F 地板下方的 1F 破洞永遠補不到）。0.03 m 用 0.3 |
| `--close-rounds` | 0 | 補洞輪數（0 = 關）。把「在表面裡面」的空格補成實心：某個軸平面（xy / xz / yz）的 8 個鄰居 ≥ 6 個有點（地板 / 牆上的洞；邊緣格最多 5），或 26 鄰居 ≥ 12 個有點（厚而雜的面裡的洞、內角）。平貼在完整平面旁的格子只看得到 9 個鄰居、自己的平面裡 0 個，所以面不會變厚、邊不會長出去；2 輪補得掉 2 格寬的縫 |
| `--workers` | 全部 CPU | 補洞分塊（每塊 200 萬格）用幾個 fork 出來的 process 平行算 |
| `--keep-tmp`, `--resume` | 關, 關 | 階段檔（體素、補洞後體素、地面）和補洞的分塊檔放在 `<out>/.tmp`；成功後整個目錄刪掉，`--keep-tmp` 留著，`--resume` 讀既有階段檔跳過前面的階段（建置中途失敗時目錄會留著） |

**樓梯配方**（dp_1012 掃描出來的最佳組合，見 [archive/scene-tuning-dp_1012-2026-10-16.md](archive/scene-tuning-dp_1012-2026-10-16.md)）：`--ground-min-cells 8 --ground-fill-guard 0.3 --ground-surface median --ground-peel 3 --ground-small-flat 100`。預設值完全不動，所以舊場景重建結果不變。

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

### 2.2 評估場景（`scripts/eval_scene.py`）

`eval_scene.py` 在 container 裡跑（要 Embree 和 MuJoCo），對一個建好的場景報告：

| 區塊 | 內容 |
|---|---|
| `ground` | `meta.json` 的地面統計（`planar_fraction`、`plane_rms_m`、`cells_filled`…） |
| `stairs` | 每個 `--stairs-box xmin,xmax,ymin,ymax[,zmin,zmax]`：平板有幾個高度層、相鄰層的階高（平均 / 標準差）、不在任何層上的格子比例、斜坡格子比例（踏板被壓成斜面）、原始可站立欄位有平板蓋到的比例（coverage）、平板和原始頂面的差（residual）、平板低於原始頂面 8 cm 以上的比例（buried）。`<scene>/eval_profile.txt` 有沿樓梯方向的 ASCII 側面圖 |
| `walk` | `--auto`：從出生點沿壓平的地板格做 BFS（8 鄰、高度差 ≤ 0.2 m，樓梯也算），每 0.3 m 一格從 BFS 的上一格「踩進去」（`IdealBase.reset` + `_check`）。機身在那格本來就放不下的（牆邊、家具）不算；有腳踩在沒平板的欄位的算 `edge_fraction`（地板邊緣或破洞）；其餘是有效取樣，`no_ground / step_up / step_down / slope / collision` 的數量就是地板缺陷，另外報有腳下方 3×3 射線打不到東西的比例 |
| `stairs_walk` | 每個 stairs box 再走一次：用設定檔的機器人限制，以及 `--climb`（預設 `robot.max_step_up=0.22,robot.max_slope_deg=50`、不檢查碰撞）—— 分開「幾何有問題」和「機器人限制不夠」 |
| `spawn` | 設定檔出生點：有沒有找到地板、有沒有被移動、地板高度 |

```bash
docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1012 --pcd Maps/dp_1012_fixed.pcd --find-stairs"   # 掃樓梯候選，印出可以直接用的 --stairs-box
docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1012 --pcd Maps/dp_1012_fixed.pcd --auto --stairs-box=-6,-3,-18,-14,-0.5,6.3"
docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1008 --poses Maps/dp_1008/poses.txt"              # 沿 SLAM keyframe 軌跡重播
```

### 2.2b OctoMap 轉回表面點雲（`scripts/bt_to_pcd.py`）

把手畫 / 修過的 `.bt` 轉成 PCD，給 `map_server_3d`（local costmap 的靜態圖）用，讓它和 sim 場景、全域規劃讀同一份幾何。只取外露面（鄰居不是 occupied 的面），每面均勻取樣，輸出 binary PCD（x y z，同座標系）。

```bash
python3 scripts/bt_to_pcd.py Maps/dp_1012_handmade.bt --dry-run                                        # 只報告 voxel / 外露面數與建議點數
python3 scripts/bt_to_pcd.py Maps/dp_1012_handmade.bt -o Maps/dp_1012_handmade_surface.pcd --points 12800000   # 精確點數、分層抖動
python3 scripts/bt_to_pcd.py Maps/dp_1012_handmade.bt -o out.pcd --grid 4                                # 每面 4×4 規則格（0.2 m → 5 cm）
```

`--skip-down` 不取朝下的面（天花板 / 樓板底面）；`--offset 0.005` 把點往面外推一點。dp_1012_handmade.bt（46 萬格）約 3 秒，12.8 M 點 154 MB。

**一條指令部署到 planner workspace**（`scripts/deploy_octomap_surface.py`）：取樣成 `Maps/<stem>_surface.pcd` → 複製到 `SyncAI-Device-Workspace/src/syncai_planner/octo_planner_3d/maps/<name>/` → `touch` 那邊的 `.bt` 和 `.layers` → 把 `maps_config.yaml` 的 `maps.<name>.pcd_path` 與 `robot01.ini` 的 `[3Dmap] pcd` 改指到新檔。`touch` 不能省：planner 只要看到 PCD 比 `.bt` 新，就會用 PCD 重建並**覆寫**手修的 `.bt`（`octo_planner_rviz_node.cpp`），`.layers` 同理。腳本永遠不覆寫既有 `.bt`：planner 那份和來源不同就停下來讓你決定。

```bash
python3 scripts/deploy_octomap_surface.py Maps/dp.bt --dry-run                    # 只印會做什麼
python3 scripts/deploy_octomap_surface.py Maps/dp.bt                              # name=dp
python3 scripts/deploy_octomap_surface.py Maps/dp_1012_handmade.bt --name dp_1012  # 自訂地圖名；--no-ini 不碰 robot01.ini
```

### 2.3 從 OctoMap 建場景（`.bt` / `.ot`）

`--map` 也吃 OctoMap 檔（副檔名決定格式）。純 numpy 解析（`pcd2sim/octomap_io.py`），不需要 octomap 套件：`.bt` 只有 occupied / free 兩值；`.ot` 帶 log-odds，機率 ≥ 0.5 算 occupied（支援 `OcTree` 和 `ColorOcTree`）。**流程**：每個 occupied 格子（prune 過的大節點先展開成樹解析度）1:1 變成一個體素 → 去雜訊 → 地面平面化（和 PCD 同一套 `--ground-*`：補地板破洞、每個地面區吸附成平面、長過雜訊欄位、丟飄浮碎塊）→ 合併成 box。`--no-flatten` 跳過平面化，保留原始 1:1 體素地板（破洞就是破洞）。PCD 那套百分位裁切和表面平滑不跑，`--voxel` 對 octree 輸入沒有作用。

**去雜訊**（兩道，和 PCD 共用同一套程式）：
1. `--min-neighbors`（預設 2）：26 鄰居中 occupied 少於這個數的體素丟掉。給 0 就是純 1:1。
2. `--min-cluster`（預設 0 = 關）：26 連通叢集少於 N 格整叢丟掉，比第一道強。

以 `Maps/dp_1008/octomap.bt`（0.1 m，123.8 萬格）對照 PCD 建的 `scenes/dp_1008`（同一個 0.1 m 格網）量過：預設 `--min-neighbors 2` 刪 22,069 格（1.8%），其中 87% 在 PCD 場景裡不存在，也就是真的雜訊；`--min-neighbors 3` 會刪 40,515 格但只有一半是雜訊，開始吃到真幾何，所以不要調高。`--min-cluster 4` 刪 19,207 格、76% 是雜訊，要更乾淨時加上。`dp_1012.bt`（0.2 m）本來就乾淨，預設只刪 677 格。

```bash
docker/run.sh "python3 -m pcd2sim.build_scene --map Maps/dp_1012.bt --out scenes/dp_1012_bt --z-min -0.8"
docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1012_bt --pcd Maps/dp_1012.bt --auto"
```

- 體素邊長 = 樹的解析度（`dp_1012.bt` 是 0.2 m）；給了 `--voxel` 只會印警告然後忽略。
- octree 沒有點數和 intensity：`voxels.npz` 的 intensity 全為 0（模擬 LiDAR 的反射率是平的），`zmean` 是格子中心 z。地面平面化內部把每格點數當 4（`ground.py` 把 ≤ 3 點當稀疏雜訊，octree 的雜訊已經由 `--min-neighbors` 處理掉）。
- 格子的表面高度是格子頂面（occupied 格是實心塊）。`--ground-layers` 維持預設 2：地板上方因雜訊多出來的那層格子會把「最上層」抬高一格，把下面兩層一起平均才接近真地板。量過 `dp_1012.bt`（0.2 m）：`--ground-layers 0` 把地板切成一格一階的 0.2 m 梯田，`eval_scene --auto` 從出生點只走得到 118 m²；預設 2 走得到 1,479 m²、失敗 0.84%。
- 地板收尾的效果：`dp_1012.bt`（0.2 m）155 個地面區全部是平面、補了 16,390 格破洞，`eval_scene --auto` 走動失敗 0.84%、腳下射線打不到東西 0.55%；`dp_1008/octomap.bt`（0.1 m）補 45,001 格、92.8% 的地面區是平面（RMS 1.8 cm），走動失敗 0.22%、腳下射線打不到東西 0.06%。
- `Maps/dp_1012.bt` 一樣含有 PCD 的合成地面層（§2.1；在 octree 裡是整層 z = −0.9 的格子，7.2 萬個）。真地板在 −0.7 / −0.5 / −0.3，用 `--z-min -0.8` 直接切掉。
- **解析度太粗，不適合當需要爬樓梯的物理場景**：0.2 m 把階高量化掉、地板稍微傾斜就出現 0.2 m 的假台階，而且沒有地面平面化來吸收（見 [pcd-to-sim-architecture.md](pcd-to-sim-architecture.md) §11）。拿來快速建粗場景、或只有 octomap 沒有 PCD 的地圖時用。
- 輸出目錄另外取名（`scenes/dp_1012_bt`），不要蓋掉 PCD 建的場景。`meta.json` 多了 `source_format`（`pcd` / `octomap`）和 `octomap_res`；`source_pcd*` 欄位照舊，裝的是來源地圖（不管格式）。
- `--no-flatten` 的場景沒有地面格，`scripts/eval_scene.py` 的走動評估（walk）只會印 `None`。
- `dp_1012.bt`（43.7 萬格，`--z-min -0.8` 後 36.1 萬格）在 Jetson 上約 1 s，出來 4.4 萬個 box。
- **目前的正式場景都是手修 octree 的純 1:1 建置**（`--no-flatten --min-neighbors 0`，不補洞、不平面化，樓梯踏板照手畫的格子走）：`scenes/dp_1012` ← `Maps/dp_1012_handmade.bt`（`config/sim.yaml` 預設，1F 原點地板 z = −0.4，樓梯 D 是 `tests/test_stairs_dp_1012.py` 的目標），`scenes/dp` ← `Maps/dp.bt`（整棟，2F 導航測試用 `config/nav_waypoints_dp_2f.yaml`）。重建：`docker/run.sh "python3 -m pcd2sim.build_scene --map Maps/dp_1012_handmade.bt --out scenes/dp_1012 --no-flatten --min-neighbors 0"`。手修 .bt 的部署見 [2.2b](#22b-octomap-轉回表面點雲scriptsbt_to_pcdpy)。

### 2.4 0.03 m 高解析度建置（dp_full_original）

機器狗的物理場景建議 2–3 cm（[pcd-to-sim-architecture.md](pcd-to-sim-architecture.md) §4、§11）。`Maps/dp_full_original.pcd`（1,956 萬點，xyz + intensity，和 `Maps/dp.pcd` 同一個檔；`fix_pcd_plane.py --dry-run` 沒找到合成地面）直接用 `--voxel 0.03` 建會是一張篩網：1,240 萬個體素裡 69% 只有 1 個點、94% ≤ 3 點，牆和地板散在 2–3 層裡，634 萬個 box 彼此不相鄰，LiDAR 會穿過去。所以多了補洞（`--close-rounds`）和幾個要跟著解析度換算的地面參數：

```bash
docker/run.sh "python3 -m pcd2sim.build_scene --pcd Maps/dp_full_original.pcd --out scenes/dp_full_003 --voxel 0.03 \
  --close-rounds 2 \
  --ground-min-cells 90 --ground-fill-guard 0.3 --ground-surface median --ground-peel 3 --ground-small-flat 1100 \
  --ground-layers 7 --ground-grow 10 --ground-fill 60 --ground-max-above 7 \
  --ground-spike-pts 0 --ground-sheet-pts 0 --ground-float-max 40 \
  --ground-roof-cell 0.15 --ground-column-gap 0.3"
```

從 0.1 m 樓梯配方換算（格數 ×11、層數 ×3.3；以公尺為單位的 `--ground-tol`、`--ground-fill-guard` 不變）：

| 參數 | 0.1 m 配方 | 0.03 m | 單位 |
|---|---|---|---|
| `--ground-min-cells` | 8 | 90 | 格（0.08 m²） |
| `--ground-small-flat` | 100 | 1100 | 格（1 m²） |
| `--ground-layers` | 2 | 7 | 體素層（0.2 m） |
| `--ground-grow` | 3 | 10 | 圈（0.3 m） |
| `--ground-fill` | 12 | 60 | 輪（1.8 m：掃描起點正下方的地板有一個半徑約 1 m 的洞，Mid-360 往下看不到） |
| `--ground-max-above` | 2 | 7 | 體素（地板上方 0.2 m 的雜訊） |
| `--ground-spike-pts` / `--ground-sheet-pts` | 3 / 2 | 0 / 0 | 每格點數 → 每格都算真的 |
| `--ground-float-max` | 4 | 40 | 每叢體素數 |
| `--ground-roof-cell` | （一格） | 0.15 | m；天花板點稀，3 cm 欄位大多沒有「屋頂」 |
| `--ground-column-gap` | （0） | 0.3 | m；1F 在 2F 平板正下方，舊規則一格一樓層、補不到 |

另外兩處依解析度自動換算、不用給參數：地面格之間連成一區（以及 5×5 中位數）看鄰居最上層體素相差 0.1 m 以內（0.1 m 時就是 ±1 層）；`--ground-fill-guard` 的「上方有實心體素」從地板上方 0.1 m 起算而不是一格（0.1 m 時一樣）。沒有這幾項時 1F 地板只剩 LiDAR 掃描線的條紋（每條 3 cm 寬、間隔 5–10 cm 沒有點），`eval_scene --auto` 從 1F 只走得到 233 m²、41% 的取樣有腳踩在沒有地板的欄位上。

實測（Jetson，14 workers）：補洞 2 輪 +324 萬格（1,108 萬 → 1,433 萬，去雜訊後），有 ≥ 12 個鄰居的體素比例 29% → 60%，補洞階段峰值記憶體 3 GB、66 s。整個建置 15 分鐘、峰值記憶體 9.8 GB（原始點陣列在體素化後就釋放，階段檔放 `<out>/.tmp`，結束自動刪掉；13 分鐘花在 `--ground-column-gap` 的多樓層補地板，每一輪都對全部地面格的 8 個鄰居提案）：地面 610 萬格、88.4% 壓成平面（RMS 1.8 cm）、補 459 萬格破洞、fill-guard 擋下 212 萬格、90 萬片 slab，最後 1,121 萬個體素合併成 309 萬個 box（`boxes.npy` 74 MB，整個場景 451 MB）。補 60 輪會從門窗開口往牆外多長約 1.8 m 的地板，目前接受。

跑這個場景要注意：

- `config/sim.yaml` 的 `robot.collision_pool` 128 / `collision_margin` 1.0 是以 0.1–0.2 m 的 box 設計的；0.03 m 時機身周圍 1 m 的視窗裡有上千個 box，`ProxyPool` 只會放最近的 128 個（`overflow` 計數會跳）。用這個場景時加 `robot.collision_pool=1024 robot.collision_margin=0.4`（`_blocked` 的分離軸檢查用整個索引，不受影響）。
- viewer 的地圖 mesh 只畫離出生點最近的 `viewer.max_map_boxes` 個 box（預設 40 萬），所以 `--viewer` 只會看到出生點附近；`viewer.max_map_boxes=1500000` 大約能看到整層（2,900 萬面，載入多 13 s）。headless 不受影響。
- `scenes/dp` 不動（還是 `Maps/dp.bt` 的 0.2 m 1:1 場景）；要用新場景就 `scene=scenes/dp_full_003`。

## 3. 跑模擬

```bash
scripts/run_sim.sh                                   # 用 config/sim.yaml（預設 scene: scenes/dp_1012、rtf 1.0）
scripts/run_sim.sh --scene scenes/dp                 # 換場景
scripts/run_sim.sh --rtf 0 --duration 60             # 全速跑 60 秒 sim time 後結束
scripts/run_sim.sh ros.publish_gt_tf=false           # 不發真值 map→odom→base_link TF（交給 Point-LIO / 定位 stack）
scripts/run_sim.sh lidar.range_noise=0.0 robot.vel_limits=[1.0,0.5,1.5]
docker/run.sh "python3 -m quadsim.sim_node --help"
```

`scripts/run_sim.sh` 把所有參數原樣交給 `python3 -m quadsim.sim_node`：

| 參數 | 說明 |
|---|---|
| `--config <yaml>` | 換整份設定檔（預設 `config/sim.yaml`） |
| `--scene <dir>` | 場景目錄，覆寫設定檔的 `scene:` |
| `--rtf <f>` | 即時倍率；0 = 能多快跑多快 |
| `--duration <s>` | 跑這麼多 sim 秒後自動結束（0 = 不停） |
| `--viewer` | 開 MuJoCo viewer（需要 `QUADSIM_GUI=1`，見第 4 章） |
| `key=value ...` | 任何設定檔欄位的覆寫（見第 9 章） |

出生點：`config/sim.yaml` 的 `spawn: [x, y, yaw, z_floor]`（目前 `[0, 0, 0, -0.4]`，dp_1012 1F）；設成 `null` 會用 `meta.json` 裡 build_scene 算出的點。`z_floor` 要給**目標樓層的地板高度**（會從它上方 0.5 m 往下找到下方 1 m）；給到兩層之間會站在雜物或牆緣上。出生點或 `set_pose` 和地圖重疊時，機身會自動移到 1 m 內最近的空位（最多轉 90°）。

啟動 log 要看的幾行：

```
scene scenes/dp_1012 (44116 boxes), spawn requested [x, y, yaw, z_floor] [0.0, 0.0, 0.0, -0.4] -> base at [0.0, 0.0, 0.1] yaw 0.00 (floor z -0.2, moved False), rtf 1.0, gt_tf True, custom_msg True
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

Frame（`ros.frames`）：`map`、`odom`、`base_link`、`livox_frame`。靜態 `base_link → livox_frame` 一律發布；`publish_gt_tf: true`（預設）時再加 `map → odom`（單位矩陣）和 `odom → base_link`（真值）。用 `ros.publish_gt_tf=false` 關掉後，這兩段由 Point-LIO / 定位 stack 發。

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
docker/run.sh "python3 -m pytest -q"               # tests/ 下 13 個檔，不用 ROS；樓梯、碰撞檢查、感測器時間的幾個需要已建好的 scenes/dp_1012，沒有就跳過
docker/run.sh "python3 -m pytest -q tests/test_ground.py -k fill"
```

涵蓋：PCD 讀寫、OctoMap `.bt` / `.ot` 解析與從 octree 建場景、體素化與 box 合併、地面平面化、合成地面層修正、碰撞代理、感測器時間、碰撞檢查、CustomMsg / PointCloud2 序列化。

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

## 7. 導航下點迴圈測試

把 map frame 的 waypoint 一個接一個下給導航 stack，並用 sim 真值驗證每一段。`scripts/nav_goal_loop.sh` 在 sim 容器內執行 `scripts/nav_goal_loop.py`，所以會在同一個 DDS domain。

前置條件：

1. sim 用 `scripts/run_sim.sh ros.publish_gt_tf=false` 啟動（TF 交給定位），`ros.topics` 的 LiDAR / IMU / cmd_vel 名稱要和導航 stack 用的一致（見第 5、9 章）。
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
| `config/nav_waypoints_dp_2f.yaml` | `scenes/dp` 2F 的 10 點路線（2026-10-22 起的主要測試集） |
| `config/archive/` | 一次性的探點集（dp_probe、西側樓梯 probe 1–4、dp_1012_3pt、custom_2026-10-27），留作紀錄 |

各檔的來龍去脈與當天的失敗分析見 [archive/nav-goal-loop-test-2026-10-13.md](archive/nav-goal-loop-test-2026-10-13.md)。

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
| `robot` | 理想底盤：機身尺寸、站高（沿腳平面法線量，機身底面在樓梯上和腳平面平行）、腳位、`max_step_up/down`、`max_feet_over_gap`、`max_slope_deg`（四隻腳的平面：手畫 0.2 m 深踏板是 46°，預設 50）、`attitude_tau`、`vel_limits` / `acc_limits`、`cmd_tau`、`cmd_timeout`、碰撞代理池大小。roll / pitch / 高度取四隻腳高度（各 3×3 射線取中位數）構成的平面，碰撞盒跟著機身一起傾斜 |
| `lidar` | Mid-360 參數：安裝位置、頻率、點數、FOV、距離、雜訊（掃描樣式固定為合成的非重複樣式） |
| `imu` | 頻率、雜訊、bias random walk（加速度單位固定為 g，和 livox_ros_driver2 相同） |
| `viewer` | `shadows`、`sync_hz` |
| `ros` | frame 名、`publish_gt_tf`、`publish_custom_msg`、`publish_pointcloud2`、`pointcloud2_format`、`publish_map_cloud`、各 topic 名 |

指令列覆寫：`區塊.欄位=值`，可以多個，`run_sim.sh`、`bench.py` 都接受。值用 YAML 解析，所以：

```bash
scripts/run_sim.sh physics.rtf=0 lidar.range_noise=0.0        # 數字
scripts/run_sim.sh robot.vel_limits=[1.0,0.5,1.5]              # list 不要有空白
scripts/run_sim.sh ros.topics.cmd_vel=/dog/cmd_vel ros.frames.base=base_footprint   # 對齊別的 stack 的名稱
scripts/run_sim.sh ros.publish_gt_tf=false ros.publish_pointcloud2=false          # 布林值
scripts/run_sim.sh --config config/sim_other.yaml              # 整份換掉
```

## 10. 常見問題

| 現象 | 原因 / 處理 |
|---|---|
| `QUADSIM_GUI=1 but DISPLAY is not set` | 不是在桌面終端機執行。改在有 X 的終端機跑，或不開 GUI |
| viewer 一幀要幾百 ms | 走到 `sw` 路徑。看啟動 log 的 `viewer: gl=`，有 NVIDIA GPU 就設 `QUADSIM_GL=nvidia`；確認 `viewer.shadows: false` |
| RViz 點雲、格線畫不出來 | GL 退到 2.1 fallback。用 `QUADSIM_GL=nvidia` 或 `sw`，不要用沒 `/dev/dri` 的 `hw` |
| 一起步就 `stuck` / `no_ground` | 場景地板有破洞。用目前預設（`--ground-fill 12`）重建場景，並確認 `robot.max_feet_over_gap: 1`。用 `scripts/eval_scene.py --auto` 找出缺陷在哪（[2.2](#22-評估場景scriptseval_scenepy)） |
| 到處 `step_down`，地板像有很多坑 | PCD 帶合成地面層（例如 dp_1012 的 z = −0.95）。先跑 `scripts/fix_pcd_plane.py` 再建場景（[2.1](#21-pcd-裡的合成地面scriptsfix_pcd_planepy)） |
| 樓梯變斜坡、或踏板留成體素 | 用樓梯配方重建（[第 2 章](#2-建場景pcd--scenes)），用 `eval_scene.py --stairs-box` 看側面圖 |
| 啟動 log 有 `moved True` 或 ROS warning 說機身被移動 | 出生點和地圖重疊，多半是 `spawn` 的 `z_floor` 給到別層。改成目標樓層的地板高度 |
| 其他節點時間不對、TF 過期 | 沒設 `use_sim_time:=true` |
| 導航 stack 看不到 sim 的 topic | RMW 或 domain 不同。sim 用 CycloneDDS、domain 1、`conf/cyclonedds.xml`；跑 `docker/run.sh "ros2 topic list"` 比對 |
| nav 測試 exit 2（preflight failed） | tracked pose 沒在發、太舊或和真值差超過 0.5 m：定位還沒收斂，或 Point-LIO 掛了（曾經 segfault） |
| nav 測試每段 0.3 s 就 `ABORTED` | controller 判起點不安全，sim 不會自己脫困。見測試紀錄 §4.2；用 teleop 推開後再跑 |
| `scenes/`、`results/`、`Maps/*.pcd` 不在 git 裡 | 本來就 gitignore；場景用第 2 章重建 |
