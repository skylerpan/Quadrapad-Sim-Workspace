# 點雲 PCD → 模擬環境：方法與技術選型

> 對象：機器狗、人型機器人、AMR｜目標模擬器：MuJoCo、Gazebo（Harmonic）、Isaac Sim / Isaac Lab
> 用途：導航演算法驗證、Sim2Real 數位孿生｜日期：2026-10-09

## 1. 現況與前提
| 項目 | 現況 |
|---|---|
| 感測器 | Livox Mid-360（非重複掃描，FOV 360° × 約 −7°~52°，約 20 萬點/秒，內建 IMU） |
| SLAM | FAST-LIO |
| 資料 | **只有合併後的 PCD**（沒有逐幀 scan 和位姿，也沒有顏色） |
| 場景 | 室內（房間、樓層） |
| 策略 | v0 先「硬轉」：不做參數化建模，直接從點雲產出三個模擬器的場景 |
| 衍生資料 | OctoMap .bt（由合併 PCD 轉出，解析度 >5 cm） |
| 優先目標 | 機器狗在 MuJoCo 中測試 3D 導航（已有自有足式策略） |
| 導航堆疊 | 魔改 Nav2 純 3D 導航：OctoMap 全域地圖、輸出 cmd_vel；定位為 Point-LIO 里程計 + VGICP/NDT 全域定位 |

## 2. 基礎觀念
- **點雲沒有面**：模擬器需要知道哪裡是實心、射線會打在哪裡，所以必須轉成 mesh、體素或基本形狀
- **SLAM 地圖的誤差有三個來源**：位姿誤差（牆變厚、鬼影）、場景變動（行人拖影）、感測器物理（玻璃、鏡面、邊緣雜點）
- **幾何表示法**

| 表示法 | 優點 | 缺點 |
|---|---|---|
| Mesh | 通用、好看 | 從點雲重建難；拿來算碰撞很貴 |
| 體素 | 簡單、穩定 | 精度等於格子大小，有方塊感 |
| 基本形狀（box、平面） | 碰撞運算最快 | 只能近似規則物體 |
| 凸包、凸分解 | 物理引擎最喜歡 | 凹形物體需要拆分 |
| Heightfield | 地形碰撞快 | 2.5D，無法表示桌底等懸空結構 |

- **視覺和碰撞分開**：視覺可以精細，碰撞要簡單，但兩者必須同源、差距要小
- **座標系**：Z-up、公尺（REP-103）；模擬座標 = 實機地圖座標
- **MuJoCo 的最大坑**：mesh 碰撞一律當成凸包，整個房間的 mesh 會變成實心巨石

## 3. 三個模擬器
| | MuJoCo | Gazebo（Harmonic） | Isaac Sim / Lab |
|---|---|---|---|
| 強項 | 接觸動力學精準、快，GPU 平行（MJX/Warp） | ROS 2 整合最深 | RTX 渲染與感測器、大規模 RL |
| 場景格式 | MJCF | SDF | USD |
| mesh 碰撞 | 當成凸包 | trimesh 可以直接用（大型 mesh 會慢） | 靜態物直接用 trimesh；動態物需要凸分解 |
| LiDAR | 沒有原生 LiDAR，用 `mj_multiRay` 自己組 | `gpu_lidar`（均勻樣式，打到視覺 mesh） | RTX LiDAR（可自訂 emitter，受材質影響） |
| 硬體 | CPU | CPU + 一般 GPU | NVIDIA RTX |
| 本專案角色 | 足式和人型機器人的 locomotion RL | Nav2 回歸測試、CI | 數位孿生、感知 Sim2Real |

## 4. 三種機器人的需求
| | 機器狗 | 人型機器人 | AMR |
|---|---|---|---|
| 最在意 | 地形、樓梯、摩擦 | 地形加可互動物件 | LiDAR 看到的幾何、可通行空間 |
| 建議體素大小 | 2–3 cm | 2–3 cm | 5 cm |

## 5. 整體架構
```
① 資料採集   Mid-360 + FAST-LIO（現在：合併 PCD；建議：rosbag + Odometry）
      ▼
② 前處理     體素化 → SOR/ROR → RANSAC 地板 → 重力對齊(Z-up, floor z=0)
             → 外輪廓過濾(窗外/鏡像) → 動態候選標記 → CloudCompare 人工確認
      ▼
③ 共用幾何層 佔用體素格（唯一的幾何真相來源）
             ├─ 合併 box        → 碰撞
             ├─ MC mesh         → 碰撞/視覺
             ├─ Poisson mesh    → 精緻視覺（可選）
             └─ 2D 佔用地圖     → Nav2（pgm+yaml）
      ▼
④ 中間格式   Scene Manifest（JSON/YAML）：T_world_map、單位、物件、資產、材質、人工修補標記
      ▼
⑤ Exporter   MJCF(MjSpec) ｜ SDF(Jinja2) ｜ USD(usd-core)
      ▼
⑥ 感測器介面 PointCloud2 → Livox CustomMsg（offset_time/line/reflectivity）
             + 雜訊後處理 + IMU 單位/同步 → 與實機相同的 Point-LIO + 全域定位 + 3D 導航
      ▼
⑦ 驗證       幾何 P95 / 感測統計與覆蓋率 / 導航行為（模擬 vs 實機）
```

### 核心設計原則
1. **單一幾何來源**：所有輸出都從同一份佔用格衍生
2. **碰撞簡單、視覺精細、兩者同源**：LiDAR 看到的就是機器人撞到的
3. **與模擬器無關的中間格式**：exporter 只是一層薄薄的翻譯
4. **座標系與實機一致**
5. **下游堆疊相同**：sim 和 real 只差資料來源

## 6. 各階段方法

### 6.1 前處理與清理（只有合併 PCD 的情況）
| 問題 | 方法 |
|---|---|
| 雙層牆、鬼影 | 檢查牆厚（點到平面距離 P95）；雙峰時牆線取靠室內一側；P95 > 5–10 cm 建議重新採集 |
| 行人、操作者拖影 | 移除結構平面後聚類，以「低密度、不貼附、細長帶狀、離地 0.1–2 m」標記候選，再人工確認 |
| 窗外點、鏡像房間 | 外牆輪廓之外的點直接刪除；室內鏡子用平面鏡射加 ICP 偵測 |
| 玻璃 | 牆線上稀疏的共線缺口 → 標成玻璃候選 → **碰撞要手動補上** |
| 雜訊、邊緣混合點 | SOR、ROR；家具的 OBB 用 trimmed 擬合 |

有逐幀資料時可以升級：HBA/BALM 位姿精修、DUFOMap/ERASOR/Removert 移除動態物（可以用 DynamicMap_Benchmark 比較）、用射線穿牆推理找鏡面和玻璃。

### 6.2 表面重建
| 方法 | 在 Mid-360 室內點雲的表現 | 定位 |
|---|---|---|
| **體素 + Marching Cubes** | 不需要法向量、不長氣泡、結果可預期；先做閉運算補洞，高斯模糊加 Taubin 平滑去階梯感 | **預設，碰撞和視覺都可以用** |
| **Screened Poisson** | 最平滑；需要定向法向量（建議「朝自由空間定向」），用 Neumann 邊界，依距離修剪掉氣泡 | 精緻視覺 |
| Ball Pivoting | 密度不均時破洞多 | 不推薦 |
| Alpha Shape | 參數敏感，容易有雜面 | 只用來預覽 |
| CGAL 平面多邊形化 | 面數少、牆角銳利，但雜物處理差 | 之後要乾淨的牆時再考慮 |
| NKSR | 品質高，需要 GPU | 選用 |

Poisson depth 參考：格子大小約等於包圍盒邊長 / 2^depth（40 m 場景：depth 10 約 3.9 cm，depth 11 約 2 cm）。

**品質指標**：Accuracy（mesh → 點雲距離 P95 < 1.5 倍體素）、Completeness（點雲 → mesh 在 5 cm 內的比例 > 95%）。

### 6.3 碰撞表示
| 方法 | 適用 |
|---|---|
| **貪婪合併 box**（約 50 行 numpy） | 三個模擬器通用，MuJoCo 首選 |
| Triangle mesh | Gazebo、Isaac 的靜態場景 |
| CoACD 凸分解 | 單一物件；不要用在整個房間 |
| Heightfield | 足式機器人的地面；不適合 AMR（無法表示桌底） |

### 6.4 感測器模擬與對齊
- **四個對齊層次**：介面（訊息格式、topic）、幾何（FOV、掃描樣式）、時間（逐點時間戳、IMU 同步）、雜訊與物理
- **介面**：轉換節點把 PointCloud2 轉成 CustomMsg，依掃描樣式回填 `offset_time`；IMU 統一單位（Livox 輸出 g，模擬器輸出 m/s²），使用 `use_sim_time`
- **雜訊後處理**：距離雜訊 σ 約 2 cm、入射角掉點、依反射率縮短量程、邊緣混合點、玻璃穿透
- **掃描樣式**：Gazebo 用均勻樣式近似（Livox 樣式外掛主要是 Gazebo Classic 版）；Isaac 用自訂 emitter；MuJoCo 用 `mj_multiRay` 讀入角度表
- **驗證**：單幀統計、積分覆蓋率、逐幀對照（需要 rosbag）、系統行為（FAST-LIO 漂移、Nav2 成功率）

## 7. 技術棧
| 層 | 工具 |
|---|---|
| 點雲處理 | Open3D、numpy/scipy、pypcd4；GUI 用 CloudCompare |
| 網格 | scikit-image（MC）、trimesh、fast-simplification 或 pymeshlab |
| 進階 | PoissonRecon、CGAL、CoACD、NKSR |
| 輸出 | `mujoco`（MjSpec）、Jinja2（SDF）、`usd-core`；Isaac Lab MeshConverter、obj2mjcf |
| 手動修補 | Blender、CloudCompare |
| 執行期 | ROS 2、ros_gz_bridge、Isaac ROS 2 bridge、FAST-LIO、Nav2 |
| 未來升級 | HBA/BALM、DUFOMap、3DGS |

## 8. 演進路線
| 階段 | 內容 | 前提 |
|---|---|---|
| v0 硬轉 | 合併 PCD → 佔用格 → box + MC → 三個模擬器 | 現在就能做 |
| v0-Q 機器狗 | 平面吸附地板 hfield + 樓梯擬合 box + MuJoCo 橋接 + 3D 導航測試 | 原始 PCD |
| v1 介面對齊 | 感測器轉換節點、雜訊模型、Nav2 地圖同源 | v0 |
| v2 資料升級 | 改錄 rosbag；HBA、自動動態移除、逐幀驗證 | 重新採集 |
| v3 結構化 | 參數化牆、樓梯，可互動物件 | 視需求 |
| v4 視覺擬真 | 3DGS 或 PBR 材質，Isaac 數位孿生展示 | 補拍影像 |

## 9. 下次採集建議
- 錄原始 bag：`ros2 bag record /livox/lidar /livox/imu`
- 離線重跑 FAST-LIO 時，同時保存 `/Odometry` 和逐幀點雲
- 設好 `blind` 參數，避免掃到自己；路線要形成閉環
- 硬體支援的話，開啟雙回波（有助於處理玻璃場景）

## 10. 待確認事項
- Isaac Sim 版本是否內建 Mid-360 的 RTX LiDAR 設定檔
- gz Harmonic 是否有可用的 Livox 掃描樣式外掛
- 實際 PCD 的欄位（intensity）、是否保留時間順序、是否被降採樣過
- 足式策略的訓練來源（判斷方法見 §12.3）
- Point-LIO 設定檔的 IMU 單位參數（例如 `acc_norm`），模擬中的 IMU 要依此發布
- 全域定位（VGICP/NDT）使用的先驗地圖格式和來源（PCD？是否和 `.bt` 同源？）
- xmSimulator / MuJoCo-LiDAR 是否支援逐點時間，以及它們的 Mid-360 樣式準不準
- Mid-360 掃描樣式表的來源（現成樣式，或從實機 rosbag 反推）

## 11. OctoMap `.bt` 評估

**定位**：`.bt` 是二值化的 OctoMap（只保留 occupied 和 free，沒有機率；機率只有 `.ot` 才會保留）。它在架構上等同於「佔用格」，但是**衍生產物，原始 PCD 才是主檔**。

| 得到的 | 失去的 |
|---|---|
| 已經完成體素化；可能包含 free space | 解析度固定（目前大於 5 cm）；intensity 遺失；無法重新設定佔用門檻 |

**風險一：單一原點射線投射**
用 `insertPointCloud(cloud, origin)` 從單一原點（通常是掃描起點）插入合併 PCD 時，射線會穿過牆壁把體素標成 free，造成牆上破洞。
- 檢查方式：統計 free 和 occupied 葉節點的數量（free 接近 0 代表安全）；在 octovis 裡檢查背對原點的牆

**風險二：解析度大於 5 cm**
- 門和走廊變窄，加上 Nav2 膨脹層後可能無法通行
- 細物件被加粗或消失
- 地板稍有傾斜就會出現 10 cm 的假台階（0.5° 傾斜約每 11 m 一階）
- 樓梯階高被量化失真

**結論**
- **`.bt` 當導航地圖沒有問題**（3D 規劃用 5–10 cm 很合理）；不適合的是拿它當**模擬器的物理幾何**，物理場景要從 PCD 用 2–3 cm 重建。所以「模擬世界」和「導航地圖」是兩份刻意分開的資料（見 §12.7）
- **機器狗和人型機器人：不使用 `.bt`**，改從 PCD 重新產生
- AMR v0 勉強可用，但地板要換成平面、門寬要比對實測值、要先檢查破牆
- 原則：先重力對齊再體素化，只標佔用、不做單一原點射線投射，各機器人依需求的解析度分別產生

**2026-10-16 更新**：`build_scene --map <x>.bt|.ot` 現在也能直接從 OctoMap 建場景（`pcd2sim/octomap_io.py` 純 numpy 讀 occupied 葉節點、展開 prune 過的節點，每格 1:1 變成一個樹解析度的體素、過 26 鄰居去雜訊（`--min-neighbors`，可選 `--min-cluster`）再走同一套地面平面化（補洞、吸附成平面）合併成 box；不做百分位裁切；用法見 [usage.md](usage.md) §2.3）。上面的原則不變：這條路給快速粗場景和只有 octomap 的地圖用，要爬樓梯的物理場景仍然從 PCD 以 2–3 cm 重建。

## 12. 機器狗：MuJoCo 3D 導航測試

### 12.1 場景表示
**陷阱**：Mid-360 的距離雜訊約 2 cm，用 2 cm 體素時，地板會變成凹凸不平的表面，假凸起會絆住機器狗的腳。所以**可行走的表面必須吸附成平面**。

| 地形 | MuJoCo 表示法 |
|---|---|
| 平地、斜坡、門檻 | **hfield（2 cm）**：每格取點高度的中位數 → 區域成長切出平面區域 → 逐區域吸附到擬合平面。門檻要依「銳利且連續的邊緣」保留下來，而不是只看高度差 |
| 樓梯 | 每一階擬合成一個 box（hfield 會把階梯邊緣內插成小斜坡） |
| 牆、家具 | 合併 box（運動帶 0–0.8 m 用 2–3 cm，以上用 5 cm；用稀疏儲存） |
| 視覺 | MC mesh，`contype=0 conaffinity=0` |

hfield 只能表示 2.5D；樓梯下方的空間和多層樓的部分要用 box 補上。

### 12.2 感測器
- **LiDAR（滾動式）**：Point-LIO 逐點融合，**必須有真實的逐點時間**。每個物理步（約 2 ms）只從當下位姿發射掃描樣式中 [t, t+dt) 的射線（用 `mj_multiRay` 讀入 Mid-360 角度表），累積 100 ms 後發布一幀，並填入 `offset_time`。運動畸變會自然產生，總計算量不變。geom 數量多時要先 benchmark；太慢的話改用 Open3D `RaycastingScene`（Embree）。實作方案見 §13
- **Height scan**：每個控制步從機身朝下打約 200 條射線，用 `bodyexclude` 排除機器人本身
- **IMU**：MuJoCo 的 accelerometer 和 gyro 感測器，安裝位置要和實機一致；必須掛在自由剛體上（**mocap body 沒有速度和加速度，IMU 會讀錯**）；**以 g 為單位發布**，Point-LIO 設定和實機一致；加入雜訊和 bias 隨機漫步；理想底盤模式可以疊加步態頻率的震動（約 2 Hz 起伏加著地衝擊）

### 12.3 策略 sim2sim 對齊清單
| 類別 | 必須和訓練時一致的項目 |
|---|---|
| 關節順序 | 觀測和動作都要重新排列（最常見的錯誤） |
| 觀測 | 預設關節角、縮放係數、投影重力、指令、上一步動作、機身線速度的來源 |
| Height scan | 範圍、間距、排列順序、偏移、裁切值、座標系（只跟 yaw 或完整姿態） |
| 動作與致動器 | action scale、kp/kd、力矩上限、armature、阻尼與摩擦、致動器網路 |
| 時間 | 物理步長 × decimation（例如 5 ms × 4 = 50 Hz） |
| 接觸 | 腳的 geom、摩擦係數、`condim`、`solref`/`solimp` |

**判斷策略的訓練來源**：
- Isaac Lab：log 目錄裡有 `params/env.yaml`、`agent.yaml`，以及 `exported/policy.pt` 或 `policy.onnx`
- legged_gym / Isaac Gym：有 Python 寫的 config 類別（`LeggedRobotCfg`）
- MuJoCo Playground / MJX：JAX/Brax 的參數檔（例如 orbax 或 flax 格式）

**驗證順序**：平地 → 斜坡 → 門檻 → 樓梯；先在 MuJoCo 的簡單地形上通過，再換成重建場景。

### 12.4 導航堆疊與模擬介面
| 方向 | Topic | 備註 |
|---|---|---|
| 發布 | `/clock` | 全部節點設 `use_sim_time` |
| 發布 | `/livox/lidar` | CustomMsg，含逐點 `offset_time`（滾動式 LiDAR） |
| 發布 | `/livox/imu` | 200 Hz，單位 g |
| 發布 | `/sim/ground_truth/odom` | **只用於評估**，不進導航的 TF 樹 |
| 訂閱 | `/cmd_vel` | 驅動理想底盤或策略 |

- **TF**：`map → odom` 由全域定位（VGICP/NDT）發布，`odom → base` 由 Point-LIO 發布，**模擬器不可發布這兩段**
- **座標系：用傾斜重力代替旋轉幾何**。`.bt`、PCD 地圖、Point-LIO 都在同一個 map 座標系（不一定水平）。模擬世界直接採用 map 座標系，用 `<option gravity="...">` 把重力設成地圖量測到的下方向，地圖檔就都不用旋轉（避免重新取樣失真）
- **cmd_vel 的速度範圍**要和實機步態控制器一致（用策略時，要在策略的訓練指令範圍內）

### 12.5 理想底盤（模式 B，導航測試主力）
- 機身是**自由剛體，以柔性 weld 約束綁在 mocap 目標上**（不能直接用 mocap body，否則 IMU 沒有讀值）
- 每個控制步：cmd_vel 在機身座標系積分出目標的 x、y、yaw；在四個腳的位置朝下打射線（每腳 3×3 取中位數），四隻腳的高度構成的平面給目標的 z、roll、pitch；機身中心沿該平面法線抬 `stand_height`，所以在樓梯上機身底面和腳平面平行（垂直抬高會讓傾斜機身的前下角插進下一階豎板）
- **模擬步態控制器特性**：速度和加速度上限、約 0.2 s 的一階延遲、追蹤誤差
- **可通行性約束**：單步高差超過可跨越上限（例如 15–18 cm）或坡度過陡時停住，記錄為「卡住」
- **碰撞**：機身外框只涵蓋身體高度（腳由射線處理），碰撞檢查用的盒子跟著機身一起 pitch / roll（2026-10-19 之前是水平盒，前緣會撞到下一階豎板）；撞到障礙物會產生真實接觸，直接記錄為撞擊事件

### 12.6 測試迴路
```
MuJoCo（自由剛體+weld 或 策略）
   │ 滾動式 LiDAR、IMU(g)、/clock、真值(僅評估)
   ▼
ROS 2 橋接(mujoco+rclpy)
   ├─► Point-LIO ──► odom→base
   ├─► VGICP/NDT 全域定位(先驗 PCD 地圖) ──► map→odom
   └─► 魔改 Nav2 純 3D 導航(OctoMap 全域地圖 + 局部 3D 地圖)
                         │ /cmd_vel
MuJoCo ◄── 理想底盤(模式 B) 或 策略推論(模式 A) ◄──┘
```
| 模式 | 機器人怎麼動 | 用途 |
|---|---|---|
| A 完整動力學 | 策略輸出關節力矩 | 端到端：會不會摔倒、被門檻絆住 |
| B 理想底盤 | 自由剛體加 weld 跟隨 cmd_vel 並貼合地形 | 只測導航和定位，排除步態問題，適合 CI |

**推進順序**：B 加真值定位（只測規劃）→ B 加 Point-LIO 和全域定位 → A 加上策略（完整 Sim2Real 鏈）。

### 12.7 測試情境與評估
**原則：刻意控制「模擬世界」和「地圖」之間的差異。** 模擬世界和定位地圖源自同一份 PCD，不加差異的話，定位會好得不真實。

| 情境 | 模擬世界 | 導航／定位地圖 | 測什麼 |
|---|---|---|---|
| 基準 | 從 PCD 重建 | 由模擬碰撞幾何體素化產生的 `.bt` | 只測規劃器 |
| 真實地圖 | 從 PCD 重建 | 實際的 `.bt` | 地圖量化、誤差的影響 |
| 未知障礙 | 加入家具、箱子 | 原地圖 | 局部 3D 建圖和重新規劃 |
| 動態障礙 | 會移動的行人（mocap） | 原地圖 | 動態避障 |
| 跨高度 | 樓梯、坡道、門檻 | — | 3D 目標、可通行性 |
| 窄通道 | 門、走廊 | — | 量化造成的假性不通 |
| 初始位姿錯誤、綁架 | 不變 | 原地圖 | 全域定位的重定位能力 |
| 退化場景 | 長走廊、對稱房間 | 原地圖 | Point-LIO 和 NDT/VGICP 的穩健性 |

**自動化**：`scenario.yaml`（起點、3D 目標、加入的障礙物、動態角色、使用的地圖）→ runner 啟動模擬和導航堆疊 → 送出導航目標並錄 rosbag → 計算指標。

**評估指標**
- 導航：成功率、時間、路徑長度、最小障礙距離、撞擊和卡住次數、重新規劃次數、3D 目標誤差
- 定位（和真值比較）：Point-LIO 的 ATE/RPE、全域定位誤差、重定位成功率和所需時間

**場景驗證**：用捲尺實測 2–3 階樓梯的階高和階深、門檻高度、坡道角度，和擬合結果比對（誤差小於 1 cm 或 1°）。

## 13. LiDAR 模擬實作方案（MuJoCo vs Gazebo）

**需求**：Point-LIO 逐點融合，需要**逐點時間**（`offset_time`），最好還要有 **Mid-360 非重複掃描樣式**；輸出格式比照 livox_ros_driver2（CustomMsg，或 PointXYZRTLT 格式的 PointCloud2：x、y、z、intensity、tag、line、timestamp）。

### 13.1 MuJoCo 現成方案
| 方案 | 做法 | Mid-360 樣式 | 逐點時間 | 備註 |
|---|---|---|---|---|
| [xmSimulator](https://github.com/rxdu/xmSimulator) | C++，`mj_multiRay`（CPU、headless），掃描樣式可替換 | 有內建 | 待確認 | 最接近需求，優先評估 |
| [MuJoCo-LiDAR](https://gittrend.io/repo/discoverse-dev/MuJoCo-LiDAR) | Python，CPU/Taichi/JAX 後端，ROS 整合 | 待確認 | 待確認 | MIT 授權 |
| [mujoco_ros2_control lidar 外掛](https://control.ros.org/humble/doc/mujoco_ros2_control/mujoco_extensions/doc/mujoco_extensions.html) | MJCF `<plugin>`，均勻網格，可設 `update_rate`、`async` | 沒有 | 沒有 | — |
| [mujoco_3d_lidar](https://index.ros.org/p/mujoco_3d_lidar/) | 類似的 `<plugin>` | 沒有 | 沒有 | `async` 時資料有延遲 |
| rangefinder 感測器 | 一個 site 一條射線 | — | — | 只適合 2D |

### 13.2 MuJoCo 滾動式 LiDAR 的核心邏輯
```
預先載入：Mid-360 樣式表 [(t_i, 方位角_i, 仰角_i)]，依時間排序（約 20 萬條/秒）

每個物理步 (dt ≈ 2 ms)：
  1. 取出 t ∈ [t_now, t_now+dt) 的射線（約 400 條）
  2. 讀 LiDAR site 的當下位姿，把方向轉到世界座標
  3. mj_multiRay(原點, 方向, geomgroup=只含場景, bodyexclude=機器人, cutoff=40m)
  4. 命中點轉回「該時刻的感測器座標」
  5. 雜訊：距離 σ≈2 cm、入射角掉點、盲區 <0.1 m
  6. 累積：offset_time = t_i − t_frame_start、reflectivity、line

每 100 ms：打包成 CustomMsg（timebase = 幀起始時間）→ /livox/lidar
```
- **排除機器人本身**：`bodyexclude` 只能排除一個 body，所以場景 geom 要放進專用的 geomgroup
- **Reflectivity**：用「geom → 反射率」對照表，再依入射角衰減
- **樣式表來源**：參考 xmSimulator 的樣式，或從實機 rosbag 的 `offset_time` 和角度反推（最準，也可以拿來驗證模擬）

**效能**（以模擬時間計約 20 萬條射線/秒，要先 benchmark）
1. 減少 geom 數：貪婪合併 box；地板和斜坡整合成單一 hfield
2. 改用 C++（xmSimulator 或 MuJoCo 外掛），去掉 Python 呼叫的負擔
3. 混合加速：靜態場景用 Embree（Open3D `RaycastingScene`），動態物體用 `mj_multiRay`，兩者取較近的距離
4. 需要 GPU 平行時：MuJoCo-LiDAR 的 Taichi/JAX 後端，或 MJX / MuJoCo Warp 的射線功能

**優點**：射線打在碰撞 geom 上（看到的就是會撞到的）；可以精確重現非重複樣式和逐點時間；結果可重現。

### 13.3 Gazebo（Harmonic）
- **原生 `gpu_lidar`**：均勻網格、GPU 渲染深度、**打到視覺 mesh**、整幀在同一時刻產生（沒有運動畸變，也沒有逐點時間），經 `ros_gz_bridge` 輸出 PointCloud2
- **Livox 外掛現況**：[fratopa/Mid360_simulation_plugin](https://github.com/fratopa/Mid360_simulation_plugin) 只支援 ROS Noetic 加 Gazebo 11，用的是固定網格，不是真正的非重複樣式。**目前沒有找到可直接用的 Harmonic 加 ROS 2 版本**

| 做法 | 內容 | 取捨 |
|---|---|---|
| **A. 高頻渲染後重新取樣**（推薦） | `gpu_lidar` 以 50–100 Hz 輸出高解析度 360° 資料 → ROS 2 節點依 Mid-360 樣式表，從時間最接近的那幀取出對應角度 → 組成 CustomMsg 並填入 `offset_time` | 不用寫 Gazebo 外掛；逐點時間被量化成 10–20 ms；GPU 負擔較高 |
| B. 自寫 gz-sim system 外掛 | 移植 fratopa 外掛的邏輯，每個物理步依樣式表發射射線 | 最準確；要寫 C++，也要追 Harmonic 的 API |

### 13.4 比較與建議
| 項目 | MuJoCo 滾動式 | Gazebo A | Gazebo B |
|---|---|---|---|
| 非重複樣式 | 精確 | 近似 | 精確 |
| 逐點時間 | 約 2 ms | 10–20 ms 量化 | 精確 |
| 打到的幾何 | 碰撞 geom | 視覺 mesh | 依實作而定 |
| 開發量 | 中（可借用 xmSimulator） | 低到中 | 高 |
| 和機器狗策略整合 | 同一個引擎 | 要另外做 sim2sim | 同左 |
| 不用 GPU 也能跑 | 可以 | 不行 | 視實作而定 |

**建議**：LiDAR 在 MuJoCo 裡做。
1. 先評估 xmSimulator（確認逐點時間，不支援就補上滾動式邏輯）
2. 從實機 rosbag 反推樣式表並用來驗證
3. 先 benchmark，太慢就改用 Embree 加 MuJoCo 的混合方案

Gazebo 留給「需要快速做 ROS 2 整合、不在意逐點時間精度」的情境，用做法 A。

**參考**：[xmSimulator](https://github.com/rxdu/xmSimulator)、[MuJoCo-LiDAR](https://gittrend.io/repo/discoverse-dev/MuJoCo-LiDAR)、[mujoco_ros2_control extensions](https://control.ros.org/humble/doc/mujoco_ros2_control/mujoco_extensions/doc/mujoco_extensions.html)、[mujoco_3d_lidar](https://index.ros.org/p/mujoco_3d_lidar/)、[fratopa/Mid360_simulation_plugin](https://github.com/fratopa/Mid360_simulation_plugin)、[NVIDIA 論壇：Livox MID360](https://forums.developer.nvidia.com/t/livox-mid360/283074)、[livox_ros_driver2 設定指南](https://openelab.io/blogs/learn/livox-mid-360-ros2-driver2-setup-guide)
