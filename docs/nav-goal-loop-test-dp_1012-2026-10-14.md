# dp_1012 三點導航迴圈測試紀錄（2026-10-14）

目的：在 quadsim `dp_1012` 場景上，用 `scripts/nav_goal_loop.sh` 連續 30 分鐘（sim time）在 3 個 waypoint 之間繞圈，驗證 robot01 3D 導航 stack 的到位率、到位精度與長時間穩定性，並把失敗分類。

## 1. 結論

- **71 段 / 66 PASS / 5 FAIL**（全部 `ABORTED`），**1218 m，平均 0.78 m/s**，24 圈，30 分鐘內沒有任何節點死掉、沒有碰撞、沒有 stuck。扣掉每圈開頭的 0 m 空段，實際行駛 48 段：**43 PASS / 5 FAIL（89.6%）**。
- 到位精度穩定：真值距 path 終點平均 0.23 m、最大 0.28 m（容差 0.35 m）；到位時定位誤差 ≤ 0.14 m。
- 5 次失敗是同一個型態：**`3D costmap 已过期（age=1.040s, timeout=1.000s）` → local costmap 進失敗安全狀態 → MPPI `Optimizer fail to compute path` → action ABORTED**。每次只持續 0.3 s，下一段立刻恢復，所以沒有觸發「連續 3 段失敗」停測。
- 探針量到的直接原因：**Point-LIO 輸出的 `/robot01/cloud_registered_body` 在 wall time 上會間歇停頓 0.5–0.9 s**（stamp 仍然連續 0.100 s，所以不是 sim 或 LiDAR 掉幀，是 Point-LIO 處理延遲），costmap 用 `now() − cloud stamp` 算年齡，超過 1.0 s 就判過期。sim `/clock` 全程無跳變（60005 筆，無 > 0.3 s 間隔）。
- 前一天 10-13 的 Point-LIO data-race 修法（`docs/nav-crash-2026-10-13.md`）**尚未套用**，但這 30 分鐘沒有再現 segfault。

## 2. 環境

| 元件 | 狀態 |
|---|---|
| quadsim | container `hopeful_bardeen`，`--scene scenes/dp_1012 --viewer --no-gt-tf`，RTF 1.00（每 10 s 區間都是 sim+10.0） |
| 導航 stack | container `robot01`：`location_3d.launch.py sim:=true`、`nav_3d.launch.py sim:=true`（17:06 啟動）、`octo_planner_rviz_node`（`src/syncai_planner/scripts/octoplanner_3d.sh`，17:27 重啟載入 17:06 重建的快取：362286 voxels） |
| 地圖 | `Maps/dp_1012.pcd`（sim 場景與 planner octomap 同一份） |
| waypoint | `config/nav_waypoints_dp_1012_3pt.yaml`：p01 (-12.62, -15.52, 0.04)、p02 (-28.30, 0.13, -0.24)、p03 (1.27, 1.49, -0.15)，17:06 定位改回 dp_1012 後重讀（定位 vs 真值 0.10 m） |
| 指令 | dry-run：`scripts/nav_goal_loop.sh --dry-run --dry-run-from matrix --waypoints config/nav_waypoints_dp_1012_3pt.yaml`（6/6 PASS，24.5–30.8 m）；live：`scripts/nav_goal_loop.sh --waypoints config/nav_waypoints_dp_1012_3pt.yaml --loops 0 --duration 1800` |
| 結果 | `results/nav_goal_loop/20261014-172817-dry/`、`results/nav_goal_loop/20261014-172830/`（report.json、legs.csv、run.log） |
| 主機負載 | 14 核，load avg ≈ 7；chrome ≈ 190%、quadsim ≈ 170%、controller_server ≈ 50%、Point-LIO ≈ 30%（launch 內的 taskset/chrt 已於 10-12 全部移除） |

測試從 17:28:30 跑到 17:58:39（sim t 3982 → 5791）。script 每圈從離機器人最近的點重新開始，所以每圈是「0 m 空段 + 2 段實走」，路線為 p01 → p02 → p03 → p01 → …。

## 3. 結果

### 3.1 各段統計（只算實走段，PASS 的）

| 段 | 次數 | PASS / FAIL | 路徑 | 耗時 平均 / 最短 / 最長 | 速度 | 真值距終點 平均 / 最大 | planner 挪點 平均 / 最大 | 途中定位誤差最大 |
|---|---|---|---|---|---|---|---|---|
| p01 → p02 | 16 | 14 / 2 | 23.9 m | 33.6 / 27.8 / 37.4 s | 0.71 m/s | 0.23 / 0.27 m | 0.22 / 0.25 m | 0.66 m |
| p02 → p03 | 16 | 15 / 1 | 29.3 m | 33.9 / 31.0 / 42.2 s | 0.87 m/s | 0.26 / 0.28 m | 0.32 / 0.36 m | 0.61 m |
| p03 → p01 | 15 | 14 / 1 | 30.4 m | 39.8 / 38.1 / 43.5 s | 0.76 m/s | 0.20 / 0.24 m | 0.51 / 0.55 m | 0.62 m |
| p03 → p02（失敗後折返） | 1 | 1 / 0 | 12.0 m | 20.9 s | 0.57 m/s | 0.20 m | 0.20 m | 0.42 m |

- 瞬時最高速度 1.17–1.25 m/s。
- 「planner 挪點」是 octo_planner 把 goal snap 到可通行體素的距離：p01 固定被挪 0.5 m（p01 附近 0.3 m 半徑放不下），p02/p03 0.2–0.35 m。script 的 `--snap-warn` 是 1.0 m，所以沒有警告。
- 途中定位誤差（Point-LIO + global_localization vs 真值）跑動時最大 0.66 m，到位停下後收斂到 0.02–0.14 m，低於 `--loc-fail` 1 m。

### 3.2 失敗清單

| # | 圈 | 段 | 中止時間 | 離終點 | controller log |
|---|---|---|---|---|---|
| 6 | 2 | p02 → p03 | 14.6 s | 10.6 m | 17:30:54 `3D costmap 已过期 age=1.040s` → `Optimizer fail to compute path` |
| 24 | 8 | p01 → p02 | 7.5 s | 8.0 m | 17:38:43 同上 |
| 25 | 9 | p02 → p01（折返） | 4.8 s | 3.0 m | 17:38:49 同上，`Controller patience exceeded` |
| 38 | 13 | p03 → p01 | 18.3 s | 18.1 m | 17:44:14 同上 |
| 62 | 21 | p01 → p02 | 9.0 s | 8.0 m | 17:54:23 同上 |

另有 17:29:59 一次 `elevation map is stale` 沒有造成中止。

### 3.3 節點存活與系統訊號（每 60 s 檢查一次，共 30 次）

| 項目 | 結果 |
|---|---|
| `/robot01/controller_server`、`/robot01/laserMapping`、`/robot01/transform_fusion`、`/robot01/global_localization`、`/octo_planner_rviz_node` | 30/30 全部在 |
| launch pane 的 `process has died` / `TF_OLD_DATA` / `exit code -11` | 0 |
| `/sim/events` collision / stuck | 0 |
| controller `Yaw-aware A* plan completed slower than one 20Hz control period` | 377 次（270–350 ms 一次） |
| controller `Control loop missed its desired rate of 20Hz` | 696 次 |
| global_localization `small_gicp 迷失 → NDT_OMP 兜底` | 33 次（全部成功，每次 ≈ 90–115 ms） |

## 4. 失敗原因分析

### 4.1 直接原因：Point-LIO 輸出延遲 > 1 s，costmap 判過期

`syncai_costmap_3d/src/costmap_3d_layer.cpp` `updateBounds()`（2230 行）：`map_age = node_->now() − map_stamp_`，`map_stamp_` 是最後一筆 `cloud_registered_body` 的 header stamp，`map_timeout` 1.0 s（`controller_server_params.yaml:301`；controller 自己也有 `elevation_map_timeout: 1.0`，22 行）。超過就 `current_ = false`，MPPI 收到失敗安全 costmap 後 `Optimizer fail to compute path`，action 立刻 ABORTED，沒有任何等待或重試。

三支探針（`SyncAI-Device-Workspace/log/looptest/*.py`，17:45–17:59）：

| 探針 | 結果 |
|---|---|
| `/clock`（200 Hz，60005 筆） | 無任何 > 0.3 s 的間隔，sim 時間沒有跳變 |
| `/robot01/cloud_registered_body`（Point-LIO 輸出，7801 筆） | 13 次 wall 間隔 0.50–0.89 s，**stamp 間隔全部仍是 0.100 s**；最後一次 17:54:23（wall 0.891 s）正好對上第 5 次中止 |
| `/robot01/Odometry` | 4 次 wall 間隔 0.40–0.45 s，stamp 仍連續 |

stamp 連續代表 LiDAR 幀都有進到 Point-LIO，是 Point-LIO 本身處理卡了 0.5–0.9 s 後一口氣補發；costmap 在補發前那一刻看到的最新 stamp 已經 1.04 s 舊（0.9 s 停頓 + 10 Hz 幀距 + 20 Hz 檢查相位）。

### 4.2 Point-LIO 為什麼會停頓（未確認）

- 時間上與 global_localization 的 `迷失 → NDT 兜底` 事件高度重疊：cloud 停頓 17:46:15/23、17:46:52、17:48:10/17、17:53:16/18、17:54:23 ↔ 迷失 17:46:20/24、17:46:56、17:48:12、17:53:20、17:54:20。但 NDT 每次只花 ≈ 100 ms，不足以解釋 0.9 s 的停頓；比較可能是同一個根因（Point-LIO odometry 延遲 → small_gicp 拿到跳動的位姿而不收斂），而不是 NDT 拖慢 Point-LIO。
- 可能的內部原因：ikd-tree 的 local map 平移/刪盒（長距離直線段固定會觸發）、或 10-13 報告指出的 `sync_packages()` 無鎖讀取 buffer 造成的異常等待；兩者都沒有 per-frame 計時 log 可以確認。
- 主機同時有 chrome ≈ 190% CPU 與 viewer 模式的 quadsim ≈ 170%，controller 的 local A* 也常常超過 20 Hz 週期（377 次），CPU 競爭會放大上述停頓，但 4 次中止前 4 s 內只有 1 次有 A* 慢速警告，所以 controller 自身的負載不是直接原因。

### 4.3 其他觀察

- p01 固定被 planner 挪 0.5 m：p01 (-12.62, -15.52) 周圍在 0.3 m 機器人半徑下放不下，goal 被 snap 到旁邊；真值仍然停在 path 終點 0.2 m 內，所以不影響判定，但代表 p01 的實際停點與設定點差 0.5 m。
- 定位在跑動時誤差 0.4–0.66 m、停下後 < 0.15 m，與速度相關的延遲型誤差，和 10-13 dp_1008 的觀察一致。
- LiDAR CustomMsg 的 python 探針（best-effort）自己會掉幀，`stamp_dt` 0.4–1.5 s 的那些紀錄是探針問題（Point-LIO 輸出 stamp 連續可證），不列為發現。

## 5. 建議

1. **costmap 過期不要立刻 abort**：`map_timeout` / `elevation_map_timeout` 1.0 s 對 10 Hz 的來源太緊，且 Point-LIO 停頓只有 0.5–0.9 s。選項：(a) 放寬到 2.0 s；(b) 過期時維持上一張 costmap 並讓 controller 回傳零速度等待（`failure_tolerance` 已有 0.3 s 的機制，可以把「costmap 過期」納入而不是直接失敗）。這 5 次中止都會消失。
2. **Point-LIO 加 per-frame 處理時間 log**（`run()` 每幀 `sync_packages → 預測 → 更新 → 發布` 的耗時，> 200 ms 就 WARN），下次測試就能確認 4.2 的停頓來源，順便套用 10-13 的 `mtx_buffer` 修法。
3. 測試時關掉 chrome / 用 headless sim（`--no-viewer`），降低 CPU 競爭後再跑一次 30 分鐘，看 5 次是否降到 0。
4. p01 往空曠處挪 0.5 m（例如 (-12.1, -15.5)），讓 snap 距離 < 0.3 m。
5. 下一輪可以把 `--duration` 拉到 3600–7200 s，驗證 Point-LIO 的 1.5 h 以上穩定性（10-13 segfault 發生在 1.5 h 後）。
