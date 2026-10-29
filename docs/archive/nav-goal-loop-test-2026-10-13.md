# dp_1008 導航下點迴圈測試紀錄（2026-10-13）

目的：用 quadsim（dp_1008 場景）自動連續下導航點，驗證 robot01 導航 stack 能不能把點走完、到位有多準，並把失敗分類。
本文記錄當天的測試方法、結果、找到的問題與已做的修正。程式碼層面的原因分析在最後一節。

## 1. 環境

| 元件 | 位置 | 備註 |
|---|---|---|
| 模擬器 quadsim | 本 repo，container `vibrant_bartik`，`python3 -m quadsim.sim_node --scene scenes/dp_1008 --viewer --no-gt-tf` | 整天都是同一個行程，載入的是**修正前**的場景 |
| 導航 stack | container `robot01`（`~/SyncAI-Device-Workspace`），`location_3d.launch.py sim:=true` + `nav_3d.launch.py sim:=true` + `octo_planner_rviz_node` | Point-LIO 定位、octo_planner 全域規劃（octomap）、syncai_controller_3d（rotation shim + MPPI，local 3D A*） |
| 地圖 | `Maps/dp_1008.pcd`（sim 場景與 planner octomap 同一份，frame 一致） | 兩層樓，測試只用 1F |
| DDS | CycloneDDS、domain 1、loopback | |

下 goal 的流程（與 `syncai_waypoint_sequencer` 相同）：`/initialpose` ← `/robot01/tracked_pose` → `/goal_pose` → 等 `/robot01/planned_path`（非空、終點在 goal 附近）→ action `/robot01/follow_path`。執行期間 planner 的 `auto_follow_path` 設 false，結束還原。

到位判定：action SUCCEEDED、真值 `/sim/ground_truth/odom` 距 path 終點 ≤ 0.35 m（controller 自己的容差是 0.25 m）、停住 1 s、沒有 `/sim/events` 的 collision / stuck、定位誤差 ≤ 1 m。

## 2. 測試程式

| 檔案 | 用途 |
|---|---|
| `scripts/nav_goal_loop.py`、`scripts/nav_goal_loop.sh` | 測試主程式（在 sim container 內跑） |
| `scripts/make_waypoints_from_poses.py` | 從 `Maps/<map>/poses.txt`（建圖 keyframe 軌跡）取樣 waypoint |
| `config/nav_waypoints_dp_1008.yaml` | 1F 軌跡每 10 m 取樣的 40 點（wp40 planner 規不到，已註解） |
| `config/nav_waypoints_dp_1008_manual.yaml` | 手動 teleop 到位後用定位讀出的 5 點 p01–p05，已排成最長路線 |
| `config/nav_waypoints_dp_1008_3pt.yaml` | p03 → p01 → p02 三角形 |
| `config/nav_waypoints_dp_1008_2pt.yaml` | p01 ↔ p02 來回 |
| `results/nav_goal_loop/<時間>/` | `report.json`、`legs.csv`、`run.log`（已 gitignore） |

用法與判定規則見 README「Navigation goal-loop test」。

## 3. 當天時間線與結果

| 時間 | 測試 | 結果 |
|---|---|---|
| 10:55 | dry-run 40 點（只規劃） | 38/40 規得出路；wp40 (2.1, -14.8) 從任何點都規不到；wp06 因 planner 把 goal 挪 0.84 m 被終點容差擋下（容差已放寬到 1.5 m） |
| 10:57 | 實走 wp29 → wp30 | 2/2 PASS，真值距終點 0.18–0.22 m |
| 10:59 | 整圈 39 點 | wp29–wp31 PASS 後，從 wp31 (-11.9, -6.5) 出發的每段都 10 s ABORTED、cmd_vel 全 0；local costmap 一半是 lethal |
| 11:06 | — | Point-LIO 行程 segfault（exit −11），定位消失，測試 preflight 擋下（定位誤差 7e6 m）。使用者重啟 location_3d / nav_3d |
| 11:24 | 整圈 39 點（重啟後） | wp01 PASS、wp02 走 6.6 m 後 ABORTED，之後全部 0.3–1.5 s ABORTED；local costmap **40000 格有 39957 格 lethal**，log：`机器人附近 0.75m 内找不到满足 0.25m 净空的连通起点，reachable_mask 保持不可达` / `MPPI master 保持失败安全状态` |
| 11:30 | 手記 5 點 p01–p05 | 定位與真值差 3–11 cm（p04 讀取時機器人還在動） |
| 11:36 | 5 點兩兩 dry-run（20 條） | 全部規得出路；最長單趟 p03 → p02 → p04 → p01 → p05 = 129.2 m |
| 11:37 | 5 點一圈 | p05 → p02 ABORTED（不動）；p05 → p04 SUCCEEDED 但途中 4 次 sim `stuck/no_ground`；p04 → p01 ABORTED，機器人在 (-35.5, 0.26) 被 sim 以 no_ground 擋住 |
| 11:54 | 3 點一圈（p03 → p01 → p02） | 3/3 PASS，69 m，平均 0.76 m/s |
| 11:58–12:16 | 3 點連續 | 54 段：37 PASS / 17 FAIL（ABORTED 13、COLLISION 4），581 m；連續 5 次失敗自動停（卡在 (-16.1, -6.5) 細柱旁） |
| 12:39 | p01 ↔ p02 來回 | 11 段後在 (-15.1, -10.7) 連續 6 次 0.3 s ABORTED（`start pose is unsafe: center traversability 0`），停 |
| 12:42–13:20 | p01 ↔ p02 來回（teleop 推開後） | 110 段（56 段真的移動）：**49 PASS / 7 FAIL**，1238 m，平均 0.74 m/s，真值距終點平均 0.23 m、最大 0.34 m，定位誤差到位時 ≤ 0.12 m；失敗 8 次全在 p02 → p01 快到終點時 |

## 4. 發現的問題

### 4.1 p02 → p01 每次都停在離終點 0.29 m 處，10 s 後 ABORTED（controller）

- 8 次失敗全部同一型態；反方向 p01 → p02 27/27 全過；p03 → p01 也同樣在第 42 s 失敗 3 次。
- controller log（odom frame）：`Distance to goal: 0.414 → 0.331 → 0.296 → 0.301 → 0.296 → 0.294 → 0.290 → 0.288 → 0.287 → 0.290 → 0.286 m`，每秒一行，接著 `Failed to make progress`。goal checker 容差 0.25 m，progress checker 10 s 內要前進 0.5 m。
- p01 (-4.76, 1.92) 周圍 2.2 m 內沒有任何離地 0.15–1.2 m 的障礙體素，所以不是被障礙物推開。
- 到 p01 的 goal yaw 是 −2.90（朝西），從 p02 過來的航向約 +70°，到點需要轉約 120°；到 p02 的 goal yaw −0.72，與來向接近。方向相依與此一致。
- 成功到達 p01 的 21 次，真值停點平均 (-4.55, 2.20)，離 path 終點 (-4.5, 1.9) 平均 0.31 m（測試容差 0.35 m 勉強過）；到 p02 的 28 次平均只差 0.17 m。也就是說到 p01 **每次都停在終點北邊約 0.3 m**，只是有時 controller 剛好算進 0.25 m。
- 全域路徑本身（用 initialpose/goal_pose 離線規劃，`auto_follow_path` 關掉）：p02 → p01 共 121 個點、25.4 m，**最後一段是從 (-10.0, -1.44) 直接跳到終點 (-4.5, 1.9) 的 6.43 m 直線，中間沒有任何點**；第一段同樣是 6.4 m 的跳躍。路徑只在圓角處每 3–5 cm 取樣，直線段只有頭尾。反方向 p01 → p02 的最後一段是 1.53 m。controller 的 local A* lookahead、MPPI 的 path-follow / path-align 都是拿路徑點當參考，最後一段太稀疏是方向相依的一個合理解釋（見第 5 節）。
  程式碼佐證：`octo_planner_3d/planner/src/global_planner.cpp` 的平滑輸出（約 1098–1190 行）只對每個轉角產生一段貝茲曲線取樣（`samples = max(6, 2·trim / sample_spacing)`），直線段只 `append` 端點；`smoothing_sample_spacing_m: 0.05` 是曲線的碰撞取樣間距，不是輸出路徑的間距。所以路徑的最後一段長度 = 最後一個轉角的出口到 goal 的直線距離，p02 → p01 剛好是 6.4 m。

### 4.2 中止後起點被判「不安全」，之後每個 goal 0.3 s 就 ABORTED（controller / local costmap）

- log：`Yaw-aware A* start pose is unsafe: footprint overlaps planning_obstacle_mask=1`、`start pose is unsafe: center traversability 0 is below minimum 0.2`、`Local 3D A* exhausted all in-map lookahead goal candidates`、`Controller patience exceeded`。
- 發生位置集中在 x −16～−14、y −12～−6（(-16.45, -11.75)、(-15.13, -10.72)），這片在修正前的 sim 場景有多群地面破洞（見 4.4），LiDAR 掃到地板缺口後 3D costmap 判成不可通行。
- 沒有 recovery：換個方向下 goal 有時可脫困（loop 9 往 p01 成功），否則要 teleop 推開。

### 4.3 p02 → p03 路線上的細柱陷阱（規劃 / 控制）

- (-16.2, -6.2～-6.8) 有一根寬 10 cm、離地 0.15–1 m 的細長物（柱子或椅腳），周圍其他地方都是空的。
- 全域規劃（機器人半徑 0.3 m）貼著它走，controller 跟過去後機身壓到它：sim 回報 `collision/contact`，之後每段一起步就 `collision/collision`（機身再往前會重疊地圖），controller 不允許倒車，連帶失敗 3–4 段。
- 3 點連續測試裡 p02 → p03 14 段失敗 6 段，大多在這裡。

### 4.4 sim 場景地面有洞（本 repo，已修）

- 現象：機器人在 (-35.1～-35.5, 0.2～0.5) 被 sim 以 `stuck/no_ground` 擋住；離線從 2 m 高往下打射線，腳下取樣點有多個完全打不到幾何。
- 原因：Mid-360 看不到機器人腳邊地板，SLAM 地圖的地板有整欄沒點的格子；`pcd2sim` 的壓平只補「8 個鄰居有 ≥ 5 個是地板」的洞、3 輪，較大的缺口留下來。沿 1F 建圖軌跡每 0.1 m 掃描：3449 格（6.8%）旁邊有地板卻沒有幾何；走路取樣窗 1.09% 無地面。
- 修正：
  - `pcd2sim/ground.py` 補洞預設改為 `fill=12, fill_nb=3`（`build_scene --ground-fill / --ground-fill-nb / --ground-max-above` 可調），`scenes/dp_1008` 已重建：沿路走路取樣無地面 0.37% → 0.17%。
  - `quadsim/base.py`：允許一隻腳踩在縫隙上（`robot.max_feet_over_gap: 1`，該腳高度取其他三腳的平面）。離線沿 1F 軌跡 702 個 keyframe 跑一遍：`stuck/no_ground` 89 → 44 次、卡住 ≥ 15 s 19 → 14 次，剩下的集中在 (17.5, 7.5)、(23.1, 49.3)、(18.6, 47.3)、(-12.8, 6.2) 幾處較大的缺口。
  - **現在跑的 sim 還沒重啟，上述修正尚未生效。**

### 4.5 其他

- Point-LIO 在 11:06 segfault（launch log：`pointlio_mapping process has died, exit code -11`），約在開始測試後 1.5 h；原因未查。
- wp40 (2.1, -14.8)（1F 往 2F 的斜坡口附近）全域規劃從任何點都規不到。
- 測試期間 robot01 內的 `teleop_twist_keyboard` 也會發 `/robot01/cmd_vel`，跑測試時鍵盤不能按。

## 5. 程式碼層面的可能原因

路徑都在 `~/SyncAI-Device-Workspace/src/`。先講 controller 每個週期的流程（`syncai_controller_3d/src/controller_server.cpp`）：

1. `updateGlobalPath()`：把全域路徑重新取樣成 0.1 m（499–533），`end_pose_` = 全域路徑最後一點（537）。
2. `computeAndPublishVelocity()`：先跑 progress checker（565–566，失敗就丟 `Failed to make progress`）→ 把路徑裁到 4 m（635）→ 依 lookahead `{3.0, 4.0, 3.5, 2.5, 2.0, 1.5}` 在全域路徑上挑候選點（706–707；goal 比 lookahead 近時候選就是全域終點，1204–1223）→ 對每個候選跑 yaw-aware local 3D A*（711–755），全失敗就丟 `Local 3D A* failed to find a valid path!`（761–783）→ 把 **local A* 的路徑** `setPlan` 給 rotation shim / MPPI（829）。
3. `isGoalReached()`（935–1004）：拿 **全域路徑終點** `end_pose_` 跟機器人比，`SimpleGoalChecker` 只看 xy ≤ 0.25 m（stateful）。

也就是 MPPI 追的是 local A* 的終點，到位判定看的是全域終點，兩者不一定是同一點。

### 5.1 停在離終點 0.29 m（問題 4.1）

| 排序 | 假設 | 程式碼 | 怎麼確認 / 快速試 |
|---|---|---|---|
| 1 | **local A* 把 goal 挪到旁邊的格子**：goal 格的 footprint 放不下（`isPoseSafe` 失敗：中心不可通行、`planning_obstacle_mask`、側向支撐不足）時，`a_star_3d.cpp:1515–1553` 以環狀掃描找第一個安全格，**不是最近的**；radius 2 時第一個候選是對角 (−2, −2)，距離 √(0.2²+0.2²) = **0.283 m**，跟觀察到的 0.286–0.296 m 幾乎一樣。MPPI 走到挪過的點就停，但判定是對原本的終點 → 永遠差 0.29 m，10 s 後 progress checker 中止。p02 在空曠處，goal 格永遠安全，所以反方向不會發生。 | `a_star_3d.cpp:364–627`（isPoseSafe）、`1515–1553`（goal 放寬）、`controller_server.cpp:935–1004` | 跑 p02 → p01 時 echo `/robot01/local_plan_3d` 最後一點，應該固定離 goal 約 0.28 m；看 `planning_obstacle_mask_2d` / `elevation_map` 在 p01 周圍。快速試：`xy_goal_tolerance` 0.25 → 0.35，或把 p01 往空曠處挪 0.3 m。程式修法：放寬時挑最近的安全格，或「到 local 終點且離全域 goal < N cm」也算到位。 |
| 2 | **rotation shim 在終點原地打轉**：`setPlan` 每個週期都被呼叫（829），`path_updated_` 一直是 true（`rotation_shim_controller.cpp:343–349`），航向檢查每週期都跑；機器人已在 local 路徑終點時目標向量趨近 (0, 0)，`atan2` 給出任意角度，超過 `angular_dist_threshold` 1.57 就以 0.6 rad/s 原地旋轉，位置不動 → progress 失敗。 | `rotation_shim_controller.cpp:157–216` | 卡住時 cmd_vel 應是 linear ≈ 0、angular ≈ 0.6。快速試：`angular_dist_threshold` 設 3.2 等於關掉 shim。 |
| 3 | **MPPI 障礙成本與 goal 成本的平衡點**：`ObstaclesCritic` 只檢查機身中心對 inscribed radius 0.235 m + `collision_margin_distance` 0.1 的圓，權重 16；goal critic 權重 5。goal 在 lethal / unknown 格 0.33 m 內時會停在外面。unknown（NaN 高程）在 MPPI 的 master grid 是 LETHAL（`costmap_3d_layer.cpp:2402` 先全填 lethal 再挖 free）。 | `obstacles_critic.cpp:139–237` | 快速試：`collision_margin_distance` 0.0 或 `critical_weight` 4，看停點距離變不變。 |

另外：`progress_checker` 只在新 action 開始時 reset（422），沒有「已在 goal 容差內」的豁免；`GoalAngleCritic` 在 yaml 裡是 `enabled: false`，所以 goal yaw 不是由 MPPI 驅動。全域路徑直線段不取樣（4.1 節）本身在 controller 端會被重新取樣成 0.1 m，不直接造成停短，但會影響 local A* 候選點與 MPPI 的參考點選取。

### 5.2 中止後起點「不安全」、0.3 s 內 ABORTED（問題 4.2）

- 起點檢查沒有任何退路：`a_star_3d.cpp:1482–1513` 用量化到 22.5° 的 yaw 檢查 footprint（角落誤差約 0.1 m），`isPoseSafe` 失敗直接 `return false`；6 個 lookahead 候選各重跑一次同樣的檢查 → `exhausted all in-map lookahead goal candidates` → 不呼叫 MPPI → `failure_tolerance` 0.3 s 一到就 `Controller patience exceeded`。
- **門檻不一致**：A* 的 `min_traversability_` 寫死 0.2（`a_star_3d.hpp:138`，沒有參數），layer 的 `min_traversability` 是 0.1，layer 把 footprint 下清掉的格子「抬到」0.1（`costmap_3d_layer.cpp:1693–1701`）對 A* 仍然不夠。
- **沒有地板就不安全**：layer 的高程取 static 點雲的最高 z（708–717），static 沒有的格子 realtime 點補不了（1255–1263）；中心格 NaN → `non-finite elevation`；4 鄰居有 NaN → 梯度算不出來，traversability 留在 0（1582–1622）→ 就是 log 的 `center traversability 0 is below minimum 0.2`；footprint 下 NaN 太多 → 側向支撐不足；機器人所在格 NaN → 整張 planning mask 變 1、costmap not current。這正是 4.4 地面破洞區域會發生的事。
- 沒有倒退：`vx_min: 0.0`（yaml:95）→ `Optimizer::applyControlSequenceConstraints` 把 vx 夾在 [0, vx_max]（`optimizer.cpp:239`），`PathAngleCritic` 因此 `reversing_allowed_ = false`；A* 只有前進與原地轉（`a_star_3d.cpp:1445–1448`）；shim 的原地旋轉遇到 LETHAL footprint 會丟例外（`rotation_shim_controller.cpp:337–339`）。
- 快速試：`a_star_yaw_bins` 16 → 32（減少量化誤差）；把 `min_traversability_` 改成參數並設 0.1；`failure_tolerance` 拉長只是多等。根本解法要有「從不安全起點脫困」的行為（倒退 primitive 或允許 vx_min < 0）。

### 5.3 細柱陷阱（問題 4.3）

- **柱子可能根本不在任何 layer 裡**：crop（1276–1289）把比 3×3 鄰居高 0.4 m 以上的格子周圍設 NaN 再補成地板高，接著 desalt（1322–1343）`similar_cnt <= 1 && ground_cnt >= 4` 就把該格換成地面平均 → 1 格寬（或 2 格跨線）的柱子會從 `filtered_elevation` 消失，traversability、obstacle mask、MPPI grid 都看不到它。
- **MPPI 碰撞模型是圓形中心點**：`consider_footprint: false`（yaml:135），只看機身中心對 inscribed 0.235 m + margin 0.1 的距離；機身角落實際伸到 0.49 m，轉彎或切彎時角落先碰到；repulsion 權重 0.7 在 3 m inflation 上幾乎是平的，拉不開。
- **全域規劃半徑 0.30 m < local footprint 半長 0.43 m**：`maps_config.yaml:21 robot_radius_m: 0.30`，路徑貼柱子 0.3–0.4 m 走，留給 controller 的橫向餘裕只有 0.07–0.17 m。
- 碰到以後：不能倒退（同 5.2），原地轉被 A* 起點檢查與 shim 的碰撞檢查擋住 → 卡死，只能靠 teleop。
- 快速試：`consider_footprint: true`、`repulsion_weight` 3–5、`collision_margin_distance` 0.2；全域 `robot_radius_m` ≥ 0.45；`vx_min: -0.3` 讓 MPPI 能倒退。確認柱子有沒有被 desalt 掉：比對 `elevation_map` 的 `elevation` 與 `filtered_elevation` 在 (−16.2, −6.5) 的值。

### 5.4 長時間測試的時間估算（p01 ↔ p02 跑 3000 次）

用 12:42–13:20 那一輪的實測：p01 → p02 單趟約 29 s、p02 → p01 約 35 s（失敗時 45–51 s 才中止），每段額外開銷約 3 s（規劃、停穩 1 s、dwell 1 s），平均每段約 38–41 s。

| 「3000 次」的定義 | 段數 | 預估 |
|---|---|---|
| 3000 段單趟（每個方向各算一次） | 3000 | 約 32–34 小時 |
| 3000 趟來回 | 6000 | 約 64–68 小時 |

sim 是 rtf 1.0，controller 已經跟不上 20 Hz，不能靠加速模擬縮短。以今天的狀態（p02 → p01 失敗率 7/28、兩次卡死需人工 teleop、Point-LIO 曾在 1.5 h 後 segfault）無法無人值守跑這麼久；要跑之前至少要：重啟 sim 載入補洞場景、處理 5.1 的 0.29 m 問題（最快是 `xy_goal_tolerance` 0.25 → 0.35 先試）、分段跑（`--duration 7200` 一段，`--max-consecutive-fail` 停下讓人處理）。指令（3000 段、不產生 0 m 空段）：

```bash
scripts/nav_goal_loop.sh --waypoints config/nav_waypoints_dp_1008_2pt.yaml --loops 1500 --no-start-nearest
```

### 5.5 建議的下一步

1. **重啟 sim** 載入補洞後的 `scenes/dp_1008` + `max_feet_over_gap: 1`，重跑 5 點與 3 點，看 4.2、4.4 類失敗是否消失。
2. **驗證 5.1 的假設**：跑一次 p02 → p01，同時記錄 `/robot01/local_plan_3d` 最後一點、`/robot01/cmd_vel`、機器人 yaw；停住時若 local 終點離 goal 0.28 m 且 cmd_vel ≈ 0 → 假設 1；若 angular ≈ 0.6 → 假設 2。
3. controller 端的修正順序（都在 robot01 workspace）：goal 放寬改挑最近安全格 / 到位判定加 local 終點條件 → `min_traversability_` 參數化 → 起點不安全時的脫困（倒退）→ MPPI `consider_footprint` 與全域半徑。
4. 把 `nav_goal_loop.py` 的報表加上「卡住時的 cmd_vel / local path 終點」欄位，之後這類問題不用再手動抓 log。

## 6. 補記 2026-10-14：dp_1012 三點測試為何全部 GT_NOT_AT_GOAL

- 環境：sim 跑 `scenes/dp_1012`（與 robot01 的 `maps/dp_1012.pcd` 同一個檔，md5 相同，建場景沒有平移/旋轉）；planner `maps_config.yaml current_map: dp_1012`。
- 16:08 實走 `config/nav_waypoints_dp_1012_3pt.yaml`（`results/nav_goal_loop/20261014-160857/`）：action 全部 SUCCEEDED、無 collision / stuck，但真值距 path 終點 p03 0.30 m、p01 1.44 m、p02 1.83 m，定位誤差同量級，全部 `GT_NOT_AT_GOAL`。dry-run matrix 6/6 規得出路。
- 原因：robot01 的 `global_localization.launch` 與 `map_server_3d.launch` 從 `config/system.ini` 讀到的是 **dp_1008.pcd**（15:57、16:19、16:22 三次 relaunch 的 launch.log 都寫 dp_1008）。主機的 `config/instances/robot01.ini` 16:19 已改成 dp_1012，但它是以「單一檔案」bind mount 進容器的 `config/system.ini`，編輯器存檔換了 inode，容器仍看到舊檔（容器內 `stat` link count 0）。定位在 dp_1008 地圖上，原點就偏 0.45 m、yaw 偏約 3.7°，28 m 外放大到 1.8 m，與觀察完全一致。
- 另外 ini 與 `maps_config.yaml` 寫的 `maps/dp_1012/dp_1012.pcd` 原本不存在（pcd 在 `maps/` 上一層），已建 symlink `maps/dp_1012/dp_1012.pcd -> ../dp_1012.pcd`。
- 處理：relaunch 時直接指定 ini，繞過壞掉的 mount（在 robot01 容器、`~/robot_ws` 下）：
  ```bash
  ros2 launch syncai_launch_ws location_3d.launch.py sim:=true system_config:=config/instances/robot01.ini
  ros2 launch syncai_launch_ws nav_3d.launch.py      sim:=true system_config:=config/instances/robot01.ini
  ```
  launch.log 應印 `PCD map from 'config/instances/robot01.ini': .../maps/dp_1012/dp_1012.pcd`。之後原點處 `/robot01/tracked_pose` 與真值應差 < 0.1 m，再跑 dry-run matrix 與 `--loops 1`。根本解法是重啟容器或改成 mount 整個 `config/instances/` 目錄。
