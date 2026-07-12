# 单人数据采集操作指南（4 个 gesture + null）

适用范围：`ios-app/RingCollector`（数据采集用，不含分类器）。目标手势：
`double_side_tap`、`double_pinch`、`pinch_hold`、`double_flick`；`null` 作为背景/负样本单独用
Null Recording 采集。

## 0. 采集前检查（每位受试者开始前都做一次）

1. 戒指电量充足，MCU 固件确认烧录的是 **data-collection 构建**
   （`overlay-none.conf` / `CONFIG_CLASSIFIER_NONE`，不是 mlc/cnn/hdc）。
2. 打开 App → 右上角齿轮 Settings：
   - **Participant ID**：填好且全场次唯一（如 `P01`）。
   - **Reps per block**：建议 **20**（见下方理由）。
   - **Gesture set version**：如果这次的指令/流程和之前批次不同，记得改版本号
     （比如 `v1` → `v2`），避免训练数据把不同协议混在一起。
   - **IMU config**：保持默认 `120hz_8g_2000dps`，需与固件实际构建一致。
   - **Re-don ring prompt between blocks**：建议保持打开——这样 4 个手势 block 之间会
     提示"重新佩戴戒指"，能采到真实的重复佩戴引入的变异性（这是论文里应该体现的噪声源）。
   - Gestures 列表：确认 4 个手势全部勾选。
3. 回到主页，确认蓝牙已连接（绿点 + "Connected"）。**新版本 UI 变化**：如果戒指没连接，
   "Guided Gestures" / "Null Recording" 按钮会直接禁用并提示"Connect the ring before
   starting a session"——这是本次修改加的保护，之前可能会在没连接的情况下误开始一个空跑
   的采集流程。

## 1. Guided（4 个手势）采集流程

进入 Guided Gestures 后，App 会把 4 个手势顺序 **随机打乱**（每次 session 都不同顺序，
减少顺序效应）。每个手势是一个 "block"：

1. **Block Intro 页**：显示手势名字 + 说明文字。
   - 新增保护：按钮位置现在会先显示 "Waiting for ring data…"（转圈），直到确认 IMU
     数据真的在流入（不是"BLE 已连接"就行，而是"真的在收样本"），才会变成可点的
     **Start Block**。正常情况这个等待 <1 秒；如果卡住超过几秒，检查戒指是否休眠/断连。
   - 点 Start Block 才真正开始这个手势的所有 reps。
2. 每个 rep 的节奏（全自动，不用手动点）：
   - 倒计时 3-2-1（震动提示）
   - **GO**（绿色闪一下 + 声音/震动）——受试者此刻开始做手势
   - **Perform** 2.5 秒 一个绿色进度条（这段时间做完手势动作，动作本身应该在这个窗口内完成）
   - **Rest** 1.5 秒（放松，准备下一个 rep）
   - 循环到设定的 rep 数
3. 采集过程中屏幕会一直显示"当前手势名 + Block X/4 · Rep Y/Z"（本次新加的常驻提示，
   之前只有 Block Intro 页会显示手势名，倒计时/perform/rest 阶段看不到，容易在长 block
   里忘记自己在做哪个手势——现在全程可见）。
4. **Redo Last**：如果受试者或实验员觉得刚才那个 rep 做错了/做变形了，点这个按钮。
   - 会记一条 `redo` marker（后处理时可以用来排除/标记那次 rep）。
   - **行为变化（本次修复）**：以前点了这个按钮只是记录一下，block 的总 rep 数不会变，
     等于那次坏数据就白丢了。现在点一次会给当前 block **多补一个 rep**，凑够干净的
     目标数量。屏幕上的 "Rep Y/Z" 分母会跟着变大，属于预期行为。
   - 用法建议：一旦发现刚才做错了，**立刻**在下一个 rep 的倒计时/perform 阶段之前点，
     不要等到 Block Complete 页面再点（那时候已经跳过 Redo Last 按钮了）。
5. 一个 block 做完后进入 **Re-don Ring** 提示：调整戒指佩戴位置/松紧，点 "Ring
   Re-donned" 继续下一个手势 block（或 "Skip" 跳过提示）。
6. 4 个 block 全部做完 → Session Complete → 点 Done 返回主页。

### Reps 数量建议

- 每 block **20 reps**、4 个手势 = 80 条 "go" cue，实际有效动作窗口按 128
  samples@120Hz(~1.07s) 分割，够训练/测试划分用。
- 如果时间允许，建议同一受试者做 **2 次独立的 Guided session**（比如上下午各一次，
  之间摘下戒指再重新佩戴），而不是只依赖 block 间的 re-don 提示——能采到更真实的
  "重新戴上戒指位置漂移"的数据分布，这对论文里对比三种范式在真实使用条件下的鲁棒性很重要。

## 2. Null（背景）采集流程

1. 主页点 "Null Recording"，会立刻开始记录（无需倒计时，自由活动）。
2. 建议采集 **5–10 分钟**，覆盖几种自然状态而不是一直静止：
   - 正常打字/刷手机
   - 走路
   - 喝水/拿东西
   - 静止放松
   这样负样本能覆盖足够多样的"非手势"运动模式，避免分类器只学会区分"完全静止 vs 手势"。
3. 屏幕上会显示计时和 "Screen lock OK · streams in background"——可以锁屏，BLE 会在后台
   继续收，不需要一直亮屏盯着。
4. 结束点 "Stop & Save"（不是 Discard，Discard 会整段丢弃）。

## 3. 采集过程中要盯的数据质量指标（DataQualityPanel，顶部常驻）

- **Hz**：应稳定在 ~120。如果长时间明显偏低，说明丢包严重。
- **Dropped**：正常应该接近 0%。超过 0.5% 会变红——如果持续变红，暂停检查蓝牙距离/干扰。
- **HW Timestamps**：正常应接近 100%（真实硬件时间戳）。如果长期显著低于 100%，说明
  固件在用插值/兜底时间戳（FIFO overrun 或 I2C 抖动),值得在 markers/meta 里留意，
  必要时重新采这段。
- **红色 "IMU stream stalled — check ring" 横幅**（本次新加）：只要曾经收到过数据、
  但连续 1 秒没有新样本进来，就会出现。出现时说明这段时间戒指实际上断流了——即使 App
  还显示 "Connected"，也不代表数据在流。看到这个立刻检查戒指/蓝牙，必要时该 session
  作废重录。

## 4. 采集结束后

1. 回主页 → Sessions，能看到这个受试者所有 session（Guided ×N + Null ×1），每条显示
   sample 数、drop%、HW timestamp %。
2. 勾选该受试者本次所有 session → Export → 通过分享面板导出 zip（AirDrop / 存到
   电脑），**不要只放在手机本地** —— 存储空间是共享给所有受试者的，攒太多没导出容易
   丢数据或占满空间。
3. 每个 session 文件夹包含：`imu.csv`（含 unwrapped_sample_id/timestamp_us/
   timestamp_ticks/timestamp_flags/ax..gz）、`markers.csv`（go/redo/block_start/
   block_end/redon/disconnect 等事件，含触发时刻的 sample id）、`meta.json`（质量汇总）。

## 5. 单个受试者时间预算参考

| 环节 | 预计时长 |
|---|---|
| 设置 + 佩戴 + 蓝牙连接 | 3–5 分钟 |
| Guided session（4 手势 × 20 reps，含 re-don 间隔） | 约 15–20 分钟 |
| （可选）第二轮 Guided session（重新佩戴） | 约 15–20 分钟 |
| Null recording | 5–10 分钟 |
| 导出 + 检查质量指标 | 2–3 分钟 |
| **合计（含第二轮）** | **约 40–55 分钟** |
