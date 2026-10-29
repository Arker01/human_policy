# GPT 异步监督器（基础版）实现说明

> 给 Codex 的实现说明。先读第 11 节"需要在仓库里确认的接口"。**不要臆造接口，也不要重写已有的 Focus 和 chunk continuity 逻辑**。不确定的地方标 `TODO`，并在 PR 描述里列出。

GPT6的调用可以用codex2的来使用，如果codex2效果差也可以用codex
---

## 0. 背景与目标

现有系统：
- **大脑（HAT）**：根据图像和指令输出 keypoint chunk（全身 keypoint 序列，包括手腕和五指）。
- **小脑**：跟踪 keypoint，驱动全身。跟踪误差约 3–5 cm。
- **chunk continuity**（保留，不修改）：切换 chunk 时，在新 chunk 里找到与已执行区间对应的位置接着执行，避免手腕因为跟踪误差卡在原地。
- **Focus 机制**（已有，复用）：手腕保持不动，只让手指动。Focus handoff 时把闭合或张开下发给实体手。

**目前的问题**：什么时候触发 Focus handoff，无法只靠 keypoint 计算出来。

**本次新增**：一个异步 GPT 监督器。它看头部相机图像和跟踪数据，每次回答两件事：
1. **执行状态**（单选）。代码根据状态触发或退出 Focus。
2. **速度档位**（三选一）。代码据此调整 chunk 的播放倍率。

**不做的事情**：
- GPT 不输出 delta action，不修改轨迹，也不直接控制手腕。
- 不修改 HAT、小脑和 chunk continuity 的匹配逻辑。
- 失败时 GPT 不出动作去纠正，只等 HAT 自己重试。

**原则**：
- 控制回路**不阻塞等待 GPT**。
- GPT 只能从固定的枚举里选。合法性、新鲜度、超时都由**代码检查**。
- 每次调用都是**无状态**的。阶段由代码维护，以摘要形式传给 GPT，不保留对话历史。

---

## 1. 架构

```
 头部相机/本体 → HAT → keypoint chunk → chunk continuity → [播放倍率] → [Focus 控制(已有)] → 小脑
                                                              ▲                 ▲
                                                              │ speed_level     │ 进入/退出 Focus
                                                       ┌──────┴─────────────────┴──────┐
                                                       │  Supervisor（新增，异步线程）    │
                                                       │  快照 → prompt → GPT → 校验      │
                                                       └─────────────────────────────────┘
```

新增模块：
- `supervisor/worker.py`：异步采集快照、组装 prompt、调用 API、解析和校验，结果写入 mailbox（只保留最新一条）。
- `supervisor/phase.py`：阶段状态机，由代码维护。
- `supervisor/apply.py`：在控制回路里读取 mailbox，校验后执行（触发或退出 Focus、设置播放倍率）。
- `supervisor/logger.py`：日志。

---

## 2. 阶段（代码维护）

| 阶段 | 含义 |
|---|---|
| `PRE_GRASP` | 还没抓住物体，正在接近或者重试 |
| `FOCUS_GRASP` | 抓取 Focus 中：手腕保持，手指闭合 |
| `HOLDING` | 已抓住，正在搬运 |
| `FOCUS_RELEASE` | 松手 Focus 中：手腕保持，手指张开 |
| `POST_RELEASE` | 已松手，执行后续动作 |

初始阶段为 `PRE_GRASP`。任务包含多次抓放时，`POST_RELEASE` 之后回到 `PRE_GRASP`（是否需要这个循环见第 11 节 TODO）。

---

## 3. GPT 输出的执行状态，以及代码动作

每个阶段只接受特定的状态，其他状态一律丢弃并记录日志。

| GPT 状态 | 合法的当前阶段 | 代码动作 | 新阶段 |
|---|---|---|---|
| `normal` | 所有阶段 | 不做任何事 | 不变 |
| `ready_focus` | `PRE_GRASP` | 触发抓取 Focus handoff：手腕保持，下发闭合 | `FOCUS_GRASP` |
| `grasp_secured` | `FOCUS_GRASP` | 退出 Focus，交还给 chunk continuity | `HOLDING` |
| `grasp_failed` | `FOCUS_GRASP` | 退出 Focus，交还给 continuity，等 HAT 重试 | `PRE_GRASP` |
| `grasp_failed` | `PRE_GRASP` | 不干预，只记录日志（HAT 正在重试或即将重试） | 不变 |
| `grasp_failed` | `HOLDING` | 物体掉落：交还给 continuity，等 HAT 重试 | `PRE_GRASP` |
| `ready_release` | `HOLDING` | 触发松手 Focus handoff：手腕保持，下发张开 | `FOCUS_RELEASE` |
| `release_done` | `FOCUS_RELEASE` | 退出 Focus，交还给 continuity | `POST_RELEASE` |
| `release_failed` | `FOCUS_RELEASE` | 退出 Focus，交还给 continuity，等 HAT 重试 | `HOLDING` |

**Focus 的进入和退出必须调用仓库里已有的 Focus 接口**，不要另写一套。退出 Focus 后，continuity 按它现有的方式从当前状态继续匹配。

**Focus 超时**：在 `FOCUS_*` 阶段停留超过 `focus_timeout_s`，并且 GPT 一直没给出合法的退出状态时，按对应的 `*_failed` 处理，并记录日志。

**最短停留时间**：进入 `FOCUS_*` 后至少停留 `min_focus_time_s`，期间忽略退出状态，给手指留出执行闭合或张开的时间。

---

## 4. 速度档位

- 三个档位：`speed_levels = [1.0, 0.5, 0.25]`（可配置）。GPT 输出档位编号 `0 / 1 / 2`。
- 实现方式是**改变 chunk 的播放倍率**：每个控制周期，播放索引前进 `s` 帧（浮点索引，帧间线性插值）。全身所有 keypoint 用同一个 `s`。
- 播放倍率加在 **continuity 输出的执行索引之后**。continuity 的匹配逻辑不修改。
- 倍率切换时做平滑：每秒变化不超过 `speed_slew_per_s`。
- GPT 没有回复或回复过期时，保持上一次被采纳的档位。启动时默认用档位 0。
- Focus 期间手腕保持不动，倍率只影响 Focus 之外的执行。

---

## 5. 异步调用与新鲜度

- **调用节奏**：同一时刻只允许一个请求在途。上一次返回后，至少间隔 `min_interval_s` 再发起下一次。
- **每个快照记录**：`obs_id`、`t_obs`、`phase_at_obs`、当时的实测手腕位姿。
- **采纳决策前检查**：
  - `now − t_obs ≤ max_decision_age_s`；
  - 当前阶段等于 `phase_at_obs`，否则丢弃这个状态（速度档位仍可采纳）；
  - 对 `ready_focus` 和 `ready_release`：从拍照到现在，手腕实测位移 < `trigger_max_drift_m`。不满足就丢弃，并立即重新查询。
- **API 错误**：超时 `api_timeout_s`，失败重试 1 次。仍然失败就当作没有决策，记录日志。

---

## 6. GPT 的输入

**图像**：头部相机当前帧，长边缩放到 `image_max_side`。

**数值**（JSON 格式，放在 user message 里）：
- `task_instruction`：任务指令原文
- `phase`、`time_in_phase_s`
- `wrist_target`：HAT 当前下发的手腕目标位置（基座系，m）
- `wrist_measured`：实测手腕位置（基座系，m）
- `tracking_error_m`：目标减去实测，给出向量和模长
- `future_wrist_disp_0.5s`、`future_wrist_disp_1.0s`：当前 chunk 剩余部分里，未来 0.5 s / 1 s 手腕目标相对实测手腕的位移
- `finger_closure_cmd`：下发给实体手的五指闭合量（0 表示张开，1 表示闭合，对应项目里已有的闭合量定义）
- `finger_closure_hat_trend`：当前 chunk 中 HAT 预测的手指闭合趋势（opening / steady / closing）
- `current_speed_level`
- `recent_decisions`：最近 3 次被采纳的状态，以及当时的阶段

---

## 7. Prompt 与输出格式

### 7.1 System prompt（直接使用，可以微调措辞）

```text
You supervise a humanoid robot doing a manipulation task. A high-level model (HAT)
predicts keypoint trajectories; a low-level tracker follows them with a few cm of
error. A "continuity" mechanism keeps the hand moving along HAT's trajectory.
You do not control motion. You only report the execution state and choose a speed
level. Your answer is applied about {latency_s}s after the image was taken.

Task: "{task_instruction}"
Current phase (tracked by code): {phase}

Phases: PRE_GRASP (not holding yet), FOCUS_GRASP (wrist held, fingers closing),
HOLDING (object held, moving), FOCUS_RELEASE (wrist held, fingers opening),
POST_RELEASE.

Choose ONE state allowed for the current phase:
- normal: nothing to change.
- ready_focus (PRE_GRASP): the hand is positioned so that closing the fingers now
  would capture the object (object between fingers/palm, fingers not blocked,
  inserted deep enough). Do not choose it if the hand is merely near the object.
- grasp_secured (FOCUS_GRASP): fingers have closed and the object is held.
- grasp_failed (PRE_GRASP / FOCUS_GRASP / HOLDING): the grasp attempt clearly
  failed or the object was dropped. HAT will retry by itself.
- ready_release (HOLDING): the object is above the intended place location with
  enough clearance that opening the hand now would place it correctly.
- release_done (FOCUS_RELEASE): the object has been released and rests correctly.
- release_failed (FOCUS_RELEASE): opening the hand did not place the object
  correctly.

Rules:
- A commanded finger closure does NOT prove a grasp; use the image.
- A commanded wrist target does NOT prove arrival; compare target and measured
  positions and the image.
- If unsure, answer "normal". Uncertainty is not a reason to change state.

Also choose a speed level for roughly the next {horizon_s}s:
0 = full speed (hand far from the object or place location, free space),
1 = half speed (approaching),
2 = quarter speed (contact or insertion imminent, or HAT is retrying near the object).

Return ONLY JSON matching the schema.
```

### 7.2 输出 schema

```json
{
  "state": "normal | ready_focus | grasp_secured | grasp_failed | ready_release | release_done | release_failed",
  "speed_level": 0,
  "confidence": 0.0,
  "evidence": "short English, <= 40 words"
}
```

- 如果 API 支持，用结构化输出 / JSON mode 强制这个 schema。
- `model`、`reasoning_effort` 都放进配置。默认用较低的推理强度来控制延迟。

---

## 8. 代码校验（必须实现）

1. JSON 解析失败，或枚举值非法：整条丢弃，记录日志。
2. `state` 不在当前阶段的合法列表里（第 3 节）：丢弃 `state`，`speed_level` 照常采纳。
3. `confidence < min_confidence_state`，并且 `state ≠ normal`：丢弃 `state`。
4. 不满足第 5 节的新鲜度条件：丢弃 `state`。
5. `speed_level` 不在 0–2 范围内：丢弃。
6. 每个决策都记录 `applied` 和 `rejected: {field: reason}`。

---

## 9. 配置（`configs/gpt_supervisor.yaml`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mode` | `shadow` | `off` / `shadow`（只记录，不执行）/ `active` |
| `model` / `reasoning_effort` | `gpt-6-astra` / `low` | 可替换 |
| `api_timeout_s` | 15 | |
| `min_interval_s` | 0.5 | 两次调用之间的最小间隔 |
| `image_max_side` | 1024 | |
| `max_decision_age_s` | 5.0 | 实测后调整 |
| `trigger_max_drift_m` | 0.02 | `ready_focus` / `ready_release` 的位移容忍 |
| `min_focus_time_s` | 0.5 | |
| `focus_timeout_s` | 10.0 | |
| `min_confidence_state` | 0.6 | |
| `speed_levels` | [1.0, 0.5, 0.25] | |
| `speed_slew_per_s` | 1.5 | |
| `horizon_s` | 3.0 | 写入 prompt |

API key 从环境变量读取，不写进仓库。

---

## 10. 日志与离线评测

**日志**（`logs/supervisor/<episode>/<obs_id>/`）：
- `head.png`（原图）、`state.json`（第 6 节的全部数值）、`request.json`、`response.json`
- `decision.json`：采纳和拒绝的字段及原因、API 延迟、token usage
- 每帧记录控制侧状态：阶段、continuity 执行索引、播放倍率 `s`、Focus 是否激活

**离线工具**：
- `tools/supervisor_replay.py`：输入已录制的 episode，按固定间隔抽取快照，调用 GPT，输出 CSV。
- `tools/supervisor_eval.py`：读取人工标注的 CSV（列：`obs_id, phase, state_gt, speed_level_gt`），输出状态判断的准确率、混淆矩阵、平均延迟和平均 token。

---

## 11. 里程碑（按顺序做）

1. **M1 控制侧（不接 GPT）**：实现阶段状态机、播放倍率（不影响 continuity 匹配），把状态到"进入/退出 Focus"的映射接到已有的 Focus 接口上。用键盘手动注入状态和速度档位来验证。
2. **M2 Supervisor + 离线评测**：完成 prompt 组装、API 调用、校验和日志，在录制数据上跑 replay 和 eval。
3. **M3 真机 shadow 模式**：GPT 实时运行，但只记录不执行。对照实际结果检查判断是否正确。
4. **M4 active 模式**：先启用 `ready_focus` 和 `grasp_secured`，再启用速度档位，然后是松手相关的状态，最后启用 `*_failed`。

---

## 12. 需要在仓库里确认的接口（不确定的标 TODO）

- 已有的 **Focus 机制**：进入和退出的接口、handoff 时下发闭合/张开的方式、Focus 期间手腕如何保持。
- **chunk continuity**：在哪个文件、哪个函数，执行索引在哪里输出（播放倍率要加在这之后）。
- HAT 推理接口、chunk 的形状和帧率（目前是 20 FPS），keypoint 索引表（手腕、五指）。
- 实测手腕位姿的读取方式（坐标系、单位）。
- 五指闭合量的计算：项目里已有通过手部 IK 转换的闭合量，直接复用。
- 头部相机图像的读取方式。
- 控制回路的频率和线程模型（决定 Supervisor 用线程还是进程）。
- 任务是否包含多次抓放（决定 `POST_RELEASE` 之后是否回到 `PRE_GRASP`）。 