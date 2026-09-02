# NERO 双臂策略控制系统

本仓库是 NERO 双臂任务的**控制代码根目录**。它负责将远程 OpenPI/π0.5 模型输出的 action chunk，以固定 30 Hz 的时钟安全地转换为双臂 NERO 的 CPV 关节位置命令。

它不包含 PICO 遥操程序、相机录制程序、数采数据或模型 checkpoint。

```text
三路相机 + 双臂 CAN 反馈 + 任务文本
                 │
                 ▼
          OpenPI 策略服务器
          输出 24 x 16 action chunk
                 │
                 ▼
       RTC 计划队列与异步推理调度
                 │
                 ▼
 OSQP 轨迹点平滑 -> CasADi 相位重定时 -> q/v/a 连续交接
                 │
                 ▼
 30 Hz streaming follower -> CPV backend -> CAN -> NERO 双臂
```

## 1. 三个本机目录的边界

| 目录 | 负责什么 | 不应该放什么 |
|---|---|---|
| `/home/dev/nero_bimanual_control` | 策略推理请求、轨迹处理、CAN/CPV 执行和诊断 | PICO 工程、视频、数采数据、checkpoint |
| `/home/dev/nero_neo_teleop` | PICO 输入、双臂回零、CAN 自动绑定、数采与相机录制 | VLA 运行时和轨迹优化器 |
| `/home/dev/nero_data` | 原始数据、筛选数据、转换数据、质量报告 | 可执行控制代码和模型 checkpoint |

模型、训练集副本、checkpoint 与训练日志只在训练服务器维护。官方 `pyAgxArm` SDK 目前仍为外部依赖，由 `NERO_ARM_SDK_ROOT` 指定。

## 2. 从哪里启动

先在本机创建路径配置：

```bash
cd /home/dev/nero_bimanual_control
cp config/paths.env.example config/paths.env
```

当前默认实机链路是 OSQP + CasADi + RTC：

```bash
# 只验证 CAN、相机、策略服务和运行时，不发送机械臂命令
./scripts/run_policy_osqp_casadi.sh --preflight-only

# 通过预检后才允许实机执行
./scripts/run_policy_osqp_casadi.sh --execute
```

`--execute` 才会使能并发送 CPV 命令。任何预检失败都不应绕过。

## 3. 当前实机调用路径

### OSQP + CasADi 正式主路径

```text
scripts/run_policy_osqp_casadi.sh
  └─ trajectory/osqp_waypoint_smoother/runtime/launch_policy.sh
       └─ trajectory/osqp_waypoint_smoother/runtime/run_policy.sh
            ├─ nero_neo_teleop/scripts/can/ensure_can_interface.sh
            ├─ scripts/bimanual_policy/ensure_bimanual_policy_server.sh
            └─ policy_runtime.py
                 └─ 进程内加载 bimanual_guarded_policy_stream.py
                    并替换 RTC queue 与 follower
```

这样做的目的，是复用已经验证的策略输入、相机与 CPV 运行器，同时在正式进程内接入 OSQP、CasADi、TOPPRA 轨迹处理逻辑。

### 每个 30 Hz tick 的数据流

1. `bimanual_guarded_policy_stream.py` 从三路相机读取最新帧，从 CAN 读取两臂关节与夹爪反馈。
2. `policy_client.py` 将 observation 和任务文本发送到 OpenPI 服务，异步获得 `24 x 16` action chunk。
3. `osqp_casadi_rtc_queue.py` 管理当前计划、待处理策略 chunk、RTC 请求和接管时刻。
4. `smoother.py` 仅平滑 14 个机械臂关节轨迹点；两个夹爪 action 保持原始事件时序。
5. `optimizer.py` 不改关节空间路径，只在固定 horizon 内调整相位推进，使速度、加速度、jerk 更可行。
6. `receding_toppra_queue.py` 根据当前命令/反馈状态处理 q、v、a 连续交接和安全余量。
7. `trajectory_executor.py` 的 follower 以 30 Hz 施加速度、加速度、jerk 与 command-feedback governor。
8. `cpv_backend.py` 检查连续性后逐关节发送 CPV position 命令。

全局时间属于 RTC。OSQP 与 CasADi 可以调整 chunk 内形状或相位，但不能把 24-step horizon 延长或缩短。

## 4. 目录与代码职责

### 顶层入口

| 文件 | 用途 |
|---|---|
| `scripts/run_policy_osqp_casadi.sh` | 当前默认入口；读取本机路径配置后启动 OSQP + CasADi 实机控制。 |
| `config/paths.env.example` | 本机目录、SDK、策略服务地址模板。复制后形成不入 Git 的 `config/paths.env`。 |
| `docs/CURRENT_CONTROL_STACK.md` | 当前控制参数、运行规则与验证命令。 |

### 双臂策略运行器基础层：`scripts/bimanual_policy/`

| 文件 | 用途 |
|---|---|
| `bimanual_guarded_policy_stream.py` | 实机主程序：相机、CAN、策略请求、夹爪、双 follower、CPV 输出、日志和安全检查。 |
| `ensure_bimanual_policy_server.sh` | 检查远程 OpenPI 服务；按需要同步并启动指定 checkpoint 的策略服务。 |
| `run_bimanual_policy_trial.sh` | 被正式 OSQP + CasADi 启动器调用的基础编排器：CAN 预检、策略预热、可选回零、执行与失败后的 hold。 |
| `bimanual_policy_dry_run.py` | 读取真实 CAN/相机并请求策略，但完全不连接机器人命令 API。 |
| `run_bimanual_policy_dry_run.sh` | dry-run 的 shell 包装器。 |

### 核心控制库：`nero_vla/`

| 文件 | 用途 |
|---|---|
| `bimanual_chunk_executor.py` | 双臂共享 phase 的 action chunk 表示、对齐、固定时域和 RTC 队列基础逻辑。 |
| `trajectory_executor.py` | 与硬件无关的关节 follower：轨迹采样、限速、限加速度、限 jerk、feedback governor。 |
| `cpv_backend.py` | NERO CPV position 流式写入适配器，拒绝不连续关节命令。 |
| `dual_can.py` | CAN 角色识别、健康流量检查、双 CAN bridge 保护与接口验证。 |
| `camera_reader.py` | 带时间戳的 V4L2 最新帧读取器，检测图像过期。 |
| `policy_client.py` | OpenPI WebSocket/msgpack 客户端与端口可达性检查。 |
| `gripper_controller.py` | 夹爪归一化位置、限速与反馈闭环控制。 |
| `robot_config.py` | NERO 关节限制、CPV 模式限制和 Home 配置常量。 |
| `guarded_policy_stream.py` | 较早的通用受保护策略流；仍提供健康检查与等待 CPV/使能状态等共享函数。 |
| `lift_assist.py` | 可选的抓取后抬升、预抓取下探和高度保护辅助逻辑；默认主路径不应随意启用。 |
| `image_tools.py` | 相机图像缩放、填充、旋转等 observation 预处理。 |
| `eth_state.py` | NERO ETH 状态读取工具。 |
| `state_monitor.py` | 单臂状态监控、固件识别和 SDK 机器人对象创建。 |
| `gripper_calibration.py` | 夹爪宽度校准工具。 |
| `observation_pipeline.py` | 合成 observation 与状态管线检查。 |
| `policy_probe.py` | 非实机策略连接和 action 输出探针。 |
| `policy_shadow.py` | 只观测、不执行的策略 shadow 工具。 |
| `real_policy_dry_run.py` | 真实相机/状态输入上的单臂无执行策略检查。 |
| `terminal_target_filter.py` | 对策略终端关节目标进行保护过滤。 |
| `trajectory_dry_run.py` | 不接硬件的 follower/trajectory 回放工具。 |
| `guarded_policy_run.py`、`guarded_policy_step.py` | 较早的单步/单臂受保护执行工具，保留用于回归与参考，不是双臂主入口。 |
| `leader_follower_bridge.py` | CAN leader-follower bridge 服务端实现；双臂策略运行前会检查它没有错误转发。 |
| `lerobot_recorder.py` | 旧版库级录制组件；当前正式录制入口在 `nero_neo_teleop`。 |

### 轨迹处理：`trajectory/`

| 路径 | 用途 |
|---|---|
| `osqp_waypoint_smoother/waypoint_smoother/smoother.py` | OSQP 凸优化：在 trust region 内平滑策略关节轨迹点，约束速度、加速度、jerk。 |
| `osqp_waypoint_smoother/runtime/osqp_casadi_rtc_queue.py` | 当前主运行时 queue：串接 OSQP、CasADi、TOPPRA 交接，并处理恢复候选。 |
| `osqp_waypoint_smoother/runtime/policy_runtime.py` | 运行时注入适配器：动态加载原生策略流，并替换 queue/follower 类。 |
| `osqp_waypoint_smoother/runtime/run_policy.sh` | 实机编排器：CAN 准备、策略服务、可选回零和 Python 环境。 |
| `osqp_waypoint_smoother/runtime/launch_policy.sh` | 当前模型与 OSQP/CasADi 参数 preset；通过环境变量暴露调参面。 |
| `casadi_fixed_horizon_retimer/fixed_phase_optimizer/optimizer.py` | 固定总时长的相位优化器，支持速度、加速度、jerk 约束和 warm-start。 |
| `casadi_fixed_horizon_retimer/runtime/casadi_rtc_queue.py` | 将 CasADi 相位优化接入滚动 RTC queue 的适配层。 |
| `toppra_fixed_horizon_retimer/fixed_path_retimer/` | 路径表示、TOPPRA 重定时与滚动计划数据结构。 |
| `toppra_fixed_horizon_retimer/runtime/receding_toppra_queue.py` | q/v/a 连续 handoff、reserve、commit、候选拒绝与重新请求策略。 |
| `toppra_fixed_horizon_retimer/runtime/follower_state_bridge.py` | 在 queue 与 follower 之间共享最后发送命令、速度、加速度和期望速度。 |

每个轨迹目录的 `tests/` 是该层的单元测试；`scripts/` 为离线回放、绘图和 action chunk 诊断；`legacy_runtime/` 为历史实机启动器，禁止直接用于当前机器人。

### 诊断与服务器部署

| 路径 | 用途 |
|---|---|
| `scripts/diagnostics/bimanual/bimanual_policy_rtc_probe.py` | 无机器人命令的 RTC 策略延迟和 chunk overlap 探针。 |
| `scripts/diagnostics/bimanual/bimanual_policy_rtc_stream_dry_run.py` | 模拟策略流与 RTC 消费的 dry-run。 |
| `scripts/diagnostics/bimanual/bimanual_policy_offline_regression.py` | 已保存 chunk 的离线回归检查。 |
| `scripts/diagnostics/bimanual/analyze_action_execution_match.py` | 对比策略 action、最终 command 与实测执行状态。 |
| `scripts/diagnostics/run_policy_native.sh` | 不注入轨迹优化器的故障隔离入口；仅用于定位策略、轨迹层或硬件层问题，不是日常实机运行方式。 |
| `server_staging/` | 训练服务器部署的转换、验证、归一化和 NERO OpenPI 配置镜像，不存放模型。 |
| `archive/legacy_policy_presets/` | 旧 checkpoint 专用启动器，仅用于复现实验。 |

## 5. 当前可调参数

推荐只通过环境变量调当前 OSQP + CasADi 正式运行时，而不要直接改 Python 默认值：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `NERO_POLICY_DURATION` | `30` | 试验最大时长，单位秒。 |
| `NERO_FOLLOWER_MAX_VELOCITY_DEG_S` | `28` | follower 关节速度上限。 |
| `NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2` | `280` | follower 关节加速度上限。 |
| `NERO_FOLLOWER_GOVERNOR_ERROR_DEG` | `1.50` | feedback governor 开始限制前馈追赶的误差。 |
| `NERO_FOLLOWER_HARD_ERROR_DEG` | `2.25` | command 与实测偏差过大时的硬保护阈值。 |
| `NERO_OSQP_TRUST_REGION_DEG` | `0.3` | OSQP 对策略原始关节轨迹的最大局部偏离。 |
| `NERO_CASADI_MAX_JERK_DEG_S3` | `8000` | CasADi jerk 上限。 |
| `NERO_TOPPRA_MIN_COMMIT_TICKS` | `8` | 新计划到达后，当前计划至少再执行的 tick 数。 |
| `NERO_TOPPRA_REPLAN_RESERVE_TICKS` | `8` | 保留给安全重规划的未来 tick 数。 |

示例：

```bash
cd /home/dev/nero_bimanual_control
NERO_POLICY_DURATION=45 \
NERO_FOLLOWER_MAX_VELOCITY_DEG_S=25 \
NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2=250 \
./scripts/run_policy_osqp_casadi.sh --execute
```

## 6. 验证顺序

先验证不会接触实机的部分：

```bash
cd /home/dev/nero_bimanual_control
./trajectory/osqp_waypoint_smoother/run_tests.sh
./trajectory/casadi_fixed_horizon_retimer/run_tests.sh
./trajectory/toppra_fixed_horizon_retimer/run_tests.sh
```

再进行实机预检：

```bash
./scripts/run_policy_osqp_casadi.sh --preflight-only
```

只有确认双 CAN、三路相机、策略服务和 Home 状态全部正确后，才使用 `--execute`。

更多运行细节见：[当前控制栈](docs/CURRENT_CONTROL_STACK.md)、[仓库目录结构](docs/REPOSITORY_LAYOUT.md)、[归档索引](docs/ARCHIVE_INDEX.md)。
