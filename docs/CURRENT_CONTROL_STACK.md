# 当前控制栈

## 运行链路

当前实机策略控制链由以下模块组成：

```text
策略 WebSocket 输出（24 x 16 action chunk）
  -> OSQP 有界轨迹点平滑
  -> 默认固定 30 Hz 快路径；可选 CasADi 固定时域相位重定时
  -> 共享的滚动 RTC 队列与 q/v/a 连续交接
  -> NERO 流式关节 follower
  -> CPV 后端 -> CAN -> NERO 双臂
```

当前实机入口为 `scripts/run_policy.sh --task towel_fold`。任务固定参数位于
`config/tasks/towel_fold.toml`，以下环境变量可在启动时覆盖其默认值：

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `NERO_POLICY_DURATION` | `30` s | 单次试验时长 |
| `NERO_FOLLOWER_MAX_VELOCITY_DEG_S` | `28` | 关节 follower 速度上限 |
| `NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2` | `280` | 关节 follower 加速度上限 |
| `NERO_FOLLOWER_GOVERNOR_ERROR_DEG` | `1.50` | command-feedback 软超前误差阈值 |
| `NERO_FOLLOWER_HARD_ERROR_DEG` | `2.25` | command-feedback 硬停止误差阈值 |
| `NERO_TOPPRA_MIN_COMMIT_TICKS` | `8` | 正常替换前当前计划最少执行 tick 数 |
| `NERO_TOPPRA_REPLAN_RESERVE_TICKS` | `8` | 触发安全重规划前必须保留的轨迹余量 |
| `NERO_TOPPRA_ALLOW_RESERVE_FOLLOWER_HANDOFF` | `0` | 当前正式运行时必须保持关闭 |

全局控制时钟固定为 30 Hz。重定时只能重新分配一个 horizon 内的相位，不能改变该 horizon 的起止 wall-clock 时间。
正式启动器默认 `NERO_OSQP_FAST_PATH=1`，OSQP 结果可行时跳过 CasADi；
两级优化器的目标、约束和源码调用关系见 [OSQP/CasADi 控制链源码解读](OSQP_CASADI_控制链源码解读.md)。

## 模块职责

| 路径 | 职责 |
|---|---|
| `trajectory/osqp_waypoint_smoother/waypoint_smoother/smoother.py` | 在策略输出邻域内做凸优化轨迹点平滑 |
| `trajectory/osqp_waypoint_smoother/runtime/osqp_casadi_rtc_queue.py` | 当前 OSQP/CasADi 队列与恢复候选轨迹选择 |
| `trajectory/casadi_fixed_horizon_retimer/fixed_phase_optimizer/optimizer.py` | 在速度、加速度、jerk 约束下优化固定总时长相位 |
| `trajectory/toppra_fixed_horizon_retimer/runtime/receding_toppra_queue.py` | q/v/a 连续交接、余量与安全决策 |
| `trajectory/toppra_fixed_horizon_retimer/runtime/follower_state_bridge.py` | 队列和流式 follower 之间的命令状态桥接 |
| `scripts/bimanual_policy/bimanual_guarded_policy_stream.py` | 策略客户端、相机、实机检查与 CPV 输出 |
| `nero_vla/trajectory_executor.py` | 流式 follower 与 feedback governor |

## 操作规则

- 轨迹模块通过进程内替换队列和 follower 接入正式运行时；日常实机运行只使用这一条 OSQP + CasADi 路径。基础策略流用于承载相机、策略与 CPV，不构成第二套正式控制方案。
- 当前启动器关闭 `reserve_exhaustion_follower_handoff`。不安全候选会被尽早丢弃并请求新的 RTC chunk。
- `rtc_queue_hold` 是轨迹余量不足或推理延迟过高的诊断信号，不能靠提高硬阈值掩盖。
- 运行诊断写入 `trajectory/osqp_waypoint_smoother/outputs/`；策略 tick 日志写入 `artifacts/logs/bimanual_policy_stream/`。

## 验证

```bash
cd /home/dev/nero_bimanual_control
./trajectory/toppra_fixed_horizon_retimer/run_tests.sh
./trajectory/casadi_fixed_horizon_retimer/run_tests.sh
./trajectory/osqp_waypoint_smoother/run_tests.sh
```

实机执行前先做预检：

```bash
./scripts/run_policy.sh --task towel_fold --preflight-only
```
