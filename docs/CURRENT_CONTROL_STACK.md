# 当前控制栈

## 运行链路

当前实机策略控制链由以下模块组成：

```text
策略 WebSocket 输出（24 x 16 action chunk）
  -> OSQP 有界轨迹点平滑
  -> CasADi 固定时域相位重定时
  -> 共享的滚动 RTC 队列与 q/v/a 连续交接
  -> NERO 流式关节 follower
  -> CPV 后端 -> CAN -> NERO 双臂
```

当前实机 A/B 入口为 `scripts/run_policy_osqp_casadi.sh`。以下环境变量是受支持的调参接口：

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `NERO_POLICY_DURATION` | `30` s | 单次试验时长 |
| `NERO_FOLLOWER_MAX_VELOCITY_DEG_S` | `28` | 关节 follower 速度上限 |
| `NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2` | `280` | 关节 follower 加速度上限 |
| `NERO_FOLLOWER_GOVERNOR_ERROR_DEG` | `1.50` | command-feedback 软超前误差阈值 |
| `NERO_FOLLOWER_HARD_ERROR_DEG` | `2.25` | command-feedback 硬停止误差阈值 |
| `NERO_TOPPRA_MIN_COMMIT_TICKS` | `8` | 正常替换前当前计划最少执行 tick 数 |
| `NERO_TOPPRA_REPLAN_RESERVE_TICKS` | `8` | 触发安全重规划前必须保留的轨迹余量 |
| `NERO_TOPPRA_ALLOW_RESERVE_FOLLOWER_HANDOFF` | `0` | 当前 A/B 中必须保持关闭 |

全局控制时钟固定为 30 Hz。重定时只能重新分配一个 horizon 内的相位，不能改变该 horizon 的起止 wall-clock 时间。

## 模块职责

| 路径 | 职责 |
|---|---|
| `trajectory/osqp_waypoint_smoother/waypoint_smoother/smoother.py` | 在策略输出邻域内做凸优化轨迹点平滑 |
| `trajectory/osqp_waypoint_smoother/ab_runtime/osqp_casadi_rtc_queue.py` | 当前 OSQP/CasADi 队列与恢复候选轨迹选择 |
| `trajectory/casadi_fixed_horizon_retimer/fixed_phase_optimizer/optimizer.py` | 在速度、加速度、jerk 约束下优化固定总时长相位 |
| `trajectory/toppra_fixed_horizon_retimer/ab_runtime/receding_toppra_queue.py` | q/v/a 连续交接、余量与安全决策 |
| `trajectory/toppra_fixed_horizon_retimer/ab_runtime/follower_state_bridge.py` | 队列和流式 follower 之间的命令状态桥接 |
| `scripts/bimanual_policy/bimanual_guarded_policy_stream.py` | 策略客户端、相机、实机检查与 CPV 输出 |
| `nero_vla/trajectory_executor.py` | 流式 follower 与 feedback governor |

## 操作规则

- 轨迹模块通过进程内替换队列和 follower 进行 A/B；不要为了测试重定时器而直接修改原生策略流。
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
./scripts/run_policy_osqp_casadi.sh --preflight-only
```
