# OSQP 固定时域轨迹点平滑器

这是当前正式实机运行时入口。它复用维护中的 Pi0.5、RTC、follower 和 CPV 基础代码，并在启动进程内加载正式的 action queue 与 follower 扩展。

完整的受支持实机命令和参数归属见 [`../../docs/CURRENT_CONTROL_STACK.md`](../../docs/CURRENT_CONTROL_STACK.md)。

## 约束契约

对于一个 `H x 16` 的策略 action chunk，平滑器：

- 只优化 14 个机械臂关节列；
- 两个夹爪列保持原值；
- 保持 `H` 与固定 30 Hz 时间轴；
- 保持每个关节轨迹点位于策略输出附近的可配置 trust region 内；
- 施加线速度、加速度、jerk 和可选关节位置约束；
- 支持精确的起点位置和起点速度边界；
- 不强制末端速度为零；
- 不可行或验证失败时返回原始 action chunk。

凸优化目标为：

```text
位置跟踪 + 速度跟踪 + 加速度正则
         + jerk 正则 + 末端速度跟踪
```

OSQP 修改关节轨迹点。随后 CasADi 只沿这条有界且更平滑的路径重新分配相位，不改变全局 horizon。

## 依赖安装

依赖隔离在本目录的 `vendor/` 中：

```bash
cd /home/dev/nero_bimanual_control/trajectory/osqp_waypoint_smoother
/home/dev/enter/envs/lerobot/bin/python -m pip install \
  --target vendor -r requirements.txt
```

## 测试

```bash
cd /home/dev/nero_bimanual_control/trajectory/osqp_waypoint_smoother
chmod +x run_tests.sh scripts/smooth_chunk.py
./run_tests.sh
```

## 离线 action chunk 检查

```bash
./scripts/smooth_chunk.py /path/to/actions.npy \
  --output outputs/actions_smoothed.npy \
  --trust-deg 0.3 \
  --max-velocity-deg-s 28 \
  --max-acceleration-deg-s2 280 \
  --max-jerk-deg-s3 8000
```

输出报告会包含原始与平滑后速度、加速度、jerk 比率，以及最大轨迹点偏离。只有离线指标显示可行性改善且轨迹偏离可接受时，才应接入 CasADi。

## 实机运行时

运行时链路保持与原生 follower 隔离：

```text
原始 RTC chunk -> OSQP 轨迹点平滑 -> CasADi 重定时 -> 既有 q/v/a 交接
```

正常链路绕过 Action Gain。每次非初始请求都会根据当前计划的 commit 与 reserve 边界预测最早真实接管 tick，并从该时刻 follower 预测的 q/v 与新 chunk 的对应 future action 构造有限的恢复候选。

真实 commit 边界仍优先使用原始候选。只有原始候选超过 q/v 硬交接限制或无法构造有界 q/v/a correction，而恢复候选可以时，才选择恢复候选。运行日志中的 `gain=bypassed`、`gain=fallback:<value>` 与 `handoff_candidate=recovery` 用于区分路径。

当前启动器使用 8 tick 最小提交与 8 tick 重规划余量。如果原始候选和恢复候选都无法安全交接，会在重规划边界丢弃候选并发送新的 RTC 请求；此路径明确禁用旧的 `reserve_exhaustion_follower_handoff`，不会在旧轨迹耗尽后强制进行无界 follower 接管。

适配器只在自身进程中替换 RTC queue，原生策略流源码保持不变：

```bash
cd /home/dev/nero_bimanual_control
NERO_POLICY_DURATION=30 \
  ./scripts/run_policy_osqp_casadi.sh --preflight-only
```

预检通过后去掉 `--preflight-only` 才会启动实机。默认运行时把跨 chunk 连续性交给既有 q/v 交接；`NERO_OSQP_ENFORCE_BOUNDARY=1` 仅用于单独的严格边界实验。

## 已录制策略回放

在不连接实机的情况下，对比原始到 CasADi 路径与 OSQP 到 CasADi 路径：

```bash
./scripts/replay_chunks_to_casadi.py \
  /path/to/chunks.jsonl \
  --trust-deg 0.3 \
  --output outputs/replay_report.json
```
