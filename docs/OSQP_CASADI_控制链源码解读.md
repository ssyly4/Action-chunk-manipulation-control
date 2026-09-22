# π0.5 控制链：OSQP、CasADi 与运行时源码

本文解释本仓库的双臂策略轨迹层。策略输出的是固定 30 Hz 的 `24 × 16` action chunk：
左臂关节为列 `0..6`，左夹爪为列 `7`，右臂关节为列 `8..14`，右夹爪为列 `15`。
关节位置在优化器内部使用 rad；速度、加速度和 jerk 分别使用 rad/s、rad/s²、rad/s³。
OSQP 和 CasADi 都处理**已有的关节轨迹**，不生成任务目标，也不直接发送 CAN/CPV 命令。

## 源码从哪里进入

```text
scripts/run_policy_osqp_casadi.sh
  → trajectory/osqp_waypoint_smoother/runtime/launch_policy.sh
  → trajectory/osqp_waypoint_smoother/runtime/run_policy.sh
  → trajectory/osqp_waypoint_smoother/runtime/policy_runtime.py
      在进程内将 BimanualRtcActionQueue 替换为 RecedingOsqpCasadiRtcQueue
  → 策略流收到 24 × 16 chunk
  → OSQP 平滑 14 个关节列
  → 默认：固定 30 Hz 快路径；可选慢路径：CasADi 相位重定时
  → toppra_fixed_horizon_retimer/runtime/receding_toppra_queue.py 的滚动计划与交接
  → nero_vla/trajectory_executor.py 的 streaming follower
  → nero_vla/cpv_backend.py → NERO CPV/CAN
```

`policy_runtime.py` 还包装原有 `RateLimitedJointFollower`，把队列产生的期望速度
传给 follower。控制 tick、反馈保护和最终硬件写入不属于 OSQP/CasADi 求解器。

## OSQP：允许小幅修改每个 waypoint

实现见 [`smoother.py`](../trajectory/osqp_waypoint_smoother/waypoint_smoother/smoother.py)：
`WaypointSmootherConfig` 定义限制和权重，`OsqpWaypointSmoother.smooth()` 接收
`H × 16` chunk，取出 14 个手臂关节组成参考轨迹 `r`。决策变量是整个未来关节序列
`x = [q₀, …, qₕ₋₁]`。`D₁/D₂/D₃` 是按 `dt = 1/action_hz` 缩放的一、二、三阶差分。

源码中的五项二次代价可写成：

```text
min_x  20   · ||(x − r) / trust||²
     + 0.25 · ||(D₁x − D₁r) / v_max||²
     + 0.10 · ||D₂x / a_max||²
     + 1.00 · ||D₃x / j_max||²
     + 1.00 · ||(D₁x)_last − (D₁r)_last||² / v_max²
```

这五个默认权重依次是 `tracking_weight`、`velocity_tracking_weight`、
`acceleration_weight`、`jerk_weight`、`terminal_velocity_weight`。
它们控制**满足约束后的取舍**：尽量贴近策略点，抑制加速度和 jerk，同时保持策略末端的速度趋势。
源码先按 trust/速度/加速度/jerk 限值归一化，再用权重相乘；不能仅比较 `20` 和 `1`
就推断实际每项的数值贡献。`_problem()` 生成 Hessian 和线性约束矩阵，
`_linear_cost()` 补上对原始轨迹的跟踪项，调用 `osqp.OSQP().solve()` 求解。

硬约束由 `_bounds()` 给出，与权重分开：

```text
|qₜ − rₜ| ≤ trust          默认每关节 0.3°
|D₁q| ≤ v_max             |D₂q| ≤ a_max             |D₃q| ≤ j_max
q₀ = boundary_position   可选：首段速度 = boundary_velocity
可选：joint lower/upper position limits
```

`smooth()` 只把优化结果写回手臂列；两个夹爪列、chunk 行数和 30 Hz 时间轴原样保留。
求解状态为 `solved` 后仍调用 `_verification_error()` 重新计算 trust 与运动学限制；
求解失败或验算失败会返回 `feasible=False` 和原始 chunk，**不是一个可直接执行的成功结果**。
相同 horizon/关节数复用问题结构，并用上次解 warm start。

运行时权重在 [`osqp_casadi_rtc_queue.py`](../trajectory/osqp_waypoint_smoother/runtime/osqp_casadi_rtc_queue.py)
中从 `NERO_OSQP_*_WEIGHT` 环境变量读取；trust、求解时间和 `v/a/j` 限值也由运行时配置传入。
这些权重不能替代硬限值。若原始 chunk 的 OSQP 问题不可行，队列先用 Action Gain
生成缩放恢复候选，再求一次 OSQP；第二次仍失败就拒绝该 chunk。

## CasADi：固定路径，只重新分配时间

实现见 [`optimizer.py`](../trajectory/casadi_fixed_horizon_retimer/fixed_phase_optimizer/optimizer.py)：
`NaturalCubicPath` 将 OSQP 后的关节 waypoint 插成固定的自然三次样条 `q_ref(s)`。
`CasadiPhaseOptimizer` 的决策变量不是关节角，而是每个输出 tick 的路径进度
`s = [s₀, …, sₜ₋₁]`，输出命令为 `qₜ = q_ref(sₜ)`。改变 `sₜ` 会改变沿同一条样条路径
前进的快慢；三次样条在 waypoint 之间的形状由输入决定，不由相位求解器重新规划。

`_solver_bundle()` 用 CasADi 建模，交给 IPOPT 求解。软目标包括：相位接近均匀进度
（默认权重 `200`）、相位速度接近名义速度（`1`）、相位加速度（`0.05`）、
相位 jerk（`0.002`），以及首尾相位速度接近给定目标（`5`）。硬约束包括：

- `s₀ = start_phase`、`sₜ₋₁ = H−1`，相位单调不减；末段相位速度有下界；
- 根据最终 30 Hz 关节命令计算的速度、加速度、jerk 不超过配置上限。

首尾相位和输出 tick 数都固定，所以它只能在**同一总时长内部**重新分配快慢，
不能把整个 24-step chunk 延长到更多 tick。24 个 30 Hz 命令占 24/30 = 0.8 s
的 RTC 边界时长，相邻首末样本间隔为 23/30 s。若固定时间内整条路径不可行，
调软权重不能使硬约束消失。`_evaluate()` 会对实际离散输出再次验算；失败结果带
`feasible=False`，由上层队列决定后续处理。

手臂使用求出的相位，夹爪仍沿原固定 tick 时间线插值，见 `_compose()`；
因此“手臂相位重定时”不等于“夹爪事件一起重定时”。

## 当前正式运行时如何选择两条路径

[`launch_policy.sh`](../trajectory/osqp_waypoint_smoother/runtime/launch_policy.sh)
默认设置 `NERO_OSQP_FAST_PATH=1`、`NERO_OSQP_SPECULATIVE_RECOVERY=0`。
队列在 `_retime_job()` 中先做 OSQP；成功后调用 `_fixed_timeline_result()`，直接用
`osqp_fixed_timeline` 状态提交固定 30 Hz 命令，并记录 `casadi_skipped=True`。
所以启动器名称虽然包含 `casadi`，**默认成功路径没有运行 CasADi**。

设置 `NERO_OSQP_FAST_PATH=0` 才进入实验性慢路径：首个 generation 使用已有的
初始 fallback，后续 generation 才调用 `CasadiPhaseOptimizer.optimize()`；慢路径可构造
并行恢复候选。运行时分别记录 OSQP、CasADi、建图/调用和总 retime 耗时，不能把
`fallback_original` 或 `osqp_fixed_timeline` 日志误读为 CasADi 优化成功。

OSQP/CasADi 生成的仍只是候选计划。共享 RTC queue 按最小提交时间与剩余计划余量
决定是否接管，并处理跨 chunk 的 q/v/a 连续性；follower 再依据实时反馈限制速度、
加速度、jerk 与命令超前误差，最后由 CPV backend 发送。这里的多层限制各有职责：
OSQP 保证局部 waypoint 有界，CasADi 可选地调整段内相位，queue 负责交接，
follower/CPV 负责逐 tick 的执行边界。

本页是源码解读，不是实机验证报告；阅读或修改文档没有执行机器人命令。
