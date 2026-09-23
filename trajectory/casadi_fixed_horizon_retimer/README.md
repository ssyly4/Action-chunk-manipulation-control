# CasADi 固定时域相位优化器

本目录提供当前 OSQP + CasADi 策略运行时使用的相位优化器。它不生成新的关节空间路径；对于固定策略路径，只重新分配局部相位，同时保持 30 Hz horizon 的边界不变。

## 当前接口

- `fixed_phase_optimizer/optimizer.py`：`CasadiPhaseOptimizer` 与 `PhaseOptimizerConfig`。
- `runtime/casadi_rtc_queue.py`：当前 OSQP 运行时继承的队列适配层。
- `tests/`：固定时域、可行性与 warm-start 覆盖。

在本目录运行测试：

```bash
./run_tests.sh
```

直接 CasADi-only 实机运行器已从主分支移除，可从提交 `08d40f1` 恢复。正式运行先用仓库根目录的 `scripts/policy_server.sh` 启动模型服务，再用 `scripts/run_control.sh` 启动控制。
