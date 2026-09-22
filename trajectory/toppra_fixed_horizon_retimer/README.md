# 共享滚动时域运行时基础

本目录包含当前 OSQP + CasADi 实机运行时共用的执行基础。保留代码负责固定频率计划采样、q/v/a 连续交接、命令状态桥接、轨迹余量处理和运行诊断。

## 当前接口

- `fixed_path_retimer/`：参考路径与重定时数据结构。
- `runtime/receding_toppra_queue.py`：`RecedingToppraRtcQueue`、交接安全契约与余量/重规划逻辑。
- `runtime/follower_state_bridge.py`：队列与流式 follower 之间传递的命令与期望速度状态。
- `tests/`：交接和固定路径回归测试。

在本目录运行测试：

```bash
./run_tests.sh
```

独立 TOPPRA 实机运行器已从主分支移除，可从提交 `08d40f1` 恢复。正式运行统一使用仓库根目录的 `scripts/run_policy.sh`。
