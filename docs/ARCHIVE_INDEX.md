# 归档索引

归档内容用于复现与对比，不属于当前实机执行链路。

| 位置 | 状态 | 原因 |
|---|---|---|
| `archive/legacy_policy_presets/` | 历史策略预设 | 绑定旧 checkpoint、旧数据集或旧任务保护逻辑的启动脚本。 |
| `trajectory/casadi_fixed_horizon_retimer/legacy_runtime/` | 已替代的直接启动器 | 直接 CasADi 实机运行器；优化器本身仍在当前链路中使用。 |
| `trajectory/toppra_fixed_horizon_retimer/legacy_runtime/` | 已替代的直接启动器 | 直接 TOPPRA 实机运行器；共享队列与桥接层仍是当前依赖。 |
| 原 `/home/dev/ros2_project/archive/` | 外部历史实验 | Ruckig PoC、旧 ROS 工作区与早期设计文档保留在旧项目中。 |

归档代码被保留而非删除。未经审查和分支恢复，不要将其中的启动器用于实机。
