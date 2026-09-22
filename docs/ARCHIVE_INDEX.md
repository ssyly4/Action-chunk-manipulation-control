# 归档索引

历史启动器不属于当前实机执行链路，已从 `main` 分支移除，避免它们被误认为受支持入口。

| 历史位置 | 最后完整提交 | 原因 |
|---|---|---|
| `archive/legacy_policy_presets/` | `08d40f1` | 绑定旧 checkpoint、旧数据集或旧任务保护逻辑。 |
| `trajectory/casadi_fixed_horizon_retimer/legacy_runtime/` | `08d40f1` | 已替代的直接 CasADi 实机运行器。 |
| `trajectory/toppra_fixed_horizon_retimer/legacy_runtime/` | `08d40f1` | 已替代的直接 TOPPRA 实机运行器。 |

查看历史文件使用 `git show 08d40f1:<路径>`。需要恢复时应新建实验分支，禁止直接把旧启动器复制回正式入口。
