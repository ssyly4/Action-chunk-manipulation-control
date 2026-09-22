# 仓库目录结构

`nero_bimanual_control` 是推理到机械臂执行代码的唯一归属目录。

```text
nero_bimanual_control/
  scripts/run_policy.sh       唯一正式策略启动入口
  config/tasks/               毛巾/单右臂任务 preset
  nero_vla/                 CPV 后端、CAN 检查、follower、策略客户端
  scripts/bimanual_policy/  策略服务与正式运行器的基础编排层
  trajectory/
    osqp_waypoint_smoother/ 关节轨迹点平滑运行时
    casadi_fixed_horizon_retimer/ 固定时域相位优化器
    toppra_fixed_horizon_retimer/ 路径重定时与连续交接
  config/                   不含密钥的本机路径模板
  artifacts/                被 Git 忽略的运行日志
```

录制实现保留在 `nero_neo_teleop`，输出必须写入 `nero_data`；转换、归一化和训练代码位于 `nero_vla_training`；模型 checkpoint 始终保留在训练服务器。这个边界避免数据、训练工具与在线运行时相互依赖。
