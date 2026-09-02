# NERO 双臂控制系统

本仓库负责从策略推理结果到两台 NERO 机械臂 CPV/CAN 命令的完整主机端控制链：

```text
OpenPI 策略服务请求
  -> action chunk 调度器 / RTC
  -> OSQP 轨迹点平滑
  -> CasADi 固定时域相位重定时
  -> CPV 反馈调速器
  -> NERO CAN 双臂
```

仓库不包含 PICO 遥操代码、数采数据或模型 checkpoint。

## 相邻目录

| 职责 | 位置 |
| --- | --- |
| PICO 遥操、回零、CAN 准备与录制 | `/home/dev/nero_neo_teleop` |
| 原始与筛选后的数采数据 | `/home/dev/nero_data` |
| 训练集、checkpoint、训练与服务日志 | 训练服务器 |

## 主要入口

- 原生受保护策略控制：`scripts/run_policy_native.sh`
- 当前 OSQP + CasADi A/B 控制：`scripts/run_policy_osqp_casadi.sh`
- 依赖与本机路径模板：`config/paths.env.example`

执行实机前，先根据模板创建本机的 `config/paths.env`，然后运行启动脚本的预检模式。启动器只通过外部路径访问遥操硬件准备脚本、官方 SDK 与远程策略服务。
