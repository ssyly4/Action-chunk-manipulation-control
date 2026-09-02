# 训练服务器部署辅助代码

本目录镜像保存了训练服务器上用于检查、转换和验证 NERO 训练数据的少量脚本。服务器上的实际模型、checkpoint 和数据副本不在本仓库。

当前工具：

- `convert_nero_bimanual_v3_to_v21.py`：LeRobot V3 到 V2.1 的转换。
- `compute_nero_bimanual_norm_stats_fast.py`：归一化统计计算。
- `validate_nero_towel_fullflow70_source.py` 与 `validate_nero_towel_fullflow70_v21.py`：数据集验证。
- `nero_action_alignment.py`：action 与 observation 对齐辅助函数。
- `openpi/`：NERO 专用 OpenPI 配置和策略定义。

历史流水线保留在原服务器/旧实验目录中。它们使用早期数据集和 action 定义，不能作为新训练的起点。
