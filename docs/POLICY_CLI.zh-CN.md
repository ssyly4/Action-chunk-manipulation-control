# Policy 启动命令

所有命令都在控制机执行：

```bash
cd /home/dev/nero_bimanual_control
```

## 当前默认毛巾模型 `119999`

查看模型：

```bash
./scripts/run_policy.sh --task towel_fold --show-config
```

启动服务并预检，不运动：

```bash
./scripts/run_policy.sh --task towel_fold --preflight-only
```

实机运行 30 秒：

```bash
NERO_POLICY_DURATION=30 \
./scripts/run_policy.sh --task towel_fold --execute
```

## 切换同一组训练的 checkpoint

例如切换到 `96000`：

```bash
./scripts/run_policy.sh --task towel_fold --checkpoint 96000 --show-config
./scripts/run_policy.sh --task towel_fold --checkpoint 96000 --preflight-only
NERO_POLICY_DURATION=30 \
./scripts/run_policy.sh --task towel_fold --checkpoint 96000 --execute
```

当前主模型可选：

```text
64000 72000 80000 88000 96000 104000 112000 119999
```

## 启动 pilot70 模型 `95999`

预检：

```bash
./scripts/run_policy.sh \
  --task towel_fold \
  --policy-config pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3 \
  --policy-source /home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3/lora_micro96000_towel_fullflow_pilot70_next_feedback_h24_eff4_v3/95999 \
  --stage-name towel_pilot70_95999 \
  --preflight-only
```

实机运行：

```bash
NERO_POLICY_DURATION=30 \
./scripts/run_policy.sh \
  --task towel_fold \
  --policy-config pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3 \
  --policy-source /home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3/lora_micro96000_towel_fullflow_pilot70_next_feedback_h24_eff4_v3/95999 \
  --stage-name towel_pilot70_95999 \
  --execute
```

## 启动右臂抓瓶放箱模型 `119999`

```bash
cd /home/dev/nero_bimanual_control
./scripts/run_policy.sh --task bottle_to_box_right --preflight-only
NERO_POLICY_DURATION=20 \
./scripts/run_policy.sh --task bottle_to_box_right --execute
```

切换到 `96000`：

```bash
./scripts/run_policy.sh --task bottle_to_box_right --checkpoint 96000 --preflight-only
NERO_POLICY_DURATION=20 \
./scripts/run_policy.sh --task bottle_to_box_right --checkpoint 96000 --execute
```

## 查看服务器服务

```bash
ssh dev@172.24.1.154 \
  "docker exec cuda12_8_torch_2_9_1_core pgrep -af '[s]erve_policy.py'; ss -ltn | grep ':8000'"
```

## 停止服务器服务

```bash
ssh dev@172.24.1.154 \
  "docker exec cuda12_8_torch_2_9_1_core pkill -f '[s]cripts/serve_policy.py' 2>/dev/null || true"
```

已有模型和 checkpoint 对应关系见
[`CHECKPOINT_REGISTRY.zh-CN.md`](https://github.com/ssyly4/NERO_VLA_training/blob/main/docs/CHECKPOINT_REGISTRY.zh-CN.md)。
