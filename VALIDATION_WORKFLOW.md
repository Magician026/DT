# UniVTAC Detail-Preserving Encoder 验证流程

## 1. 先给结论：是否需要重新训练 encoder？

需要，但要区分“代码验证”和“科研对比”。

| 阶段 | 是否需要重训 encoder | 说明 |
| --- | --- | --- |
| Import / forward / backward / decoder shape smoke | 不需要 | 只验证代码、shape、梯度和 checkpoint 接口。 |
| Encoder reconstruction 对比 | 需要 | detail_v1 的 trunk/state keys 与 Original 不同，必须产生 V1 自己的 encoder checkpoint。 |
| 公平 baseline | 建议重跑 Original | Original 也要在同一数据 schema、resize、抽样、seed、loss weights 和训练预算下重跑。 |
| Policy training | 需要训练 policy | 两个 encoder 版本接入同一个 ACT policy architecture，分别训练 policy。 |
| Official evaluation | 不重新训练 | 只加载已经训练好的 policy checkpoint，在相同 task/seed/episode 设置下评估。 |

当前 /usr1/home/s126mdg41_04/UniVTAC details/checkpoints/encoder.pth 是 Original 结构的既有 checkpoint。它已通过 strict load 验证，但不能直接作为 detail_v1 权重，也不能把它的结果直接当作 V1 科研对比结果。

## 2. 实验对象与控制变量

最小实验矩阵：

    Original encoder
        -> same ACT policy architecture
        -> same policy training protocol
        -> official Insert HDMI evaluation

    DetailPreservingEncoderV1
        -> same ACT policy architecture
        -> same policy training protocol
        -> official Insert HDMI evaluation

    lift_bottle 保留为后续/辅助对照，不作为第一项主要结论来源。

第一阶段只改变 encoder representation：

- Original：ResNet-18，512-D latent。
- V1：layer2 detail branch（4×4 spatial pooling，256-D）与 layer4 semantic branch（GAP，256-D）拼接为 512-D。
- 不修改 decoder、ACT Transformer、action head、temporal aggregation 或 simulator。
- 两个版本使用相同 dataset root、schema、resize、sample count、seed、split、loss weights、epoch、batch size 和 policy num_steps。

## 3. 环境初始化

服务器：mlda2。

开发目录：

    /usr1/home/s126mdg41_04/UniVTAC details

官方 eval 文档：

    /usr1/home/s126mdg41_04/UniVTAC/eval/lift_bottle_official_univtac_eval.md

该文档记录的是 lift bottle 的已验证 official pipeline；Insert HDMI 沿用其中的环境初始化、路径、GPU、checkpoint 和 simulator 排查原则，但 task、dataset、policy checkpoint 和结果目录必须替换为 Insert HDMI 对应值。正式 Insert HDMI eval 前仍需完整重读该文档，不能把 lift bottle 的结果或参数直接当作 Insert HDMI 结果。

每次运行 Isaac/PyTorch 前：

    BASE=/usr1/home/s126mdg41_04
    ISAAC=$BASE/isaacsim-4.5.0
    set +u
    source "$BASE/miniconda3/etc/profile.d/conda.sh"
    conda activate UniVTAC
    source "$ISAAC/setup_conda_env.sh"
    set -u

涉及 simulator/evaluation 的命令使用：

    "$ISAAC/kit/python/bin/python3"

不要把 Conda Python 当作 official simulator evaluation Python。正式 eval 前必须再次完整阅读官方文档，并遵守其中的 libstdc++、scikit-image、torch_scatter、imageio-ffmpeg 和 ffmpeg 处理要求。

## 4. 每次实验前的 GPU 与 Git 检查

训练或 evaluation 前执行：

    nvidia-smi
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits

规则：

- encoder/policy training：剩余显存必须大于 12 GB。
- official evaluation：尽量选择真正空闲 GPU。
- 不杀其他进程；没有合规 GPU 时停止并记录。
- 使用 CUDA_VISIBLE_DEVICES=<id>，并记录物理 GPU、free memory、utilization 和已有进程。

Git 检查：

    cd "/usr1/home/s126mdg41_04/UniVTAC details"
    git branch --show-current
    git rev-parse HEAD
    git status --short
    git remote -v

不要使用 git clean、git reset --hard、force checkout 或 force push。

## 5. Phase A：代码和接口 smoke（不需要重训）

### A1. Encoder smoke

    cd "/usr1/home/s126mdg41_04/UniVTAC details/encoder"
    "$ISAAC/kit/python/bin/python3" smoke_test.py

确认 Original/V1 forward、512-D latent、V1 layer2/layer4 shape、finite gradients、五类 decoder shape，以及 V1 strict checkpoint round-trip。

### A2. Policy tactile backbone smoke

    cd "/usr1/home/s126mdg41_04/UniVTAC details"
    "$ISAAC/kit/python/bin/python3" policy/ACT/tactile_backbone_smoke_test.py

确认：

- original/detail_v1 都支持 feat/full；
- feat feature 为 [B,512,1,1]；
- full 默认 layer4 feature 为 [B,512,8,8]；
- 原有 policy checkpoint tactile key/shape 兼容；
- 不用 strict=False 隐藏问题。

当前 PositionEmbeddingSine 的 position batch 维为 1，例如 [1,512,8,8]，由 ACT 广播使用；不要为测试修改它。

### A3. Data adapter smoke

    cd "/usr1/home/s126mdg41_04/UniVTAC details"
    "$ISAAC/kit/python/bin/python3" encoder/data_adapter_smoke_test.py

当前 adapter：

- 默认 legacy_contact_gs：保持旧的 left_tactile/right_tactile 和旧路径逻辑；
- 显式 gsmini：支持 left_gsmini/right_gsmini；
- 自动选择 actor/prism 或 actor/bottle；
- gsmini 默认把 RGB/depth target 对齐到 256×256。

## 6. Phase B：Encoder reconstruction training

### B1. 固定数据处理

当前可用数据目录：

    ../data/insert_HDMI/clean
    ../data/lift_bottle/clean

正式 baseline/V1 必须选定同一个 root。当前 mini training 使用：

    schema: gsmini
    data_root: ../data/insert_HDMI/clean
    effective image size: 256×256
    data_num: 8
    seed: 42

### B2. Short real-data smoke

训练前重新执行 nvidia-smi。命令模板：

    cd "/usr1/home/s126mdg41_04/UniVTAC details/encoder"
    CUDA_VISIBLE_DEVICES=<GPU_ID> \
      "$ISAAC/kit/python/bin/python3" train.py all 8 \
      --encoder_type original \
      --schema gsmini \
      --data_root ../data/insert_HDMI/clean \
      --epochs 1 \
      --batch_size 2 \
      --num_workers 0 \
      --seed 42

V1 只将 --encoder_type original 改成 --encoder_type detail_v1。确认 loss finite、无 NaN/OOM、train/validation loop 完成、checkpoint 独立保存且不覆盖旧 checkpoint。

### B3. 正式 Original/V1 encoder training

    cd "/usr1/home/s126mdg41_04/UniVTAC details/encoder"
    CUDA_VISIBLE_DEVICES=<GPU_ID> \
      "$ISAAC/kit/python/bin/python3" train.py all <DATA_NUM> \
      --encoder_type original \
      --schema gsmini \
      --data_root <SAME_DATA_ROOT> \
      --epochs <SAME_EPOCHS> \
      --batch_size <SAME_BATCH_SIZE> \
      --num_workers <SAME_NUM_WORKERS> \
      --seed 42

V1 只改 --encoder_type detail_v1。

输出路径：

    Original:
    encoder/ablation/data_<DATA_NUM>/resnet18/<TIMESTAMP>/best.pth

    Detail V1:
    encoder/ablation/data_<DATA_NUM>/detail_v1/resnet18/<TIMESTAMP>/best.pth

必须记录 Git commit、schema、root、文件数、sample count、seed、loss weights、GPU、显存、checkpoint、各项 reconstruction loss、参数量和可用时的 latency。

1 epoch/8 samples 的 smoke loss 不能用于判断 hypothesis。

## 7. Phase C：Policy training

### C1. 先检查 policy dataset

ACT policy loader 需要：

    /action
    /observations/qpos
    /observations/images/<camera_name>
    /observations/images/<tactile_name>

当前第一任务使用 `sim-insert_HDMI-demo-50`。ACT 数据已在服务器本地找到：

    /usr1/home/s126mdg41_04/UniVTAC/policy/ACT/data/sim-insert_HDMI/demo-50

该目录包含 50 个 episode HDF5，约 4.7 GB；代表性 episode 已确认：

    /action                                  [117, 8] float32
    /observations/qpos                       [117, 8] float32
    /observations/images/cam_high            [117, 270, 480, 3] uint8
    /observations/images/tac_left            [117, 240, 320, 3] uint8
    /observations/images/tac_right           [117, 240, 320, 3] uint8

`details/policy/ACT/SIM_TASK_CONFIGS.json` 已保留 lift bottle 条目，并新增 `sim-insert_HDMI-demo-50`。两个条目引用原始项目中的只读 ACT 数据路径；dataset 不复制到 DT，也不会提交 GitHub。`details/data/insert_HDMI/clean` 仍仅用于 encoder reconstruction，不能直接代替 ACT episode dataset。

policy training 的前置条件：

    1. 检查 `sim-insert_HDMI-demo-50` 的 episode 数量和只读路径；
    2. 检查 camera/tactile key、action shape、qpos shape，并用 ACT `TacArenaDataset` 读取一个样本；
    3. 由同一数据和同一 split 生成/核对 `dataset_stats.pkl`；
    4. 先让 Original policy 跑通；
    5. 再用完全相同的数据和训练预算训练 V1 policy。

### C2. 两个 policy config

Original：

    tactile_encoder_type: original
    tactile_ckpt: <ORIGINAL_ENCODER_BEST_PTH>
    tactile_type: feat
    tactile_masks: false

V1：

    tactile_encoder_type: detail_v1
    tactile_ckpt: <DETAIL_V1_ENCODER_BEST_PTH>
    tactile_type: feat
    tactile_masks: false

V1 模板：

    /usr1/home/s126mdg41_04/UniVTAC details/policy/ACT/train_config_detail_v1.yml

训练前把 tactile_ckpt: null 替换为已核验存在且非空的 V1 checkpoint。两个版本保持相同 ACT hyperparameters、data split、stats、seed、batch size、learning rate、weight decay、num_steps 和 output directory 规则。

### C3. Policy training command

仅在 C1 通过后运行：

    cd "/usr1/home/s126mdg41_04/UniVTAC details/policy/ACT"
    CUDA_VISIBLE_DEVICES=<GPU_ID> \
      "$ISAAC/kit/python/bin/python3" imitate_episodes.py \
      --ckpt_dir <POLICY_OUTPUT_DIR> \
      --task_name sim-insert_HDMI-demo-50 \
      --config_path <TRAIN_CONFIG_YAML> \
      --seed 42

Original/V1 使用不同 output directory，禁止覆盖已有 official/local policy checkpoint。

## 8. Phase D：Official evaluation

正式 evaluation 前再次完整阅读：

    /usr1/home/s126mdg41_04/UniVTAC/eval/lift_bottle_official_univtac_eval.md

按文档逐项确认 checkpoint、stats、环境初始化、simulator、task、deploy config、seed、episode 和 GPU。这里的 task 使用 `insert_HDMI`，policy 使用对应的 Original/V1 policy checkpoint；选择真正空闲 GPU，不杀其他用户进程。

命令模板：

    cd "/usr1/home/s126mdg41_04/UniVTAC details"
    CUDA_VISIBLE_DEVICES=<FREE_GPU> \
      "$ISAAC/kit/python/bin/python3" scripts/eval_policy.py \
      insert_HDMI demo ACT/deploy --headless --total_num 100

具体 environment workaround、checkpoint 和变量以官方 lift bottle 文档中已验证的流程为准；Insert HDMI 的 task config、policy checkpoint、seed 和结果目录必须单独记录。记录完整命令、checkpoint、config、eval result directory 和 success count。

## 9. 结果记录模板

每个正式实验记录：

    experiment name
    Git commit
    encoder type
    dataset schema / data root / file count / sample count
    input and target size
    seed
    architecture differences
    parameter count
    GPU ID / start free memory / peak memory
    encoder training command
    encoder checkpoint
    encoder reconstruction losses
    policy training command
    policy checkpoint
    policy validation loss
    official eval command
    official eval result directory
    success / total
    failure type: code / training / research
    rollback point
    next step

最小对比表：

| 项目 | Original | Detail V1 |
| --- | --- | --- |
| latent dimension | 512 | 512 |
| encoder parameter count | 11,439,168 backbone params | 11,850,048 backbone params |
| reconstruction loss | 待正式实验 | 待正式实验 |
| policy validation loss | 待 Insert HDMI policy training | 待 Insert HDMI policy training |
| manipulation success rate | 待 Insert HDMI official eval | 待 Insert HDMI official eval |
| GPU memory | 待正式实验 | 待正式实验 |

## 10. 结果解释

- reconstruction 和 policy success 都提升：支持 spatial detail preservation 假设，但仍需检查参数量和显存差异。
- reconstruction 提升、policy success 不变：可能 policy 没有利用 detail。
- reconstruction 不提升、policy 提升：检查 reconstruction loss 与 manipulation target 的相关性。
- 两者都不提升：记录 negative result，不立即堆叠 attention/frequency/temporal module。
- shape error、OOM、NaN：先按 code/training failure 排查，不解释为 research failure。

## 11. 回退与安全

- Original 可通过显式 encoder_type=original 运行，不依赖 Git 回退。
- 失败实验保留 commit，使用 Git revert 或文件级新 commit 回退。
- 不删除 dataset、checkpoint、log、eval result 或旧代码。
- 不 force push，不覆盖 DT 远程历史。

## 12. 当前状态（2026-09-04）

已完成：

- Original/V1 encoder、decoder、policy tactile backbone smoke；
- Original policy checkpoint tactile key/shape compatibility；
- Original encoder checkpoint policy-side strict loader；
- gsmini adapter 对 insert_HDMI 与 lift_bottle 的真实 HDF5 读取；
- GPU2 合成 mini-step；
- GPU2 真实 insert_HDMI 8-sample、1-epoch Original/V1 encoder training smoke；
- DT 仅发布实验 overlay，没有 dataset/checkpoint/upstream history。

尚未完成：

- 正式规模 Original/V1 encoder reconstruction training；
- ACT policy dataset 的生成/适配与 policy training；
- Original/V1 policy controlled comparison；
- official Insert HDMI evaluation；
- lift bottle secondary comparison（暂不作为第一项任务）。

当前结论：V1 已完成接口级和短训练可行性验证，但尚未证明 performance improvement。Insert HDMI 的 policy-format dataset 已在原始项目 ACT 目录中找到并完成 schema 事实核验；下一步是先完成 details 配置与 loader smoke，再按本文流程进行 Original/V1 成对 encoder 重训、policy training 和 official eval。
