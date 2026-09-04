# UniVTAC details 实验修改日志

本日志只记录 `/usr1/home/s126mdg41_04/UniVTAC details` 内的实验性开发。原始参考项目 `/usr1/home/s126mdg41_04/UniVTAC` 不属于本仓库的修改范围；除非得到明确授权，本项目不在原始项目中写入代码、不删除服务器文件、不覆盖 checkpoint、不清理工作树、不 force push。

## 2026-09-04 16:15 +08:00：Phase 0–5 调查、架构分析与 V1 计划

### 当前 Git branch

- 调查开始时为 `main`。
- 当前稳定基线 commit：`05bcd3edb92237107efa40105292a24f1a9fd761`（`fix installation typo`）。
- 计划建立实验 branch：`exp/detail-preserving-v1`。

### 当前 commit

`05bcd3e`，与当前 details 仓库的 `origin/main` 一致。

### 环境与 Git 范围检查

- SSH 别名 `mlda2` 当前连接到主机 `gpu41`，用户为 `s126mdg41_04`。
- 实际开发目录：`/usr1/home/s126mdg41_04/UniVTAC details`。
- `git rev-parse --show-toplevel` 确认 Git 根目录就是上述 details 目录，没有越界到其父目录。
- details 已是独立 Git repository；现有 `origin` 是 `https://github.com/univtac/UniVTAC.git`，不是实验仓库 DT。
- GitHub `https://github.com/Magician026/DT.git` 的 `HEAD`、`main`、`master` 和 tags 查询均没有返回 ref；当前未发现可覆盖的远程历史。
- 调查时工作树已有、且不是本轮产生的状态：约 2441 个 tracked deletion，以及 4 个 untracked 项：`.eval_deps/`、`checkpoints/`、`eval/`、`policy/ACT/SIM_TASK_CONFIGS.json`。本轮没有执行 `git reset --hard`、`git clean`、恢复/删除文件或覆盖这些内容。
- details 当前缺少工作树中的 ACT 源文件（例如 `policy/ACT/act_policy.py`、`policy/ACT/detr/models/backbone.py`、`scripts/eval_policy.py`），但这些文件仍在 Git HEAD 或原始项目中。这个现状必须保留并在 policy 集成前显式处理；不能借机恢复整个原始仓库或修改原始目录。

### 修改目标

在保持现有 encoder 输入/输出与 downstream latent 接口的前提下，先实现可解释、最小的 `Detail-Preserving Encoder V1`，验证中层空间细节是否比单一路径 GAP/fc 表征更有利于 tactile reconstruction 和 manipulation policy。

### 修改原因 / hypothesis

当前 ResNet-18 在早期下采样并在 `avgpool -> fc` 中压缩为 `[B,512]`。该过程可能丢失 contact boundary、局部 deformation、shear、marker displacement 和 small contact region 的位置信息。V1 只改变 representation：保留一个中层高分辨率 detail branch，并与深层 semantic branch 融合；不引入 temporal module、频域模块、额外 detail loss、decoder 改造或 policy architecture 改造。

### 修改前网络结构

当前实现位置：

- 训练 encoder：`encoder/network.py` 的 `Tactile`。
- 训练入口：`encoder/train.py`。
- 训练数据：`encoder/dataloader.py`，读取 HDF5 的 tactile RGB/marked RGB、depth、marker、pose 等字段。
- policy-side tactile loader（参考原始项目）：`policy/ACT/detr/models/backbone.py` 的 `TactileBackbone`。

实际 `Tactile` 代码使用 `torchvision.models.resnet18/34/50(num_classes=latent_dims)`。当前 V1 的基线是 ResNet-18、`latent_dims=512`，没有传入 `weights` 或 `pretrained`，因此该 encoder 构造本身是随机初始化，不是 ImageNet pretrained。训练时可选的 decoder 为 RGB/marked RGB、depth、marker、pose。

对 ResNet-18 的实际 forward smoke 检查结果如下：

| 输入（经过代码中的 transform） | conv1 | maxpool/layer1 | layer2 | layer3 | layer4 | avgpool | fc 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[1,3,320,240]` | `[1,64,160,120]` | `[1,64,80,60]` | `[1,128,40,30]` | `[1,256,20,15]` | `[1,512,10,8]` | `[1,512,1,1]` | `[1,512]` |
| `[1,3,256,256]` | `[1,64,128,128]` | `[1,64,64,64]` | `[1,128,32,32]` | `[1,256,16,16]` | `[1,512,8,8]` | `[1,512,1,1]` | `[1,512]` |

- encoder 训练 dataloader 的实际输出 tactile tensor 是 `[B,3,320,240]`；policy dataloader/deployment 将 tactile RGB resize 为 `[B,3,256,256]`。
- 当前 baseline ResNet-18 backbone 参数量为 `11,439,168`。
- 当前 `encoder.pth` 对全量五类 decoder 的 `Tactile` 参数量为 `41,339,340`；该 checkpoint 的 247 个 key 已通过 `strict=True` 全部加载。
- `RGBDecoder` 的输入接口是 `[B,512]`，并通过 `Linear(512,256*8*8)` 解码；marker head 输出 `[B,63,2]`，pose head 输出 `[B,7]`。因此只要 latent 仍为 `[B,512]`，这些 decoder 不需要改。

### 下游 policy 接口与实际调用

- ACT 的 `TactileBackbone` 构造一个带 `FrozenBatchNorm2d` 的 ResNet-18，并尝试从配置的 `tactile_ckpt` 加载；当前配置声明的相对路径 `encoder/checkpoints/resnet18/20251128-125750/best.pth` 在 details 和原始项目中都不存在，当前代码对缺失路径静默跳过。
- 官方 ACT 配置没有 `tactile_type`，`build_tactile_backbone` 因此默认 `feat`：使用 ResNet `fc` 输出 `[N,512]`，再 reshape 为 `[N,512,1,1]` 并送入 `tactile_input_proj`。
- 单独的 `train_config_tactile_full.yml` 使用 `tactile_type: full`，通过 `IntermediateLayerGetter` 暴露 layer4，256×256 policy 输入时为 `[N,512,8,8]`；这不是官方默认 eval 配置，但属于需要保留的现有接口。
- ACT 官方 policy checkpoint 是一个 449-key state dict，包含 103 个 `model.backbones.1.*` tactile-backbone keys；当前官方 eval 加载的是 policy checkpoint 自身的完整权重，而不是上面那个不存在的独立 `tactile_ckpt`。
- DETR/ACT 的 `tactile_input_proj` 只要求 512-channel feature map；因此 V1 在 policy side 需要显式提供同样的 `feat -> [N,512,1,1]` 适配，不能把一个只会返回 ResNet module 的自定义类直接交给当前 `IntermediateLayerGetter`。
- 当前 details 工作树缺少 ACT source 的大部分文件。policy 集成前只允许在 details 内采取最小 overlay/新增文件方案，不能修改原始 `/UniVTAC`，也不能恢复并上传整个原始仓库。

### 修改前 checkpoint / baseline 记录

- details `checkpoints/encoder.pth` 与原始项目 `checkpoints/encoder.pth` 内容一致；大小均为 165,482,707 bytes，当前 encoder strict load 全部匹配。
- 原始项目已有 local ACT baseline：
  `/usr1/home/s126mdg41_04/UniVTAC/policy/ACT/act_ckpt/act-lift_bottle/demo-50/train_config/policy_last.ckpt`。
- 官方 UniVTAC lift_bottle checkpoint：
  `/usr1/home/s126mdg41_04/UniVTAC/policy/ACT/act_ckpt/act-lift_bottle/demo-50/train_config_official_univtac/policy_last.ckpt`，官方文档记录 SHA-256 为 `a6d5d8c0513357fdbafa1cc658cde03770d434a0dd08993697203e89b0f5d14a`。
- 官方 stats：同目录 `dataset_stats.pkl`，官方文档记录 SHA-256 为 `d0839807c19dadac83188ad8f451d29e1b6b8a9c4212ccf002859f57b3f1af36`。
- 官方文档中的 local baseline policy checkpoint SHA-256 为 `6cedd68d26b4facfd5f04d13779ef2c18bcbebb7481412bd8a7d73782c94a012`；在正式对比前仍需按文档重新执行 hash preflight，不能把不同 config 的结果混为 baseline。
- 当前已有的 eval 不是本轮启动：GPU0 上 PID `929110` 正运行 `TRAIN_CONFIG=train_config_univtac_latest` 的 lift_bottle ACT eval；截至调查时最新结果目录 `eval_result/ACT/lift_bottle/deploy/2026-09-04_15:50:11` 为 18 个 seed 中 14 成功（77.78%），仍在运行。该结果不能标为 official checkpoint 的最终 baseline。

### 官方 eval 文档与必须规避的坑

已完整阅读原始项目的 248 行文档：
`/usr1/home/s126mdg41_04/UniVTAC/eval/lift_bottle_official_univtac_eval.md`；details 下的同名副本当前内容一致。

后续正式 eval 必须遵循：

1. 从项目 root 执行 `sha256sum` 和 `test -s`，区分 official checkpoint、local baseline、trained policy 以及 stats。
2. 先激活 `UniVTAC` Conda，再 `source isaacsim-4.5.0/setup_conda_env.sh`，让 Conda 的 `libstdc++.so.6` 优先；不要直接用 Conda env Python 作为 Isaac eval entry point，使用 Isaac Kit Python。
3. Isaac Kit Python 已有针对性的 `scikit-image`、`imageio-ffmpeg` 修复；需确认 `torch_scatter`、`skimage`、`ffmpeg`，不进行 broad pip upgrade。
4. 启动前用 `ps` 和 `nvidia-smi` 检查运行状态、checkpoint、环境变量、GPU；不要用会匹配 launcher 自身的 `pgrep -f` 模式。
5. 评估选择真正空闲 GPU；绝不 kill 他人进程。调查时 GPU 状态为：GPU0 已用约 9663/24564 MiB、利用率 54%，GPU1/2/3 分别约 18/23/16 MiB，后续启动前仍需重新检查。
6. 保留官方 `scripts/eval_policy.py lift_bottle demo ACT/deploy --headless --total_num 100` 流程和 seed 约定；结果目录、`log.log`、metadata/video 路径按文档记录。

### 准备改哪些文件

正式 V1 architecture modification 前，预期只在 details 内新增或修改：

- `encoder/network.py`：保留原始实现，增加 `OriginalUniVTACEncoder` / `DetailPreservingEncoderV1` 和明确的 encoder factory/switch；外层 `Tactile` 维持现有 decoder 接口。
- `encoder/train.py` 或新增轻量 config/entry：增加 `encoder_type: original|detail_v1`，默认仍为 `original`；修复/增加的参数必须保持 baseline 命令可追踪。
- `tests/` 或 `encoder/tests/`：增加 import、forward shape、backward、checkpoint load 的 smoke test；不触碰数据、checkpoint、eval_result。
- policy-side 只在确定 details 内的最小 source overlay 后再改；首选新增可测试 adapter，不改 decoder/ACT transformer/policy architecture。
- `.gitignore`：仅在确认当前 untracked 大文件/运行产物范围后，增加必要的 checkpoint、cache、log、video 忽略规则；不改变或删除已有用户文件。

### 设计中的修改后网络结构（Detail-Preserving V1）

输入保持 RGB、外层调用保持 `[B,3,H,W]`。

```text
ResNet-18 shared trunk
       |-----------------------------|
       v                             v
     layer2                         layer4
 [B,128,H/8,W/8]              [B,512,H/32,W/32]
       |                             |
  1x1 Conv projection             GAP
       |                             |
  AdaptiveAvgPool(4,4)         Linear(512,256)
       |                             |
  Flatten -> Linear(...,256)  z_semantic [B,256]
       |                             |
  z_detail [B,256]                  |
       |____________ concat ________|
                    v
             z [B,512]
```

- 第一版固定从 layer2 取 detail，从 layer4 取 semantic；不改 conv1/maxpool/stride，不引入 dilation。
- detail 分支保留 4×4 粗空间布局，不做直接 GAP；semantic 分支保留现有深层语义聚合。
- `z_detail=256`、`z_semantic=256`，直接 concat 为 `[B,512]`；如实现需要 LayerNorm/最终 Linear，必须记录额外参数并保持 latent 维度不变。
- 默认新 encoder 不新增 auxiliary loss；现有 RGB/marked RGB/depth/marker/pose decoder 继续接收 512-D latent。
- 第一轮只改变 encoder representation；Experiment B 的 reduced early downsampling、Experiment C 的 dilation/high-resolution deep feature、未来 detail supervision 均另开逻辑 commit，不与 V1 混合。

### 是否改变 encoder 输入输出接口

不改变：输入仍为 `[B,3,H,W]`，V1 输出仍为 `[B,512]`；encoder training 的 reconstruction decoder、ACT policy 的 `feat` adapter 和现有 latent consumer 必须保持兼容。`full` 模式若保留，必须通过显式 adapter 定义其 feature-map 输出，而不是改变默认 policy 接口。

### checkpoint compatibility

- `encoder_type=original` 必须继续严格加载旧 `encoder.pth`；V1 不得把旧 checkpoint 以 `strict=False` 静默吞掉。
- V1 的新 projection/MLP 参数无法从旧 ResNet checkpoint 直接得到；加载旧 checkpoint 时必须打印并记录 missing/unexpected keys、成功加载的 shared ResNet keys、随机初始化的新增 keys。
- 旧 ACT policy checkpoint 只与原始 `model.backbones.1.backbone.*` 结构直接兼容；V1 policy 必须重新构造并重新训练/保存。若尝试 warm-start，必须显式核对 key mapping 和 load status，禁止全局 `strict=False`。
- 旧 baseline checkpoint、数据、eval_result 不覆盖、不移动、不删除。

### Rollback plan

1. 稳定基线是 `main` 的 `05bcd3e`；实验在 `exp/detail-preserving-v1`，不直接在 main 上开发。
2. 建 branch 时保留当前工作树原有的 2441 个 deletion 和 4 个 untracked 状态，不对它们做恢复/清理；本实验只 stage 自己明确新增/修改的文件。
3. V1 失败时保留失败 commit 作为 negative-result 记录；回退 baseline 使用 branch 指针/commit 对比，或新建 revert commit，不 rewrite history。
4. 任何涉及 policy-side source materialization、路径重定向或 checkpoint copy 的动作只在 details 内执行；若需要修改原始项目、删除/覆盖文件、恢复整棵缺失源码树，先停止并征求用户意见。
5. push 只推送 details repository 的实验 branch，先配置单独的 DT remote（保留原 `origin`），普通 push，不 force push、不删除远程 branch。

### 验证方法

正式 V1 实现后按顺序：

1. import test；
2. original dummy forward，确认 `[B,512]`；
3. detail_v1 dummy forward，确认 `[B,512]`；
4. backward/gradient finite；
5. original checkpoint strict load；
6. V1 checkpoint save/reload，并检查 missing/unexpected keys；
7. 只在 GPU 检查后选择剩余显存大于 12 GB 的 GPU做极短训练 smoke test，记录 GPU ID、显存、command、config、checkpoint；
8. smoke test 全部通过后才讨论 full encoder training、同架构 policy training 和 official eval。

### 使用的训练 / eval command

本调查阶段未启动训练、未启动新的 evaluation。已记录官方 eval 命令，但正式 eval 前必须再次完整阅读上述官方文档并重新做 preflight。

### 使用的 GPU

调查阶段未占用 GPU 运行实验。只读检查时：GPU0 约 9663/24564 MiB 且已有他人/既有任务 PID 929110；GPU1/2/3 约 18/23/16 MiB。后续短训练需重新执行 `nvidia-smi`，正式 eval 需选择真正空闲卡。

### 实验结果

本阶段无新实验结果。已确认 baseline/official eval 现有运行不能与未来 V1 结果混淆。

### 存在的问题

- details 工作树预先存在大规模 deletion/untracked 状态，必须避免任何会隐式恢复、清理或覆盖的 Git 命令。
- details 缺少大部分 ACT policy/eval source，V1 policy 集成尚未执行；必须设计最小 details 内 overlay，不能修改原始项目。
- 配置中的独立 `tactile_ckpt` 路径缺失；未来训练命令必须明确给出实际 checkpoint 路径，并在日志中记录实际是否加载。
- 现有 GPU0 eval 正在运行，不能占用或终止；其结果不是本轮 V1 baseline。

### 回退方法

见本条目的 Rollback plan。当前尚未写入 architecture code，因此恢复当前实现只需保持 `main@05bcd3e` 和未触碰的原有工作树状态。

### 下一步计划

1. 在不触碰原有 dirty 文件的情况下建立 `exp/detail-preserving-v1`。
2. 提交仅包含本日志的 docs commit，验证 staged path 不包含 deletion、checkpoint、dataset、eval_result 或其他 untracked 产物。
3. 再实现最小 V1 encoder 与 tests；保持 Original/V1 switch 和 512-D latent。
4. 完成 smoke test 后，再决定 policy-side adapter 的最小 materialization 方案，并在任何 policy training 前重新检查 GPU 与 official eval 文档。

## 2026-09-04 16:23 +08:00：Phase 7–8 Detail-Preserving Encoder V1 实现与 smoke test

### 当前 Git branch

`exp/detail-preserving-v1`。

### 当前 commit

当前 HEAD：`392c423`。

- `786750b docs: add UniVTAC modification log`
- `a7b3ede feat: add configurable detail-preserving encoder v1`
- `392c423 test: add encoder forward-shape smoke test`

### 修改目标

在不改变原始 encoder 默认行为和 512-D latent 接口的前提下，落地第一版 multi-scale spatial-detail representation，并建立可重复的最小验证入口。

### 修改前网络结构

Original 模式保持 torchvision ResNet-18 的 `backbone.*` state-dict 命名、early downsampling、`avgpool -> fc -> [B,512]` 路径和现有 decoder 调用方式。

### 修改后网络结构

- 新增 `DetailPreservingEncoderV1`，共享 ResNet-18 convolutional trunk。
- detail：`layer2 [B,128,H/8,W/8] -> 1x1 Conv(128->128) -> norm/GELU -> AdaptiveAvgPool(4,4) -> Linear(2048->256) -> LayerNorm/GELU`。
- semantic：`layer4 [B,512,H/32,W/32] -> GAP -> Linear(512->256) -> LayerNorm/GELU`。
- fusion：`concat(z_semantic, z_detail) -> [B,512]`。
- V1 不使用 ResNet 原始 fc；不改 conv1/maxpool stride、不改 decoder、不改 ACT/Transformer、不加 temporal/dilation/auxiliary loss。
- `Tactile(..., encoder_type='original'|'detail_v1')` 提供明确 switch；`encoder/train.py` 增加 `--encoder_type`，默认 `original`，旧 positional 命令仍可用。

### 修改的文件

- `encoder/network.py`
- `encoder/train.py`
- `encoder/smoke_test.py`

### 是否改变 encoder 输入输出接口

没有改变：Original 与 V1 都接收 `[B,3,H,W]` 并输出 `[B,512]`。现有 RGB/marked RGB/depth/marker/pose decoder 未改。

### checkpoint compatibility

- Original `Tactile` state-dict 结构保持兼容；调查/测试中 `checkpoints/encoder.pth` 用 `strict=True` 加载，247 keys 全部匹配。
- V1 的新增 `trunk`、detail projection/head、semantic head 是新参数，旧 checkpoint 不会自动映射；V1 checkpoint 需要重新训练或显式 key mapping，不能用全局 `strict=False` 隐藏问题。
- V1 的内存序列化 reload 用 `strict=True` 通过；未创建或覆盖服务器上的 checkpoint 文件。
- 旧 ACT policy checkpoint 的 103 个 tactile-backbone keys 仍只匹配原始 policy-side ResNet；policy-side V1 adapter 尚未实现。

### 参数与预期风险

CPU smoke test 测得（仅 backbone、`supervise=[]`）：

- Original：`11,439,168` parameters。
- V1：`11,850,048` parameters。
- 增加：`410,880`，约 `3.59%`；最终训练时 decoder 参数另计。

主要风险是 policy-side 当前代码期待 ResNet module/`IntermediateLayerGetter`，而 details 工作树缺少 ACT source；下一阶段必须在 details 内完成最小 adapter/overlay，并显式验证 `feat` 与可选 `full` 模式，不能修改原始 `/UniVTAC`。

### 验证方法

`$ISAAC/kit/python/bin/python3 encoder/smoke_test.py`（先激活 `UniVTAC` Conda 并 source Isaac Sim setup）已通过：

- Original checkpoint strict load：`All keys matched successfully`。
- Original forward：`(2,512)`。
- V1 forward：`(2,512)`。
- V1 layer2：`(2,128,32,32)`；layer4：`(2,512,8,8)`。
- Original/V1 backward loss 均 finite，所有 trainable gradients finite。
- V1 checkpoint in-memory save/reload strict：`All keys matched successfully`。
- 最终输出：`SMOKE_TEST_PASSED`。
- `python3 -m py_compile` 和 `git diff --check` 通过。

### 使用的训练 / eval command

本阶段没有启动训练或新的 eval。encoder smoke command：

```bash
BASE=/usr1/home/s126mdg41_04
ISAAC=$BASE/isaacsim-4.5.0
set +u
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate UniVTAC
source "$ISAAC/setup_conda_env.sh"
set -u
cd "/usr1/home/s126mdg41_04/UniVTAC details"
"$ISAAC/kit/python/bin/python3" encoder/smoke_test.py
```

### 使用的 GPU

本阶段只在 CPU 做 smoke test，未占用 GPU。GPU0 上已有 PID `929110` 的 `train_config_univtac_latest` eval 未触碰；后续短训练前仍须重新执行 `nvidia-smi`，选择剩余显存大于 12 GB 且不影响他人任务的 GPU。

### 实验结果

这是 code/shape validation，不是 reconstruction 或 manipulation performance 结果。V1 已证明可以在 CPU 上完成 forward/backward 和 checkpoint round-trip；研究效果尚未结论化。

### 回退方法

- 只运行 baseline：`encoder/train.py` 不传 `--encoder_type`，或显式 `--encoder_type original`。
- 完整代码回退：保留失败 commit，使用 `main@05bcd3e` 对比或在实验 branch 上建立 revert commit；不 reset/clean dirty worktree，不删除 checkpoint。

### 下一步计划

1. 在 details 内确认 policy-side 最小 adapter 的落地方式，尤其是官方默认 `feat` 的 `[N,512,1,1]` 接口和 `tactile_type: full` 的兼容性。
2. 补充 detail decoder reconstruction smoke（只检查 shape/interface，不改 decoder）。
3. 在开始任何 short training 前重新执行 `nvidia-smi`，选择合规 GPU，记录显存/command/config/checkpoint。
4. policy training、正式 evaluation 前再次完整阅读官方 eval 文档；先建立 Original baseline 对照，再运行 V1。

## 2026-09-04 16:25 +08:00：decoder interface smoke 补充

### 当前 Git branch

`exp/detail-preserving-v1`。

### 当前 commit

`d1adb25 test: cover decoder compatibility in encoder smoke test`。

### 验证方法与结果

在同一 CPU smoke test 中用 V1 的 `[1,512]` latent 调用现有五类 decoder，未修改 decoder 代码，输出 shape 全部符合原接口：

- `rgb`: `[1,3,256,256]`
- `marked_rgb`: `[1,3,256,256]`
- `depth`: `[1,1,256,256]`
- `marker`: `[1,63,2]`
- `pose`: `[1,7]`

输出仍为 `SMOKE_TEST_PASSED`。该验证只覆盖 encoder/decoder shape compatibility，不代表 reconstruction quality 提升。

### 回退方法

该 commit 只修改 `encoder/smoke_test.py`；删除其效果时使用 Git revert 或回到前一实验 commit，不删除服务器上的任何数据/checkpoint。

## 2026-09-04 16:27 +08:00：GitHub 发布范围保护

### 当前 Git branch

`exp/detail-preserving-v1`。

### 当前 commit

`88e87ee chore: ignore local experiment artifacts`。

### 修改目标与关键变化

- 在 `.gitignore` 中新增 `/.eval_deps/` 和 `/checkpoints/`，防止 root 级环境目录与 165 MB encoder checkpoint 被误加入提交。
- 这些服务器文件仍原样存在，没有删除、移动或覆盖。

### 验证方法

`git diff --check` 通过；提交 staged path 仅为 `.gitignore`。检查确认 details 下 `checkpoints/encoder.pth` 仍存在且大小为 165,482,707 bytes。

### Push 约束

当前 details 的本地 history 起点是完整的 upstream UniVTAC history；若直接把 `exp/detail-preserving-v1` 推到空的 `Magician026/DT`，GitHub 会接收原始仓库的历史/对象，违反本项目“不上传原始 UniVTAC 整个仓库”的约束。因此本轮不做直接 push，也不改写现有 branch/history。

后续发布必须使用只包含实验性文件/修改的 patch-only orphan 发布分支或等价的无 upstream history 方案，正常 push 到 DT；先检查远程仍为空，绝不 force push。当前 `origin` 继续保留为 `https://github.com/univtac/UniVTAC.git`，未覆盖。

## 2026-09-04 16:30 +08:00：Policy-side source materialization 前置记录

### 当前 Git branch

`exp/detail-preserving-v1`。

### 当前 commit / rollback point

当前稳定 rollback point：`6cfe41b`。本次 policy 集成将在其后进行；如失败，保留失败 commit，使用文件级 revert 或回到该 commit 对比，不执行 reset/clean，不删除服务器文件。

### 修改目标

使 details 内的 ACT training/eval pipeline 能实际调用 `encoder_type=original|detail_v1`，以便后续保持同一 ACT/Transformer policy 做公平对比。

### 计划恢复的最小文件

仅从原始项目复制当前 details 工作树中缺失、且 ACT 运行所需的源码/配置：

- `policy/ACT/act_policy.py`
- `policy/ACT/imitate_episodes.py`
- `policy/ACT/utils.py`
- `policy/ACT/detr/main.py`
- `policy/ACT/detr/models/{__init__,backbone,detr_vae,network,position_encoding,transformer}.py`
- `policy/ACT/detr/util/{__init__,misc}.py`
- `policy/ACT/deploy.yml`
- `policy/ACT/deploy_policy.py`
- `policy/_base_policy.py`
- `policy/task_settings.json`
- `task_config/demo.yml`
- `scripts/eval_policy.py`

不恢复整个 `policy/`、`scripts/`、`third_party/` 或其他 2441 个 deletion；原始项目目录不写入任何内容。恢复的未修改文件只用于让 details 运行环境闭合，后续发布到 DT 时仍只选择实验性 diff/patch 文件。

### V1 policy-side modification plan

- 将 details 的 `policy/ACT/detr/models/network.py` 改为从 details `encoder/network.py` 复用 `Tactile`/factory，避免维护第二份 encoder 实现。
- `TactileBackbone` 增加 `tactile_encoder_type` 配置；默认 `original`，保持旧 policy checkpoint 的原始 key layout。
- `feat` 模式：Original 继续输出 `[N,512,1,1]`；V1 通过其 512-D latent 输出同样的 shape。
- `full` 模式：Original 继续用 layer maps；V1 通过显式 `forward_features()` 提供 layer maps，默认选择 layer4，保持 512-channel ACT input。
- checkpoint 加载必须显式输出 status；旧 policy checkpoint 不对 V1 静默 `strict=False`。
- 不改 ACT Transformer、decoder、action head、temporal aggregation 或 policy architecture。

### 风险与验证

- 恢复文件可能使此前的 `D` 状态变为 clean，但不应 stage/commit 未修改源文件；需在恢复后用路径清单检查。
- 先做 import test、Original policy model construction、V1 `feat/full` shape test；随后才考虑短训练。
- 当前 GPU0 仍有既有 eval，恢复源码不启动 GPU 任务；训练前必须重新执行 `nvidia-smi`。

## 2026-09-04 16:41 +08:00：Policy-side V1 adapter 与 smoke validation 完成

### 当前 Git branch

`exp/detail-preserving-v1`。

### 当前 commit

`0737aad test: add policy tactile backbone smoke test`。

前置逻辑提交：

- `1916061 feat: add policy tactile encoder switch`
- `818b4bd exp: add detail v1 policy config`
- `0737aad test: add policy tactile backbone smoke test`

### 修改目标 / hypothesis

使 details 内的 ACT policy 在不改变 Transformer、action head、temporal aggregation 和 decoder 接口的前提下，能够显式选择 `original` 或 `detail_v1` tactile encoder。这样后续可以用相同 policy architecture 对比 encoder representation 是否保留了更多 manipulation-relevant spatial detail。

### 实际修改与源文件范围

- `policy/ACT/detr/models/network.py` 改为 import shim，统一复用 `details/encoder/network.py` 中的 `Tactile` 与 encoder factory，避免 encoder pretraining 与 policy 侧维护两套实现。
- `policy/ACT/detr/models/backbone.py` 增加 `tactile_encoder_type: original|detail_v1`；默认值为 `original`。
- `feat` 模式中两种 encoder 均输出 `[N,512,1,1]`；`full` 模式中两种 encoder 默认输出 layer4 `[N,512,8,8]`（256×256 policy 输入）。
- `policy/ACT/train_config_detail_v1.yml` 基于官方 ACT 配置建立，保持 policy 超参数不变，仅设置 `tactile_encoder_type: detail_v1`；`tactile_ckpt` 暂设为 `null`，待 encoder V1 正式训练产生并核验 checkpoint 后填写，避免误用不存在的旧路径。
- `policy/ACT/tactile_backbone_smoke_test.py` 覆盖 Original/V1、feat/full、旧 policy tactile state key/shape 和 V1 strict reload。

此前按授权从 Original 项目复制到 details 的 ACT 最小源文件均使用 `cp -p`，逐个通过 `cmp` 与原文件核对；没有修改 Original 项目，也没有恢复整个缺失的 policy/scripts 树。已有的 checkpoint、dataset、eval 文件和其余历史 deletion 均未删除、清理或加入本次提交。

### 是否改变 encoder 输入输出接口

没有改变 encoder 输入 `[N,3,H,W]` 或 latent `[N,512]` 接口。decoder 仍接收 `[N,512]`。ACT 的 `feat`/`full` 外部 feature 接口保持原样。

### Checkpoint compatibility

- Original encoder checkpoint 仍由共享 `Tactile` 实现按原 state-dict layout 读取；V1 采用不同的 trunk/state keys，不对旧 encoder checkpoint 静默兼容。
- 旧官方 policy checkpoint 的 tactile 子模块实际包含 103 个参数键；与 `original + feat` 的 policy backbone 逐键、逐 shape 比较全部通过。
- V1 policy backbone 的内存 state dict 使用 `strict=True` 保存/重载通过。
- checkpoint 加载路径存在时使用 `strict=True` 并打印 status；路径不存在时明确打印 `Tactile checkpoint not found`。未使用 `strict=False` 隐藏 missing/unexpected keys。

### 验证方法与结果

使用 Isaac Kit Python 做 CPU-only smoke test（未启动训练/eval、未占用 GPU）：

```bash
BASE=/usr1/home/s126mdg41_04
ISAAC=$BASE/isaacsim-4.5.0
set +u
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate UniVTAC
source "$ISAAC/setup_conda_env.sh"
set -u
cd "$BASE/UniVTAC details"
"$ISAAC/kit/python/bin/python3" policy/ACT/tactile_backbone_smoke_test.py
```

结果：

- `original/feat`: feature `[2,512,1,1]`，position `[1,512,1,1]`
- `detail_v1/feat`: feature `[2,512,1,1]`，position `[1,512,1,1]`
- `original/full`: feature `[2,512,8,8]`，position `[1,512,8,8]`
- `detail_v1/full`: feature `[2,512,8,8]`，position `[1,512,8,8]`
- 原有 policy checkpoint tactile key/shape compatibility：`103 keys` 全部通过
- V1 policy checkpoint reload：`<All keys matched successfully>`
- 最终输出：`POLICY_SMOKE_TEST_PASSED`

其中 position batch 维为 1 是当前 UniVTAC `PositionEmbeddingSine` 的既有实现，ACT 通过广播使用；测试已按实际接口记录，未擅自修改 position encoding。

### GPU / 训练与 eval 状态

本次为 CPU-only code validation，GPU 记为 `N/A`；没有运行训练或 evaluation。此前 GPU0 上已有他人/既有 eval 进程，本次未触碰。任何 short training 或 official eval 开始前必须重新执行 `nvidia-smi`，按本日志前述规则重新选择 GPU，并记录当时剩余显存。

### 存在的问题

当前 details 工作树在本任务开始前就存在大量 tracked deletion 及若干 untracked 数据/结果目录；本次没有清理或恢复这些无关内容。Git 提交只包含上述 policy 适配、V1 配置和 smoke test，未包含 checkpoint 或数据。

### 回退方法

- 回退 policy 适配：对 `1916061` 使用 Git revert，或切换到前一 commit；不执行 reset/clean。
- 回退 V1 policy 配置：对 `818b4bd` 使用 Git revert。
- 回退 smoke test：对 `0737aad` 使用 Git revert。
- 保留并可对照的稳定点为 `c692392`；Original baseline 仍可显式使用 `encoder_type=original`，不依赖 Git 回退。

### 下一步计划

1. 在正式训练前核对 encoder training 的工作目录、输出路径和实际 checkpoint 格式。
2. 先运行 `nvidia-smi`，做合规的极短 baseline/V1 training smoke，记录 GPU、显存、config 和 checkpoint。
3. 通过 smoke 后再分别进行 baseline 与 V1 encoder training；policy architecture 保持不变。
4. policy training 与 official evaluation 前再次完整阅读官方 eval 文档，并严格复用其中的环境、checkpoint、simulator、seed 和 episode 设置。

## 2026-09-04 16:48 +08:00：GPU mini-step smoke 与真实数据入口审计

### 当前 Git branch / commit

`exp/detail-preserving-v1` / `a1d39ab`（日志更新前）。本次只补充验证记录，没有修改模型或数据管线。

### GPU 状态与实验 command

启动前执行 `nvidia-smi`。GPU2 状态为：总显存 `24564 MiB`，已用 `6410 MiB`，剩余 `17837 MiB`，GPU utilization `76%`；已有 PID `473862` 使用约 `5852 MiB`，未杀进程、未修改其他任务。按 encoder training 规则，GPU2 满足剩余显存大于 12 GB 的门槛。

实验使用：

```bash
CUDA_VISIBLE_DEVICES=2 \
  /usr1/home/s126mdg41_04/isaacsim-4.5.0/kit/python/bin/python3 <CPU/GPU mini-step script>
```

脚本对 `original` 与 `detail_v1` 各执行一次合成 tactile batch 的 `reconstruct → MSE loss → backward → Adam step`，并在内存中用 `torch.save`/`load_state_dict(strict=True)` 做 checkpoint round-trip。该实验不是实际数据训练，也不产生 reconstruction 或 manipulation 结果。

### 结果

- Original：loss `0.0958312377`，梯度 finite，strict reload 通过，峰值显存约 `529.3 MiB`。
- Detail V1：loss `0.0919276103`，梯度 finite，strict reload 通过，峰值显存约 `542.7 MiB`。
- 输出：`GPU_TRAINING_SMOKE_PASSED`。

显存差异约 `13.4 MiB`，但这是 batch=2、仅 `marked_rgb` decoder、单步合成输入的 code smoke，不能外推正式训练成本。

### 真实数据入口审计

- `encoder/train.py` 当前硬编码查找 `../data/contact-gs/<prism>/hdf5`，在 details 中找到 `0` 个文件。
- details 现有数据目录为 `data/insert_HDMI/clean` 和 `data/lift_bottle/clean`，每个目录目前有 100 个 HDF5 文件。
- 现有 HDF5 的 tactile group 是 `tactile/left_gsmini` 与 `tactile/right_gsmini`，而 `encoder/dataloader.py` 使用 `left_tactile/right_tactile`；`lift_bottle` 的 actor key 也不是 `actor/prism`。
- 只读试读 `insert_HDMI` 时，当前 dataloader 在旧 key 上触发 `KeyError`。因此暂不修改 dataloader、数据 schema 映射、输入/target resize 或训练入口；否则会同时改变 baseline 的数据处理，污染第一版 encoder 对比。

### 当前结论与阻塞边界

模型与 policy 接口已经完成 code-level validation；真实 reconstruction training 尚未开始。下一步若要取得真实 encoder training result，需要单独决定并记录一个兼容数据源/数据适配方案，同时先让 Original baseline 在同一数据处理下跑通。该问题属于既有数据管线，不把 GPU mini-step 或现有 checkpoint 当作正式 baseline result。

### 回退方法

本次没有持久化模型/数据文件，也没有产生需删除的服务器 artifact。代码回退点仍为 `c692392`；结果记录可通过 Git revert 回退日志 commit。

## 2026-09-04 16:50 +08:00：Policy encoder checkpoint loader 集成验证

### 验证方法

使用 details 现有非空 checkpoint：

`/usr1/home/s126mdg41_04/UniVTAC details/checkpoints/encoder.pth`

只读构造 `TactileBackbone(tactile_encoder_type="original", tactile_type="feat")`，由 policy-side adapter 实际执行 `torch.load(..., weights_only=True)` 与 `load_state_dict(strict=True)`，随后用 `[1,3,256,256]` dummy tactile image 做 forward。

### 结果

- loader 输出：`<All keys matched successfully>`。
- feature shape：`[1,512,1,1]`。
- position shape：`[1,512,1,1]`。
- 输出：`POLICY_ENCODER_CHECKPOINT_LOAD_PASSED`。

该验证没有写入或覆盖 checkpoint；`checkpoints/encoder.pth` 仍由 `.gitignore` 排除，不上传到 DT。V1 checkpoint 尚未产生，因为真实 encoder training 被上一节记录的数据入口不匹配阻塞。

## 2026-09-04 16:52 +08:00：ACT 启动上下文 import 验证

### 验证方法与结果

在官方脚本式工作目录 `policy/ACT` 下运行 Isaac Kit Python，成功 import：

- `act_policy.ACTPolicy`
- `imitate_episodes.make_policy`
- `detr.models.detr_vae.build`

输出：`ACT_SCRIPT_CONTEXT_IMPORT_PASSED`。从 details 项目根目录将 `imitate_episodes` 作为 package import 会因原始源码的 `from utils import load_data` 失败；这是既有脚本启动约束。未做无关 import 重构，后续训练按 `policy/ACT` 的官方工作目录执行。

### Evaluation 备注

`scripts/eval_policy.py` 在 import 时会解析命令行并启动 Isaac App，因此不采用普通 package import 作为验证方式；正式 evaluation 仍必须按官方文档给出的脚本命令、环境初始化和 GPU 规则执行。

## 2026-09-04 16:58 +08:00：配置化 encoder 数据 schema adapter 修改前计划

### 当前稳定 rollback point

- branch：`exp/detail-preserving-v1`
- commit：`2cc34a8 docs: record act import context`
- 当前 details 工作树仍保留任务开始前的历史 deletion、1 个既有 modified 文件和 untracked 数据/配置；本次只对下列明确文件做修改，不执行 reset/clean。

### 修改目标

让 encoder reconstruction 能在当前 details 已有的 HDF5 数据上进行可复现的 baseline/V1 公平对比，同时不改变原有 legacy 数据入口的默认行为。

### 计划修改文件

- `encoder/dataloader.py`：增加 schema 配置。保留默认 `legacy_contact_gs`；新增显式 `gsmini`，映射 `left_gsmini/right_gsmini`，自动选择 `actor/prism` 或 `actor/bottle`，并在新 schema 下把 RGB/depth target 对齐到 decoder 的 256×256 输出。
- `encoder/train.py`：增加 `--schema`、`--data_root`、`--image_size`、`--epochs`、`--batch_size`、`--num_workers` 参数；旧参数默认值与旧路径逻辑保持一致；修复空/单 batch 时的平均 loss 除数问题；输出路径继续按 encoder type 隔离。
- `encoder/data_adapter_smoke_test.py`：只读验证当前 `insert_HDMI` HDF5 可被新 schema 解析，检查样本 key、shape、dtype。
- `VALIDATION_WORKFLOW.md`：记录从 code smoke、encoder retraining、policy retraining 到 official eval 的完整流程和 checkpoint 判定规则。
- `MODIFICATION_LOG.md`：记录实际结果、数据边界和 GPU/command。

### 不在本次范围内

- 不修改 decoder、ACT Transformer、action head、temporal module 或 policy action semantics。
- 不把 `gsmini` 数据强行映射成 policy 训练所需的 `/action` 与 `/observations` schema；policy 数据适配另行审计。
- 不覆盖现有 `checkpoints/encoder.pth`，不删除数据/log/checkpoint，不修改 Original 项目。

### 预期风险与回退

- legacy 默认路径必须保持原样；新训练只能显式使用 `--schema gsmini`。
- 新 schema 的 256×256 resize 是为匹配当前 decoder 接口；Original 与 V1 必须使用完全相同 schema、resize、split seed 和 loss weights。
- 如 smoke 或训练失败，保留 commit，使用 Git revert 回退本次文件级修改；不删除失败结果，不重写 history。

## 2026-09-04 17:10 +08:00：配置化 encoder data adapter 与验证流程文档完成

### 当前 Git branch / rollback point

- branch：`exp/detail-preserving-v1`
- 当前代码 commit（日志提交前）：`d16335d docs: add encoder validation workflow`
- 本次修改前 rollback point：`333741f docs: plan configurable data schema adapter`

### 修改目标与实际变化

按授权在 details 内增加配置化 encoder 数据入口：

- `encoder/dataloader.py` 增加 `legacy_contact_gs` 与 `gsmini` schema。
- 默认仍为 `legacy_contact_gs`，旧的 `left_tactile/right_tactile` 和 `../data/contact-gs` 查找逻辑不被静默替换。
- `gsmini` 显式支持当前数据的 `left_gsmini/right_gsmini`，自动解析 `actor/prism` 或 `actor/bottle` pose。
- `gsmini` 默认将 RGB 和 depth target 对齐为 256×256，匹配现有 decoder 输出；Original/V1 使用同一处理时仍可公平对比。
- `encoder/train.py` 增加 `--schema`、`--data_root`、`--image_size`、`--epochs`、`--batch_size`、`--num_workers`、`--seed`。
- 修正 train/validation loss 在 batch 数量很小时的平均值除数；不改变模型结构。
- 增加 `encoder/data_adapter_smoke_test.py`。
- 增加 `VALIDATION_WORKFLOW.md`，说明何时需要重训 encoder、policy 和 official eval 的完整顺序。

没有修改 decoder、ACT policy architecture、action semantics、Original 项目或现有 checkpoint；没有删除服务器文件。

### Data adapter 验证

结果：

- `insert_HDMI/clean`：100 files，单文件 236 samples，样本 shape 为 RGB `[3,256,256]`、marked RGB `[3,256,256]`、depth `[1,256,256]`、marker `[63,2]`、pose `[7]`。
- `lift_bottle/clean`：100 files，单文件 622 samples，shape 同上。
- 缺失 legacy root 时显式报 `FileNotFoundError`，不会静默使用另一套数据。
- 输出：`DATA_ADAPTER_SMOKE_PASSED`，py_compile 通过。

### 真实数据 short training smoke

启动前 `nvidia-smi`：GPU2 总显存 `24564 MiB`，已用 `23 MiB`，剩余 `24224 MiB`，utilization `0%`；没有占用或杀掉其他进程。

两个运行使用完全相同的：

- `schema=gsmini`
- `data_root=../data/insert_HDMI/clean`
- `data_num=8`
- `epochs=1`
- `batch_size=2`
- `num_workers=0`
- `seed=42`
- 默认 loss weights

结果：

- Original validation loss：`0.263397`。
- Original checkpoint：`encoder/ablation/data_8/resnet18/20260904-170250/best.pth`。
- Detail V1 validation loss：`0.266245`。
- Detail V1 checkpoint：`encoder/ablation/data_8/detail_v1/resnet18/20260904-170312/best.pth`。

两者均完成真实 HDF5 读取、train/validation loop 和 checkpoint 保存；checkpoint 分目录隔离，未覆盖 `checkpoints/encoder.pth`。这些是 8-sample/1-epoch code smoke，不能当作性能结论。

### 验证文档结论

`VALIDATION_WORKFLOW.md` 已写明：

- 代码 smoke 不需要重训。
- V1 reconstruction 对比需要重新训练 V1 encoder。
- 为了公平，Original 也应在同一新 schema、resize、seed、loss 和训练预算下重跑；旧 checkpoint 只作为兼容性参考。
- policy 必须在 encoder checkpoint 准备好后，对 Original/V1 使用同一个 ACT architecture 分别训练。
- official eval 不重新训练，只加载 policy checkpoint，并且必须再次阅读官方 eval 文档、选择空闲 GPU。
- 当前 policy dataset 仍是独立阻塞：`SIM_TASK_CONFIGS.json` 指向不存在的 `data/sim-lift_bottle/demo-50`，raw HDF5 不能未经语义验证直接改成 ACT action schema。

### 回退方法

- data adapter：对 `ba084fa` 使用 Git revert，或回到 `333741f` 后建立文件级回退 commit。
- training CLI：对 `7538e5c` 使用 Git revert。
- smoke test：对 `d280340` 使用 Git revert。
- validation document：对 `d16335d` 使用 Git revert。
- 不执行 reset/clean，不删除上述 smoke checkpoint、数据或日志。

## 2026-09-04 17:35 +08:00：Insert HDMI 第一任务切换修改前计划

### 当前 Git branch

- `exp/detail-preserving-v1`

### 当前稳定 commit / rollback point

- `ded9666e137d001284170c14048db8b0341cdf4b`
- 本次修改失败时保留修改历史，并以该 commit 为回退参照；不使用 `git reset --hard`、`git clean` 或删除实验产物。

### 修改目标

将第一阶段正式 downstream 验证任务从 `lift_bottle` 调整为 `Insert HDMI`。原因是已有 UniVTAC lift bottle 结果的成功率较高，存在 ceiling effect；Insert HDMI 更适合作为第一项检验 Detail-Preserving Encoder V1 是否能保留并利用局部 tactile spatial detail 的任务。

### 调查事实与研究判断

- ACT 所需的 lift bottle 数据位于原始项目的 `policy/ACT/data/sim-lift_bottle/demo-50`，不在项目根目录 `data/`。
- ACT 所需的 Insert HDMI 数据位于原始项目的 `policy/ACT/data/sim-insert_HDMI/demo-50`，包含 50 个 HDF5 episode，约 4.7 GB。
- Insert HDMI HDF5 已确认包含 `/action`、`/observations/qpos`、`/observations/images/cam_high`、`tac_left`、`tac_right`，可直接作为 ACT loader 的输入。
- `details/data/insert_HDMI/clean` 是 encoder reconstruction 使用的 raw tactile schema，不等同于 ACT episode 数据；本次不做未经验证的 schema 改名或复制。

### 准备修改的文件

- `policy/ACT/SIM_TASK_CONFIGS.json`：保留原有 lift bottle 条目，新增 `sim-insert_HDMI-demo-50`；配置引用服务器上原始项目的 ACT 数据目录，只读使用，不复制数据。
- `VALIDATION_WORKFLOW.md`：将 Insert HDMI 写为第一项正式对比任务，明确 encoder 是否需要重训、policy 是否需要重训、official eval 只做推理，以及 lift bottle 的次要对照定位。
- `MODIFICATION_LOG.md`：记录任务选择、数据路径、schema 检查、验证命令和后续结果。

### 不在本次修改范围内

- 不修改原始 `/usr1/home/s126mdg41_04/UniVTAC` 项目。
- 不复制、删除或覆盖任何 dataset、checkpoint、log 或 simulation output。
- 不修改 decoder、ACT policy architecture、action semantics、temporal module 或 simulator。
- 不启动正式 policy training 或 official evaluation；先完成配置解析和数据 loader smoke。

### 预期验证与风险

- 先确认 JSON 可解析、Insert HDMI 50 个 episode 可访问、代表性 HDF5 keys/shapes 正确，并能被 ACT 的 `TacArenaDataset` 读取。
- 训练前仍需重新执行 `nvidia-smi`；policy training 要求剩余显存大于 12 GB，official evaluation 尽量使用真正空闲 GPU。
- 绝对路径依赖当前服务器环境，因此文档会明确指出该配置是 mlda2 上的只读数据引用，不代表 DT 仓库携带数据。

### 回退方法

- 本次修改前稳定点为 `ded9666e137d001284170c14048db8b0341cdf4b`。
- 通过保留 lift bottle 条目、单独提交 Insert HDMI 配置和文档，必要时使用文件级 revert 回退；不删除已有数据或 checkpoint。

## 2026-09-04 17:55 +08:00：Insert HDMI 任务配置与 ACT 数据验证完成

### 当前 Git branch / commit

- branch：`exp/detail-preserving-v1`
- 修改前 commit：`84d4b28 docs: plan insert hdmi validation task`
- 本次配置和文档修改尚未覆盖任何已有 checkpoint 或数据目录。

### 修改目标

将第一项 downstream 验证固定为 `sim-insert_HDMI-demo-50`，并使验证文档回答清楚：代码 smoke 不需要重训；正式 reconstruction 对比需要 Original/V1 成对重训 encoder；policy 需要在相同 ACT 数据和预算下分别训练；official evaluation 不重训，只加载对应 policy checkpoint。

### 修改的文件

- `policy/ACT/SIM_TASK_CONFIGS.json`
  - 保留 `sim-lift_bottle-demo-50`；将其路径修正为原始项目中实际存在的只读 ACT 数据路径。
  - 新增 `sim-insert_HDMI-demo-50`，指向：
    `/usr1/home/s126mdg41_04/UniVTAC/policy/ACT/data/sim-insert_HDMI/demo-50`
- `VALIDATION_WORKFLOW.md`
  - 将 Insert HDMI 设为第一项正式对比任务。
  - 补充 ACT HDF5 schema、数据边界、loader smoke、policy training 和 task-specific official eval 命令。
  - 明确 lift bottle 只作为后续/辅助对照。

### 数据验证结果

- 配置 JSON 解析通过。
- lift bottle 与 Insert HDMI 两个 ACT 数据目录均实际存在，各有 50 个 `episode_*.hdf5`。
- Insert HDMI 代表性 episode：`action [117,8] float32`、`observations/qpos [117,8] float32`、`cam_high [117,270,480,3] uint8`、`tac_left/tac_right [117,240,320,3] uint8`。
- 使用 `UniVTAC` Conda Python 进行 CPU-only `TacArenaDataset` smoke：输出 `cam [1,3,256,256]`、tactile `[2,3,256,256]`、qpos `[8]`、action chunk `[50,8]`、padding `[50]`，结果 `ACT_DATASET_SMOKE_PASSED`。
- 直接使用 Isaac Kit 裸 `python3` 做该 loader smoke 时因其环境没有 `numpy` 而失败；这是解释器选择问题，不是数据失败。后续纯数据/loader 检查使用 `UniVTAC` Conda Python；涉及 simulator 的 official eval 仍按官方文档使用 Isaac Kit 环境。

### GPU / command / checkpoint

- GPU：N/A；本次只读配置和 CPU 数据 loader 验证，没有训练或 evaluation。
- 验证命令：在 `details/policy/ACT` 下使用 `/usr1/home/s126mdg41_04/miniconda3/envs/UniVTAC/bin/python` 解析 `SIM_TASK_CONFIGS.json`、检查两个目录和 HDF5 keys/shapes，并读取 Insert HDMI 的 `TacArenaDataset` 样本。
- checkpoint：N/A；没有覆盖或生成 checkpoint。

### 是否改变 encoder / downstream 接口

- 没有改变 encoder、decoder、ACT policy architecture 或 simulator 接口。
- 只增加/修正 task dataset config 和验证文档；Insert HDMI policy dataset 仍在原始项目中只读引用，不复制到 DT、不提交 GitHub。

### 当前结论与下一步

ACT 所需的 `data/sim-insert_HDMI/demo-50` 已确认存在于原始 UniVTAC 的 `policy/ACT/data` 下，不需要去官网下载。下一步先重新执行 GPU 状态检查，再进行同一 Insert HDMI raw data、同一训练预算下的 Original/V1 encoder reconstruction training；只有两套 encoder checkpoint 都完成并通过严格加载后，才进入成对 ACT policy training。正式 eval 前必须再次完整阅读 lift bottle official eval 文档并按 Insert HDMI 的 task/checkpoint 替换参数。

### 回退方法

- 配置/文档修改可通过文件级 revert 回退到 `84d4b28`。
- 若需要完全回到任务切换前的稳定代码，回退参照仍为 `ded9666`；不删除数据、checkpoint 或日志。

## 2026-09-04 18:02 +08:00：Insert HDMI 正式 encoder training 首轮 OOM

### 实验配置

- branch：`exp/detail-preserving-v1`
- commit：`5b4d177`
- task/data：Insert HDMI；`schema=gsmini`；`../data/insert_HDMI/clean`；100 个 HDF5 文件；随机抽取 1000 samples；image/depth target `256×256`；seed `42`。
- Original 与 Detail V1：均为 5 epochs、batch size 64、num_workers 8、lr `1e-3`、相同 reconstruction loss weights。

### GPU 与命令

- GPU0：启动时约 24,230 MiB free，Original 进程 PID 494589。
- GPU1：启动时约 24,229 MiB free，Detail V1 进程 PID 494590；由于 `CUDA_VISIBLE_DEVICES=1`，PyTorch 报错中的 logical GPU 0 对应物理 GPU1。
- 两张卡启动时均无 compute process；GPU2/3 上已有其他用户任务，未触碰。

### 结果与分类

- 两个进程均完成数据 metadata 读取并进入 epoch 1，但在第一个 reconstruction batch 的 decoder convolution 触发 `torch.OutOfMemoryError`。
- 报错需要额外约 2.00 GiB，进程当时约使用 22.5 GiB；没有生成 `best.pth` 或 epoch checkpoint。
- 这是 `B. Training failure / OOM`，不是 `C. Research failure`；不能据此判断 Original/V1 的重建效果。
- 已保留两个空的外层 run log 和包含 OOM traceback 的 training log，未删除任何文件。

### 重试计划

- 仅将共同的 `batch_size` 从 64 调整为 16，其他变量保持不变；Original/V1 继续分别使用 GPU0/GPU1 和独立输出目录。
- 重试前重新执行 `nvidia-smi`。若 batch size 16 仍 OOM，再停止并记录，不自动继续降低或修改网络。

### 回退方法

- 本轮没有代码结构修改；OOM 运行不会覆盖既有 checkpoint。
- 后续如需回看首轮状态，使用本条记录和 commit `5b4d177`；不删除 OOM 日志。

## 2026-09-04 18:08 +08:00：Insert HDMI Original/V1 encoder training batch16 完成

### 实验配置

- branch：`exp/detail-preserving-v1`
- code commit：`6cf02b6`（本次只运行既有代码，没有 architecture 修改）
- task/data：Insert HDMI；`schema=gsmini`；`../data/insert_HDMI/clean`；100 个 HDF5 文件；随机抽取 1000 samples；RGB/depth target `256×256`；seed `42`。
- Original 与 Detail V1 使用完全相同的 5 epochs、batch size 16、num_workers 8、lr `1e-3` 和 loss weights：`marked_rgb=1.0`、`rgb=1.0`、`depth=0.5`、`marker=0.5`、`pose=0.5`。
- batch size 16 是在 batch64 首轮 OOM 后的共同资源修正；两者仍保持相同设置，可进行成对比较。

### GPU 与命令

- Original：物理 GPU0；启动前 `24,230 MiB` free；观察到约 `9,792 MiB` used；外层 PID `658633`。
- Detail V1：物理 GPU1；启动前 `24,229 MiB` free；观察到约 `9,837 MiB` used；外层 PID `658634`。
- GPU2/3 当时由其他用户任务使用，未触碰；没有 kill 或修改其他进程。
- 两个命令分别为：
  - `CUDA_VISIBLE_DEVICES=0 ... train.py all 1000 --encoder_type original --schema gsmini --data_root ../data/insert_HDMI/clean --epochs 5 --batch_size 16 --num_workers 8 --seed 42`
  - `CUDA_VISIBLE_DEVICES=1 ... train.py all 1000 --encoder_type detail_v1 --schema gsmini --data_root ../data/insert_HDMI/clean --epochs 5 --batch_size 16 --num_workers 8 --seed 42`

### checkpoint 与结果

- Original checkpoint directory：`encoder/ablation/data_1000/resnet18/20260904-175305/`
- Original `best.pth`：165,482,707 bytes；SHA256 `830ff60ceb6cab1535aee81140244be8e93cfc1ef1c27165e0e65fbc56018c2c`
- Original best validation loss：`0.003893`；训练过程无 NaN/OOM。
- Detail V1 checkpoint directory：`encoder/ablation/data_1000/detail_v1/resnet18/20260904-175305/`
- Detail V1 `best.pth`：167,132,311 bytes；SHA256 `624ce147b2b8beebb09fd5bb1ebbce6578f452e937f30ca65df64e3120ac5555`
- Detail V1 best validation loss：`0.004047`；训练过程无 NaN/OOM。
- 两个目录均生成 `ep0.pth`–`ep4.pth` 和 `best.pth`，没有覆盖既有 `checkpoints/encoder.pth`。

### 严格加载验证

- Original 和 Detail V1 均使用对应 encoder type、完整 supervision decoder 配置和 `strict=True` 加载通过，输出 `All keys matched successfully`。
- 完整训练模型参数量：Original `41,339,340`；Detail V1 `41,750,220`。
- 该参数量差异来自 V1 detail branch/head；latent dimension 仍为 512，decoder interface 未改变。

### 结果解释与下一步

本轮只说明两种 encoder 都能在 batch16 下稳定完成 5-epoch reconstruction training，并得到可严格加载的 checkpoint。当前 V1 validation loss 略高于 Original，不能据此否定 spatial-detail hypothesis；需要在相同 Insert HDMI ACT 数据、相同 policy architecture 和训练预算下进行 downstream policy 对比。下一步是先用这两份 checkpoint 做 policy-side backbone/load smoke，再分别训练 Original/V1 ACT policy；正式 evaluation 前重新阅读 official eval 文档并重新检查空闲 GPU。

### 回退方法

- 训练结果是新增独立目录；失败或不采用时保留 checkpoint 和日志，不删除文件。
- 代码回退参照为 `6cf02b6` 之前的稳定 details commit；本轮无代码改动，不需要 reset/clean。
