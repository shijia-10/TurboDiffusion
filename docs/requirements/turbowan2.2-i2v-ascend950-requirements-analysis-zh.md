# TurboWan2.2 I2V Ascend 950 适配需求分析报告

## 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档名称 | TurboWan2.2 I2V Ascend 950 适配需求分析报告 |
| 文档版本 | V1.0 |
| 编写日期 | 2026-09-15 |
| 代码基线 | `TurboDiffusion-official` `main`，基线提交 `cd0bd03` |
| 目标模型 | TurboWan2.2-I2V-A14B-720P |
| 目标硬件 | Ascend 950 系列 NPU |
| 文档状态 | 待评审 |

## 1. 编写目的

本文档定义 TurboWan2.2 I2V 在 Ascend 950 系列 NPU 上运行所需的功能、性能、精度、兼容性和验收要求。文档回答以下问题：

1. 本次适配要解决什么问题；
2. 哪些能力属于本期交付范围；
3. 单卡和八卡分别必须达到什么结果；
4. 如何判断功能、精度和性能是否通过；
5. 当前实现与最终目标之间还存在哪些差距。

本文档不描述具体代码实现细节。实现结构、数据流和模块接口见《TurboWan2.2 I2V Ascend 950 功能设计说明书》。

## 2. 项目背景

TurboDiffusion 已提供 TurboWan2.2 I2V 的推理代码和 High-noise、Low-noise 两阶段 DiT 权重，但原始实现及其依赖主要面向 CUDA/GPU 生态。目标环境使用 Ascend 950 系列 NPU，需要处理设备初始化、算子替换、数据布局、混合精度、模型加载、显存管理以及多卡通信等差异。

本项目不是重新实现 Wan2.2，也不是重新训练模型，而是在保留 TurboWan 推理语义的前提下：

- 让非量化 TurboWan2.2 I2V 在 Ascend 950 上稳定运行；
- 接入 MindIE-SD 已有的 Dense Attention、Sparse Linear Attention 和 Fast LayerNorm 能力；
- 支持单卡推理和单机八卡 Ulysses 序列并行；
- 在标准 720P、81 帧场景下达到目标性能；
- 保持生成视频可用，且不存在明显精度异常。

## 3. 术语和缩写

| 术语 | 含义 |
| --- | --- |
| I2V | Image-to-Video，输入首帧图片和文本提示词生成视频 |
| DiT | Diffusion Transformer，本项目中的主要去噪网络 |
| High-noise model | 负责高噪声时间步的 TurboWan DiT |
| Low-noise model | 负责低噪声时间步的 TurboWan DiT |
| SLA | Sparse Linear Attention，MindIE-SD 提供的稀疏注意力实现 |
| Dense Attention | 不做稀疏筛选的完整注意力计算 |
| BNSD | Attention 张量布局 `[Batch, NumHeads, Sequence, HeadDim]` |
| Ulysses | 通过 All-to-All 在序列维与 Head 维之间交换切分方式的序列并行方案 |
| HCCL | Ascend 集合通信后端 |
| Warmup | 正式计时前执行模型前向，使算子编译、缓存和运行时状态稳定 |
| `s/it` | 进度条显示的平均每个正式采样迭代耗时 |

## 4. 需求目标

### 4.1 总体目标

在 Ascend 950 系列 NPU 上完成 TurboWan2.2-I2V-A14B-720P 的 BF16 非量化推理适配，使同一套推理入口同时支持：

- 单卡生成视频；
- 单机八卡 Ulysses 共同生成一个视频；
- MindIE SLA Self Attention；
- MindIE Dense Attention 回退和对照路径；
- High/Low 两阶段模型切换；
- 720P、81 帧、4 步 ODE 推理；
- 正式推理前固定 Warmup；
- 生成 MP4 文件并正常退出。

### 4.2 关键成功指标

| 编号 | 指标 | 目标 |
| --- | --- | --- |
| KPI-01 | 单卡功能 | 成功生成可播放的 720P、81 帧视频 |
| KPI-02 | 八卡功能 | 8 个 rank 协同完成推理，仅 rank 0 保存视频，所有 rank 正常退出 |
| KPI-03 | 单卡性能 | 正式采样进度条平均耗时 `< 10.5 s/it` |
| KPI-04 | 八卡性能 | 正式采样进度条平均耗时 `≤ 1.6 s/it` |
| KPI-05 | 精度 | 人工检查无明显黑屏、纯噪声、严重闪烁、主体崩坏或首帧条件失效 |
| KPI-06 | 权重 | 使用 BF16 非量化 High/Low checkpoint，不依赖 `-quant` 权重 |

## 5. 范围定义

### 5.1 本期范围

1. 模型范围：Wan2.2 I2V A14B，不扩展其他 Wan 模型；
2. 设备范围：Ascend 950 系列 NPU；
3. 精度范围：BF16 非量化推理；
4. 并行范围：单卡和单机八卡 Ulysses；
5. Attention 范围：`sla` 和 `original`；
6. 输出范围：单视频 MP4；
7. 性能范围：正式采样迭代耗时；
8. 精度验收：当前以人工同条件对比为主；
9. 运行方式：命令行脚本和 Python 推理入口。

### 5.2 非本期范围

- Linear INT8 或其他量化推理；
- Tensor Parallel；
- Ulysses 与 Tensor Parallel 混合并行；
- 多机推理；
- 训练、微调或权重转换流程改造；
- T2V、VACE 或其他模型任务；
- 服务化并发、请求调度和生产部署；
- VBench 自动化精度验收；
- 端到端启动耗时指标；
- VAE 编解码性能指标。

说明：代码中可能保留历史量化或服务化参数，但它们不属于本期交付和验收对象。

## 6. 用户和使用场景

### 6.1 主要用户

- 模型适配开发人员：进行 NPU 算子适配和性能优化；
- 测试人员：执行单卡、八卡功能与性能验收；
- 算法人员：人工检查输出视频精度；
- 维护人员：根据日志定位设备、权重、通信和算子问题。

### 6.2 核心场景

#### 场景 A：单卡 SLA 推理

用户指定一张物理 NPU、输入图片、提示词和权重目录，运行单卡脚本，程序完成文本编码、图片编码、四步采样和视频保存。

#### 场景 B：单卡 Dense 对照推理

用户将 `ATTENTION_TYPE` 设置为 `original`，使用同一套 TurboWan checkpoint 运行 MindIE Dense Self Attention，用于功能回退、精度对照和性能分析。

#### 场景 C：八卡 Ulysses 推理

用户通过 `torchrun --nproc_per_node=8` 启动 8 个进程。每个进程绑定一个本地 NPU，使用 HCCL WORLD 组执行 Ulysses Self Attention，共同生成一个视频。

#### 场景 D：性能验收

用户在固定输入、固定 seed、固定模型和固定环境下运行内置 Warmup，读取正式 `Sampling` 进度条最终显示的平均 `s/it`，与单卡、八卡指标比较。

## 7. 功能需求

### 7.1 NPU 环境和设备管理

| 编号 | 需求 |
| --- | --- |
| FR-001 | 程序应检测 `torch_npu` 和可用 Ascend NPU；依赖缺失或设备不可用时给出明确错误。 |
| FR-002 | 单卡模式应根据 `--device_id` 绑定逻辑 NPU，并统一使用 BF16 tensor placement。 |
| FR-003 | 多卡模式应读取 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`，将每个进程绑定到对应本地 NPU。 |
| FR-004 | 多卡模式应校验 `ulysses_size == WORLD_SIZE`，避免进程数与并行度不一致。 |
| FR-005 | 多卡通信后端固定使用 HCCL，不要求用户额外传递 backend 参数。 |

### 7.2 模型和权重加载

| 编号 | 需求 |
| --- | --- |
| FR-010 | 程序应加载 High-noise 和 Low-noise 两个非量化 TurboWan DiT checkpoint。 |
| FR-011 | 程序应加载 umT5-XXL 文本编码器、对应本地 tokenizer 和 Wan2.1 VAE。 |
| FR-012 | DiT 模型应先在 Meta device 上构造，再加载实际权重，降低初始化峰值。 |
| FR-013 | `sla` 模式应严格加载 SLA 所需的 `proj_l` 参数。 |
| FR-014 | `original` 模式应仅忽略 SLA 专属 `proj_l.weight/bias`，其他缺失或多余参数仍应报错。 |
| FR-015 | High/Low 模型应在正式采样期间同时驻留 NPU，模型切换仅切换引用，不执行 CPU/NPU 搬运。 |

### 7.3 文本和图像条件处理

| 编号 | 需求 |
| --- | --- |
| FR-020 | 程序应根据提示词生成 umT5 embedding，并在使用后释放 umT5 模型占用。 |
| FR-021 | 多卡模式下，各 rank 应对称执行 umT5 加载和前向，避免单 rank NPU 计算与其他 rank HCCL 等待发生冲突。 |
| FR-022 | 程序应读取输入图片并转换为 RGB。 |
| FR-023 | 自适应分辨率模式应保持目标面积近似不变，并根据输入图片宽高比计算输出尺寸。 |
| FR-024 | 程序应使用 VAE 编码首帧，并构造图像条件 mask 与 latent condition。 |
| FR-025 | 多卡模式应在采样前一次性同步文本和图像静态条件，避免每个 DiT forward 重复同步。 |

### 7.4 Attention 和归一化

| 编号 | 需求 |
| --- | --- |
| FR-030 | Self Attention 和 Cross Attention 的 Q/K/V 应采用 MindIE 所需的 BNSD 布局。 |
| FR-031 | `attention_type=sla` 时，仅将 Wan2.2 Self Attention 替换为 MindIE `SparseLinearAttention`。 |
| FR-032 | SLA 的 `topk` 应由 `--sla_topk` 控制，验收默认值为 `0.1`。 |
| FR-033 | `attention_type=original` 时，Self Attention 应使用 MindIE Dense Attention。 |
| FR-034 | Cross Attention 在 `sla` 和 `original` 两种模式下均应使用 MindIE Dense Attention。 |
| FR-035 | MindIE Fast LayerNorm 应通过进程启动前的 `FAST_LAYERNORM=1` 启用。 |
| FR-036 | RMSNorm 应保持输入 BF16 dtype，避免无意的 FP32 输出影响算子兼容性和性能。 |

### 7.5 采样流程

| 编号 | 需求 |
| --- | --- |
| FR-040 | 程序应支持 1～4 个蒸馏采样步，验收使用 4 步。 |
| FR-041 | 程序应根据 `boundary` 在 High-noise 和 Low-noise 模型之间切换。 |
| FR-042 | 程序应支持 ODE 推理，验收默认启用 `--ode`。 |
| FR-043 | 正式采样前，应分别对实际会使用的 High/Low 模型执行一次 Warmup。 |
| FR-044 | Warmup 不应出现在正式 `Sampling` 进度条统计中。 |
| FR-045 | rank 0 显示正式采样进度条；其他 rank 不应重复输出进度条。 |
| FR-046 | 相同 seed、输入、提示词和配置应具备可复现性。 |

### 7.6 Ulysses 多卡并行

| 编号 | 需求 |
| --- | --- |
| FR-050 | 八卡模式应使用 HCCL WORLD 组作为 Ulysses process group。 |
| FR-051 | 程序应保证视频 token 数可被 Ulysses size 整除；不能整除时应以最小面积增量对齐宽或高。 |
| FR-052 | DiT 输入应按 sequence 维切分到各 rank。 |
| FR-053 | 每层 Self Attention 应执行 `sequence shard/all heads → full sequence/head shard` 的输入 All-to-All。 |
| FR-054 | 本地 SLA 计算结束后，应执行反向 All-to-All，恢复 `sequence shard/all heads`。 |
| FR-055 | Cross Attention 不做 Ulysses All-to-All，直接在本地 sequence shard 上执行 Dense Attention。 |
| FR-056 | DiT head 输出应在各 rank 间 AllGather，恢复完整 latent 输出。 |
| FR-057 | 推理结束前所有 rank 应 barrier，并正常销毁进程组，避免 torchrun 异常退出。 |

### 7.7 输出处理

| 编号 | 需求 |
| --- | --- |
| FR-060 | 多卡模式下仅 rank 0 执行 VAE decode 和视频保存。 |
| FR-061 | 输出目录不存在时，运行脚本应自动创建。 |
| FR-062 | 输出文件应为可正常播放的 MP4，默认帧率为 16 FPS。 |
| FR-063 | 推理结束后应释放 High/Low 模型及 NPU cache。 |

## 8. 非功能需求

### 8.1 性能需求

标准性能场景固定为：

| 项目 | 配置 |
| --- | --- |
| 模型 | TurboWan2.2-I2V-A14B-720P High/Low |
| 精度 | BF16 非量化 |
| 输出 | 720P、81 帧 |
| 样本数 | 1 |
| 采样步数 | 4 |
| 采样器 | ODE |
| Attention | SLA |
| SLA top-k | 0.1 |
| Fast LayerNorm | 开启 |
| Warmup | High/Low 各一次，位于正式进度条之前 |

性能口径：

- 使用 tqdm `Sampling` 进度条完成后的平均 `s/it`；
- 不包含模型加载、umT5、VAE encode、Warmup、VAE decode 和视频写盘；
- Profiler 或强制同步环境变量开启时的数据不得作为验收数据；
- 建议同一环境连续运行至少 3 次，记录每次结果并取中位数；重复次数和统计方法在正式验收前确认。

### 8.2 精度需求

当前阶段采用人工验收：

1. 输入图片主体应在视频中可识别；
2. 视频内容应与提示词语义基本一致；
3. 输出不应为黑屏、纯色、随机噪声或无法解码文件；
4. 不应出现全局严重闪烁、颜色异常或明显数值爆炸；
5. High/Low 模型切换前后不应出现突发性画面崩坏；
6. 单卡与八卡在相同输入和 seed 下应保持可接受的一致性。

VBench 数据集已经具备，但自动化 VBench 评测暂不作为本期阻塞项，可在人工精度稳定后补充。

### 8.3 稳定性需求

- 单卡和八卡各连续运行不少于 3 次，不应出现随机崩溃；
- 不应出现 HCCL collective 参数不一致；
- 不应出现 rank 长时间停留在 UMT5 或 barrier；
- 不应出现 torchrun 在视频成功保存后仍异常退出；
- 错误必须包含 rank、设备或阶段信息，便于定位。

### 8.4 可维护性需求

- NPU 适配逻辑应集中在 Wan2.2 I2V 主流程和相关算子适配点；
- 不为本期不使用的 TP、量化或 backend 增加额外接口；
- 运行脚本应提供可直接复现的默认配置；
- 诊断功能优先使用系统 Profiler，不将临时同步计时作为正式 CLI 参数；
- 关键布局、dtype 和通信行为应有自动化测试保护。

## 9. 外部依赖和运行约束

### 9.1 软件依赖

- Ascend CANN；
- PyTorch 与 `torch_npu`；
- MindIE-SD；
- Triton-Ascend；
- TurboDiffusion Python 依赖；
- HCCL 单机通信环境。

具体兼容版本以交付镜像的软件清单为准，不在本文中写死版本号。

### 9.2 权重目录约束

建议目录结构：

```text
TurboWan2.2-I2V-A14B-720P/
├── TurboWan2.2-I2V-A14B-high-720P.pth
├── TurboWan2.2-I2V-A14B-low-720P.pth
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.1_VAE.pth
└── google/
    └── umt5-xxl/
        ├── special_tokens_map.json
        ├── spiece.model
        ├── tokenizer.json
        └── tokenizer_config.json
```

### 9.3 硬件和资源约束

- 八卡模式要求 8 张 NPU 同时可用；
- 每个 torchrun rank 对应一张本地 NPU；
- High/Low 两个 DiT 同时驻留时，每卡显存应满足峰值要求；
- 多 rank 对称加载 umT5 会增加主机内存和 checkpoint I/O 压力；
- 八卡性能受 HCCL 链路、CPU 绑核、任务队列和运行时版本影响。

## 10. 验收方案

### 10.1 单卡功能验收

执行：

```bash
bash scripts/inference_wan2.2_i2v_npu_single.sh
```

通过条件：

- 进程返回码为 0；
- High/Low Warmup 完成；
- 正式采样完成 4/4；
- 输出 MP4 存在且可播放；
- 人工精度检查通过；
- 平均耗时 `<10.5 s/it`。

### 10.2 八卡功能验收

执行：

```bash
bash scripts/inference_wan2.2_i2v_npu_8card.sh
```

通过条件：

- 8 个 rank 完成 NPU 与 HCCL 初始化；
- 不出现 collective shape、dtype、root rank 不一致；
- Warmup 和 4 步正式采样完成；
- 仅 rank 0 保存一个 MP4；
- 所有 rank barrier 后正常退出；
- 人工精度检查通过；
- 平均耗时 `≤1.6 s/it`。

### 10.3 Attention 回退验收

单卡设置：

```bash
ATTENTION_TYPE=original \
    bash scripts/inference_wan2.2_i2v_npu_single.sh
```

通过条件：同一套 TurboWan checkpoint 能够加载，Dense Attention 完成视频生成，且除 SLA 专属 `proj_l` 外仍保持严格权重校验。

### 10.4 自动化回归

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=turbodiffusion:turbodiffusion/inference \
pytest -q -p no:cacheprovider tests/inference
```

自动化测试至少覆盖：

- 单卡/八卡脚本参数；
- NPU 设备绑定；
- HCCL 初始化顺序；
- Ulysses 分辨率对齐；
- BNSD Attention 布局；
- SLA 替换范围；
- Dense checkpoint 兼容加载；
- Warmup High/Low 调度；
- 静态条件只同步一次；
- All-to-All 和输出形状恢复；
- rank 0 输出与进程组销毁。

## 11. 当前状态和差距

### 11.1 已确认能力

- 单卡能够成功生成视频；
- 四卡能够成功生成视频，历史结果约 `3.82 s/it`；
- 八卡 Ulysses 能够成功生成视频，历史基线约 `2.02 s/it`；
- 静态条件一次 Broadcast 优化已验证有收益；
- High/Low 模型切换搬运问题已通过双模型常驻解决；
- MindIE SLA、Dense Attention 和 Fast LayerNorm 已接入；
- Warmup 已固定在正式采样之前；
- `original` 已兼容包含 SLA `proj_l` 的 TurboWan checkpoint。

上述耗时为研发过程中的阶段性数据，不替代正式验收记录。静态条件优化后的八卡精确结果尚未写入本文。

### 11.2 未闭环项

| 编号 | 差距 | 影响 |
| --- | --- | --- |
| GAP-01 | 八卡性能尚无 `≤1.6 s/it` 的正式记录 | 阻塞性能验收 |
| GAP-02 | 单卡 720P/81f 最新标准测试记录未固化 | 需要补齐验收证据 |
| GAP-03 | 精度仍以人工检查为主 | 缺少量化精度报告 |
| GAP-04 | 最新软件镜像和依赖版本未固化 | 影响复现性 |
| GAP-05 | 八卡 Profiler 热点分析尚未形成正式结论 | 性能优化方向仍需数据支撑 |

## 12. 风险分析

| 风险 | 可能表现 | 应对措施 |
| --- | --- | --- |
| HCCL 通信暴露 | 八卡加速比低，A2A/AllGather 等待明显 | 使用系统 Profiler 分析通信与计算重叠，按收益优化 |
| 多 rank UMT5 资源压力 | 启动时主机内存和磁盘读取显著增加 | 监控 RSS、I/O 和各 rank 阶段日志，保证对称执行 |
| BNSD/dtype 不一致 | Dense/SLA 算子 shape 或 dtype 报错 | 在接口边界保持 BNSD 和 BF16 契约，使用单测保护 |
| 分辨率不可整除 | `L % ulysses_size != 0` | 在 VAE/DiT 前进行最小分辨率对齐 |
| 临时计时干扰性能 | `torch.npu.synchronize()` 拉低并发 | 验收关闭诊断同步，使用 msprof/torch_npu Profiler |
| SLA top-k 影响质量 | 稀疏比例过高导致细节损失 | 固定验收值 0.1，并与 Dense/人工精度对照 |
| 环境漂移 | 同代码在不同镜像上性能波动 | 固化 CANN、torch_npu、MindIE-SD 和驱动信息 |

## 13. 需求追踪矩阵

| 需求组 | 主要实现位置 | 主要验证位置 |
| --- | --- | --- |
| NPU 初始化 | `inference/wan2.2_i2v_infer.py` | `test_wan22_npu_device.py` |
| 单卡脚本 | `scripts/inference_wan2.2_i2v_npu_single.sh` | `test_wan22_npu_single_script.py` |
| 八卡 Ulysses | `wan2.2_i2v_infer.py`、`wan2pt2.py`、`a2a_cp.py` | `test_wan22_npu_8card_script.py`、`test_wan22_sla_attention.py` |
| SLA/Dense | `modify_model.py`、`wan2pt2.py` | `test_wan22_sla_attention.py`、`test_wan22_original_attention.py` |
| Fast LayerNorm | `wan2pt2.py` | `test_wan22_dit_device.py` |
| Warmup与模型常驻 | `wan2.2_i2v_infer.py` | `test_wan22_npu_device.py` |
| 静态条件同步 | `wan2.2_i2v_infer.py`、`wan2pt2.py` | `test_wan22_npu_device.py`、`test_wan22_dit_device.py` |

## 14. 待确认事项

1. 正式验收使用的 CANN、torch_npu、MindIE-SD、Triton-Ascend 版本；
2. 单卡和八卡各重复运行次数及最终统计规则；
3. 固定精度验收样例数量、提示词和输入图片；
4. 静态条件 Broadcast 优化后的八卡精确 `s/it`；
5. VBench 自动化评测是否纳入下一阶段。
