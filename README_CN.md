---
pipeline_tag: text-to-video
frameworks:
  - PyTorch
hardwares:
  - NPU
  - Ascend 950
license: apache-2.0
---

# TurboWan2.2 I2V 推理指导

## 目录

- [一、准备运行环境](#一准备运行环境)
- [二、下载权重](#二下载权重)
- [三、TurboWan2.2 使用](#三turbowan22-使用)
  - [3.1 开始前必读](#31-开始前必读)
  - [3.2 下载到本地，安装模型依赖](#32-下载到本地安装模型依赖)
  - [3.3 TurboWan2.2-I2V-A14B](#33-turbowan22-i2v-a14b)
- [四、推理结果参考](#四推理结果参考)
- [五、声明](#五声明)
- [六、常见问题](#六常见问题)

## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.11.10 | - |
  | torch | 2.9.0 | - |

**注意**：
- 该模型也支持torch 2.1.0等版本

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Ascend 950PR
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 1.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.4 Torch_npu安装
请从[昇腾社区资源下载中心](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)下载与当前PyTorch版本、Python版本和CPU架构匹配的`pytorch_v{pytorchversion}_py{pythonversion}.tar.gz`软件包。Torch_npu版本配套关系可参考[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)。

安装前可通过以下命令确认Python版本和CPU架构：
```shell
python3 --version
uname -m
```

解压并安装Torch_npu：
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，先确认whl包的实际文件名
find . -name "*.whl"
# 安装Torch_npu依赖
pip install pyyaml setuptools
# 安装与当前Python版本和CPU架构匹配的whl包，文件名以实际解压结果为准
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

安装完成后执行验证：
```shell
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('NPU available:', torch.npu.is_available())"
```

### 1.5 gcc、g++安装
```shell
# 若环境镜像中没有gcc、g++，请用户自行安装
yum install gcc
yum install g++

# 导入头文件路径
export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
```
注：若使用openeuler镜像，需要配置gcc、g++环境，否则会导致`fatal error: 'stdio.h' file not found`

## 二、下载权重

### 2.1 权重文件说明

推理需要以下四个非量化权重文件：

| 文件 | 用途 |
| --- | --- |
| `TurboWan2.2-I2V-A14B-high-720P.pth` | High-noise DiT |
| `TurboWan2.2-I2V-A14B-low-720P.pth` | Low-noise DiT |
| `models_t5_umt5-xxl-enc-bf16.pth` | umT5 文本编码器 |
| `google/umt5-xxl/` | umT5 tokenizer |
| `Wan2.1_VAE.pth` | Wan2.2 I2V 兼容 VAE |

TurboWan2.2 权重可从 [TurboWan2.2-I2V-A14B-720P](https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P) 获取。

### 2.2 权重目录

建议将权重整理为：

```text
/path/to/TurboWan2.2-I2V-A14B-720P/
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

其中 `google/umt5-xxl` 是本地 umT5 tokenizer 目录。推理程序会根据 `models_t5_umt5-xxl-enc-bf16.pth` 所在目录，自动查找同级的 `google/umt5-xxl`，不需要单独传递 tokenizer 路径。

本指导不使用文件名中带 `-quant` 的量化权重。

## 三、TurboWan2.2 使用

### 3.1 开始前必读

当前推理配置如下：

| 项目 | 配置 |
| --- | --- |
| 模型 | TurboWan2.2-I2V-A14B-720P |
| 硬件 | Ascend 950 系列 NPU |
| 数据类型 | BF16 非量化 |
| 分辨率 | 720P，自适应输入图片宽高比 |
| 视频帧数 | 81 帧 |
| 采样步数 | 4 steps |
| 采样方式 | ODE |
| Attention | `sla`：MindIE SLA Self Attention；`original`：MindIE Dense Self Attention |
| SLA top-k | 0.1 |
| 单卡 | 支持 |
| 多卡 | 单机八卡 Ulysses，`ulysses_size=8` |

当前 TurboWan High/Low checkpoint 同时包含公共 DiT 权重和 SLA 专属的 `proj_l` 参数。同一套权重可以运行两种 Attention：

```bash
--attention_type sla
--attention_type original
```

使用 `original` 时，加载器只忽略不参与 Dense Attention 计算的 `proj_l.weight` 和 `proj_l.bias`，其他权重仍然严格校验。本文性能测试和验收配置使用 `sla`。

推理程序会在正式采样前分别对 High-noise 和 Low-noise 模型执行一次 Warmup。Warmup 为固定流程，不需要额外传递参数，并且不计入 tqdm 显示的正式采样时间。

### 3.2 下载到本地，安装模型依赖

下载本仓库后进入代码目录：

```bash
git clone https://modelers.cn/MindIE/TurboWan2.2.git
cd TurboDiffusion
pip3 install -r requirements.txt
```

本项目依赖 CANN、PyTorch NPU、MindIE-SD 和 Triton-Ascend，推荐直接使用配套昇腾镜像。完成环境准备后，通过 `PYTHONPATH` 使用源码：

```bash
export PYTHONPATH=turbodiffusion${PYTHONPATH:+:$PYTHONPATH}
```

当前上游安装脚本包含 CUDA 扩展，不要在 NPU 环境直接执行会编译 CUDA 算子的安装命令。

### 3.3 TurboWan2.2-I2V-A14B

#### 3.3.1 单卡性能测试

仓库提供单卡运行脚本：

```bash
export PYTHONPATH=turbodiffusion
export FAST_LAYERNORM=0
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export ASCEND_RT_VISIBLE_DEVICES=3

python turbodiffusion/inference/wan2.2_i2v_infer.py \
    --model Wan2.2-A14B \
    --low_noise_model_path /mnt/share/weights/TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-low-720P.pth \
    --high_noise_model_path /mnt/share/weights/TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-high-720P.pth \
    --text_encoder_path /mnt/share/weights/TurboWan2.2-I2V-A14B-720P/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path /mnt/share/weights/TurboWan2.2-I2V-A14B-720P/Wan2.1_VAE.pth \
    --resolution 720p \
    --adaptive_resolution \
    --image_path assets/i2v_inputs/i2v_input_0.jpg \
    --prompt "POV selfie video, ultra-messy and extremely fast. A white cat in sunglasses stands on a surfboard with a neutral look when the board suddenly whips sideways, throwing cat and camera into the water; the frame dives sharply downward, swallowed by violent bursts of bubbles, spinning turbulence, and smeared water streaks as the camera sinks. Shadows thicken, pressure ripples distort the edges, and loose bubbles rush upward past the lens, showing the camera is still sinking. Then the cat kicks upward with explosive speed, dragging the view through churning bubbles and rapidly brightening water as sunlight floods back in; the camera races upward, water streaming off the lens, and finally breaks the surface in a sudden blast of light and spray, snapping back into a crooked, frantic selfie as the cat resurfaces." \
    --num_samples 1 \
    --num_steps 4 \
    --default_norm \
    --attention_type sla \
    --sla_topk 0.1 \
    --save_path output/generated_video_layernorm_0.mp4 \
    --num_frames 81 \
    --ode
```

参数说明：


#### 3.3.2 八卡性能测试

仓库提供单机八卡 Ulysses 运行脚本：

```bash
export PYTHONPATH=turbodiffusion

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export FAST_LAYERNORM=1
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

torchrun --nproc_per_node=8 turbodiffusion/inference/wan2.2_i2v_infer.py \
    --model Wan2.2-A14B \
    --low_noise_model_path /mnt/share/weights/TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-low-720P.pth \
    --high_noise_model_path /mnt/share/weights/TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-high-720P.pth \
    --text_encoder_path /mnt/share/weights/TurboWan2.2-I2V-A14B-720P/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path /mnt/share/weights/TurboWan2.2-I2V-A14B-720P/Wan2.1_VAE.pth \
    --resolution 720p \
    --aspect_ratio 16:9 \
    --adaptive_resolution \
    --image_path assets/i2v_inputs/i2v_input_0.jpg \
    --prompt "POV selfie video, ultra-messy and extremely fast. A white cat in sunglasses stands on a surfboard with a neutral look when the board suddenly whips sideways, throwing cat and camera into the water; the frame dives sharply downward, swallowed by violent bursts of bubbles, spinning turbulence, and smeared water streaks as the camera sinks. Shadows thicken, pressure ripples distort the edges, and loose bubbles rush upward past the lens, showing the camera is still sinking. Then the cat kicks upward with explosive speed, dragging the view through churning bubbles and rapidly brightening water as sunlight floods back in; the camera races upward, water streaming off the lens, and finally breaks the surface in a sudden blast of light and spray, snapping back into a crooked, frantic selfie as the cat resurfaces." \
    --num_samples 1 \
    --num_steps 4 \
    --num_frames 81 \
    --attention_type sla \
    --sla_topk 0.1 \
    --default_norm \
    --ulysses-size 8 \
    --save_path output/ulysses8.mp4 \
    --ode
```

脚本使用以下并行配置：

```text
torchrun --nproc_per_node=8
distributed backend = HCCL
ulysses_size = 8
```

八卡共同生成同一个视频。只有 Rank 0 显示采样进度条并执行最终 VAE 解码和 MP4 保存。

如果默认通信端口被占用，可以在脚本的 `torchrun` 命令中增加：

```bash
--master_port 29511
```

## 四、推理结果参考

Ascend 950PR 性能数据

| 场景 | 分辨率         | 帧数 | 迭代次数 | 单步迭代耗时  |
| --- |-------------| ---: |-----:|---------|
| 单卡 SLA | 1280 x 720P | 81 帧 |    4 |  待补充 |
| 八卡 Ulysses SLA | 1280 x 720P | 81 帧 |    4 | 待补充 |


## 五、声明

本项目基于 TurboDiffusion、Wan2.2 和 MindIE-SD 相关开源能力进行昇腾 NPU 适配。使用本项目时，请同时遵守上游代码、模型权重及依赖软件的许可证和使用条款。

本仓库代码采用 Apache License 2.0，详情参见 [LICENSE](LICENSE)。

如果本项目对您的研究有帮助，请引用 TurboDiffusion：

```bibtex
@article{zhang2025turbodiffusion,
  title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
  author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.16093},
  year={2025}
}
```

## 六、常见问题

1. 找不到 `torch_npu` 或 `mindiesd`。请检查 CANN、PyTorch NPU 和 MindIE-SD 是否正确安装，并确认已经加载相应环境变量。可先执行 [1.3 验证运行环境](#13-验证运行环境)。

2. HCCL 初始化或 collective 报错，请依次检查：`torchrun --nproc_per_node=8` 是否与可见设备数一致；`--ulysses-size 8` 是否与 `WORLD_SIZE=8` 一致；
