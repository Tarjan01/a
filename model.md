# AIGC 人物异常检测算法 - 技术报告

## 一、算法梗概

### 1. 课题算法当前具备的能力

**主要功能：**
- 判断一张含人物的图片是"正常"还是"异常"
- 如果异常，进一步识别属于哪种类型的异常（8 类）

**通俗理解：** 可以把算法想象成一个"AI 图片质检员"，自动批量检查 AI 生成的图片质量，过滤掉不合格的图片。

**支持的检测场景（8 类异常）：**

| 编号 | 异常类型 
|------|---------|
| 1 | 头部异常 | 
| 2 | 颈部异常 | 
| 3 | 身体异常 | 
| 4 | 手臂异常 | 
| 5 | 手异常 | 
| 6 | 腿异常 | 
| 7 | 脚异常 | 
| 8 | 其他 未归入上述类别的异常 |

**核心检测能力：**
- 判断一张含人物的图片是"正常"还是"异常"（二分类）

**支持的任务类型：**
- 异常检测（二分类）
- 异常定位（空间热力图可视化）
- 跨域迁移（域自适应微调）

---

### 2. 当前算法的精度与指标表现

**通俗解释：** 我们用准确率（Accuracy）和召回率（Recall）两个指标来衡量算法好不好用。

| 指标 | 含义 | 要求 |
|------|------|------|
| **准确率** | 算法判断正确的图片占总图片的比例 | >= 80%（基础要求） |
| **召回率** | 在所有真正的异常图片中，算法能找出多少比例 | >= 85%（基础要求） |

**当前测试结果（实际测试）：**

| 评估集 | 准确率 | 召回率 | 精确率 | 说明 |
|--------|--------|--------|--------|------|
| 验证集（huawei_data/train） | **100.00%** | **100.00%** | **100.00%** | 微调后在该域过拟合（正常现象） |
| 测试集（huawei_data/test） | **91.94%** | **95.16%** | **89.39%** | 独立测试，用于最终考核 |

> **注意：** 验证集 100% 准确率是因为 248 张数据量小，域自适应微调后又专门针对 huawei_data/train 优化过，所以验证集过拟合是正常的。测试集表现更真实地反映了算法能力。

**通俗理解：** 算法大概能判断对 85%-95% 左右的图片，在所有真正的异常图片中，能找出 95% 左右。

---

### 3. 测试结果示意图

**3.1 模型结构图**

> **模型整体架构图**
![](./pic/model.png)

**3.2 骨架热力图示例**

> **骨架热力图示例**
![](./pic/skeleton1.png)

![](./pic/skeleton2.png)

人体骨架热力图（第 3 通道）和手部骨架热力图（第 4 通道）生成示例：
```
原始图像 → RTMPose 人体/手部检测 → 关键点坐标 → 高斯模糊 → 连续概率热力图
```

**3.3 Patch 与neck结构**

> **patch结构**
![](./pic/patch.png)

> **neck结构**
![](./pic/neck.png)

**3.4 合成数据示例**

> **合成数据**
![](./pic/fusion1.png)
![](./pic/fusion2.png)
![](./pic/fusion3.png)

合成数据生成示例：
```
┌──────────────────────────────────────────────────────────────┐
│               合成数据生成 Pipeline                            │
│                                                              │
│  源图片 A (含手部)     目标图片 B (含完整人体)                │
│       │                          │                           │
│       ▼                          ▼                           │
│  ┌─────────┐              ┌─────────┐                       │
│  │ mmdet   │              │ mmdet   │                       │
│  │ 检测人体 │              │ 检测人体 │                       │
│  └────┬────┘              └────┬────┘                       │
│       ▼                       ▼                              │
│  ┌─────────┐              ┌─────────┐                       │
│  │ mmpose  │              │ mmpose  │                       │
│  │ 关键点检测 │             │ 关键点检测 │                       │
│  └────┬────┘              └────┬────┘                       │
│       │ (elbow, wrist)        │ (elbow, wrist)               │
│       ▼                       ▼                              │
│  ┌─────────┐              ┌─────────┐                       │
│  │  SAM    │              │  无需SAM │                       │
│  │ 提取手部 │              │          │                       │
│  │ patch+mask│             │          │                       │
│  └────┬────┘              └────┬────┘                       │
│       │                       │                               │
│       ▼                       ▼                               │
│  ┌─────────────────────────────────────┐                    │
│  │   仿射变换对齐 + 融合               │                    │
│  │   - 以 Elbow 为锚点对齐             │                    │
│  │   - 随机角度扰动模拟解剖学异常      │                    │
│  │   - 按 mask 覆盖到目标图上          │                    │
│  └───────────────┬─────────────────────┘                    │
│                  ▼                                           │
│         合成图片 (手部异常样本)                                │
│         打上异常类型标签 → "1" (手部畸形)                     │
└──────────────────────────────────────────────────────────────┘
```

**3.5 推理调试可视化示例**

> **推理调试可视化图**
![](./pic/goodcase1.png)

debug.py 可输出 11 个中间过程可视化子图，包括：
1. RGB 输入
2. 人体骨架热力图
3. 手部骨架热力图
4. Stem 输出（最大激活值）
5. Stage 1-4 输出
6. 空间注意力图
7. Patch Score S4/S3

---

### 4. 算法的局限性与能力边界

#### 可检测的异常类型

| 异常类型 | 检测效果 | 说明 |
|---------|---------|------|
| 手部畸形 | 较好 | 手指数量不对、手指融合等常见异常 |
| 肢体缺失/增加 | 较好 | 明显缺失或多余肢体 |
| 脸部畸形 | 较好 | 面部比例失调、五官错位 |
| 身体畸形 | 一般 | 比例异常程度明显时有效 |

#### 不可检测或效果较弱的异常类型

| 异常类型 | 效果 | 原因 |
|---------|------|------|
| 背景异常 | 较弱 | 算法聚焦人物，对背景不敏感 |
| 光影/色调问题 | 较弱 | 属于风格问题而非质量缺陷 |
| 文字/水印瑕疵 | 较弱 | 不在训练数据集中 |
| 极小人物图片 | 较弱 | 人物太小，特征不清晰 |

#### Good Case 与 Bad Case

**Good Case（算法判断正确的典型例子）：**

| 示例 | 场景 | 说明 |
|------|------|------|
| Case 1 | 图片中有明显手臂冗余 | 准确识别为异常 |
| Case 2 | 图片中多出一只手臂 | 准确识别为异常 |
| Case 3 | 图片中多出一只手 | 准确识别为异常 |

![](./pic/goodcase1.png)

![](./pic/goodcase2.png)

![](./pic/goodcase3.png)


**Bad Case（算法判断错误的典型例子）：**

| 示例 | 场景 | 说明 |
|------|------|------|
| Case 1 | 异常区域很小的图片 | 误判为正常（漏检） |
| Case 2 | 人物异常情况特殊的图片 | 可能误判（误检） |

![](./pic/badcase1.png)
![](./pic/badcase2.png)


---

## 二、算法推理速度与资源占用

### 1. 推理性能

| 指标 | 值 | 说明 |
|------|------|------|
| 训练速度 (单卡) | 约 140 秒/epoch | RTX 4090, batch=64 |
| 推理速度 (单张) | 0.3-2 张/秒 | RTX 4090,含文件IO |
| 推理速度 (单张) | 100 张/秒 | RTX 4090,不含文件输入输出 |
| 推理速度 (batch=64) | 200-400 张/秒 | RTX 4090,含文件IO |
| 推理速度 (batch=64) | 667 张/秒 | RTX 4090,不含文件IO |

**说明：**
- 该算法需要 GPU 才能高效运行，CPU 推理速度太慢，不适合实际生产
- 支持 batch 推理，同时处理多张图片可以大幅提升吞吐量

### 2. 资源占用

| 资源 | 占用 | 说明 |
|------|------|------|
| GPU 显存 | 约 23 GB | 模型加载后，batch=64 时约 23GB |
| 计算资源 | 中等 | 模型为 ConvNeXt-Tiny 级别，属于轻量级模型 |

---

## 三、材料与部署说明

### 1. 训练接口（Train）

#### 数据集格式准备

每个数据集子集（train/val/test）目录结构如下：

```
数据集/
├── washed_images/              # 图片文件 (JPG/PNG)
├── washed_multiple_labels/     # 多标签文件 (*.txt)
└── cache_skeleton/             # 骨架热力图缓存（预生成）
```

**原始数据目录结构（以 HumanRefiner 为例）：**

```
humanrefiner_data/
├── images/                    # 原始图片（JPG/PNG）
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── labels/                    # 标签文件（与图片同名）
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
└── ...
```

**标签文件格式（`.txt`）：**

每个标签文件与图片同名（如 `1.jpg` 对应 `1.txt`），每行格式为：

```
异常编号  x_min y_min x_max y_max
```

- 一行表示一个异常标注
- 异常编号 1-8 对应 8 类异常，编号 9 表示非人类样本
- 一个文件可以有多行，表示同一张图片存在多个异常
- bbox 信息用于原始数据标注，本项目**不使用**，仅读取异常编号

**标签格式（重组后）：**
- 正常图片：标签文件内容为 `0`
- 异常图片：标签文件内容为 `1,3,7`（表示同时存在第 1、3、7 类异常）

**数据来源：**

| 数据集 | 来源 | 说明 |
|--------|------|------|
| 训练集 | HumanRefiner + CrowdPose + 合成数据 | 50,459 张，含 8 类异常标签 |
| 验证集 | HumanRefiner + CrowdPose + 合成数据 | 14,417 张，含 8 类异常标签 |
| 测试集 | HumanRefiner + CrowdPose + 合成数据 | 7,209 张，含 8 类异常标签 |
| 华为域 train | 内部数据（248 张） | 域自适应微调训练数据 |
| 华为域 test | 内部数据（248 张） | 域自适应微调后评估数据 |

**CrowdPose 数据说明：**
- 来源：https://people.eecs.berkeley.edu/~pwang060/CrowdPose/
- 用途：外部公开数据集，引入复杂场景下的正常图片，提升模型鲁棒性
- 目录：`data/crowdpose/washed_images/` 和 `data/crowdpose/washed_multiple_labels/`

**合成数据说明：**
- 来源：sam_fusion.py（mmdet + mmpose + SAM pipeline）
- 用途：扩充手部畸形类别的训练数据
- 目录：`data/fusion_result_*/washed_images/`

**训练命令：**

```bash
# 单卡训练
python sk_resnet.py

# 多卡训练（DDP）
torchrun --nproc_per_node=4 sk_resnet.py
```

**关键超参数配置（config.yaml）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 64 | 每批处理图片数 |
| num_epochs | 40 | 训练总轮数 |
| 输入尺寸 | 224x224 | 图片统一缩放到此尺寸 |
| backbone_lr | 1e-4 | backbone 学习率 |
| weight_decay | 3e-3 | 正则化系数 |
| freeze_backbone_epochs | 8 | 前 8 轮冻结 backbone |
| use_ddp | False | 是否使用 DDP |

---

### 2. 评估接口（Evaluation）

**独立评估脚本：**

```bash
python eval.py
```

**功能：**
- 从验证集收集所有预测结果
- 在 [0.2, 0.8] 范围内搜索最优分类阈值
- 使用最优阈值评估验证集和测试集
- 输出准确率、召回率、精确率等指标

**复现实验结果：**

```bash
# 1. 准备测试数据集（格式同上）
# 2. 运行评估
python eval.py

# 输出：
# - 验证集和测试集的准确率、召回率、精确率
# - 最优分类阈值
# - 测试集预测结果（每个样本的异常概率）
# - 概率分布直方图
```

**实际测试结果（可复现）：**

| 评估集 | 准确率 | 召回率 | 精确率 | 说明 |
|--------|--------|--------|--------|------|
| 验证集（huawei_data/train） | 100.00% | 100.00% | 100.00% | 微调后在该域过拟合（正常） |
| 测试集（huawei_data/test） | 91.94% | 95.16% | 89.39% | 独立测试，用于最终考核 |

> **验证集 100% 准确率说明：** 248 张数据量小且经过筛选，又专门针对 huawei_data/train 域自适应微调优化过，所以验证集过拟合是正常的。

---

## 四、Methodology

### 1. 模型设计

#### 1.1 整体架构

本模型基于 **ConvNeXt Tiny**（ImageNet 预训练）进行多层次改造，核心设计思路是将 CNN 的空间感知能力与骨骼关键点的结构先验相结合，使模型能够精准定位并识别人物部位的异常。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        输入层 [B, 5, 224, 224]                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │  R 通道  │  │  G 通道  │  │  B 通道  │  │  人体骨架 │  │ 手部骨架││
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └─────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stem 层：Conv2d(5→96, kernel=4, stride=4) → 56×56×96               │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼─────────────────────────┐
        ▼                       ▼                         ▼
┌───────────────┐      ┌───────────────┐         ┌───────────────┐
│   Stage 1     │      │   Stage 2     │         │   Stage 3     │
│  56×56×96     │      │  28×28×192    │         │  14×14×384    │
│               │      │               │         │               │
│               │      │ ┌───────────┐ │         │ ┌───────────┐ │
│               │      │ │Cross-Attn │ │         │ │Cross-Attn │ │
│               │      │ │(sk_dim=2) │ │         │ │(sk_dim=2) │ │
│               │      │ └───────────┘ │         │ └───────────┘ │
└───────────────┘      └───────────────┘         └───────┬───────┘
                                                        │
                                ┌───────────────────────┤
                                ▼                       ▼
                    ┌──────────────────┐    ┌──────────────────┐
                    │  PatchAnomaly   │    │  PatchAnomaly    │
                    │    Head S3      │    │    Head S4       │
                    │  [B,8,14,14]    │    │  [B,8,7,7]       │
                    └────────┬─────────┘    └───────┬──────────┘
                             │                      │
        ┌────────────────────┘                      └──────────────────┐
        ▼                                                             │
┌─────────────────────────────────────────────────────────────────────┤
│  Stage 4：7×7×768 ──→ SpatialAttention(CBAM) ──→ 特征聚合           │
│                     ┌───────────────────┐                           │
│           ┌────────│  多层特征融合       │                           │
│           ▼        └────────┬───────────┘                           │
│  feat(768)+S3_Avg(384)+    │                                        │
│  S3_Max(384)+S4_Avg(768)+  │                                        │
│  S4_Max(768)+patch_s3(16)+ │                                        │
│  patch_s4(16) = 4520 维    │                                        │
└────────────────────────────┼───────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
          ┌───────────────┐  ┌───────────────┐
          │ 8类异常分类头 │  │ 二分类异常头   │
          │ [B, 8]       │  │ [B, 1]        │
          └───────────────┘  └───────────────┘
```

> **模型整体架构图**
![](./pic/model.png)


Backbone：**ConvNeXt Tiny**（来自 Facebook AI，ImageNet-21K 预训练），具体网络结构：

| Stage | 输出尺寸 | 通道数 | 模块组成 |
|-------|---------|--------|---------|
| Stem | 56×56 | 96 | Conv2d(5→96, stride=4) + LayerNorm + GELU |
| Stage 1 | 56×56 | 96 | ConvNeXt Block |
| Stage 2 | 28×28 | 192 | ConvNeXt Block × 2 + Cross-Attention 注入 |
| Stage 3 | 14×14 | 384 | ConvNeXt Block × 2 + Cross-Attention 注入 |
| Stage 4 | 7×7 | 768 | ConvNeXt Block × 2 + SpatialAttention |

#### 1.2 输入层改造

**5 通道多模态输入设计**

原始 ConvNeXt 接受 3 通道 RGB 输入，本项目将其扩展为 5 通道，引入人体结构先验：

| 通道 | 来源 | 说明 |
|------|------|------|
| 0-2 | RGB 图像 | 原始彩色图像，通过 PIL 读取后标准化 |
| 3 | 人体骨架热力图 | 通过 OpenMMLab RTMPose 检测全身关键点，经高斯模糊生成概率图 |
| 4 | 手部骨架热力图 | 通过 RTMPose 检测手部关键点，经高斯模糊生成概率图 |

**Stem 改造实现：**

```python
# 将原始 3 通道卷积扩展为 5 通道
old_conv = base_model.features[0][0]  # 原始 Conv2d(3, 96)
self.stem_conv = nn.Conv2d(5, 96, kernel_size=4, stride=4)
# 保留 RGB 通道的预训练权重
self.stem_conv.weight[:, :3, :, :] = old_conv.weight
self.stem_conv.bias = old_conv.bias
# 骨架通道初始化为 0，通过微调学习
self.stem_conv.weight[:, 3:, :, :] = 0.0
```

**骨架热力图生成流程：**
```
原始图像 → RTMPose 人体检测 → 关键点坐标 → 高斯模糊(kernel=45) → 人体骨架热力图(224x224)
原始图像 → RTMPose 手部检测 → 关键点坐标 → 高斯模糊(kernel=25) → 手部骨架热力图(224x224)
```

**骨架热力图生成与缓存机制（预生成模式）：**

骨架热力图计算较慢，项目采用**预生成模式**——在加载数据集之前先生成所有骨架缓存，而不是在训练时懒加载计算。

```python
# 训练/评估前必须先生成缓存（内置函数，sk_resnet.py 自动调用）：
batch_pre_generate_all_caches(train_dataset, mode='both')
batch_pre_generate_all_caches(val_dataset, mode='both')
batch_pre_generate_all_caches(test_dataset, mode='both')
```

**旧机制（懒加载，已弃用）：** 在 `sk_dataset.py` 中缓存自动保存的逻辑已被注释：
```python
# torch.save(skeleton, skeleton_cache_path)  # 已弃用
```

**新机制（预生成）：** 训练、验证、评估之前，统一先生成所有骨架缓存，再加载模型进行推理。

**缓存目录结构（独立）：**

```
data/rearranged_train/cache_skeleton/
data/rearranged_val/cache_skeleton/
data/rearranged_test/cache_skeleton/
```

由于重组后每个目录使用带前缀的文件名（如 `train_xxx.jpg`），缓存文件名也带前缀（`train_xxx.pt`），不会冲突。

> **骨架热力图示例图**
![](./pic/skeleton1.png)

#### 1.3 Cross-Attention 骨骼注入模块

**设计动机：** CNN 的特征图本身对人体的结构位置不敏感，通过 Cross-Attention 将骨骼关键点信息注入到中间特征层，引导模型在特征空间中关注人物的关键部位（手、臂、腿等）。

**模块结构：**

```
┌──────────────────────────────────────────────┐
│           StandardCrossAttention              │
│                                               │
│  输入: x [B, C, H, W]     (CNN 特征图)       │
│  输入: sk [B, 2, H, W]    (骨骼热力图)        │
│                                               │
│  Q = Conv2d(C → C//8)    从特征图提取查询     │
│  K = Conv2d(2 → C//8)    从骨骼图提取键       │
│  V = Conv2d(2 → C)       从骨骼图提取值       │
│                                               │
│  Attention = Softmax(QK^T * scale)            │
│  Output = Attention @ V                       │
│                                               │
│  输出: x + γ * Dropout(Output)  (残差连接)    │
└──────────────────────────────────────────────┘
```

**关键设计参数：**

| 参数 | 值 | 说明 |
|------|------|------|
| inter_dim | C // 8 | 降维后的注意力维度，减少计算量（192→24, 384→48） |
| sk_dim | 2 | 骨骼通道数（人体+手部） |
| dropout | 0.2-0.3 | Dropout2d 防止过拟合 |
| gamma | 可学习参数 | 初始化 0，训练后自动调节骨骼信息的注入强度 |

**注入位置：**

| 注入点 | Stage 输出 | 特征尺寸 | 通道数 | 说明 |
|--------|-----------|---------|--------|------|
| attn2 | Stage 2 后 | 28×28 | 192 | 注入第一次骨骼信息 |
| attn3 | Stage 3 后 | 14×14 | 384 | 注入第二次骨骼信息 |

**代码实现：**

```python
class StandardCrossAttention(nn.Module):
    def __init__(self, dim, sk_dim=2, dropout=0.2):
        super().__init__()
        self.inter_dim = dim // 8       # 降维到 24/48
        self.scale = self.inter_dim ** -0.5

        self.q = nn.Conv2d(dim, self.inter_dim, 1)        # Q 来自特征图
        self.k = nn.Conv2d(sk_dim, self.inter_dim, 1)     # K 来自骨架图
        self.v = nn.Conv2d(sk_dim, dim, 1)                # V 来自骨架图
        self.gamma = nn.Parameter(torch.zeros(1))         # 可学习缩放系数
        self.dropout = nn.Dropout2d(dropout)               # Dropout 防止过拟合

    def forward(self, x, sk):
        B, C, H, W = x.shape
        # 1. 对齐骨架图尺寸
        sk_res = F.interpolate(sk, size=(H, W), mode='bilinear', align_corners=False)

        # 2. 生成 Q, K, V
        proj_query = self.q(x).view(B, self.inter_dim, -1).permute(0, 2, 1)  # [B, N, C']
        proj_key   = self.k(sk_res).view(B, self.inter_dim, -1)              # [B, C', N]
        proj_value = self.v(sk_res).view(B, C, -1)                           # [B, C, N]

        # 3. 计算注意力矩阵
        energy = torch.bmm(proj_query, proj_key) * self.scale  # [B, N, N]
        attention = F.softmax(energy, dim=-1)                  # 对空间像素归一化

        # 4. 注意力加权叠加
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.dropout(out)

        # 5. 残差连接：将骨架注意力引导的信息注入原特征
        return x + self.gamma * out
```

#### 1.4 SpatialAttention（CBAM 空间注意力）模块

在 Stage4 之后引入 CBAM（Convolutional Block Attention Module）的空间注意力子模块：

```
┌──────────────────────────────────────────┐
│          SpatialAttention (CBAM)          │
│                                          │
│  输入: x [B, 768, 7, 7]                  │
│                                          │
│  1. 通道维度取 Max → [B, 1, 7, 7]        │
│  2. 通道维度取 Avg → [B, 1, 7, 7]        │
│  3. Concat → [B, 2, 7, 7]               │
│  4. Conv2d(2→1, kernel=7, pad=3)         │
│  5. Sigmoid → [B, 1, 7, 7] 权重图        │
│                                          │
│  输出: x * sa_map  (逐元素相乘)           │
└──────────────────────────────────────────┘
```

该模块学习"图像中**哪里**是重要的"，增强模型对异常区域的空间感知能力。

**代码实现：**

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        # 输入 2 通道（Max 和 Avg 合并），输出 1 通道权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 通道维度取 Max → [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 2. 通道维度取 Avg → [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 3. 拼接 → [B, 2, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 4. 卷积 + Sigmoid 生成权重图 → [B, 1, H, W]
        out = self.conv1(x_cat)
        return self.sigmoid(out)
```

#### 1.5 PatchAnomalyHead 补丁级异常检测头

在 Stage3 和 Stage4 分别接一个补丁级异常检测头，实现像素级（更精确地说是 patch 级）的异常定位：

**网络结构：**

```
┌─────────────────────────────────────┐
│         PatchAnomalyHead             │
│                                      │
│  输入: x [B, C, H, W]                │
│                                      │
│  Conv2d(C, C//2, 3, pad=1) → GELU   │
│  Conv2d(C//2, num_classes, 1)       │
│                                      │
│  输出: [B, 8, H, W]                  │
│        (每个 patch 的 8 类异常 logits) │
└─────────────────────────────────────┘
```

**Patch 级聚合 - LSE（Log-Sum-Exp）方法：**

将空间维度 [H, W] 聚合为 [B, K]，使用 Log-Sum-Exp Trick 防止 exp 运算溢出：

```python
def lse_aggregate(patch_logits, r=10):
    """
    将空间维度 [H, W] 聚合为 [B, K]
    r=10 为温度系数，比简单 Max/Avg Pool 能更好地捕捉异常区域的峰值响应
    """
    B, K, H, W = patch_logits.shape
    x = patch_logits.view(B, K, -1)

    # Log-Sum-Exp Trick: 减去最大值防止 exp 溢出
    x_max, _ = torch.max(x, dim=-1, keepdim=True)
    lse = x_max + (1.0 / r) * torch.log(
        torch.mean(torch.exp(r * (x - x_max)), dim=-1, keepdim=True)
    )
    return lse.squeeze(-1)  # [B, K]
```

**Patch 级输出：**

| 输出 | 维度 | 说明 |
|------|------|------|
| `patch_maps['s3']` | [B, 8, 14, 14] | Stage 3 补丁级异常 logits |
| `patch_maps['s4']` | [B, 8, 7, 7] | Stage 4 补丁级异常 logits |
| `patch_score['s3']` | [B, 8] | Stage 3 经 LSE 聚合后的分数 |
| `patch_score['s4']` | [B, 8] | Stage 4 经 LSE 聚合后的分数 |

> **patch 热力图示例**
![](./pic/patch.png)

#### 1.6 多层特征融合策略

最终分类器融合来自网络不同层次、不同聚合方式的 9 路特征（4520 维）：

| 特征来源 | 维度 | 说明 |
|---------|------|------|
| neck feat | 768 | Stage4 经 SpatialAttention 后经可学习权重融合（alpha*Avg + beta*Max）+ 投影 + LayerNorm |
| S3 Avg Pool | 384 | Stage3 全局平均池化 |
| S3 Max Pool | 384 | Stage3 全局最大池化 |
| S4 Avg Pool | 768 | Stage4 全局平均池化 |
| S4 Max Pool | 768 | Stage4 全局最大池化 |
| Patch S3 Avg | 8 | PatchAnomalyHead S3 的 LSE 聚合 |
| Patch S3 Max | 8 | PatchAnomalyHead S3 的 Max 聚合 |
| Patch S4 Avg | 8 | PatchAnomalyHead S4 的 LSE 聚合 |
| Patch S4 Max | 8 | PatchAnomalyHead S4 的 Max 聚合 |
| **合计** | **4520** | 768 + 384×2 + 768×2 + 8×4 = 4520 |

**可学习融合权重：**

```python
self.fusion_weights = nn.Parameter(torch.ones(2))

# 通过 softmax 自动学习 Avg 和 Max 的权重比例
weights = F.softmax(self.fusion_weights, dim=0)
alpha, beta = weights[0], weights[1]

feat_avg = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)     # [B, 768]
feat_max = F.adaptive_max_pool2d(x_refined, (1, 1)).flatten(1)  # [B, 768]
feat_max = self.max_pool_dropout(feat_max)

feat = alpha * feat_avg + beta * feat_max  # 加权融合
feat = self.neck_projector(feat)           # MLP(768→768) + LayerNorm + GELU + Dropout(0.4)
feat = self.final_norm(feat)               # LayerNorm(eps=1e-6)
```

**输出融合权重示例（训练时打印）：**

```
--- Feature Fusion Weights ---
Avg Pool Weight (Alpha): 0.5234
Max Pool Weight (Beta):  0.4766
------------------------------
```

#### 1.7 多任务输出头

模型同时输出 4 种预测结果：

| 输出 | 维度 | 作用 |
|------|------|------|
| `multi_logits` | [B, 8] | 8 类异常的独立预测分数 |
| `binary_logit` | [B, 1] | 整体异常/正常的二分类分数 |
| `patch_score` | {s3: [B,8], s4: [B,8]} | 补丁级聚合分数，支持可解释性 |
| `patch_maps` | {s3: [B,8,H,W], s4: [B,8,H,W]} | 空间热力图，可视化模型关注区域 |

**代码实现：**

```python
self.combined_dim = 4520  # 768 + 384×2 + 768×2 + 8×4
self.combined_multi_classifier = nn.Linear(self.combined_dim, 8)
self.combined_binary_classifier = nn.Linear(self.combined_dim, 1)
```

#### 1.8 损失函数

采用多任务联合优化：

| Loss | 权重 | 说明 |
|------|------|------|
| Focal Loss（多标签） | 1.0 | 解决 8 类异常的类别不平衡问题，gamma=2.5 |
| Focal Loss（二分类） | 5.0 | 整体异常/正常分类，放大二分类损失以加强监督信号 |
| Patch Loss | 0.0 | 可选，当前未启用，未来可通过调整 `loss_patch_para` 开启 |

**总损失公式：**

```
total_loss = 1.0 × loss_multi + 5.0 × loss_binary + 0.0 × loss_patch
```

**Focal Loss 实现：**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 正样本权重
        self.gamma = gamma  # 聚焦参数（默认 2.5）

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce_loss

        if self.alpha is not None:
            alpha_factor = torch.where(targets == 1, self.alpha, torch.ones_like(targets))
            loss = alpha_factor * loss

        return loss.mean()
```

**pos_weights 计算：**

```python
# 训练前一次性计算，不会每个 epoch 重新统计
# 在 sk_resnet.py 第 278 行调用，数据在 train_dataset 初始化时统计
label_counter = train_dataset.get_label_count()
num_samples = len(train_dataset)

for i in range(1, NUM_ABNORMAL_CLASSES + 1):
    pos = label_counter.get(i, 0)      # 正样本数
    neg = num_samples - pos            # 负样本数
    pos_weights.append(min(max((neg / max(pos, 1)) ** 0.5, 1.0), 10.0))
```

`label_counter` 是在 `ImageWithSKMultiLabelDataset` 初始化时通过遍历所有文件统计的（见 `sk_dataset.py` 第 170-200 行），**每个 epoch 不会重新统计**。对于重组后 `rearranged_train/` 目录下的文件，`label_counter` 统计的是重组后的文件数，和原始文件无关。

---

### 2. 数据构建

#### 2.1 数据来源

- **主要来源：** HumanRefiner 数据集（MIT License），包含大量人工标注的 AIGC 人物图片
- **辅助来源：** CrowdPose 数据集，引入复杂场景下的正常图片
- **合成数据：** sam_fusion.py 生成的手部异常样本

#### 2.2 数据清洗流程

```
原始数据 → 标签合并 → 过滤非人类 → 统计分布
```

**核心函数：**

| 函数 | 功能 | 输出 |
|------|------|------|
| `find_label_files()` | 将多异常标签合并为单一异常标签 1，正常标签为 0 | `washed_labels/` |
| `find_multiple_label_files()` | 保留多标签格式（如 `1,3,7`） | `washed_multiple_labels/` |
| `label_count()` | 统计各异常类别数量 | 控制台输出 |

**标签规则：**
- 异常编号 1-8 表示对应异常类型
- 编号 9 表示非人类样本（过滤）
- 一个文件可以有多行，每行一个异常编号
- 同时存在多个异常编号表示多标签样本

#### 2.3 标注规范

- **单标签：** `0` = 正常，`1` = 异常
- **多标签：** `1,3,7` = 同时存在第 1、3、7 类异常（共 8 种异常类型）

#### 2.4 数据增强

| 增强操作 | 训练集 | 验证/测试集 |
|---------|--------|-----------|
| Resize | 224x224 | 224x224 |
| 水平翻转 | 概率 0.5 | 无 |
| 随机旋转 | 5 度，概率 0.5 | 无 |
| 亮度调整 | 5%，概率 0.5 | 无 |
| 对比度调整 | 5%，概率 0.5 | 无 |
| 归一化 | ImageNet stats [0.485,0.456,0.406] / [0.229,0.224,0.225] | ImageNet stats |

**增强代码实现（sk_dataset.py · JointTransform）：**

```python
class JointTransform:
    def __init__(self, input_size=224, hflip_prob=0.5, rotate_deg=5,
                 rotate_prob=0.5, brightness=0.05, contrast=0.05,
                 brightness_prob=0.5, contrast_prob=0.5):
        # ...

    def __call__(self, image, skeleton, hand_skeleton):
        # 1. Resize
        image = TF.resize(image, (self.input_size, self.input_size))

        # 2. Random Flip
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            skeleton = torch.flip(skeleton, dims=[2])      # 水平翻转人体骨架
            hand_skeleton = torch.flip(hand_skeleton, dims=[2])  # 水平翻转手部骨架

        # 3. Random Rotate（对 RGB、人体骨架、手部骨架同时旋转）
        if self.rotate_deg > 0 and random.random() < self.rotate_prob:
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            skeleton = TF.rotate(skeleton.unsqueeze(0), angle,
                                 interpolation=InterpolationMode.BILINEAR).squeeze(0)
            hand_skeleton = TF.rotate(hand_skeleton.unsqueeze(0), angle,
                                      interpolation=InterpolationMode.BILINEAR).squeeze(0)

        # 4. Brightness & Contrast
        if random.random() < self.brightness_prob:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            image = TF.adjust_brightness(image, brightness_factor)

        if random.random() < self.contrast_prob:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            image = TF.adjust_contrast(image, contrast_factor)

        # 5. RGB to Tensor + Normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, skeleton, hand_skeleton
```

#### 2.5 合成数据生成（可选）

除了人工标注数据外，本项目还通过 **mmdet + mmpose + SAM** 三阶段 pipeline 合成带有手部异常的数据样本，用于扩充手部畸形类别的训练数据：

**合成原理概述：**

```
┌──────────────────────────────────────────────────────────────┐
│               合成数据生成 Pipeline                            │
│                                                              │
│  源图片 A (含手部)     目标图片 B (含完整人体)                │
│       │                          │                           │
│       ▼                          ▼                           │
│  ┌─────────┐              ┌─────────┐                       │
│  │ mmdet   │              │ mmdet   │                       │
│  │ 检测人体 │              │ 检测人体 │                       │
│  └────┬────┘              └────┬────┘                       │
│       ▼                       ▼                              │
│  ┌─────────┐              ┌─────────┐                       │
│  │ mmpose  │              │ mmpose  │                       │
│  │ 关键点检测 │             │ 关键点检测 │                       │
│  └────┬────┘              └────┬────┘                       │
│       │ (elbow, wrist)        │ (elbow, wrist)               │
│       ▼                       ▼                              │
│  ┌─────────┐              ┌─────────┐                       │
│  │  SAM    │              │  无需SAM │                       │
│  │ 提取手部 │              │          │                       │
│  │ patch+mask│             │          │                       │
│  └────┬────┘              └────┬────┘                       │
│       │                       │                               │
│       ▼                       ▼                               │
│  ┌─────────────────────────────────────┐                    │
│  │   仿射变换对齐 + 融合               │                    │
│  │   - 以 Elbow 为锚点对齐             │                    │
│  │   - 随机角度扰动模拟解剖学异常      │                    │
│  │   - 按 mask 覆盖到目标图上          │                    │
│  └───────────────┬─────────────────────┘                    │
│                  ▼                                           │
│         合成图片 (手部异常样本)                                │
│         打上异常类型标签 → "1" (手部畸形)                     │
└──────────────────────────────────────────────────────────────┘
```

**详细流程：**

| 步骤 | 工具 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 1. 人体检测 | mmdet (RTMDet-Person) | 原始图像 | 人体 bounding box | 筛选置信度 > 0.4 的检测结果 |
| 2. 姿态估计 | mmpose (RTMPose-L Body) | 人体框 | elbow/wrist 关键点坐标 | COCO 17 keypoints，左侧 wrist=9, elbow=7; 右侧 wrist=10, elbow=8 |
| 3. 手部分割 | SAM (Segment Anything) | 关键点提示 (wrist, elbow) | 手部 mask + patch | 使用 point prompt 模式（wrist 为正样本，elbow 为负样本），提取手部区域的精确分割 |
| 4. 仿射变换 | 自定义算法 | 源 patch+mask, 目标关键点 | 变换后 patch+mask | 以 elbow 为锚点，向量对齐 + 随机角度扰动 (±15°) |
| 5. 图像融合 | 自定义算法 | 目标图+变换后 patch | 合成图片 | 按 mask 布尔索引覆盖，保留目标图其他部分 |

**仿射变换对齐逻辑：**

```
源向量 (elbow→wrist) → 计算角度和长度
目标向量 (elbow→wrist) → 计算角度和长度
仿射变换参数:
  - 旋转角度 = dst_angle - src_angle + 随机扰动 (±15°)
  - 缩放比例 = dst_length / src_length
```

**随机角度扰动是关键设计：** ±15° 的扰动可以模拟手部位置不正常的解剖学异常，使合成的样本更接近真实的手部畸形场景。

**仿射变换完整参数：**

| 参数 | 计算公式 | 说明 |
|------|---------|------|
| 旋转角度 | `dst_angle - src_angle + random(-15°, 15°)` | 向量对齐 + 随机扰动 |
| 缩放比例 | `dst_length / src_length` | 源向量与目标向量长度比 |
| 锚点 | 源 elbow 坐标 | 以肘部为旋转中心 |
| Padding | `max(patch_width, patch_height)` | 防止旋转裁剪 |

**合成数据标注：**

| 文件名 | 标签文件 | 标签内容 | 说明 |
|--------|---------|---------|------|
| `fusion_result/washed_images/result_xxx.jpg` | `result_xxx.txt` | `1` | 手部畸形样本（异常） |

**合成代码文件：**

| 文件 | 功能 |
|------|------|
| `sam_fusion.py` | 核心合成逻辑：mmdet+mmpose+SAM 完整 pipeline |
| `fusion_test/fusion.py` | 合成参考实现（含仿射变换和对齐逻辑） |
| `mmpose_sk.py` | 人体/手部关键点检测和骨架热力图生成 |
| `sam.py` | SAM 资产库构建脚本 |

**SAM 关键点提示说明：**

| 点 | 坐标来源 | 标签 | 作用 |
|----|---------|------|------|
| 手腕 (wrist) | RTMPose 关键点 | `1`（前景） | 告诉 SAM "这里是要分割的区域" |
| 肘部 (elbow) | RTMPose 关键点 | `0`（背景） | 告诉 SAM "这里不是要分割的区域" |

**合成数据输出目录：**

| 目录 | 内容 |
|------|------|
| `data/fusion_result_1/` | 第 1 批次合成数据 |
| `data/fusion_result_2/` | 第 2 批次合成数据 |
| `data/fusion_result_3/` | 第 3 批次合成数据 |

每个目录包含：
- `washed_images/` — 合成后的图片 (JPG)
- `washed_multiple_labels/` — 对应的标签文件 (TXT)

> **合成结果示例图**
![](./pic/fusion1.png)

#### 2.6 数据集统计

| 数据集 | 图片数 | 正常 | 异常 | 主要异常分布 |
|--------|--------|------|------|-------------|
| **训练集** | 50,459 | 21,036 | 29,423 | 5:27,266, 1:5,356, 7:4,214, 4:3,338, 6:2,095, 8:709, 3:654, 2:129 |
| **验证集** | 14,417 | 6,098 | 8,319 | 5:7,687, 1:1,449, 7:1,209, 4:948, 6:634, 8:199, 3:178, 2:30 |
| **测试集** | 7,209 | 2,982 | 4,227 | 5:3,912, 1:756, 7:633, 4:496, 6:348, 8:128, 3:97, 2:25 |
| **华为域 train** | 248 | 124 | 124 | 1:124（简化为 2 类） |
| **华为域 test** | 248 | 124 | 124 | 1:124（简化为 2 类） |

**说明：**
- 主数据集（rearranged_*）包含 8 类异常，其中第 5 类（脸部畸形/身体畸形）占绝大多数
- 华为域数据简化为 2 类：normal 和 1（异常），用于域自适应微调评估

---

### 3. 训练策略

#### 3.1 分阶段训练

| 阶段 | Epoch | 操作 |
|------|-------|------|
| 第一阶段 | 1-8 | 冻结 backbone，仅训练分类头和注入层 |
| 第二阶段 | 9-40 | 解冻 backbone，全参数训练 |

**关键变量说明：**

```python
# load_parameter 是硬编码的 bool 变量（sk_resnet.py 第 360 行）：
load_parameter = False

# 前 free_epochs(8) 轮冻结 backbone
if epoch == free_epochs or load_parameter:
    for p in backbone_params:
        p.requires_grad = True
```

#### 3.2 优化手段

| 手段 | 说明 | 效果 |
|------|------|------|
| Warmup | 前 10 个 epoch 线性增加学习率 | 避免训练初期不稳定 |
| Cosine Annealing | 10 个 epoch 后余弦退火 | 平滑降低学习率 |
| EMA | 维护模型权重的指数移动平均 | 提升泛化能力 |
| 混合精度训练 | 使用 AMP (autocast + GradScaler) 进行 half-precision 计算 | 节省显存，加速训练 |
| 多损失联合优化 | Focal Loss 解决类别不平衡 | 提升异常样本召回率 |
| 梯度缩放 | GradScaler 防止梯度溢出 | 稳定训练过程 |

**训练输出示例：**

```
Epoch [1/40] Train Loss: 0.5678 Val Loss: 0.4321 EMA Val Loss: 0.4200
Train Acc: 0.8500 Val Acc: 0.8200 EMA Val Acc: 0.8300
Train Recall: 0.8600 Val Recall: 0.8400 EMA Val Recall: 0.8500
Train Precision: 0.8400 Val Precision: 0.8300 EMA Val Precision: 0.8400
--- Feature Fusion Weights ---
Avg Pool Weight (Alpha): 0.5234
Max Pool Weight (Beta):  0.4766
------------------------------
LR: 0.000034
```

**训练时间估算：**
- 单卡 (RTX 4090)：约 1.5 小时

#### 3.3 域自适应微调

针对跨域场景（从源域数据迁移到目标域数据），提供专门的微调脚本：

```bash
python domain_fine_tuning.py
```

**微调策略：**
- 低学习率（3e-5）：避免破坏预训练权重
- 混合数据采样：30% 源域 + 70% 目标域
- Backbone 冻结：前 5 个 epoch 冻结，防止灾难性遗忘
- Focal Loss：关注难样本
- EMA：decay=0.999

**DomainFineTuner 类：**

```python
class DomainFineTuner:
    def __init__(self, model, device='cuda', lr=3e-5, weight_decay=3e-3, ema_decay=0.999):
        # 分层学习率
        self.optimizer = optim.Adam([
            {"params": fusion_params, "lr": lr * 10},   # 5e-4
            {"params": backbone_params, "lr": lr},       # 3e-5
            {"params": head_params, "lr": lr},           # 3e-5
            {"params": inject_params, "lr": lr},         # 3e-5
        ], weight_decay=weight_decay)

        self.criterion_multi = FocalLoss(gamma=2.5)
        self.criterion_binary = FocalLoss(gamma=2.5)

        self.ema_model = AveragedModel(model, avg_fn=lambda averaged, current, num:
                                       ema_decay * averaged + (1 - ema_decay) * current)
```

**微调流程：**

```python
# 1. 加载预训练模型
model = ConvNeXtTinyFusion(num_classes=8, input_size=224)
model.load_state_dict(torch.load('models_parameter/resnet/best_model_ema_opt.pth'))

# 2. 加载数据
# 源域：rearranged_train (HumanRefiner 数据, 50,459 张)
# 目标域：huawei_data/train (华为域数据, 248 张)

# 3. 创建混合数据加载器
# 采样比例：源域 30%，目标域 70%
source_samples = max(1, int(len(source_train_dataset) * 0.3))  # 15,138
target_samples = max(1, int(len(target_train_dataset) * 0.7))  # 174

source_loader = DataLoader(source_train_dataset, batch_size=32, sampler=source_sampler)
target_loader = DataLoader(target_train_dataset, batch_size=32, sampler=target_sampler)

# 4. 执行微调
fine_tuner = DomainFineTuner(model, device=device, lr=3e-5, weight_decay=3e-3, ema_decay=0.999)

fine_tuner.fine_tune(
    source_loader=source_loader,
    target_loader=target_loader,
    num_epochs=15,
    save_path='models_parameter/finetuned/cross_domain_fine_tuned.pth',
    freeze_backbone=True,
    free_epochs=5,
    source_weight=0.3,
    target_val_loader=target_val_loader,
    batch_size=64
)
```

**微调参数：**

| 参数 | 值 | 说明 |
|------|------|------|
| lr | 3e-5 | 低学习率防止灾难性遗忘 |
| weight_decay | 3e-3 | 正则化系数 |
| num_epochs | 15 | 微调轮数 |
| freeze_backbone | True | 冻结 backbone |
| free_epochs | 5 | 前 5 轮冻结 backbone |
| source_weight | 0.3 | 源域数据在混合数据中的比例 |
| ema_decay | 0.999 | EMA 衰减率 |
| batch_size | 64 | 批次大小 |

**华为域微调数据可视化：**

> **此处添加 finetune 数据集 train 和 test 中每一张图片用 debug.py 生成的可视化图片**


#### 3.4 华为域微调使用说明

| 复现者类型 | 微调用数据 | 说明 |
|-----------|-----------|------|
| 华为内部 | `huawei_data/` | 可直接使用内部数据域自适应微调 |
| 外部复现者 | 自有数据集 | 可替换为自己的域数据，或跳过域自适应微调直接使用预训练模型 |

**华为域数据来源：**
- Z-Image ComfyUI 生成的图片
- 华为内部的业务图片
- 248 张数据量，每张图片都需要人工审核质量

**使用建议：** 外部复现者如果无法获取华为内部数据，可以：
1. 直接使用 `best_model_ema_opt.pth`（预训练模型）进行推理
2. 用自己的目标域数据替换 `huawei_data/` 目录，重新执行域自适应微调

---

## 五、算法中间过程可视化

### 1. 空间异常热力图

模型输出 `patch_maps.s3` 和 `patch_maps.s4`，可可视化模型在图像上的关注区域：

- 热力图颜色越亮，表示模型认为该区域异常可能性越高
- 可用于解释模型判断依据

### 2. 骨架热力图

中间过程可展示：
- 原始 RGB 图像
- 人体骨架关键点检测结果
- 手部骨架关键点检测结果
- 生成的骨架热力图（通道 3 和通道 4 的输入）

### 3. 推理调试可视化（debug.py）

`debug.py` 可以可视化模型推理的 11 个中间过程：

| 子图编号 | 内容 | 说明 |
|---------|------|------|
| 1 | RGB 输入 | 原始输入图像（224x224） |
| 2 | 人体骨架热力图 | 输入的第 3 通道 |
| 3 | 手部骨架热力图 | 输入的第 4 通道 |
| 4 | Stem 输出 | 最大激活值（96 通道，56x56 → 224x224） |
| 5 | Stage 1 输出 | 最大激活值（96 通道，56x56 → 224x224） |
| 6 | Stage 2 输出 | 最大激活值（192 通道，28x28 → 224x224） |
| 7 | Stage 3 输出 | 最大激活值（384 通道，14x14 → 224x224） |
| 8 | Stage 4 输出 | 最大激活值（768 通道，7x7 → 224x224） |
| 9 | 空间注意力图 | CBAM 空间注意力（7x7 → 224x224） |
| 10 | Patch Score S4 | 补丁级异常分数（S4 层） |
| 11 | Patch Score S3 | 补丁级异常分数（S3 层） |

**使用示例：**

```python
# 运行推理调试
python debug.py

# 修改 debug.py 中的图片路径以测试不同图片：
# image_path = './debug/a_223.png'  # 替换为你的图片路径

# 输出：
# - 推理调试图：./debug/detailed_debug_result.png（11 个子图）
# - 推理结果：Binary Probability、Multi Logit Mean
```

---

## 六、算法细节补充

### 各子算法的作用

| 子算法 | 作用环节 | 效果 |
|--------|---------|------|
| RTMPose | 预处理（骨架检测） | 生成人体和手部骨架热力图，为模型提供人体结构先验 |
| SAM | 数据增强（可选） | 分割手部 patch 并融合，增加手部异常样本 |
| Cross-Attention | 特征提取阶段 | 将骨架信息注入 CNN 特征图，引导模型关注人物部位 |
| SpatialAttention | 特征融合阶段 | 增强空间感知能力，关注异常区域 |
| PatchAnomalyHead | 异常检测阶段 | 输出补丁级异常分数，支持可解释性 |
| Focal Loss | 损失计算阶段 | 解决类别不平衡问题，提升难样本学习效果 |
| EMA | 训练后处理 | 维护模型权重的移动平均，提升泛化能力 |

### 模型参数分类与学习率

| 参数组 | 包含模块 | 学习率 | 说明 |
|--------|---------|--------|------|
| backbone_params | ConvNeXt 主干网络（除 stem、attn、fusion） | 1e-4 | 预训练权重，小学习率 |
| head_params | classifier, head | 1e-4 | 随机初始化的分类头，较大学习率 |
| inject_params | attn, stem_conv | 1e-4 | 新增的注入层 |
| fusion_params | fusion_logits | 5e-3 | 融合权重，最大学习率 |

**优化器配置：**

```python
optimizer = optim.Adam([
    {"params": fusion_params, "lr": 5e-3},
    {"params": backbone_params, "lr": 10e-5},
    {"params": head_params, "lr": 10e-5},
    {"params": inject_params, "lr": 10e-5},
], weight_decay=3e-3)
```

**学习率调度：**

```python
# Warmup + 余弦退火
warmup_epochs = 10
free_epochs = 8  # 前 8 轮冻结 backbone

warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_then_const)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=3e-5)

def warmup_then_const(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # 0.1 → 1.0
    return 1.0
```

**参数分组逻辑（sk_resnet.py）：**

```python
for name, p in model.named_parameters():
    # 逻辑：只要包含 classifier 或 head，就是分类头
    if "classifier" in name or "head" in name:
        head_params.append(p)
    # 只要包含 attn 或 cross_attn 或 inject，就是注入层
    elif "attn" in name or "inject" in name or "stem_conv" in name:
        inject_params.append(p)
    else:
        backbone_params.append(p)
```

### 版本兼容问题（MMCV 伪装）

mmpose 0.10.7 和 mmdet 3.3.0 对 mmcv 版本的需求不同：
- mmpose 需要 `mmcv>=1.5.0`（旧版 API）
- mmdet 3.3.0 需要 `mmcv>=2.0.0`（新版 API）

两者对 mmcv 版本的要求是冲突的。代码在 `mmpose_sk.py` 中通过版本伪装绕过检查：

```python
import mmcv
mmcv.__version__ = '2.1.0'  # 必须在 import mmdet 之前执行
import mmdet
```

**原理：** 代码先导入原始 mmcv 版本，然后通过修改 `mmcv.__version__` 字符串让 mmdet 的版本检测逻辑认为 mmcv 是 2.1.0 版本，从而通过 mmdet 的兼容性检查。

**实际运行时使用的 mmcv 版本：** `mmcv>=2.0.0`（即 2.1.0），通过 `pip install mmcv==2.1.0` 安装。

**不做这个 hack 的后果：** mmdet 的版本检测会失败，报错类似 `ValueError: mmcv version error`，导致 mmdet 无法导入。

**正确的 import 顺序：**

```python
# 必须在 import mmdet 之前执行
import mmcv
mmcv.__version__ = '2.1.0'   # 欺骗 mmdet，让它以为我们用的是 2.1.0
import mmdet
from mmdet.apis import init_detector, inference_detector
```

### 预训练权重下载

**自动下载（需网络通畅）：**

项目代码中会通过 `init_detector` 和 `init_model` 自动从 OpenMMLab 下载以下预训练权重：

| 模型 | 完整 URL |
|------|---------|
| RTMDet-Person | `https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth` |
| RTMPose-L Body | `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth` |
| RTMDet-Hand | `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth` |
| RTMPose-M Hand | `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth` |

**权重保存位置：** 下载到 `~/.cache/torch/hub/checkpoints/` 或当前工作目录。

**SAM 模型（手动下载）：**

```bash
wget https://github.com/facebookresearch/segment-anything/releases/download/v1.0/sam_vit_h_4b8939.pth -O sam_vit_h_4b8939.pth
```

需手动下载 `sam_vit_h_4b8939.pth`（2.5 GB）到项目根目录。

---

## 七、能力边界

### 能力边界表

| 场景 | 能否检测 | 效果 | 原因 |
|------|---------|------|------|
| 手部畸形 | 能 | 好 | 训练数据覆盖充分 |
todo:

### 测试充分性

已覆盖的测试场景：

| 测试维度 | 测试内容 |
|---------|---------|
| 正常图片 | 无异常的 AIGC 图片 |
| 各类异常 | 8 种异常类型的图片 |
| 不同分辨率 | 不同尺寸的图片 |
| 不同背景 | 简单背景、复杂背景 |
| 不同姿势 | 正面、侧面 |
| 跨域数据 | 源域（HumanRefiner）和目标域（华为域） |

### Good Case 与 Bad Case

**Good Case（算法判断正确的典型例子）：**

| 示例 | 场景 | 说明 |
|------|------|------|
| Case 1 | 图片中有明显手部畸形（6 根手指） | 准确识别为异常 |
| Case 2 | 图片中脸部五官错位 | 准确识别为异常 |
todo:

**Bad Case（算法判断错误的典型例子）：**

| 示例 | 场景 | 说明 |
|------|------|------|
| Case 1 | 模糊程度很轻的图片 | 误判为正常（漏检） |
| Case 2 | 人物姿势特殊、背景复杂的图片 | 可能误判（误检） |
todo:

---

## 八、可维护性与可魔改点

### 当前算法的局限性

| 局限性 | 说明 |
|--------|------|
| 输入分辨率低 | 输入的图片尺寸会转化成224*224，丢失部分信息 |
| 依赖骨架检测 | 如果 RTMPose 检测失败会影响效果 |
| 异常类型固定 | 仅支持 8 种异常类型，不支持自定义 |
| 需要标注数据 | 每增加新的异常类型需要重新训练 |

### 可魔改的点（交付后可自行修改）

| 模块 | 可修改内容 | 修改难度 | 文件位置 |
|------|-----------|---------|---------|
| Backbone | 替换为其他模型（ResNet, ViT 等） | 中等 | `models/ConvNeXt.py` |
| Loss 函数 | 修改损失函数权重或类型 | 简单 | `loss_calculator.py` |
| 异常类别 | 增加/减少异常分类数 | 中等 | `models/ConvNeXt.py` 输出层 |
| 数据增强 | 修改增强策略 | 简单 | `sk_dataset.py` |
| 阈值搜索 | 修改阈值搜索范围 | 简单 | `eval.py` |
| 注意力模块 | 替换 Cross-Attention 实现 | 中等 | `models/ConvNeXt.py` |
| 训练策略 | 修改学习率调度、冻结策略 | 中等 | `sk_resnet.py` |
| 推理接口 | 封装为 API 服务 | 简单 | 自行扩展 |

**主要可维护点：**
1. **模型结构清晰**：各模块解耦，可独立替换
2. **配置化管理**：超参数集中在 `config.yaml`
3. **模块化设计**：数据集、模型、评估各模块独立
4. **易于扩展**：新增异常类型只需修改输出层和训练数据

---

## 九、项目文件索引

| 文件 | 功能 |
|------|------|
| `sk_resnet.py` | 主训练脚本（训练 + EMA + 评估 + 保存） |
| `domain_fine_tuning.py` | 域自适应微调脚本（DomainFineTuner 类） |
| `eval.py` | 独立评估脚本（阈值优化 + 测试集评估 + 概率分布可视化） |
| `debug.py` | 推理调试脚本（可视化中间过程） |
| `sk_dataset.py` | 数据集类 + JointTransform + FocalLoss + LabelSmoothingBCEWithLogitsLoss |
| `loss_calculator.py` | 损失函数计算器（多任务损失 + 阈值优化 find_optimal_threshold） |
| `evaluate/evaluate.py` | 评估指标计算（calc_metrics_balance, calc_tf） |
| `evaluate/ml_evaluate.py` | 多标签评估（evalModel_multiple） |
| `models/ConvNeXt.py` | 主模型 ConvNeXtTinyFusion（含 SpatialAttention, StandardCrossAttention, PatchAnomalyHead） |
| `models/fs_resnet.py` | 备选模型 MultiScaleFusionResNet（基于 ResNet50） |
| `mmpose_sk.py` | 人体/手部关键点检测 + 骨架热力图生成（RTMPose-L Body, RTMPose-M Hand） |
| `sam_fusion.py` | SAM 手部 patch 融合到目标图（完整 pipeline） |
| `dataWash.py` | 数据集清洗脚本（find_label_files, find_multiple_label_files, label_count） |
| `rearrange_data.py` | 数据集合并重组（7:2:1 划分，软链接） |
| `rearrange_finetune_data.py` | 微调数据集重组 |
| `config.yaml` | 全局配置文件 |
| `sam_vit_h_4b8939.pth` | SAM ViT-H 模型权重（2.5 GB） |

---

## 十、环境依赖

### 核心依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10 | 编程语言 |
| PyTorch | 2.4.0+cu121 | 深度学习框架 |
| CUDA | 可用 | GPU 加速 |
| torchvision | >= 0.19 | 图像处理和模型 |
| OpenMMLab MMPose | 0.10.7 | 人体关键点检测 |
| MMDetection | 3.3.0 | 目标检测 |
| MMCV | 2.1.0 | 基础框架（需伪装版本号） |
| NumPy | - | 数值计算 |
| Pillow | - | 图像处理 |
| Matplotlib | - | 可视化 |
| OpenCV | - | 图像操作 |

### 安装步骤

```bash
# 使用 conda 环境
conda create -n aigc python=3.10 -y
conda activate aigc

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 OpenMMLab 系列
pip install mmpose==0.10.7
pip install mmdet==3.3.0
pip install mmcv==2.1.0

# 安装 Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git

# 其他依赖
pip install numpy Pillow matplotlib tqdm pyyaml opencv-python
```

### 模型参数量与训练资源

| 指标 | 值 | 说明 |
|------|------|------|
| 模型参数量 | ~28M | ConvNeXt-Tiny 级别，轻量级模型 |
| 输入尺寸 | 224×224 | 固定输入尺寸 |
| 输入通道 | 5 | RGB(3) + 人体骨架(1) + 手部骨架(1) |
| 特征维度 | 4520 | 9 路特征融合后的总维度 |
| 分类头输出 | 9 | 8(多标签) + 1(二分类) |
| 显存占用 (batch=64) | ~23GB | GPU 显存 |
| 训练速度 (单卡) | 约 140 秒/epoch | RTX 4090, batch=64 |
| 推理速度 (单张) | 0.3-2 张/秒 | RTX 4090,含文件IO |
| 推理速度 (单张) | 100 张/秒 | RTX 4090,不含文件输入输出 |
| 推理速度 (batch=64) | 200-400 张/秒 | RTX 4090,含文件IO |
| 推理速度 (batch=64) | 667 张/秒 | RTX 4090,不含文件IO |

---

---

## 附录

### 附录A：华为测试集推理可视化

本项目的完整推理可视化结果请参见附录文档：[finetune_huawei_test_app.md](finetune_huawei_test_app.md)

该附录包含华为数据集（huawei_data/test）中所有测试图片的详细推理过程可视化，包括：
- 每张图片的11子图详细推理可视化（RGB输入、骨架热力图、Stem/Stage激活、注意力图、Patch Scores）
- 预测概率和真实标签
- 正确/错误预测标记
- 统计信息汇总

**相对路径引用：** [docs/finetune_huawei_test_app.md](./finetune_huawei_test_app.md)

---

---

## 附录

### 附录A：华为测试集推理可视化

本项目的完整推理可视化结果请参见附录文档：[finetune_huawei_test_app.md](finetune_huawei_test_app.md)

该附录包含华为数据集（huawei_data/test）中所有测试图片的详细推理过程可视化，包括：
- 每张图片的11子图详细推理可视化（RGB输入、骨架热力图、Stem/Stage激活、注意力图、Patch Scores）
- 预测概率和真实标签
- 正确/错误预测标记
- 统计信息汇总

**相对路径引用：** [docs/finetune_huawei_test_app.md](./finetune_huawei_test_app.md)

---

## 附录

### 附录A：华为测试集推理可视化

本项目的完整推理可视化结果请参见附录文档：[finetune_huawei_test_app.md](finetune_huawei_test_app.md)

该附录包含华为数据集（huawei_data/test）中所有测试图片的详细推理过程可视化，包括：
- 每张图片的11子图详细推理可视化（RGB输入、骨架热力图、Stem/Stage激活、注意力图、Patch Scores）
- 预测概率和真实标签
- 正确/错误预测标记
- 统计信息汇总

**相对路径引用：** [docs/finetune_huawei_test_app.md](./finetune_huawei_test_app.md)

---

## 附录

### 附录A：华为测试集推理可视化

本项目的完整推理可视化结果请参见附录文档：[finetune_huawei_test_app.md](finetune_huawei_test_app.md)

该附录包含华为数据集（huawei_data/test）中所有测试图片的详细推理过程可视化，包括：
- 每张图片的11子图详细推理可视化（RGB输入、骨架热力图、Stem/Stage激活、注意力图、Patch Scores）
- 预测概率和真实标签
- 正确/错误预测标记
- 统计信息汇总

**相对路径引用：** [docs/finetune_huawei_test_app.md](./finetune_huawei_test_app.md)

---

## 附录

### 附录A：华为测试集推理可视化

本项目的完整推理可视化结果请参见附录文档：[finetune_huawei_test_app.md](finetune_huawei_test_app.md)

该附录包含华为数据集（huawei_data/test）中所有测试图片的详细推理过程可视化，包括：
- 每张图片的11子图详细推理可视化（RGB输入、骨架热力图、Stem/Stage激活、注意力图、Patch Scores）
- 预测概率和真实标签
- 正确/错误预测标记
- 统计信息汇总

**相对路径引用：** [docs/finetune_huawei_test_app.md](./finetune_huawei_test_app.md)

---

## 附录

### 附录A：华为测试集推理可视化

本项目的完整推理可视化结果请参见附录文档：[finetune_huawei_test_app.md](finetune_huawei_test_app.md)

该附录包含华为数据集（huawei_data/test）中所有测试图片的详细推理过程可视化，包括：
- 每张图片的11子图详细推理可视化（RGB输入、骨架热力图、Stem/Stage激活、注意力图、Patch Scores）
- 预测概率和真实标签
- 正确/错误预测标记
- 统计信息汇总

**相对路径引用：** [docs/finetune_huawei_test_app.md](./finetune_huawei_test_app.md)
