# Molecular GTN Contrastive Pipeline

一个面向分子表征学习的预训练项目：把 SMILES 分子字符串转换成图结构，用 Graph Transformer Network（GTN）做编码，再用对比学习把分子映射到向量空间。

这个 README 既写给第一次接触项目的新手，也写给想快速接手代码的人。你可以把它当成：

- 项目介绍
- 技术设计说明
- 代码结构地图
- 环境与运行手册
- 常见参数与输出说明

---

## 1. 这个项目在做什么？

项目目标是学习一个通用的分子向量表示（embedding）：

- 输入：分子 SMILES
- 中间：转成 PyTorch Geometric 的图对象
- 模型：6 层 Graph Transformer
- 训练目标：对比学习（NT-Xent / InfoNCE）
- 输出：每个分子的向量表示，可用于检索、聚类、下游 QSAR/分类/回归任务

简单理解：

- **同一个分子** 的不同扰动视图，模型应该学得“更像”
- **不同分子** 的表示，模型应该学得“更不像”

所以，这个项目不是做监督分类，而是在做**自监督预训练**

---

## 2. 技术思路总览

整个流程分成 5 步：

1. **读取 CSV**
   - 从 `pretraining.csv` 中读入 SMILES

2. **分子图预处理**
   - 用 RDKit 解析 SMILES
   - 提取原子特征和键特征
   - 计算 Laplacian Positional Encoding（LapPE）
   - 存成 LMDB，避免训练时重复做昂贵 CPU 计算

3. **图增强**
   - 对原子和键随机 mask
   - 每个分子生成多个 masked views
   - 原图作为 anchor，masked 图作为 positive

4. **GTN 对比学习预训练**
   - 模型输出分子级向量
   - 再接一个 2 层 MLP projection head
   - 用 NT-Xent loss 训练

5. **推理导出**
   - 读取训练好的模型
   - 对前 10 个分子做编码
   - 导出分子向量 CSV 文件

---

## 3. 为什么这样设计？

### 3.1 为什么用图而不是直接用字符串？

SMILES 是文本格式，但分子本质上更像一张图：

- 原子 = 节点
- 化学键 = 边

图神经网络/图 Transformer 更容易显式利用结构关系。

### 3.2 为什么预处理后存成 LMDB？

因为分子预处理很贵：

- RDKit 解析耗 CPU
- 特征提取耗 CPU
- LapPE 特征分解耗 CPU

如果训练时每个 batch 都现算，会导致 GPU 等 CPU，训练很慢。

LMDB 的作用：

- 让预处理和训练解耦
- 适合大规模样本
- 顺序读写快
- 比大量小文件更稳定

### 3.3 为什么采用 Producer-Consumer？

多进程都直接写 LMDB，很容易有：

- 事务冲突
- 写锁竞争
- 性能下降
- 极端情况下数据库损坏

所以这里采用：

- **多个 Producer**
  - 只负责 RDKit 解析、特征提取、LapPE 计算、序列化
- **一个 Consumer**
  - 只负责写 LMDB

这是更稳妥的并发设计。

### 3.4 为什么要 Laplacian Positional Encoding？

标准图 Transformer 只看邻接结构时，不一定能很好理解“节点在图里的拓扑位置”。

LapPE 用图拉普拉斯矩阵的特征向量来描述节点的结构位置，可以帮助模型更好理解：

- 节点在图中的角色
- 图的整体拓扑结构
- 局部与全局关系

### 3.5 为什么用对比学习？

因为大规模分子数据通常没有标签，但数量很多。

对比学习能在无标签数据上训练，让模型先学到“好的表示”，后续下游任务再微调。

---

## 4. 输入数据说明

### 4.1 `pretraining.csv`

当前项目默认输入文件。

字段示例：

- `smiles`
- `source_file`
- `source_line`

其中：

- `smiles` 是实际的分子字符串
- `source_file` / `source_line` 主要用于追踪来源、日志和错误定位

### 4.2 `分子信息汇总.md`

这是项目的特征设计依据文档，主要描述：

- 原子层特征
- 键层特征
- 如何把某些边信息池化到节点上

当前代码使用了文档中的核心思路，并将边相关统计拼接到节点特征中。

---

## 5. 当前实现的核心特征

### 5.1 原子特征

当前实现位于 `mol_gtn/features.py:85`，主要包括：

- 原子序数
- 形式电荷
- 自由基电子数
- 芳香性标记
- 环结构标记
- 隐式价
- 手性标签 one-hot
- 杂化状态 one-hot
- 邻接键类型计数
  - single
  - double
  - triple
  - aromatic
- 邻接立体化学状态
  - cis
  - trans
  - none

### 5.2 键特征

位于 `mol_gtn/features.py:99`，包括：

- 键类型 one-hot
- 键立体化学 one-hot
- 键方向 one-hot
- 共轭标记
- 是否在环中

### 5.3 LapPE

位于 `mol_gtn/lap_pe.py:7`。

实现逻辑：

- 计算图拉普拉斯矩阵
- 做特征分解
- 跳过最平凡的第一个特征向量
- 取前 `k` 个有效特征向量
- 不足 `k` 时用 0 填充

默认：

- `lap_pe_dim = 8`

### 5.4 关于 padding mask

预处理阶段会存：

- `lap_pe`
- `lap_pe_valid_mask`
- `padding_mask`

其中：

- `lap_pe_valid_mask` 表示 LapPE 的哪些维度是真实特征向量，哪些是补零
- `padding_mask` 当前实现中对单个图的真实节点全部为 `False`
- 真正在 Transformer 中用于屏蔽“批内补齐节点”的 mask，是 dense batching 时动态生成的节点 mask

也就是说：

- **图内真实节点不会被误屏蔽**
- **批处理时为了对齐长度新增的 padded nodes 会被 attention/pooling 屏蔽**

这部分逻辑位于 `mol_gtn/model.py:89`。

---

## 6. 模型设计

模型定义在 `mol_gtn/model.py:58`。

### 6.1 Backbone：6-layer GTN

模型的核心结构：

- 输入：节点特征 + LapPE 拼接
- 线性投影到隐藏维度
- 多层 Graph Transformer blocks
- 使用边特征构建 attention bias
- 输出节点级表示

默认超参数：

- `hidden_dim = 256`
- `num_layers = 6`
- `num_heads = 8`
- `dropout = 0.1`

### 6.2 Edge Bias

边特征不是简单丢掉，而是进入 attention bias：

- 先把边特征投影到多头维度
- 再作为注意力分数中的偏置信号

这样做的好处是：

- 键类型会影响注意力
- 立体化学会影响注意力
- 共轭/环结构会影响注意力

### 6.3 三层输出语义

模型保留 3 种级别表示：

- **Node-level**
  - 每个原子的表示
- **Bond-level**
  - 每条边的表示
- **Mol-level**
  - 整个分子的表示

其中对比学习真正使用的是：

- `mol_embeddings`

### 6.4 Global Attention Pooling

位于 `mol_gtn/model.py:44`。

作用：

- 把节点表示聚合成一个分子向量
- 不是简单平均，而是让模型学习“哪些节点更重要”

并且会在 softmax 前把 padded nodes 设成 `-inf`，避免污染最终分子向量。

---

## 7. 对比学习设计

### 7.1 增强策略

增强逻辑在 `mol_gtn/augment.py:22`。

对于每个分子：

- 保留 1 份原始图
- 生成若干份随机 masked 图

默认配置：

- 正式预训练：`1 original + 6 masked`
- smoke test：`1 original + 2 masked`

支持通过参数调整 masked views 数量：

- `num_masked_views`
- `smoke_num_masked_views`

### 7.2 Mask 的内容

当前实现是最直接的特征 mask：

- 随机选一些节点，把节点特征置零
- 随机选一些边，把边特征置零

默认：

- `mask_ratio = 0.1`

### 7.3 正负样本构造

在 `mol_gtn/dataset.py:38` 中实现。

规则是：

- 原图 = anchor
- 每个 masked view = 一个 positive
- 同一 batch 中其他样本 = negatives

为了保持标准 pairwise NT-Xent 形式，代码会把：

- 1 个原图
- N 个 masked 图

展开成 N 对 `(anchor, positive)`。

### 7.4 NT-Xent Loss

实现位于 `mol_gtn/losses.py:6`。

核心思路：

- 先对向量做归一化
- 计算所有样本之间余弦相似度
- 排除自己和自己的相似度
- 让正样本对更接近
- 让其他样本更远

默认温度参数：

- `temperature = 0.07`

---

## 8. 为什么需要 Gradient Accumulation？

因为现在硬件是：

- 1 张 GPU
- 但模型和 batch 需求较大

Gradient Accumulation 的做法是：

- 每次只跑一个较小 mini-batch
- 先不立刻 `optimizer.step()`
- 累积若干次梯度后再更新

这样可以模拟更大的有效 batch size，而不必真的把所有样本一次塞进显存。

代码在 `mol_gtn/train.py:19`。

有效 pair 数大致可以理解为：

`batch_size × grad_accum_steps × num_masked_views`

---

## 9. 为什么代码看起来像是为多 GPU 留了接口？

虽然当前只跑单卡，但设计时尽量避免把代码写死成“永远只能单卡”。

所以当前结构已经为未来迁移到 DDP 留好了基础：

- 配置集中管理
- 数据集和 collate 分离
- 日志与输出路径统一
- batch 级别正负样本构造清晰

目前还**没有真正实现 DDP 训练**，但代码组织方式已经方便后续扩展。

---

## 10. 项目目录结构

```text
.
├── mol_gtn/
│   ├── __init__.py
│   ├── augment.py
│   ├── check_env.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── infer.py
│   ├── lap_pe.py
│   ├── lmdb_io.py
│   ├── losses.py
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── runtime.py
├── pretraining.csv
├── 分子信息汇总.md
├── run_full_pipeline.sh
├── setup_env.sh
├── smoke_test.sh
└── README.md
```

---

## 11. 每个 Python 文件是做什么的？

### `mol_gtn/config.py`

统一管理配置参数。

你可以把它理解成全项目的“默认设置表”。

包括：

- 数据路径
- 输出路径
- 模型超参数
- 训练超参数
- smoke test 参数

### `mol_gtn/features.py`

负责把 RDKit 的分子对象变成数值特征：

- 原子特征
- 键特征
- 特征维度校验

### `mol_gtn/lap_pe.py`

负责计算图的 Laplacian positional encoding。

### `mol_gtn/lmdb_io.py`

负责：

- 打开 LMDB
- 序列化 `Data`
- 反序列化 `Data`

### `mol_gtn/preprocess.py`

负责预处理主流程：

- 读取 CSV
- 多进程处理分子
- Producer 把样本放进 Queue
- Consumer 单独写 LMDB

这是数据准备阶段最关键的文件之一。

### `mol_gtn/augment.py`

负责图增强：

- 原子特征 mask
- 边特征 mask
- 生成多个 masked views

### `mol_gtn/dataset.py`

负责训练数据读取与拼 batch：

- 从 LMDB 读回图
- 构造 anchor / positive
- 用 PyG `Batch` 打包

### `mol_gtn/model.py`

定义模型结构：

- Transformer 层
- attention bias
- pooling
- projection head

### `mol_gtn/losses.py`

定义 NT-Xent loss。

### `mol_gtn/train.py`

负责训练主流程：

- 创建 DataLoader
- 前向传播
- loss 计算
- AMP
- gradient accumulation
- 保存最优权重

### `mol_gtn/infer.py`

负责推理和导出 embedding。

### `mol_gtn/check_env.py`

用于检查环境是否安装正确。

### `mol_gtn/utils/logging.py`

统一日志输出：

- 终端日志
- 文件日志
- 多进程日志队列

### `mol_gtn/utils/runtime.py`

负责随机种子等基础运行辅助功能。

---

## 12. 环境要求

项目默认目标环境：

- Conda
- Python 3.11
- PyTorch 2.5.1
- CUDA 版 PyTorch
- torch-geometric
- rdkit
- lmdb

当前安装脚本在 `setup_env.sh:1`。

---

## 13. 如何从零开始运行？

### 13.1 第一步：创建 Conda 环境

```bash
./setup_env.sh mol_gtn
```

创建完成后：

```bash
conda activate mol_gtn
```

### 13.2 第二步：检查环境

```bash
python -m mol_gtn.check_env --output-dir /hy-tmp/result
```

会检查：

- 关键包是否存在
- GPU 是否可见
- `/hy-tmp/result` 是否可写

### 13.3 第三步：先跑烟雾测试

```bash
./smoke_test.sh
```

它默认会：

- 只用前 100 行数据
- 只训练 1 个较小规模流程
- 使用 `1 original + 2 masked views`

### 13.4 第四步：跑完整流程

```bash
./run_full_pipeline.sh
```

---

## 14. Shell 脚本说明

### `setup_env.sh`

作用：

- 创建新的 Conda 环境
- 安装 PyTorch CUDA 版本
- 安装 PyG、RDKit、LMDB 等依赖
- 自动做一次环境检查

### `smoke_test.sh`

作用：

- 用小规模数据快速验证整个链路是否通

适合：

- 第一次部署环境
- 改了代码后快速回归
- 确认 GPU/LMDB/RDKit/训练逻辑都没坏

### `run_full_pipeline.sh`

作用：

- 一键串起：
  - 环境检查
  - 预处理
  - 训练
  - 推理

这是生产入口脚本。

---

## 15. `run_full_pipeline.sh` 里可以调哪些参数？

位于 `run_full_pipeline.sh:4` 的 `# CONFIGURATION` 段。

常见参数如下：

- `CSV_PATH`
  - 输入 CSV 路径
- `OUTPUT_DIR`
  - 输出目录
- `LMDB_PATH`
  - 正式预处理 LMDB 路径
- `SMOKE_LMDB_PATH`
  - 烟雾测试 LMDB 路径
- `MODEL_PATH`
  - 模型保存路径
- `EMBEDDINGS_PATH`
  - 推理结果导出路径
- `BATCH_SIZE`
  - 单步 batch size
- `GRAD_ACCUM_STEPS`
  - 梯度累计步数
- `LEARNING_RATE`
  - 学习率
- `MASK_RATIO`
  - mask 比例
- `NUM_MASKED_VIEWS`
  - 正式预训练每个分子生成多少 masked views
- `SMOKE_NUM_MASKED_VIEWS`
  - smoke test 时每个分子生成多少 masked views
- `CPU_WORKERS`
  - 预处理 CPU worker 数
- `EPOCHS`
  - 训练轮数
- `TEMPERATURE`
  - NT-Xent 温度参数
- `HIDDEN_DIM`
  - 隐层维度
- `LAP_PE_DIM`
  - LapPE 维度
- `QUEUE_SIZE`
  - Producer-Consumer 队列长度
- `WRITER_BATCH_SIZE`
  - LMDB writer 每次提交事务的 batch 大小
- `SMOKE_TEST`
  - 是否开启 smoke 模式，`1` 表示开启

示例：

```bash
BATCH_SIZE=8 \
GRAD_ACCUM_STEPS=4 \
EPOCHS=10 \
NUM_MASKED_VIEWS=6 \
CPU_WORKERS=48 \
./run_full_pipeline.sh
```

---

## 16. 输出文件都在哪里？

所有大文件默认都写到：

```text
/hy-tmp/result
```

常见输出包括：

- `project.log`
  - 项目日志
- `pretraining.lmdb` 或 `smoke_pretraining.lmdb`
  - 预处理后的图数据库
- `model_final.pth`
  - 当前最佳模型权重
- `top10_embeddings.csv`
  - 推理导出的前 10 个分子向量

---

## 17. 训练阶段做了哪些工程优化？

### 17.1 多进程预处理

利用 CPU 做：

- RDKit 解析
- 特征提取
- LapPE 计算

### 17.2 单写者 LMDB

避免多进程同时写库导致问题。

### 17.3 GPU 加速

模型训练默认走 CUDA。

### 17.4 AMP

训练中使用自动混合精度：

- 更省显存
- 往往更快

### 17.5 Gradient Accumulation

支持单卡模拟更大有效 batch。

### 17.6 `tqdm` + `logging`

训练和预处理都有：

- 终端进度条
- 持久日志文件

---

## 18. 新手最容易困惑的几个点

### Q1：为什么预处理慢？

因为不是简单读 CSV，而是在做：

- RDKit 化学解析
- 图特征构造
- LapPE 特征分解
- 序列化

这是正常的。

### Q2：为什么 smoke test 只取 100 条？

目的是：

- 快速发现环境问题
- 验证代码逻辑
- 节省时间

不是为了得到高质量模型。

### Q3：为什么 loss 看起来没有很快降？

对比学习预训练本来就不像监督分类那样直观，而且 smoke test 数据太少、epoch 太少，所以只看 loss 不能说明全部问题。

### Q4：为什么一个分子会生成多个 masked views？

因为这样可以增加正样本对数量，让模型看到同一个分子的多种扰动版本。

### Q5：LMDB 里存的是增强后的数据吗？

不是。

LMDB 只存：

- 原始图
- 原始特征
- LapPE

增强是在训练时动态生成的。

---

## 19. 当前实现的已知简化

为了先把主干链路跑通，当前实现做了一些合理简化：

- 目前不做 3D 构象生成
- 当前 mask 是“特征置零”风格，不是更复杂的化学规则增强
- 当前 DDP 只是结构上预留，并未完整实现
- `padding_mask` 的实际主要作用体现在批内 dense padding 节点屏蔽

这些都不影响项目作为一个完整可运行的预训练基线。

---

## 20. 如果你想继续增强这个项目，可以从哪里开始？

建议的下一步方向：

- 增加更丰富的化学增强策略
- 增加验证集与更稳定的 checkpoint 选择标准
- 支持真正的 DDP 多卡训练
- 加入下游任务微调脚本
- 加入 TensorBoard / WandB 等实验追踪
- 增加单元测试和回归测试
- 对超大规模数据增加分片式 LMDB/manifest 管理

---

## 21. 一份最常用的运行清单

### 初始化环境

```bash
./setup_env.sh mol_gtn
conda activate mol_gtn
```

### 快速验证

```bash
./smoke_test.sh
```

### 正式预训练

```bash
NUM_MASKED_VIEWS=6 \
BATCH_SIZE=8 \
GRAD_ACCUM_STEPS=4 \
CPU_WORKERS=48 \
EPOCHS=10 \
./run_full_pipeline.sh
```

### 查看结果

```bash
ls -lh /hy-tmp/result
tail -n 50 /hy-tmp/result/project.log
```

---

## 22. 关键文件速查

- 项目配置：`mol_gtn/config.py:8`
- 特征工程：`mol_gtn/features.py:85`
- LapPE：`mol_gtn/lap_pe.py:7`
- 预处理主流程：`mol_gtn/preprocess.py:106`
- 数据集与 batch：`mol_gtn/dataset.py:11`
- 图增强：`mol_gtn/augment.py:22`
- 模型定义：`mol_gtn/model.py:58`
- 损失函数：`mol_gtn/losses.py:6`
- 训练入口：`mol_gtn/train.py:19`
- 推理入口：`mol_gtn/infer.py:19`
- 环境检查：`mol_gtn/check_env.py:13`
- 全流程脚本：`run_full_pipeline.sh:1`

---

## 23. 总结

这个项目的核心价值在于：

- 把大规模分子预处理和训练流程真正串起来
- 兼顾工程稳定性和研究扩展性
- 让新手也能从 `setup_env.sh` 到 `smoke_test.sh` 一路跑通

如果你是新手，推荐顺序是：

1. 先看本 README
2. 跑 `./setup_env.sh mol_gtn`
3. 跑 `./smoke_test.sh`
4. 再看 `mol_gtn/preprocess.py`、`mol_gtn/model.py`、`mol_gtn/train.py`

这样最容易理解项目全貌。
