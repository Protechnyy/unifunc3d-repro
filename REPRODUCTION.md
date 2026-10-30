# UniFunc3D-8B 独立复现

## RynnBrain-8B 完整验证集实验（已完成）

本轮按用户指定改用 `models/RynnBrain-8B`，保留全部提示词补充。
该权重的本地配置是 `Qwen3VLForConditionalGeneration`，因此复用现有 Qwen3-VL 后端；
代码中的 `qwen_model` 字段及日志的 `Qwen` 标签表示架构，并不代表本轮加载原始 Qwen 权重。
设置见 `configs/rynnbrain_8b.json`：BF16、greedy、K=4、N=64、全部视频及细定位窗口。
这是替换为 RynnBrain 后对论文方法的实验，不是相同 checkpoint 的严格复现。

已于北京时间 **2026-10-29 23:42** 完成 **30 场景、445 查询** 的完整运行及评测，
`complete_split=true`，4 条为空预测（已纳入指标）。从首次启动至最终评测约 31 小时 11 分钟，
包含中途启用图像缓存时的重启。
启动前用本地标注重算作者全部 445 条结果并逐条核验通过；作者完整验证集 mIoU 为
20.9185%，AP50 为 23.8202%，与论文四舍五入值一致。

| 指标（%） | RynnBrain 本实现 | 论文 8B | 差距（百分点） |
|---|---:|---:|---:|
| mIoU | 14.99 | 20.92 | -5.93 |
| AP50 | 17.53 | 23.82 | -6.29 |
| AP25 | 29.44 | 44.04 | -14.60 |
| AR50 | 38.43 | 46.07 | -7.64 |
| AR25 | 46.07 | 55.51 | -9.44 |

**结论：当前 RynnBrain-8B 独立实现尚未达到论文报告的 8B 水平。**
mIoU 相对低约 28.3%，不能用先前 17 条小子集的接近结果代替本次完整验证结论。
尚未用原始 Qwen3-VL-8B 在完整验证集进行受控对照，因此不能把差距单独归因于 RynnBrain。

本轮输出目录为 `outputs/rynnbrain_val/`：

- `status.json`：运行状态、进程号、已完成场景/查询数量；`failed` 时包含错误原因。
- `run.log`：逐次模型输出；`supervisor.log`：后台管理进程日志。
- `reference_check.json`：445 条作者结果的预核验。
- `evaluation_progress.json`：有新结果时约每五分钟更新，属于部分数据指标。
- `evaluation.json`、`comparison.json`：**只有推理结束并评测后才生成**；完整覆盖验证通过后
  `status.json` 才标记 `completed`。`comparison.json` 给出相对论文各指标的百分点差。
- `predictions/`、`traces/`：本轮独立预测和过程记录，与先前 Qwen 实验分开。

查看进度：

```bash
cat outputs/rynnbrain_val/status.json
tail -n 20 outputs/rynnbrain_val/run.log
```

如进程中断，用相同配置恢复并自动完成评测：

```bash
python tools/run_validation.py --config configs/rynnbrain_8b.json \
  --output outputs/rynnbrain_val \
  --metadata-overrides configs/val_description_overrides.json
```

该命令要求完整验证集，不允许静默跳过缺失场景。已有结果自动复用；配置改变需新目录。
本次全量实际墙钟耗时约 31 小时 11 分钟；场景长度差异较大，不能仅用前几个场景的速度外推。

运行中加入了不改变图像内容的 CPU 加速：并行解码/旋转/缩放粗定位图像，并在每个场景内
缓存最多 1024 张缩放后的图片（切换场景清空）。实测一批 64 帧原为 3.02 秒，冷缓存
1.25 秒，命中缓存 0.03 秒；图像像素和 processor 输出张量均逐元素完全一致，17 项测试通过。
这只是图像准备阶段的加速，不能直接作为整条查询的加速倍数；原记录约七成耗时在模型生成。
优化后使用原配置和缓存恢复推理，保留已完成预测。重启前源码快照、进度及优化记录位于
`outputs/rynnbrain_val/`，详见 `image_cache_optimization.json` 和 `image_cache_benchmark.json`。

**数据版本例外：**全部 30 场景资产已完整，但 437294 的描述 SHA256 与代码发布记录不同。
重新下载官方文件后确认与本地逐字节一致；其裁剪掩码、标注指纹和 17 条作者参考指标均匹配。
因此使用 `configs/val_description_overrides.json` 精确允许该描述哈希，其他检查保持有效，
不改原始数据或 `datasets/val.json`。证据存于 `metadata_exception_evidence.json`。
作者原始描述文件不可得，因此不能证明两版描述文本的语义完全一致；评测报告显式记录
`metadata_overrides` 和 `release_fingerprints_exact=false`。

---

这是基于 [arXiv:2603.23478v2](https://arxiv.org/html/2603.23478v2) 的独立实现，
不是作者尚未发布的核心源码。原始 `predictions/`、`reference_results/` 和评测器保持原样。
新预测仅由本地 Qwen3-VL-8B-Instruct、SAM3、RGB-D、相机位姿及输入描述产生。
核心推理不读取标注掩码或作者预测，也不使用 30B。资产审计单独检查标注文件的存在与指纹。

## 安装与运行

Python 3.12。RTX 5090 需要支持 Blackwell 的 PyTorch；本机原有 2.5.1+cu124 不支持其 CUDA 内核。
推荐在单独环境安装：

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
python tools/check_inference_environment.py --output outputs/environment.json
python infer.py audit --output outputs/assets
# 可选：逐张解码检查（读取大量数据，会较慢）
python infer.py audit --decode --output outputs/assets_decoded

# 一个完整场景；不缩减论文 K=4、N=64 或细定位时间窗口
python infer.py run --config configs/8b_reproduction.json --visits 467330 --output outputs/reproduction_8b

# 所有已下载完整且与发布版指纹一致的验证场景，自动跳过其余场景
python infer.py run --config configs/8b_reproduction.json --output outputs/reproduction_8b

python evaluate_inference.py --predictions outputs/reproduction_8b/predictions --output outputs/reproduction_8b/evaluation.json
python -m unittest discover -s tests -v
```

`--limit-queries 1` 可先测试首条查询。`--visits`、`--desc-ids` 支持显式选择。
所有路径默认相对于仓库根目录。模型路径可在 JSON 配置中覆盖。
保持配置不变、使用相同输出目录即可断点续跑。换设置须使用新目录，以免复用不同设置的缓存。
不要把尚未处理的查询写为空预测：评测器显式报告实际查询数量和 `complete_split`。

## 实现与论文对应

| 阶段 | 实现 |
|---|---|
| 粗定位，式 (1) | 每个视频 K=4 次错位采样，每次 N=64；局部 `<frame i>` 标签，完整画面低分辨率输入 |
| 联合语义/时空推理 | 附录粗定位提示词，同时预测功能部件名称、帧编号、点位置 |
| 细定位，式 (3–4) | 每个候选帧时间戳 ±视频时长/[2(N−1)]；窗口内每帧独立高分辨率推理，保留完整画面 |
| SAM3 分割 | 官方 `enable_inst_interactivity=True` + `predict_inst`；每个正点生成掩码，置信度 ≥0.5 |
| 掩码核验，式 (5) | 红色 alpha=0.7 覆盖，复用相同 Qwen；仅 YES 掩码进入 3D |
| 多视角融合，式 (6) | 原始相机网格、深度可见性判断、各帧投票；min/max 归一化后严格 >0.7 |
| 评测 | 复用 `release_lib.common`，验证数据指纹，按相同查询比较作者 8B 发布结果 |

实现位于 `unifunc3d/`。每条查询的 `traces/<visit>/<desc>/` 保存原始模型输出、
粗定位采样索引、核验覆盖图、各帧投影点索引、稀疏投票和摘要。
同一帧在多个窗口重复出现时合并掩码，每帧最多贡献一票。
裁剪采用 `crop_mask == 1`，严格保留 PLY 原始顶点顺序。
`inputs/` 保存各场景的输入文件快照，防止数据变动后错误复用旧预测。

## 未公开细节与复现假设

论文没有完整定义以下工程细节，因此不能把独立实现称为逐位复现：

- 使用本地 **Instruct** 权重，BF16、确定性 greedy 解码；论文未提供解码参数或 checkpoint revision。
- 默认 `linear_patch_embed=true`：将预先切好的独立 patch 上的 Conv3d 改写为数学等价的
  `F.linear`，复用原权重，规避本机 5090/cuDNN 的大 batch Conv3d 严重性能问题。
  该优化不改变模型结构、输入分辨率或采样；浮点累加顺序可能有微小差异。
  可设为 false 使用原算子。
- Qwen 的坐标明确为 [0,1000]，帧号从 0 开始；附录提示词补充该输出约定与不可见时 NONE。
- `8b_paper.json` 使用附录短提示词；`8b_reproduction.json`（推荐）另外明确
  “功能目标是手直接操作的部件，而不是父物体”。这是针对本地 Instruct 模型将 radiator dial
  误解成整个 radiator 的实测语义错误所做的通用提示词澄清，并非论文公开的原始提示词。
  两种设置必须使用不同输出目录；不能将后者的指标称为原提示词的精确复现。
- SAM3 多掩码候选取预测 IoU 最高者，再应用 0.5 阈值。论文未说明候选选择规则。
- 按 Fun3DU 重力方向把图片转正，投影前把掩码逆旋转；竖屏低分辨率用 384×512。
- 高分辨率上限 1920×1440，Qwen processor 仍会按 patch 大小对齐，不裁剪。
- 几何投影按 SceneFun3D/Fun3DU：hires COLMAP 位姿的旋转向量/平移是 world-to-camera，
  深度毫米转换为米，深度相对容差 0.25。无需额外 ARKit transform。
- 融合使用 Fun3DU 的 min/max 归一化，避免把 0.7 误当作“必须被所有帧的 70% 看见”。
- `max_fine_frames`、`max_videos` 默认 0（不限制）。设为正数属于加速/调试设置，应另报指标。

## 指标解释

原仓库的 AP25/AP50 是**点 precision 达到阈值的查询比例**，AR 使用点 recall；
不是标准实例检测 AP。mIoU 为逐描述点集 IoU 的平均值。
论文 8B 完整验证集（30 场景、445 查询）目标：AP50 23.82、AP25 44.04、
AR50 46.07、AR25 55.51、mIoU 20.92（百分数）。
部分场景只与作者同一子集比较，不能直接宣称达到完整验证集指标。
最终结论必须以 `evaluate_inference.py` 的实际新推理结果为准。

## 本机实测与结果文件

已在 RTX 5090 32GB 上完成真实 Qwen3-VL-8B + SAM3 推理，全程使用本地权重。
最终已跑完场景 **421393（13 条）和 467330（4 条）**，共 17 条，使用
`configs/8b_reproduction.json`，未缩减采样或细定位窗口，也未运行 30B。

| 两个完整场景，17 条查询（百分数） | mIoU | AP50 | AP25 | AR50 | AR25 |
|---|---:|---:|---:|---:|---:|
| 本实现，部件提示词版本 | **16.98** | 17.65 | 35.29 | 58.82 | 58.82 |
| 作者发布的相同 17 条预测 | **16.52** | 17.65 | 29.41 | 52.94 | 52.94 |

完整报告为 `outputs/reproduction_8b/evaluation.json`，逐条预测在同目录的 `predictions/`。
`complete_selected_scenes=true`，但 `complete_split=false`；这不是完整验证集复现结论。
追加的 12 条保持配置冻结，单独 mIoU 为 **14.16%**，作者相同 12 条为 **18.19%**，
见 `evaluation_additional12.json`。因此整体接近不代表每类任务都达到作者水平；
例如顶层抽屉本实现 IoU 为 0%，作者为 55.92%。

17 条累计查询耗时约 **47.1 分钟**，平均 **166 秒/条**；包含模型加载及逐查询处理，
不包含前期安装、资产审计、调试和两次启动之间的等待。
共执行 152 次粗定位，处理 924 个细定位帧/部件候选；每 5 秒采样的最高 GPU 显存
为 **26,622MiB（约 26.0GiB）**，不是精确瞬时峰值。详见 `runtime_summary.json`。
14 项单元测试通过，几何可见点集合与上游 Fun3DU 对照一致；真实推理和评测均已完成。

### 首批调试对照

首批五条查询包括场景 421393 的首条描述以及场景 467330 的全部四条描述。
这些查询参与了工程调试和通用提示词澄清，因此不能当作独立测试集。

| 首批五条查询（百分数） | mIoU | AP50 | AP25 | AR50 | AR25 |
|---|---:|---:|---:|---:|---:|
| 附录短提示词版本 | 10.41 | 20.00 | 20.00 | 20.00 | 20.00 |
| 部件提示词版本 | 23.73 | 40.00 | 40.00 | 60.00 | 60.00 |
| 作者发布的相同五条预测 | 12.51 | 20.00 | 20.00 | 20.00 | 20.00 |

原始报告分别保存在 `outputs/reproduction_8b_paper_prompt/evaluation.json` 和
`outputs/reproduction_8b/evaluation_smoke5.json`。后者固定保存首批五条结果，续跑不会覆盖。
场景 421393 的首个暖气旋钮仍定位错误：模型选择了暖气顶部圆盖，IoU 为零。
保留该失败样例，没有按标注修改坐标或为单个场景调整阈值。

以下命令运行这两个场景的全部 17 条查询；本机已经完成，重跑会自动跳过已有查询：

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
python infer.py run --config configs/8b_reproduction.json --visits 421393 467330 --output outputs/reproduction_8b
python evaluate_inference.py --predictions outputs/reproduction_8b/predictions --output outputs/reproduction_8b/evaluation.json
```

当前资产审计中，30 个验证场景有 27 个下载完整，其中 26 个与发布版指纹一致。
466916、467115、467139 缺少部分 RGB/深度/内参；437294 的描述文件指纹不一致。
下载继续进行后请重新审计；程序默认跳过不满足条件的场景。

几何约定参考本地 SceneFun3D toolkit 与 Fun3DU；上游授权见 `THIRD_PARTY_NOTICES.md`。
