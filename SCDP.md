SCDP-Lite for HAT：轨迹条件化的视觉特征注入
给 Codex 的实现任务。目标是在现有 V-JEPA + HAT 大脑中，以最小改动加入 SCDP 的核心机制：用当前预测的未来手腕轨迹，在当前头部 RGB 对应的空间特征图上采样局部视觉信息，并反馈到动作预测头。这是 SCDP 的非 Diffusion 简化适配，不声称复现原论文。

参考：SCDP（CoRL 2026）https://arxiv.org/abs/2606.14535 ，项目 https://scdp-project.github.io/

0. 硬约束（必须遵守）
- 不改数据集：只使用已有 RGB、状态、未来 KP/动作标签、已有相机标定；不增加深度图、分割、热力图标签，不采新数据。
- 不接入小脑：训练/推理均不读取 ScaleBFM/SONIC 跟踪误差、执行进度或其他低层控制反馈；保留 HAT 原有输入和输出接口。
- 保留 V-JEPA + HAT 主干，不替换 encoder，不引入 Diffusion、新 world model，也不新增独立的抓取/合手分类器。
- 不改 chunk 调度逻辑；本任务只改善输出的未来手腕 KP。第一版不改变手指闭合、root、头部等其他通道的语义/维度。
- 先检查现有代码：查明 V-JEPA 是否保留空间 patch tokens，HAT 是否有逐时刻动作 hidden states，左右腕 KP 的索引/归一化方式、相机标定和每帧头部位姿来自哪里。不能凭空假定它们存在。
1. 目标 pipeline
当前头部 RGB ──> V-JEPA ──> spatial patch map F [B,C,Hf,Wf]
                         └──> 原视觉输入 ─┐
当前状态/历史 ────────────────────────────┼──> 原 HAT ──> coarse future KP [B,T,...]
                                         │              └──> 逐时刻 hidden H [B,T,D]
                                         │
coarse 左右腕 xyz ──> 坐标反归一化与 FK/坐标变换（按仓库现有表示）
                  ──> 当前相机坐标系 ──> K 投影到当前 RGB 像素
                  ──> grid_sample(F, uv) ──> local features [B,T,2,C]
                                                │
H + local features ──> Spatial FiLM / gated fusion ──> 原 KP 输出头 ──> refined future KP
与原论文的区别：原 SCDP 在每一步 Diffusion 去噪中重新投影中间轨迹，并使用多尺度 ResNet 特征；这里没有去噪循环，因此只采用 coarse → sample → refine 一轮。第一版直接使用 V-JEPA 原生空间 patch map，不要把低分辨率特征上采样伪装成高分辨率信息。
2. 最小实现方案
A. 获取 V-JEPA 空间特征
- 从编码器获取当前 RGB 对应的 patch tokens，正确还原为 [B,C,Hf,Wf]；若原输入是视频 clip，明确选择与当前 observation image 对应的时间 slice（或有依据的融合），绝不随意把全部时间 token flatten 后当空间位置。
- 保留原 HAT 使用的全局/序列视觉特征路径。新增 spatial map 是一条旁路；不要求改原输入接口。
- 如果只保存了 pooled embedding，先改 encoder 输出/缓存以保留真实 patch tokens；禁止用 pooled vector 重复铺开冒充 spatial map。
B. 手腕轨迹的 3D→2D 投影
- 从 coarse KP 取未来左右腕 xyz（默认两侧均使用；索引从原仓库配置读取），还原到物理单位米和实际坐标系；绝不能直接投影归一化数值。
- 使用与输入 RGB 同一时刻 的相机外参：p_cam = T_cam_from_kp_frame(t) @ p_kp，再用内参 K 投影为 (u,v)。若相机位姿相对头部或 world 动态变化，必须包含当前头部姿态；不要默认头部相机固定。
- 原始像素坐标要经过与 V-JEPA 输入完全相同的 resize/crop/pad/flip 映射，再归一化到 grid_sample 坐标。
- 严格处理 z_cam <= 0、越界、NaN 和缺失标定：mask 无效 anchor。标定缺失时回退原模型，不要自行估算外参；单独统计可用训练样本比例。
- 人类示教（如 Vision Pro）与机器人（如 ZED）必须各用自己的标定与 KP 坐标约定。不能用 ZED 标定投影人类示教。
C. 局部视觉采样与融合
- 推荐起始配置：左右腕、沿整个预测 horizon 均匀选取 8 个时间 anchor/手；可再对抓取前 0.5 s 增加采样密度，但先别引入新 focus 规则。
- 用 torch.nn.functional.grid_sample 对每个投影点进行双线性采样。首版只采点；如果结果不好，再测试每点周围 3×3 个 feature-grid 邻点（不是裁 RGB）。
- 将有效点局部特征做带 mask 的时间聚合，得到每条手腕的 spatial context；也可按时间插值广播给每个 action token。先选择仓库中改动最小的一种，不要同时开发两套。
- 推荐 零初始化 FiLM 适配器：从 spatial context 预测 (gamma,beta)，调制已有 action hidden state：H' = H + gamma * LN(H) + beta。将预测 (gamma,beta) 的最后一层权重与 bias 初始化为 0，使初始行为与 baseline 一致；不要同时将所有乘性门控和调制分支都初始化为 0，以免梯度完全消失。所有 anchor 无效时显式强制 H'=H。
- 用同一个原 KP head 从 H' 生成 refined KP；不要额外引入视觉 residual network 或渲染回 RGB 的循环。若原 HAT 没有可访问的逐时刻 hidden/head，先记录这一阻碍，尽量在原 action head 前增加条件接口，不要偷偷改成另一种模型。
- 第一版只替换 refined 输出中的左右手腕 xyz；其他通道沿用 coarse 原输出。保留 coarse 和 refined 用于日志、消融、测试。
D. 梯度与训练
- 第一轮使用已有 checkpoint：冻结 V-JEPA 与大部分 HAT，训练新 spatial conditioner、gate；必要时允许微调原 KP head，尽量避免改动其它预测头。
- 由 coarse.detach() 生成投影 anchor，避免第一轮通过采样坐标产生不稳定的梯度反馈；保留对融合模块和动作头的正常梯度。
- 监督完全沿用原有未来 KP ground truth，不新增标签。保留你当前已经验证有效的 抓取前关键阶段加权损失 与原 latent loss；避免偷偷更改已有 loss 系数。
- 可以使用 L = L_KP(refined, gt) + 0.2 * L_KP(coarse, gt) + L_other_existing 作为实验起点，其中 L_other_existing 沿用原仓库定义，避免重复累计现有项；若原代码已在外部计算同样的 loss，不要重复。
- 验证后再考虑联合 fine-tune HAT 最后几层。不要一开始解冻整个 V-JEPA。
3. 代码交付要求
1. 独立模块，例如 SpatialTrajectoryConditioner：输入 feature map、coarse wrist xyz、camera calibration、HAT hidden；输出融合后的 hidden + valid masks。文件位置遵循仓库现有结构。
2. 在配置中加开关 scdp_lite.enabled=false（默认关闭），以及 anchor 数、FiLM 隐层维度、冻结策略；关闭时与原 checkpoint 输出完全一致。
3. 保持已有 forward() 返回格式、训练入口与推理接口兼容；新增 coarse/refined/projection_debug 只作为可选辅助输出。
4. 单元测试：投影内外参、resize/crop 对齐、grid_sample 已知格点、左右腕索引、镜头后方/越界/无标定 mask、全无效时 baseline 回退、关闭开关的数值等价、形状和梯度正常。
5. 实现可视化 debug：在对应 RGB 上叠加当前左右腕与 coarse 未来轨迹投影（仅用作调试，不回喂模型），保存有效投影占比；用于人工核对坐标系统是否正确。
6. 先跑一个 mini-batch 过拟合测试，再小规模离线评估；不要擅自开展真机控制测试或改变现有部署参数。
4. 对照实验和判定标准
只比较：
- Baseline：原 HAT（相同 checkpoint / 数据划分）。
- SCDP-Lite：唯一新增 spatial feature sampling + FiLM；其它训练配置保持一致。
- Shuffled-anchor（必要消融）：将 anchor 打乱/替换为固定采样点，验证收益是否真的来自轨迹条件采样，而非参数量增加。
重点指标：
- 原整体 MPJPE（防止为了末端改善破坏全身性能）。
- 合手前 0.5 s 左右腕预测误差，按相机坐标的 x/y/z 分开报，尤其关注 depth/z 误差。
- 保持状态/历史不变、瓶子位置变化时，预测抓取 KP 的位置响应比例（只在有合法成对数据/可控干预时计算）。
- 与 baseline 相同的离线推理时延和显存；真实插入/抓取成功率需后续正常真机测试确认。
不保证一定改善。如果投影大多无效、V-JEPA patch grid 过粗、真实相机外参不可靠，先修正基础条件；不能用错误投影继续训练并声称模型未利用视觉。

5. Codex 执行顺序
1. 先输出代码审计结果：现有模块路径、V-JEPA patch token shape、HAT hidden shape、KP 坐标系与左右腕索引、各来源数据的相机标定可用性、图像预处理链路。
2. 按 A→B→C→D 实现最小可开关模块和测试，不重构整体训练框架。
3. 运行测试 + 一个 mini-batch 前后向；若缺标定或空间 tokens，明确报告阻碍，不要假造参数或默默改做热力图/深度方案。
4. 最后提供改动文件列表、命令（使用仓库中真实存在的入口）、实验配置与 debug 可视化保存位置。 