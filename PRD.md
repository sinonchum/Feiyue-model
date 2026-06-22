# Feiyue-Model PRD: Qwen3-4B 微调，服务 Feiyue+Hermes 系统

> **Version**: v1.0 · **Date**: 2026-06-22 · **Status**: Draft → Strong Model Review

---

## 1. 目标

微调 Qwen3-4B-Instruct-2507，使其成为 Feiyue+Hermes 系统中的高效本地 worker，替代云端 API（DeepSeek/GPT）。

**不是**重建系统架构。**是**让模型更好地服务于已有的 Feiyue 验证循环 + Hermes 工具链。

---

## 2. 系统上下文（不可变）

- **编排器**：Feiyue（验证门控、Curator、Capability History、Asset Promotion）
- **Agent 框架**：Hermes Agent（工具调用、记忆、技能执行）
- **Worker 模型**：Qwen3-4B-Instruct-2507（待微调）
- **验证器**：强模型（GPT-5.5/Claude，双轨验证：规范+效率）
- **硬件**：RTX 5060 8GB
- **训练栈**：trl.SFTTrainer + peft + bitsandbytes（PatentFlow 已验证）

---

## 3. 待解决问题

Qwen3-4B 是通用模型，不理解 Feiyue+Hermes 的工作方式：
- 不知道如何接收 TaskContract 并拆解为 Hermes 工具调用序列
- 不知道如何格式化输出给 Feiyue 验证器
- 不知道如何处理验证失败反馈并自我纠正
- 不知道如何利用 Hermes 的技能库和记忆系统

---

## 4. 微调策略（三阶段）

### 阶段一：基础对齐

**目标**：教会模型 Feiyue+Hermes 的基本"规则"。

**训练数据**：
- TaskContract → Hermes 工具调用序列的正确示范
- 工具调用的正确 JSON 格式（9 个细粒度工具）
- 验证器期望的输出格式

**数据来源**：
- Feiyue 执行证据（152 文件）→ 提取成功轨迹
- Feiyue 测试 fixtures（100+ 测试文件）→ 提取 inline 示例
- Feiyue Phase B-F 脚本 → 提取端到端工作流
- Hermes 会话转储 → 提取真实 tool call 模式

### 阶段二：高级推理与自纠正

**目标**：教会模型思考、规划、从错误中恢复。

**训练数据**（研究论文启发的数据构造方法）：

**自纠正对**（Reflexion 模式）：
- 强模型验证器纠正 worker 的错误 → 自动变成训练样本
- 格式：(初始错误输出, 验证器反馈) → 纠正后输出
- 来源：Feiyue 执行证据中的 teacher-retry 对

**链式思维规划**：
- 模型先输出分步计划 → 验证器检查逻辑 → 再执行
- 来源：强模型生成的高质量规划示范

**失败学习**（Learning From Failure 论文启发）：
- 将失败轨迹 + 错误原因 + 正确做法 → 构造训练样本
- 负样本比正样本更有信息量

### 阶段三：自进化飞轮

**目标**：训练模型主动参与系统知识积累。

**训练数据**：
- 模型生成的新方案 → 经验证器验证 → Curator 提升为资产 → 反哺训练
- Capability History 中的成功轨迹 → 定期重训

**飞轮机制**（用 Feiyue 已有组件）：
```
模型生成方案 → Feiyue 验证器验证
→ 通过 → Curator 提升为资产（技能/回归评估/任务模板）
→ 资产加入训练数据 → 模型更优 → 生成更好方案
```

---

## 5. 数据生成与管理

### 数据来源优先级

1. **验证器生成**（主要）：强模型作为"教授"，生成高质量 (指令, 回答) 对
2. **反馈蒸馏**（次要）：worker 与验证器的每次交互中，验证器的纠正 → 自动变训练样本
3. **成功轨迹复用**（持续）：Feiyue 系统中验证通过的任务 → 加入"黄金数据集"

### 研究论文启发的数据质量优化

| 研究启发 | 应用方式（不是建组件，是改进数据） |
|---|---|
| synth_gen 的自博弈方法 | 用强模型生成问题变体 → worker 解答 → 验证器验证 → 通过则入训练集 |
| Dynamic Unit Tests 论文 | 验证器不仅返回 pass/fail，还返回难度分级 → 训练数据标注难度等级 |
| Learning From Failure | 失败轨迹自动转 (错误, 纠正) 训练对 |
| Symbolic Learning 的反思方法 | 验证器的纠正理由 → 蒸馏为模型的 chain-of-thought 训练样本 |
| Scaf-GRPO 的中间奖励 | RL 阶段的奖励信号从二值 → 连续分数（基于验证器的信心度） |
| Compute-Optimal TTS | 合理设置测试时采样数 N，用验证器选最优 |

### 数据格式

统一 ChatML 多轮对话 + JSON 工具调用。工具返回注入为 user role。

### 数据管理

- **黄金数据集**：持续增长的验证通过轨迹
- **负样本库**：失败轨迹 + 纠正（用于 DPO/GRPO 的 rejected 样本）
- **合成数据**：强模型生成 + 验证器过滤
- 定期用 Feiyue 的 Curator 机制清理和提升数据质量

---

## 6. 训练配置

### SFT 阶段

```yaml
model: Qwen/Qwen3-4B-Instruct-2507
quantization: 4-bit NF4, double quant (BitsAndBytes)
lora: r=16, alpha=32, dropout=0.05, all projections
max_seq_length: 2048
batch: 4, grad_accum: 2 (effective 8)
lr: 1e-4, cosine, warmup 5%
epochs: 3
loss: completion-only (DataCollatorForCompletionOnlyLM)
response_template: "<|im_start|>assistant\n"
framework: trl.SFTTrainer + peft + bitsandbytes
```

### RL 阶段（SFT 之后）

```yaml
method: DPO（起步稳定）→ 效果不足换 GRPO
data: 验证器通过 = chosen, 验证器失败 = rejected
reward: 验证器信心度分数（非二值，研究论文启发）
framework: trl.DPOTrainer / trl.GRPOTrainer
```

### 测试时

```
TaskContract → 生成 N 候选 → Feiyue 验证器逐条验证 → 选最优
N 的选择基于 Compute-Optimal TTS 研究的最优分配策略
```

---

## 7. 验证（双轨，强模型执行）

### Specs Track
- 训练配置是否符合 PRD 规范
- completion-only loss 是否正确实现
- 数据格式是否与 Feiyue 系统兼容

### Code Efficiency Track
- 训练脚本是否有 bug
- VRAM 预算是否在 8GB 内
- 错误处理和 checkpoint 恢复

---

## 8. 成功指标（系统内部，不写 benchmark）

| 指标 | 含义 | 衡量方式 |
|---|---|---|
| **纵向能力增益** | 系统是否随时间持续提升 | Feiyue Curator 的 capability history |
| **首次通过率** | worker 不需纠正的比例 | 验证器日志统计 |
| **验证复杂度下降** | 模型输出越来越容易验证 | 验证器 token 用量趋势 |
| **新方案生成率** | 生成训练数据中没有的新解法 | Curator 提升的新资产数 |
| **成本效率** | 强模型调用量下降 | 月 token 用量 |
| **延迟** | 单次 worker 调用时间 | < 2s（本地推理） |
| **月成本** | 从 $5-15 降至 $0（推理成本） | 账单 |

---

## 9. 分阶段计划

| 阶段 | 周 | 内容 |
|---|---|---|
| Phase 0 | W1 | 评估管道搭建 + 基线测量 |
| Phase 1 | W1-2 | 数据提取 + 强模型生成 + 验证器过滤 |
| Phase 2 | W2 | 数据比例实验 |
| Phase 3 | W2-3 | SFT 训练 + 双轨验证 |
| Phase 4 | W3 | RL 训练（DPO/GRPO） |
| Phase 5 | W3-4 | 系统集成 + A/B 测试 |
| Phase 6 | W4 | 上线 + 监控 |

---

## 10. 风险

| 风险 | 缓解 |
|---|---|
| 合成数据质量差 | 多步验证 + 人工抽检 10% |
| 模型过拟合验证器风格 | 验证器 prompt 多样化 |
| VRAM 不足 | 严格 QLoRA 配置 |
| 自进化飞轮冷启动 | 初始数据集来自 Feiyue 已有证据 |

---

## 11. 不做的事

- 不重建 Feiyue 或 Hermes 架构
- 不添加新的持久化微服务
- 不追求公开 benchmark 分数
- 不引入视觉/多模态能力
