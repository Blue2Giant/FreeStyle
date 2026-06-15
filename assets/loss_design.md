# SRef Loss 设计文档（Enrichment + Entropy）

> 配置：`configs/train/0420_qwen_h200_sref_entropy.yaml`
> 实现：`vgo/train_engines/naive_policy.py`（loss 组合）、`vgo/models/transformers/layers.py: compute_sref_attention_aux()`（指标计算）、`vgo/models/transformers/model.py`（timestep 加权 / schedule）

---

## 一、整体目标

主任务是 flow-matching 扩散损失（让 DiT 学会从噪声预测速度场生成目标图）。在此之上，针对 **参考图 (SRef)** 引入两个轻量级正则项，约束模型在 **block-0（第一个 double-attention block）** 里对参考图的注意力行为：

- **Enrichment**：约束目标图 token 对参考图区域的**关注强度**——别看得太多。
- **Entropy**：约束注意力在参考图内部各 token 上的**分布形状**——别太集中、也别太散。

总损失：

```
L_total = L_diffusion + λ_enrich · L_enrichment + λ_entropy · L_entropy
        = L_diffusion + 0.1 · L_enrichment + 0.1 · L_entropy
```

两个权重都是 0.1，属于"轻推一把"的正则化量级，不会盖过主损失。

---

## 二、注意力指标的计算基础

两个 loss 都建立在同一个 attention 矩阵之上（只在 block-0 算一次）：

- **Q** = 目标图（noisy）的 image token
- **K** = 整条 image 序列（参考图 token + 目标图 token）
- 注意力：`A = softmax(Q·Kᵀ / √D)`，shape `(H, L_q, L_kv)`，H = heads

通过 `sref_key_ranges = (k_start, k_end)` 定位**参考图 token** 在 K 中的位置；
通过 `sref_query_ranges = (0, L_q)` 限定只用**目标图 token 作为 query**（由 `sref_enrichment_noise_query_only=true` 开启）。

---

## 三、Enrichment Loss —— 关注强度

**指标定义：**

```
mass_sref   = Σ A[:, q, k]    （query 落在参考图 key 上的注意力质量）
mass_total  = Σ A[:, q, :]    （query 在所有 key 上的总质量，clamp ε）
ratio       = mass_sref / mass_total

uniform     = max(L_ref / L_kv, ε)        （均匀注意力下的期望占比）
enrichment  = ratio / uniform
```

- `enrichment = 1` → 关注度恰好等于"随机均匀"的预期
- `enrichment > 1` → 过度关注参考图
- `enrichment < 1` → 关注不足

**惩罚（双 hinge 平方）：**

```
penalty = relu(α_low − enrichment)² + relu(enrichment − α_high)²
L_enrichment = Σ penalty / (H · L_q)
```

**本配置取值：**

| 超参 | 值 | 含义 |
|------|----|----|
| `sref_enrichment_loss_weight` | 0.1 | 正则化权重 λ_enrich |
| `sref_enrichment_lower_bound` α_low | **0.0** | 下界=0 → **不惩罚关注不足**，只防过度 |
| `sref_enrichment_upper_bound` α_high | **0.6** | 关注度超过 0.6× 均匀预期就惩罚 |
| `sref_enrichment_eps` | 1e-6 | 防除零 |

> 设计意图：`lower_bound=0.0` 说明目标不是"强迫模型看参考图"，而是"防止它过度依赖参考图"导致细节照搬、丧失生成多样性。

**Timestep 加权（已开启）：**

```
w_t = (1 − t)^power = (1 − t)^1.0
penalty ← penalty · w_t
```

| 超参 | 值 |
|------|----|
| `sref_enrichment_timestep_weighting` | true |
| `sref_enrichment_timestep_weight_power` | 1.0（线性） |

> t 大=高噪声早期，此时参考图最难被有效利用，权重 `(1-t)` 反而让低 t（去噪后期）惩罚更强——即在图像结构逐渐成形、参考图影响真正落地的阶段加大约束力度。

---

## 四、Entropy Loss —— 注意力分布形状

只看注意力**在参考图 key token 上**的分布，把它当成概率分布算归一化熵：

```
k_mass[k]  = Σ_q A[:, q, k]                       （仅 k ∈ 参考图范围）
k_probs    = k_mass / (Σ k_mass + ε)
entropy    = − Σ k_probs · log(k_probs+ε) / log(L_ref)   ∈ [0,1]
```

- 熵低 → 注意力 collapse 到参考图的少数 token（只盯局部）
- 熵高 → 注意力撒得太均匀（没抓住重点）
- 仅在参考图 token 数 `L_ref > 1` 时计算；总 mass 过小的 head 被 mask 置零

**惩罚（双 hinge 平方）：**

```
penalty = relu(β_low − entropy)² + relu(entropy − β_high)²
L_entropy = Σ penalty / (H · L_q)
```

**本配置取值：**

| 超参 | 值 | 含义 |
|------|----|----|
| `sref_entropy_loss_weight` | 0.1 | 正则化权重 λ_entropy |
| `sref_entropy_lower_bound` β_low | **0.06** | 熵低于 0.06 惩罚（防 collapse） |
| `sref_entropy_upper_bound` β_high | **0.14** | 熵高于 0.14 惩罚（防过散） |
| `sref_entropy_eps` | 1e-6 | 防除零 / 防 log(0) |

> 目标区间 [0.06, 0.14] 较窄，要求参考图内部注意力维持一个"适度聚焦"的状态。

**Entropy Schedule（本配置未启用）：**
代码支持让 β_low 随训练动态升高（`schedule_enabled=true` 时），但本 yaml 未配置该项，故全程使用静态 [0.06, 0.14]。相关参数 `sref_entropy_schedule_*` 均走默认值、不生效。

---

## 五、超参数速查

```yaml
policy:
  # ---- Enrichment（关注强度，只防过度）----
  sref_enrichment_loss_weight: 0.1      # 权重
  sref_enrichment_lower_bound: 0.0      # 下界=0，不罚关注不足
  sref_enrichment_upper_bound: 0.6      # 上界，超出即罚
  sref_enrichment_eps: 1.0e-06
  sref_enrichment_noise_query_only: true        # 只用目标图做 query
  sref_enrichment_timestep_weighting: true      # 按 (1-t)^p 加权
  sref_enrichment_timestep_weight_power: 1.0

  # ---- Entropy（分布形状，双边约束）----
  sref_entropy_loss_weight: 0.1         # 权重
  sref_entropy_lower_bound: 0.06        # 防 collapse
  sref_entropy_upper_bound: 0.14        # 防过散
  sref_entropy_eps: 1.0e-06
  # schedule 相关参数本配置未启用
```

---

## 六、数据流

```
样本 (ref_images, target_image, text)
  → VAE encode → x0；采样噪声 x1；xt = t·x1+(1-t)·x0；v_target = x1-x0
  → DiT forward
       block-0 attention: Q=目标图, K=[参考图+目标图]
         ├─ enrichment: 参考图区域注意力质量 / 均匀预期 → 双hinge → L_enrichment
         └─ entropy:    参考图 key 上的分布熵          → 双hinge → L_entropy
  → L_total = L_diffusion + 0.1·L_enrichment + 0.1·L_entropy
```

---

## 七、设计要点回顾

1. **只在 block-0 算**：省开销，且首层 attention 最能反映模型对参考图的原始关注模式。
2. **Enrichment 单边（lower=0）**：核心是抑制"过度照搬参考图"，而非强制使用。
3. **Entropy 双边窄区间**：维持"适度聚焦"，既不 collapse 也不发散。
4. **Enrichment 配 timestep 加权**：在去噪后期（参考图影响真正落地）加大约束。
5. **两项权重均 0.1**：辅助正则，不喧宾夺主。
