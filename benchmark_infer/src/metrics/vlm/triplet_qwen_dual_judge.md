# `triplet_qwen_dual_judge.py` 判别逻辑总结

本文档总结 `/data/FreeStyle/benchmark_infer/src/metrics/vlm/triplet_qwen_dual_judge.py` 中“判别”阶段的实现方式。这个脚本的核心目标是：对一张 `style_and_content` 生成图，同时检查它是否 **保留了目标 content**，以及是否 **匹配了目标 style**；只有两者都通过，最终才算正样本。

---

## 1. 总体结论

脚本做的是一个 **双重判别 Dual Judge**：

```text
一张候选图 main_img
    ├── 和若干 content reference 图做“主体内容一致性”判别
    ├── 和若干 style reference 图做“画风/视觉风格一致性”判别
    └── content_pass AND style_pass => dual_pass
```

最终输出的 `all.json / pos.json / neg.json` 都是基于 `dual_pass`：

- `dual_pass = True`：写入 `pos.json`，`all.json[key] = 1`
- `dual_pass = False`：写入 `neg.json`，`all.json[key] = 0`

---

## 2. 输入数据组织方式

脚本支持两种输入方式。

### 2.1 目录模式：`--root`

要求 `root` 下有：

```text
root/
  style_and_content/   # 待判别主图 main_img
  content_1/
  content_2/
  ...
  style_1/
  style_2/
  ...
```

判别时，脚本会从 `style_and_content/` 里枚举候选图片名，例如：

```text
style_and_content/xxx.png
```

然后用同名图片去各个 `content_*` 和 `style_*` 目录里找 reference：

```text
content_1/xxx.png
content_2/xxx.png
style_1/xxx.png
style_2/xxx.png
```

如果某个 reference 不存在，该 reference 会记录为 `exists: False`，不会参与通过计数，但分母仍然是目录数量。

### 2.2 JSONL 模式：`--input_jsonl`

每行是一个 JSON，对象里至少需要有：

```json
{
  "style_and_content": "主图路径",
  "content_1": "content参考图路径",
  "content_2": "content参考图路径",
  "style_1": "style参考图路径",
  "style_2": "style参考图路径"
}
```

脚本会收集所有 key 以 `content_` 开头的字段作为 content references，收集所有 key 以 `style_` 开头的字段作为 style references，并按字段名尾部数字排序。

---

## 3. 单次图片对判别：`direct_judge_images_generic`

无论 content 还是 style，本质上都调用同一个函数：

```python
direct_judge_images_generic(path_a, path_b, system_prompt, user_instruction)
```

其中：

- `path_a`：候选主图，一般是 `style_and_content` 图
- `path_b`：reference 图，可以是 content reference 或 style reference
- `system_prompt/user_instruction`：根据 content 或 style 任务切换

### 3.1 图片预处理

每张图会被处理成 OpenAI-compatible chat API 的 `image_url` data URI：

1. 读取本地路径或 `s3:// / oss://` 路径。
2. 用 PIL 解码。
3. 转成 RGB。
4. 长边缩放到不超过 `1024`。
5. 以 JPEG 格式重新编码，质量 `85`。
6. base64 后构造成：

```text
data:image/jpeg;base64,...
```

对应常量：

```python
RESIZE_MAX_SIDE = 1024
JPEG_QUALITY = 85
```

### 3.2 模型调用方式

默认模型和地址：

```python
MODEL = "Qwen3-VL-30B-A3B-Instruct"
BASE_URL = "http://10.201.19.61:22002/v1"
```

实际运行时可以通过参数覆盖：

- `--model`
- `--base_url`
- `--endpoint`

请求接口：

```text
POST {BASE_URL}/chat/completions
```

关键请求参数：

```python
temperature = 0.0
max_tokens = 1
logprobs = True
top_logprobs = 8
top_k = 8
```

也就是说，模型只允许输出 **1 个 token**，并且脚本要求模型只输出：

```text
0 或 1
```

### 3.3 输出解析

模型返回后，脚本会：

1. 取 `choices[0].message.content`。
2. 去掉可能的 markdown code fence。
3. 找到第一个非空白字符。
4. 如果不是 `0` 或 `1`，则这次判别无效。

含义统一为：

```text
1 = 一致 / 通过
0 = 不一致 / 不通过
```

---

## 4. 置信度计算：基于 `0/1` 的 logprob

脚本不是只看模型输出的字符，还会取模型返回的 top logprobs。

### 4.1 提取 logprob

函数 `_extract_01_logprobs` 会从返回结构里找 token 为 `"0"` 和 `"1"` 的 logprob：

```python
logp0 = logprob("0")
logp1 = logprob("1")
```

如果 top logprobs 里找不到 `0` 或 `1`，这次判别会被视为无效：

```text
无法提取 0/1 top_logprobs
```

### 4.2 二分类 softmax

然后只在 `0` 和 `1` 两个 token 之间做一次二分类 softmax：

```text
p0 = exp(logp0) / (exp(logp0) + exp(logp1))
p1 = exp(logp1) / (exp(logp0) + exp(logp1))
```

如果模型输出 `1`，该次判别置信度是 `p1`；如果模型输出 `0`，置信度是 `p0`。

```python
conf = p1 if pred_is_consistent else p0
```

所以返回结果类似：

```text
pred=1, conf=0.873 (p0=0.127, p1=0.873)
```

注意：这里的 `conf` 表示“模型对自己当前输出字符的置信度”，不是固定的 `p1`。

---

## 5. 多次投票：`judge_pair_voting`

为了让单个 pair 的判断更稳定，脚本对同一对图片可以重复判别多次。

核心参数：

```python
judge_times  # 总共判几次，默认 3
min_true     # 至少多少次有效正判才算 pair 通过，默认 2
conf_thr     # 单次判别的置信度阈值，默认 0.5
```

默认 content 参数：

```text
--content_judge_times 3
--content_min_true    2
--content_conf_thr    0.5
```

默认 style 参数：

```text
--style_judge_times 3
--style_min_true    2
--style_conf_thr    0.5
```

### 5.1 单次有效判别的标准

每一次调用模型后，脚本判断这次是否有效：

```python
is_valid = pred 是 bool 且 conf 是数字 且 conf > conf_thr
```

也就是说，必须同时满足：

1. 模型输出能解析成 `0/1`。
2. 成功提取 `0/1` logprobs。
3. 当前输出字符的置信度 `conf` 大于阈值。

### 5.2 只统计“有效且为 True”的次数

投票时只累计：

```python
if is_valid and pred is True:
    good_true += 1
```

也就是说：

- 高置信度输出 `1`：计入 `good_true`
- 高置信度输出 `0`：不计入 `good_true`
- 低置信度输出 `1`：不计入 `good_true`
- 解析失败 / 缺 logprob：不计入 `good_true`

最后：

```python
passed = good_true >= min_true
```

默认情况下，同一图片对要在 3 次判别里至少有 2 次是“高置信度 1”，这个 pair 才算通过。

---

## 6. Content 判别逻辑

Content 判别关注的是：**主体内容和主题是否一致**。

对应 prompt：

- `CONTENT_SYSTEM_PROMPT`
- `CONTENT_USER_INSTRUCTION`

核心要求：

```text
只看画面里“是什么”和“在做什么”，忽略画风、线条、色彩、渲染方式等风格差异。
```

具体规则大致是：

### 6.1 人物主体

关注是否是同一个角色或极为相似的角色，包括：

- 性别
- 年龄段
- 身材
- 发型
- 头发颜色
- 肤色
- 服装类型
- 服装主色调
- 主要配饰

姿势、朝向、镜头视角可以变化；但如果明显是不同人物或完全不同造型，则判不一致。

### 6.2 单一物体主体

关注物体类别和结构形状是否一致，例如：

- 都是跑车
- 都是 SUV
- 都是圆桌

颜色可以不同；但如果只是大类相似，例如都包含“车”，但车型结构明显不同，则判不一致。

### 6.3 复杂场景

关注：

- 场景类型
- 主要元素组合
- 布局
- 核心构图和主体物体

只是都在室内或都在室外不够，核心内容需要高度一致。

### 6.4 Content pair 通过标准

每个 content reference 都和主图做一次 `judge_pair_voting`。

如果某个 content reference pair 通过，则：

```python
passed_content += 1
```

最后计算：

```python
content_r = passed_content / total_content
content_pass = content_r >= args.content_ratio
```

默认：

```text
--content_ratio 0.66
```

也就是默认至少约三分之二的 content references 通过，整个样本的 content 才算通过。

---

## 7. Style 判别逻辑

Style 判别关注的是：**画风 / 视觉风格是否一致**。

对应 prompt：

- `STYLE_SYSTEM_PROMPT`
- `STYLE_USER_INSTRUCTION`

核心要求：

```text
只看视觉表现形式，忽略人物/物体身份、动作含义、故事语义、场景类别、构图内容是否相似。
```

重点维度：

1. 媒介与渲染方式：摄影、3D、插画、水彩、油画、厚涂、赛璐璐、像素、素描等。
2. 笔触与线条体系：有无线稿、线条粗细、边缘处理、笔触颗粒等。
3. 材质与纹理生成方式：表面质感、噪声颗粒、细节组织方式等。
4. 光影模型与对比：硬/软阴影、体积光、漫反射、镜面、高反差等。
5. 色彩策略：饱和度、色相偏好、综合色调、复古/冷暖/霓虹等调色方式。

允许变化的内容：

- 主体不同
- 场景不同
- 构图不同
- 视角不同
- 细节密度不同
- 轻微色相、亮度、对比、压缩噪声差异

明显不一致的例子：

- 真实摄影 vs 插画/渲染
- 线稿体系突变
- 油画厚涂 vs 平涂赛璐璐 vs 3D 塑料感
- 光影模型明显变化
- 整体调色策略完全不同

### 7.1 Style pair 通过标准

默认和 content 一样，每个 style reference 都会调用 `judge_pair_voting`：

```python
style_r = passed_style / total_style
style_pass = style_r >= args.style_ratio
```

默认：

```text
--style_ratio 0.66
```

也就是默认至少约三分之二的 style references 通过，整个样本的 style 才算通过。

### 7.2 特殊参数：`--style_repeat_only_style1`

如果开启这个参数：

```text
--style_repeat_only_style1
```

则 style 判别会变成：

- 对 `style_1`：仍然使用多次投票 `judge_pair_voting`
- 对其他 style reference：只做单次 `direct_judge_images_generic`

其他 style reference 单次通过条件为：

```python
pred is True and conf > args.style_conf_thr
```

这个参数的作用是减少 style 侧重复判别次数，从而降低 API 调用量。

---

## 8. 单个样本最终判别流程

对一个样本，流程可以概括为：

```text
main_img = style_and_content 图

1. Content Judge
   for each content reference:
       对 main_img 和 content_ref 做多次 0/1 判别
       如果高置信度正判次数 >= content_min_true，则该 content_ref 通过

   content_ratio = 通过的 content_ref 数 / content_ref 总数
   content_pass = content_ratio >= content_ratio_threshold

2. Style Judge
   for each style reference:
       对 main_img 和 style_ref 做多次 0/1 判别
       如果高置信度正判次数 >= style_min_true，则该 style_ref 通过

   style_ratio = 通过的 style_ref 数 / style_ref 总数
   style_pass = style_ratio >= style_ratio_threshold

3. Dual Judge
   dual_pass = content_pass and style_pass
```

默认可以理解成一个两级投票：

```text
pair 级别：同一图片对判 3 次，至少 2 次高置信度判 1 才通过。
样本级别：content_refs/style_refs 中，至少约 66% 的 pair 通过才通过对应维度。
最终级别：content 和 style 两个维度都通过才是正样本。
```

---

## 9. 失败和跳过逻辑

### 9.1 API 重试

连接失败时使用：

```text
--conn_retry_times 默认 5
--conn_retry_delay 默认 2.0 秒
```

如果所有重试都失败，`direct_judge_images_generic` 返回：

```text
pred = None
reason = "API 重试耗尽"
conf = None
```

### 9.2 任意一次重试耗尽会跳过整个样本

在 `judge_pair_voting` 中，如果某一次 API 重试耗尽，会返回 `retry_exhausted=True`。

上层 `_process_one_name` / `_process_one_record` 收到后会直接：

```python
return None, 0, True
```

也就是说，该样本会被计入 `skipped`，不会写入正负结果。

### 9.3 其他无效情况不会直接跳过

例如：

- 图片编码失败
- 模型输出不是 `0/1`
- 没有提取到 `0/1` logprobs
- 置信度低于阈值

这些情况通常只会导致该次判别不计入 `good_true`，从而更难通过，但不会立刻跳过整个样本。

---

## 10. 输出文件含义

### 10.1 `--out_all`

一个 map，key 是图片 basename 去掉扩展名，value 是 dual 判别结果：

```json
{
  "xxx": 1,
  "yyy": 0
}
```

### 10.2 `--out_pos`

只保存 dual pass 的样本：

```json
{
  "xxx": 1
}
```

### 10.3 `--out_neg`

只保存 dual failed 的样本：

```json
{
  "yyy": 0
}
```

### 10.4 `--out_detail`

如果指定，会额外保存详细结果：

```json
{
  "summary": {
    "root": "...",
    "picked": 100,
    "processed": 98,
    "content_ok": 80,
    "style_ok": 70,
    "dual_ok": 65,
    "skipped": 2,
    "args": {}
  },
  "results": [
    {
      "name": "...png",
      "main_img": "...",
      "content_pass": true,
      "content_ratio": 0.75,
      "content_passed_cnt": 3,
      "content_total": 4,
      "content_details": [],
      "style_pass": true,
      "style_ratio": 0.75,
      "style_passed_cnt": 3,
      "style_total": 4,
      "style_details": [],
      "dual_pass": true
    }
  ]
}
```

其中 `content_details` / `style_details` 会保存每个 reference 的判别详情，包括每次模型调用的 `pred/conf/valid/reason`。

---

## 11. 并发和多 endpoint

脚本支持两种并发方式。

### 11.1 多 endpoint：`--endpoint`

可以传多个 endpoint：

```bash
--endpoint "Qwen3-VL-30B-A3B-Instruct@http://host1/v1"
--endpoint "Qwen3-VL-30B-A3B-Instruct@http://host2/v1"
--procs_per_endpoint 4
```

脚本会为每个 endpoint 启动若干进程，并把样本 round-robin 分配给这些 worker。

### 11.2 单 endpoint 多进程：`--num_procs`

如果没有传 `--endpoint`，但传了：

```bash
--num_procs N
```

则用 `multiprocessing.Pool` 并发跑。

---

## 12. Resume 逻辑

如果没有指定 `--overwrite`，并且 `--out_all` 已经存在，脚本会读取已有 key，并跳过已经处理过的样本。

逻辑：

```python
processed_keys = set(out_all.keys())
```

采样后会过滤掉 basename 已经在 `processed_keys` 里的样本。

---

## 13. 需要注意的点

### 13.1 缺失 reference 会降低通过率

如果某个 `content_*/xxx.png` 或 `style_*/xxx.png` 不存在：

- 它不会产生判别调用。
- 但 `total_content` / `total_style` 仍然按目录总数计算。

因此缺图会降低最终 ratio，更容易导致不通过。

### 13.2 阈值使用的是严格大于

单次有效判别条件是：

```python
conf > conf_thr
```

不是 `>=`。例如阈值是 `0.5` 时，`conf == 0.5` 不算有效。

### 13.3 `conf` 是输出 token 的置信度

如果模型输出 `0` 且 `p0=0.9`，那么 `conf=0.9`，但这仍然是一个高置信度负判，不会计入 `good_true`。

### 13.4 `--content_id_txt` / `--style_id_txt` 过滤

函数 `extract_content_style_ids` 返回顺序是：

```python
return content_id, style_id
```

目录模式过滤也按这个顺序使用：

```python
content_id, style_id = extract_content_style_ids(name)
if content_id_set and content_id not in content_id_set:
    continue
if style_id_set and style_id not in style_id_set:
    continue
```

因此 `--content_id_txt` 过滤 content id，`--style_id_txt` 过滤 style id。

---

## 14. 简化伪代码

```python
for sample in picked_samples:
    main = sample.style_and_content

    # content 侧
    content_passed = 0
    for content_ref in sample.content_refs:
        good_true = 0
        for _ in range(content_judge_times):
            pred, conf = qwen_judge(main, content_ref, content_prompt)
            if pred == True and conf > content_conf_thr:
                good_true += 1
        if good_true >= content_min_true:
            content_passed += 1

    content_score = content_passed / len(content_refs)
    content_pass = content_score >= content_ratio

    # style 侧
    style_passed = 0
    for style_ref in sample.style_refs:
        good_true = 0
        for _ in range(style_judge_times):
            pred, conf = qwen_judge(main, style_ref, style_prompt)
            if pred == True and conf > style_conf_thr:
                good_true += 1
        if good_true >= style_min_true:
            style_passed += 1

    style_score = style_passed / len(style_refs)
    style_pass = style_score >= style_ratio

    dual_pass = content_pass and style_pass
```

---

## 15. 一句话概括

这个脚本用 Qwen3-VL 对主图和 reference 图做二分类 `0/1` 判别：content 侧判断“主体内容是否一致”，style 侧判断“画风机制是否一致”；每个 pair 用 logprob 置信度过滤并多次投票，每个样本再按 reference 通过比例做二次投票，最终只有 content 和 style 都达到阈值才算 `dual_pass`。
