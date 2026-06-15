import os
import re
import json
import time
import uuid
import random
import asyncio
import tempfile
import argparse
import concurrent.futures
import signal
import threading
from typing import Optional, List, Dict, Tuple, Any
import aiohttp
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from tqdm import tqdm
from comfykit import ComfyKit
from megfile.smart import (
    smart_listdir,
    smart_exists,
    smart_makedirs,
    smart_open as mopen,
    smart_copy as mcopy,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "of", "in", "on", "at",
    "for", "to", "from", "by",
    "with", "without", "about",
    "above", "below", "under", "over",
    "between", "into", "through", "during",
    "before", "after", "around", "across",
    "behind", "along", "within", "outside",
    "near", "inside", "upon", "off",
    "up", "down", "out",
    "as", "than", "so", "that",
    "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
}
RE_OUT_IMG = re.compile(r"^(\d{5})_(\d+)\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE)
DEFAULT_PROBE_TIMEOUT_SEC = 2.0
DEFAULT_PROBE_CONCURRENCY = 256
STOP_EVENT = threading.Event()


def _is_stopping() -> bool:
    return STOP_EVENT.is_set()


def _handle_stop_signal(signum, frame):
    if STOP_EVENT.is_set():
        os._exit(130)
    STOP_EVENT.set()
    print("[STOP] 捕获中断信号，正在停止任务。再次 Ctrl+C 将立即退出。")


def _normalize_probe_host(host: str) -> Tuple[str, str]:
    s = str(host).strip().rstrip("/")
    if s.startswith("http://"):
        return s, s[len("http://"):]
    if s.startswith("https://"):
        return s, s[len("https://"):]
    return f"http://{s}", s


async def _probe_one_host_port(host_with_scheme: str, host_for_tcp: str, port: int, timeout_sec: float) -> Optional[Tuple[str, int]]:
    try:
        fut = asyncio.open_connection(host_for_tcp, int(port))
        reader, writer = await asyncio.wait_for(fut, timeout=timeout_sec)
        writer.close()
        await writer.wait_closed()
        return host_with_scheme, int(port)
    except Exception:
        return None


def probe_available_ports(host_port_list: Optional[List[Tuple[str, int]]] = None) -> List[Tuple[str, int]]:
    pairs = host_port_list or [("http://127.0.0.1", 8188)]

    async def _run() -> List[Tuple[str, int]]:
        sem = asyncio.Semaphore(DEFAULT_PROBE_CONCURRENCY)

        async def _task(host: str, port: int):
            host_with_scheme, host_for_tcp = _normalize_probe_host(host)
            async with sem:
                return await _probe_one_host_port(
                    host_with_scheme=host_with_scheme,
                    host_for_tcp=host_for_tcp,
                    port=int(port),
                    timeout_sec=DEFAULT_PROBE_TIMEOUT_SEC,
                )

        tasks = [_task(h, p) for h, p in pairs]
        results = await asyncio.gather(*tasks)
        return [x for x in results if x is not None]

    available = asyncio.run(_run())
    if not available:
        raise RuntimeError("No reachable ComfyUI endpoints after port probing")
    return available


def build_reachable_endpoints(comfy_hosts: List[str], base_port: int, num_workers: int) -> List[str]:
    candidates: List[Tuple[str, int]] = []
    slots_per_host = max(1, int(num_workers))
    for host in comfy_hosts:
        for i in range(slots_per_host):
            candidates.append((host, int(base_port) + i))
    if not candidates:
        raise RuntimeError("没有构造出任何 ComfyUI endpoint")
    reachable = probe_available_ports(candidates)
    return [f"{host}:{port}" for host, port in reachable]


def is_remote_path(path: str) -> bool:
    """判断路径是否为远端对象存储路径。"""
    return str(path).startswith("s3://") or str(path).startswith("oss://")


def join_path(root: str, name: str) -> str:
    """拼接统一格式的路径字符串。"""
    return str(root).rstrip("/") + "/" + str(name).lstrip("/")


def safe_makedirs(path: str):
    """安全创建目录并兼容本地/远端场景。"""
    p = path if str(path).endswith("/") else str(path) + "/"
    if is_remote_path(p):
        return
    try:
        smart_makedirs(p, exist_ok=True)
    except TypeError:
        try:
            smart_makedirs(p)
        except Exception as e:
            if "File exists" not in str(e):
                raise
    except Exception as e:
        if "File exists" not in str(e):
            raise


def ensure_only_images(local_dir: str):
    """清理目录中的非图片文件。"""
    if not os.path.isdir(local_dir):
        return
    for fname in os.listdir(local_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            try:
                os.remove(os.path.join(local_dir, fname))
            except Exception:
                pass


def scan_done_prompt_indices(eval_dir: str) -> List[int]:
    """扫描输出目录并提取已完成的 prompt 索引。"""
    if not smart_exists(eval_dir):
        return []
    done = set()
    try:
        for name in smart_listdir(eval_dir):
            n = str(name).split("/")[-1]
            m = RE_OUT_IMG.match(n)
            if not m:
                continue
            done.add(int(m.group(1)))
    except Exception:
        return []
    return sorted(done)


def _iter_meta_files_for_model(meta_root: str, model_id: str) -> List[str]:
    """收集某个模型在 meta 根目录下可能关联的文件路径。"""
    out: List[str] = []
    dir_path = join_path(meta_root, model_id)
    if smart_exists(dir_path):
        try:
            for n in smart_listdir(dir_path):
                out.append(join_path(dir_path, n))
        except Exception:
            pass
    try:
        for n in smart_listdir(meta_root):
            bn = str(n).split("/")[-1]
            if bn.startswith(model_id + "_") or bn == f"{model_id}.json":
                out.append(join_path(meta_root, bn))
    except Exception:
        pass
    uniq = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def build_meta_index(meta_root: str) -> Dict[str, List[str]]:
    """建立 model_id 到 meta 文件名列表的索引。"""
    index: Dict[str, List[str]] = {}
    try:
        names = smart_listdir(meta_root)
    except FileNotFoundError:
        return index
    for name in names:
        bn = str(name).split("/")[-1].rstrip("/")
        m = re.match(r"(\d+)", bn)
        if m:
            model_id = m.group(1)
            index.setdefault(model_id, []).append(bn)
    return index


def read_json(path: str) -> Dict[str, Any]:
    """读取 JSON 文件并返回字典。"""
    with mopen(path, "r", encoding="utf-8") as f:
        return json.load(f)


def heuristic_from_json(j: Dict[str, Any]):
    """从模型 JSON 中提取 trigger 和图片 prompt 等信息。"""
    def strip_html(s: str) -> str:
        """函数用途说明：执行该步骤的核心逻辑。"""
        return re.sub(r"<[^>]*>", "", s or "").strip()

    v = (j.get("version") or {})
    trig_words = v.get("trigger_words") or []
    if isinstance(trig_words, dict):
        trig_words = list(trig_words.values())
    trigger_words = " ".join([tw for tw in trig_words if isinstance(tw, str)]).lower()
    images = j.get("images") or []
    image_prompts: List[str] = []
    for im in images:
        pos = (im.get("prompt") or "").strip()
        image_prompts.append(pos)
    return {
        "trigger_words": trigger_words,
        "_image_prompts": image_prompts,
        "description": strip_html(j.get("description") or ""),
    }, ""


def longest_common_substring_two(a: str, b: str) -> str:
    """计算两个字符串的最长公共子串。"""
    if not a or not b:
        return ""
    a = a.lower()
    b = b.lower()
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    max_len = 0
    end_pos = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > max_len:
                    max_len = curr[j]
                    end_pos = i
        prev = curr
    return a[end_pos - max_len:end_pos]


def longest_common_substring_multi(strings: List[str]) -> str:
    """计算多字符串的最长公共子串。"""
    strings = [s.lower() for s in strings if isinstance(s, str) and s.strip()]
    if not strings:
        return ""
    lcs = strings[0]
    for s in strings[1:]:
        lcs = longest_common_substring_two(lcs, s)
        if not lcs:
            break
    lcs = lcs.strip(" ,;:()[]{}<>\"'")
    if len(lcs) < 2:
        return ""
    if len(lcs) > 120:
        lcs = lcs[:120].rstrip()
    return lcs


def compute_trigger_from_image_prompts(prompts: List[str]) -> str:
    """基于示例 prompt 自动估计 trigger 词。"""
    prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    if len(prompts) < 2:
        return ""
    token_lists: List[List[str]] = []
    for p in prompts:
        norm = re.sub(r"[^a-zA-Z0-9\s]", " ", p.lower())
        tokens = [t for t in norm.split() if t]
        if tokens:
            token_lists.append(tokens)
    if len(token_lists) < 2:
        return ""
    max_ngram = 6

    def collect_ngrams(tokens: List[str]) -> set:
        """函数用途说明：执行该步骤的核心逻辑。"""
        ngrams = set()
        L = len(tokens)
        for i in range(L):
            for l in range(1, max_ngram + 1):
                j = i + l
                if j > L:
                    break
                span = tokens[i:j]
                if not any(tok not in STOPWORDS for tok in span):
                    continue
                ngrams.add(" ".join(span))
        return ngrams

    ngram_sets = []
    for toks in token_lists:
        cand = collect_ngrams(toks)
        if cand:
            ngram_sets.append(cand)
    if not ngram_sets:
        return ""
    common = set(ngram_sets[0])
    for s in ngram_sets[1:]:
        common &= s
        if not common:
            break
    if common:
        best = max(common, key=lambda s: (len(s.split()), len(s)))
        if len(best) > 120:
            best = best[:120].rstrip()
        if len(best) < 2:
            return ""
        return best
    stripped_prompts = []
    for toks in token_lists:
        content_toks = [t for t in toks if t not in STOPWORDS]
        if content_toks:
            stripped_prompts.append(" ".join(content_toks))
    if len(stripped_prompts) < 2:
        return ""
    return longest_common_substring_multi(stripped_prompts)


def parse_trigger_from_meta(model_id: str, meta_root: str, meta_index: Dict[str, List[str]]):
    """从 meta 信息中解析模型 trigger 与示例 prompts。"""
    names = meta_index.get(model_id, []) or []
    json_path = None
    for n in names:
        p = join_path(meta_root, n)
        if str(n) == f"{model_id}.json" and smart_exists(p):
            json_path = p
            break
    if json_path is None:
        for p in _iter_meta_files_for_model(meta_root, model_id):
            if str(p).lower().endswith(".json"):
                if os.path.basename(str(p)) == f"{model_id}.json":
                    json_path = p
                    break
    if json_path is None:
        return "", []
    try:
        j = read_json(json_path)
    except Exception:
        return "", []
    heur, _ = heuristic_from_json(j)
    return (heur.get("trigger_words") or "").strip(), (heur.get("_image_prompts") or [])


def copy_meta_for_model(model_id: str, meta_root: str, model_output_dir: str, meta_index: Dict[str, List[str]]):
    """把模型相关 meta 文件复制到输出目录。"""
    safe_makedirs(model_output_dir)
    demo_dir = join_path(model_output_dir, "demo_images")
    safe_makedirs(demo_dir)
    files = _iter_meta_files_for_model(meta_root, model_id)
    if not files:
        names = meta_index.get(model_id, [])
        for n in names:
            files.append(join_path(meta_root, n))
    for src in files:
        bn = os.path.basename(str(src))
        ext = os.path.splitext(bn)[1].lower()
        if ext == ".json":
            dst = join_path(model_output_dir, bn)
        elif ext in IMG_EXTS:
            dst = join_path(demo_dir, bn)
        else:
            continue
        try:
            mcopy(src, dst, overwrite=True)
        except Exception:
            pass


def add_trigger_to_img_jsons(model_id: str, model_output_dir: str, trigger_word: str, from_website: bool):
    """把 trigger 回写到 model_id_img*.json。"""
    try:
        names = smart_listdir(model_output_dir)
    except Exception:
        return
    for name in names:
        bn = str(name).split("/")[-1]
        if not bn.lower().endswith(".json"):
            continue
        if not bn.startswith(f"{model_id}_img"):
            continue
        path = join_path(model_output_dir, bn)
        try:
            j = read_json(path)
        except Exception:
            continue
        j["trigger_word"] = (trigger_word or "").strip()
        j["trigger_from_websit"] = bool(from_website)
        try:
            with mopen(path, "w", encoding="utf-8") as f:
                json.dump(j, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def prepare_prompt_txt(prompt_txt: str, local_tmp_root: str) -> str:
    """准备 prompt 文本（远端会先下载到本地）。"""
    if not is_remote_path(prompt_txt):
        return prompt_txt
    os.makedirs(local_tmp_root, exist_ok=True)
    local_prompt = os.path.join(local_tmp_root, "prompts_base.txt")
    with mopen(prompt_txt, "r", encoding="utf-8") as fin, open(local_prompt, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)
    return local_prompt


def load_prompts_from_txt(txt_path: str) -> List[str]:
    """按行读取 prompt 列表并过滤空行与注释。"""
    prompts = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            prompts.append(s)
    return prompts


def _iter_nodes(wf: Dict[str, Any]):
    """遍历 workflow 的节点字典。"""
    if isinstance(wf, dict):
        for nid, node in wf.items():
            if isinstance(node, dict):
                yield str(nid), node


def _deepcopy(obj):
    """对对象做 JSON 级深拷贝。"""
    return json.loads(json.dumps(obj))


def _find_sampler_nodes(wf: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """查找 workflow 中的采样器节点。"""
    out = []
    for nid, node in _iter_nodes(wf):
        ct = node.get("class_type", "")
        if ct in ("KSampler", "KSamplerAdvanced"):
            out.append((nid, node))
    return out


def _resolve_ref_node(wf: Dict[str, Any], ref) -> Optional[Tuple[str, Dict[str, Any]]]:
    """根据 ComfyUI 引用解析目标节点。"""
    if isinstance(ref, list) and len(ref) >= 1:
        nid = str(ref[0])
        node = wf.get(nid)
        if isinstance(node, dict):
            return nid, node
    return None


def _set_text_in_node(node: Dict[str, Any], text: str) -> bool:
    """向文本编码节点写入 prompt 文本。"""
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        return False
    hit = False
    if "text" in inputs:
        inputs["text"] = text
        hit = True
    if "clip_l" in inputs:
        inputs["clip_l"] = text
        hit = True
    if "t5xxl" in inputs:
        inputs["t5xxl"] = text
        hit = True
    if "prompt" in inputs:
        inputs["prompt"] = text
        hit = True
    return hit


def _find_checkpoint_nodes(wf: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """定位 workflow 中的底模加载节点。"""
    out = []
    for nid, node in _iter_nodes(wf):
        ct = node.get("class_type", "")
        if ct in ("CheckpointLoaderSimple", "CheckpointLoader"):
            out.append((nid, node))
            continue
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and ("ckpt_name" in inputs or "checkpoint" in inputs):
            out.append((nid, node))
    out.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 10**9)
    return out


def _find_lora_nodes(wf: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """定位 workflow 中可注入 LoRA 的节点。"""
    out = []
    for nid, node in _iter_nodes(wf):
        ct = node.get("class_type", "")
        if ct in ("LoraLoader", "LoraLoaderModelOnly", "LoraLoader|pysssss"):
            out.append((nid, node))
            continue
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and "lora_name" in inputs:
            out.append((nid, node))
    out.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 10**9)
    return out


def _depth_to_checkpoint_via_model_chain(wf: Dict[str, Any], start_nid: str, checkpoint_ids: set, max_hops: int = 50) -> int:
    """计算 LoRA 节点沿 model 链到 checkpoint 的距离。"""
    hops = 0
    cur_nid = start_nid
    seen = set()
    while hops < max_hops:
        if cur_nid in seen:
            return 10**9
        seen.add(cur_nid)
        node = wf.get(cur_nid)
        if not isinstance(node, dict):
            return 10**9
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            return 10**9
        ref = inputs.get("model")
        parent = _resolve_ref_node(wf, ref)
        if parent is None:
            return 10**9
        pid, pnode = parent
        hops += 1
        if pid in checkpoint_ids or pnode.get("class_type", "") in ("CheckpointLoaderSimple", "CheckpointLoader"):
            return hops
        cur_nid = pid
    return 10**9


def _get_filename_from_url(url: str) -> str:
    """从图片 URL 中提取文件名。"""
    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        filename_list = query.get("filename")
        if filename_list:
            return filename_list[0]
        path = parsed.path or ""
        if path:
            return path.split("/")[-1]
    except Exception:
        pass
    return str(url).split("/")[-1]


def _append_text_line(path: str, line: str):
    """向文本日志追加一行内容。"""
    try:
        with mopen(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _inject_workflow_payload(
    wf: Dict[str, Any],
    *,
    seed: int,
    base_model_name: str,
    lora_name: str,
    strength_model: float,
    strength_clip: float,
    positive_prompt: str,
    filename_prefix: str,
) -> Dict[str, int]:
    """将 seed/base model/lora/positive/save prefix 一次性注入 workflow。"""
    seed_hits = 0
    seed = int(seed)
    for _, node in _find_sampler_nodes(wf):
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and "seed" in inputs:
            inputs["seed"] = seed
            seed_hits += 1
    for _, node in _iter_nodes(wf):
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and "seed" in inputs:
            if inputs["seed"] != seed:
                inputs["seed"] = seed
                seed_hits += 1

    bm_hits = 0
    base_model_name = (base_model_name or "").strip()
    if base_model_name:
        for _, node in _find_checkpoint_nodes(wf):
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            if "ckpt_name" in inputs:
                inputs["ckpt_name"] = base_model_name
                bm_hits += 1
            elif "checkpoint" in inputs:
                inputs["checkpoint"] = base_model_name
                bm_hits += 1

    lora_hits = 0
    lora_name = (lora_name or "").strip()
    if lora_name:
        all_loras = _find_lora_nodes(wf)
        if all_loras:
            candidate = []
            for nid, n in all_loras:
                inputs = n.get("inputs", {})
                if not isinstance(inputs, dict):
                    continue
                cur_name = str(inputs.get("lora_name", "")).lower()
                if "smooth" not in cur_name:
                    candidate.append((nid, n))
            target_loras = candidate if candidate else all_loras
            ckpts = _find_checkpoint_nodes(wf)
            checkpoint_ids = {nid for nid, _ in ckpts}
            best = None
            for nid, node in target_loras:
                d = _depth_to_checkpoint_via_model_chain(wf, nid, checkpoint_ids)
                if best is None:
                    best = (d, nid, node)
                else:
                    bd, bid, _ = best
                    nid_key = int(nid) if str(nid).isdigit() else 10**9
                    bid_key = int(bid) if str(bid).isdigit() else 10**9
                    if d < bd or (d == bd and nid_key < bid_key):
                        best = (d, nid, node)
            if best is not None:
                _, _, target_node = best
                inputs = target_node.get("inputs", {})
                if isinstance(inputs, dict):
                    if "lora_name" in inputs:
                        inputs["lora_name"] = lora_name
                        lora_hits += 1
                    if "strength_model" in inputs:
                        inputs["strength_model"] = float(strength_model)
                    if "strength_clip" in inputs:
                        inputs["strength_clip"] = float(strength_clip)

    pos_hits = 0
    samplers = _find_sampler_nodes(wf)
    for _, s_node in samplers:
        s_inputs = s_node.get("inputs", {})
        if not isinstance(s_inputs, dict):
            continue
        pos_ref = s_inputs.get("positive")
        pos_target = _resolve_ref_node(wf, pos_ref)
        if pos_target is None:
            continue
        _, pos_node = pos_target
        if _set_text_in_node(pos_node, positive_prompt):
            pos_hits += 1

    prefix_hits = 0
    for _, node in _iter_nodes(wf):
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if "filename_prefix" in inputs:
            inputs["filename_prefix"] = filename_prefix
            prefix_hits += 1

    return {
        "seed_hits": seed_hits,
        "base_hits": bm_hits,
        "lora_hits": lora_hits,
        "positive_hits": pos_hits,
        "prefix_hits": prefix_hits,
    }


def _write_binary(path: str, content: bytes):
    """把二进制内容写入目标路径。"""
    with mopen(path, "wb") as f:
        f.write(content)


def _write_json(path: str, data: Dict[str, Any]):
    """把字典以 JSON 格式写入目标路径。"""
    with mopen(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def _download_worker(
    session: aiohttp.ClientSession,
    download_queue: "asyncio.Queue[Optional[Dict[str, Any]]]",
    max_retries: int,
    retry_delay: float,
):
    """下载协程：消费队列中的下载任务并复用 keep-alive session。"""
    while True:
        if _is_stopping() and download_queue.empty():
            break
        task = await download_queue.get()
        if task is None:
            download_queue.task_done()
            break
        img_url = str(task["img_url"]).strip()
        img_path = str(task["img_path"])
        meta_path = str(task["meta_path"])
        meta = task.get("meta")
        sel_idx = int(task.get("sel_idx", -1))
        ok = False
        for attempt in range(max(1, int(max_retries))):
            if _is_stopping():
                break
            try:
                async with session.get(img_url) as resp:
                    if resp.status != 200:
                        if attempt + 1 < max_retries:
                            await asyncio.sleep(float(retry_delay) * (2 ** attempt))
                        continue
                    content = await resp.read()
                await asyncio.to_thread(_write_binary, img_path, content)
                if isinstance(meta, dict):
                    await asyncio.to_thread(_write_json, meta_path, meta)
                ok = True
                break
            except Exception:
                if attempt + 1 < max_retries:
                    await asyncio.sleep(float(retry_delay) * (2 ** attempt))
        if not ok:
            print(f"[WARN] sel_idx={sel_idx} 下载失败: {img_url}")
        download_queue.task_done()


async def run_comfy_batch_workflow(
    *,
    wf_json: str,
    base_model_name: str,
    lora_name: str,
    prompts_to_run: List[Tuple[int, str]],
    save_dir: str,
    strength_model: float,
    strength_clip: float,
    url: str,
    gen_seed: Optional[int] = None,
    save_prompt_json: bool = True,
    exec_timeout: Optional[float] = None,
    download_queue: Optional["asyncio.Queue[Optional[Dict[str, Any]]]"] = None,
):
    """按批次执行 ComfyUI 工作流并下载结果图片。"""
    safe_makedirs(save_dir)
    with mopen(wf_json, "r", encoding="utf-8") as f:
        wf_template = json.load(f)
    async with ComfyKit(comfyui_url=url) as kit:
        for sel_idx, pos_prompt in prompts_to_run:
            if _is_stopping():
                break
            wf = _deepcopy(wf_template)
            if gen_seed is None:
                new_seed = int.from_bytes(os.urandom(4), byteorder="big", signed=False)
            else:
                new_seed = int(gen_seed) + int(sel_idx)
            raw_prefix = f"{lora_name}_{sel_idx:05d}_{uuid.uuid4().hex}"
            filename_prefix = re.sub(r"[^0-9A-Za-z_\-]+", "_", raw_prefix)
            _inject_workflow_payload(
                wf,
                seed=int(new_seed),
                base_model_name=base_model_name,
                lora_name=lora_name,
                strength_model=float(strength_model),
                strength_clip=float(strength_clip),
                positive_prompt=pos_prompt,
                filename_prefix=filename_prefix,
            )

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(wf, tf, ensure_ascii=False, indent=2)
                temp_json = tf.name

            try:
                if exec_timeout and float(exec_timeout) > 0:
                    result = await asyncio.wait_for(kit.execute(temp_json), timeout=float(exec_timeout))
                else:
                    result = await kit.execute(temp_json)
            except asyncio.TimeoutError:
                continue
            finally:
                try:
                    os.remove(temp_json)
                except OSError:
                    pass

            if _is_stopping():
                break

            images_all = getattr(result, "images", None) or []
            images: List[str] = []
            mismatch_log = os.path.join(save_dir, "mismatch_log.txt")

            for item in images_all:
                if isinstance(item, str):
                    url_i = item
                elif isinstance(item, dict):
                    url_i = item.get("url") or item.get("image_url") or item.get("filename") or ""
                    if not url_i:
                        continue
                else:
                    url_i = str(item)
                fname = _get_filename_from_url(url_i)
                if fname.startswith(filename_prefix):
                    images.append(url_i)
                else:
                    _append_text_line(mismatch_log, f"sel_idx={sel_idx} prefix={filename_prefix} filename={fname} url={url_i}")

            if not images:
                _append_text_line(mismatch_log, f"sel_idx={sel_idx} prefix={filename_prefix} no_valid_image, images_all_len={len(images_all)}")
                continue
            images = list(dict.fromkeys([u for u in images if u]))[:1]
            for img_idx, img_url in enumerate(images):
                if _is_stopping():
                    break
                prefix = f"{sel_idx:05d}_{img_idx}"
                img_path = join_path(save_dir, prefix + ".png")
                meta_path = join_path(save_dir, prefix + ".json")
                img_url = img_url.strip() if isinstance(img_url, str) else str(img_url)
                meta = None
                if save_prompt_json:
                    meta = {
                        "base_model": base_model_name,
                        "lora_name": lora_name,
                        "strength_model": float(strength_model),
                        "strength_clip": float(strength_clip),
                        "positive_prompt": pos_prompt,
                        "seed": int(new_seed),
                        "workflow": wf_json,
                        "mode": "custom_workflow",
                        "sel_idx": int(sel_idx),
                        "endpoint": url,
                    }
                if download_queue is not None:
                    await download_queue.put(
                        {
                            "img_url": img_url,
                            "img_path": img_path,
                            "meta_path": meta_path,
                            "meta": meta,
                            "sel_idx": int(sel_idx),
                        }
                    )
                else:
                    max_retries = 3
                    retry_delay = 2
                    success = False
                    for attempt in range(max_retries):
                        if _is_stopping():
                            break
                        try:
                            req = Request(img_url, headers={"User-Agent": "Mozilla/5.0", "Connection": "close"})
                            with urlopen(req, timeout=300) as resp:
                                status_code = getattr(resp, "status", 200)
                                if status_code != 200:
                                    if attempt < max_retries - 1:
                                        time.sleep(retry_delay * (2 ** attempt))
                                    continue
                                content = resp.read()
                            _write_binary(img_path, content)
                            if isinstance(meta, dict):
                                _write_json(meta_path, meta)
                            success = True
                            break
                        except (HTTPError, URLError):
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay * (2 ** attempt))
                        except Exception:
                            break
                    if not success:
                        continue


async def _run_with_download_queue(
    *,
    wf_json: str,
    base_model: str,
    lora_name: str,
    prompts_to_run: List[Tuple[int, str]],
    selected_prompts: List[str],
    indices_to_run: List[int],
    eval_dir: str,
    strength_model: float,
    strength_clip: float,
    comfy_url: str,
    gen_seed: Optional[int],
    exec_timeout: Optional[float],
    download_retry_rounds: int,
    download_retry_wait: float,
    download_workers: int,
) -> List[Tuple[int, str]]:
    """在每个 LoRA 处理周期内维护下载队列与 keep-alive 会话。"""
    timeout = aiohttp.ClientTimeout(total=300, connect=20, sock_connect=20, sock_read=300)
    connector = aiohttp.TCPConnector(
        limit=max(32, int(download_workers) * 8),
        limit_per_host=max(8, int(download_workers) * 4),
        ttl_dns_cache=300,
    )
    download_queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue(maxsize=max(64, int(download_workers) * 8))
    remain_to_run = list(prompts_to_run)
    total_rounds = max(0, int(download_retry_rounds)) + 1
    # 创建一个长链接下载会话，供给当前lora的所有下载任务使用
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers={"User-Agent": "Mozilla/5.0", "Connection": "keep-alive"},
    ) as session:
        #创建异步下载worker的数量，每个 worker 都拿到同一个 session 和同一个 download_queue。把同一个长连接会话分发给多个下载协程复用
        workers = [
            asyncio.create_task(_download_worker(session, download_queue, max_retries=3, retry_delay=2.0))
            for _ in range(max(1, int(download_workers)))
        ]
        try:
            for round_idx in range(total_rounds):
                if _is_stopping() or not remain_to_run:
                    break
                await run_comfy_batch_workflow(
                    wf_json=wf_json,
                    base_model_name=base_model,
                    lora_name=lora_name,
                    prompts_to_run=remain_to_run,
                    save_dir=eval_dir,
                    strength_model=strength_model,
                    strength_clip=strength_clip,
                    url=comfy_url,
                    gen_seed=gen_seed,
                    exec_timeout=exec_timeout,
                    download_queue=download_queue,
                )
                await download_queue.join()
                if _is_stopping():
                    break
                done_after = set(scan_done_prompt_indices(eval_dir))
                remain_to_run = [(i, selected_prompts[i]) for i in indices_to_run if i not in done_after]
                if remain_to_run and round_idx + 1 < total_rounds:
                    await asyncio.sleep(max(0.0, float(download_retry_wait)))
        finally:
            for _ in workers:
                await download_queue.put(None)
            await asyncio.gather(*workers, return_exceptions=True)
    return remain_to_run


def _normalize_prompt_for_dedup(prompt: str) -> str:
    """规范化 prompt 以便去重比较。"""
    s = (prompt or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(",.;: ")
    return s


def _normalize_text_for_match(text: str) -> str:
    """规范化文本用于短语包含判断。"""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_phrase(text: str, phrase: str) -> bool:
    """判断文本中是否已包含某个短语。"""
    t = _normalize_text_for_match(text)
    p = _normalize_text_for_match(phrase)
    if not p:
        return False
    return p in t


def _dedup_prompts(prompts: List[str]) -> List[str]:
    """对 prompt 列表做去重。"""
    seen = set()
    out: List[str] = []
    for p in prompts:
        p = (p or "").strip()
        if not p:
            continue
        key = _normalize_prompt_for_dedup(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _compose_prompt(prompt: str, trigger_word: str, prefix_phrase: str) -> str:
    """组装单条 prompt（trigger/前缀 + 原始 prompt）。"""
    body = (prompt or "").strip()
    if not body:
        return ""
    head_parts: List[str] = []
    pp = (prefix_phrase or "").strip()
    tw = (trigger_word or "").strip()
    if tw and not _contains_phrase(body, tw):
        head_parts.append(tw)
    if pp and not _contains_phrase(body, pp):
        head_parts.append(pp)
    if not head_parts:
        return body
    return f"{', '.join(head_parts)}, {body}"


def _attach_prefix_and_trigger_to_prompts(prompts: List[str], trigger_word: str, prefix_phrase: str) -> List[str]:
    """为整批 prompt 添加 trigger 与固定前缀。"""
    out: List[str] = []
    for p in prompts:
        composed = _compose_prompt(p, trigger_word, prefix_phrase)
        if composed:
            out.append(composed)
    return out


def _extract_trigger_from_json_data(j: Dict) -> str:
    """从模型 JSON 结构中提取 trigger 字段。"""
    if not isinstance(j, dict):
        return ""
    tw = j.get("trigger_word")
    if isinstance(tw, str) and tw.strip():
        return tw.strip()
    tws = j.get("trigger_words")
    if isinstance(tws, str) and tws.strip():
        return tws.strip()
    if isinstance(tws, list):
        vals = [str(x).strip() for x in tws if str(x).strip()]
        if vals:
            return " ".join(vals).strip()
    v = j.get("version")
    if isinstance(v, dict):
        v_tws = v.get("trigger_words")
        if isinstance(v_tws, list):
            vals = [str(x).strip() for x in v_tws if str(x).strip()]
            if vals:
                return " ".join(vals).strip()
        if isinstance(v_tws, str) and v_tws.strip():
            return v_tws.strip()
    return ""


def _read_trigger_word_from_model_meta(model_id: str, meta_root: str) -> str:
    """优先从 model_id 对应 meta json 读取 trigger。"""
    candidate_paths = [
        join_path(join_path(meta_root, model_id), f"{model_id}_img1.json"),
        join_path(join_path(meta_root, model_id), f"{model_id}.json"),
        join_path(meta_root, f"{model_id}_img1.json"),
        join_path(meta_root, f"{model_id}.json"),
    ]
    for p in candidate_paths:
        if not smart_exists(p):
            continue
        try:
            with mopen(p, "r", encoding="utf-8") as f:
                j = json.load(f)
            tw = _extract_trigger_from_json_data(j)
            if tw:
                return tw
        except Exception:
            pass
    return ""


def _prepare_workflow_with_negative(
    wf_json: str,
    negative_prompt: str,
    negative_node_id: int,
    local_tmp_root: str,
    model_id: str,
) -> str:
    """生成注入 negative prompt 的临时 workflow。"""
    neg = (negative_prompt or "").strip()
    if not neg:
        return wf_json
    with mopen(wf_json, "r", encoding="utf-8") as f:
        wf = json.load(f)
    node_key = str(int(negative_node_id))
    node = wf.get(node_key)
    if not isinstance(node, dict):
        raise RuntimeError(f"workflow 节点不存在: {node_key}")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError(f"workflow 节点 inputs 非法: {node_key}")
    if "text" in inputs:
        inputs["text"] = neg
    elif "prompt" in inputs:
        inputs["prompt"] = neg
    elif "clip_l" in inputs:
        inputs["clip_l"] = neg
    else:
        inputs["text"] = neg
    wf_tmp_dir = os.path.join(local_tmp_root, "wf_tmp")
    os.makedirs(wf_tmp_dir, exist_ok=True)
    wf_tmp_path = os.path.join(wf_tmp_dir, f"{model_id}_neg_{int(time.time() * 1000)}.json")
    with open(wf_tmp_path, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
    return wf_tmp_path


def _tokenize_prompt(prompt: str) -> List[str]:
    """把 prompt 分词为英文词元和中文词元。"""
    s = (prompt or "").lower()
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", s)


def _jaccard_distance(a: set, b: set) -> float:
    """计算两个词集合的 Jaccard 距离。"""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union <= 0:
        return 0.0
    return 1.0 - (inter / union)


def _select_diverse_indices(prompts: List[str], k: int, rng: random.Random) -> List[int]:
    """使用贪心 max-min 选择更分散的 prompt 索引。"""
    n = len(prompts)
    if k >= n:
        return list(range(n))
    token_sets: List[set] = [set(_tokenize_prompt(p)) for p in prompts]
    first_idx = rng.randrange(n)
    selected: List[int] = [first_idx]
    remaining = set(range(n))
    remaining.remove(first_idx)
    while len(selected) < k and remaining:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            cand_set = token_sets[idx]
            min_dist = 1.0
            for sid in selected:
                d = _jaccard_distance(cand_set, token_sets[sid])
                if d < min_dist:
                    min_dist = d
                if min_dist <= best_score:
                    break
            score = min_dist + rng.random() * 1e-9
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            best_idx = remaining.pop()
            selected.append(best_idx)
        else:
            remaining.remove(best_idx)
            selected.append(best_idx)
    return selected


def _stable_model_offset(model_id: str) -> int:
    """生成与 model_id 相关的稳定随机偏移。"""
    if model_id.isdigit():
        return int(model_id)
    return sum(ord(ch) for ch in model_id) % 1000003


def _selected_prompts_manifest_path(eval_dir: str) -> str:
    """返回采样 manifest 的保存路径。"""
    return join_path(eval_dir, "selected_prompts_diverse.json")


def _load_selected_prompts_if_any(eval_dir: str) -> Optional[Dict[str, Any]]:
    """读取已有采样 manifest。"""
    mpth = _selected_prompts_manifest_path(eval_dir)
    if not smart_exists(mpth):
        return None
    try:
        with mopen(mpth, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_selected_prompts(eval_dir: str, manifest: Dict[str, Any]):
    """保存采样 manifest。"""
    mpth = _selected_prompts_manifest_path(eval_dir)
    try:
        with mopen(mpth, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def select_diverse_prompts_for_model(
    *,
    model_id: str,
    all_prompts: List[str],
    num_prompts: int,
    prompt_seed: int,
    trigger_word: str,
    prefix_phrase: str,
    eval_dir: str,
    overwrite: bool,
) -> List[str]:
    """按模型执行可复现的多样化 prompt 采样。"""
    num_prompts = int(num_prompts)
    if num_prompts <= 0:
        raise ValueError("--num-prompts 必须 > 0")
    if not all_prompts:
        raise ValueError("prompt 列表为空")
    safe_makedirs(eval_dir)
    strategy = "greedy_maxmin_jaccard"
    k = min(num_prompts, len(all_prompts))
    if not overwrite:
        old = _load_selected_prompts_if_any(eval_dir)
        old_prompts = old.get("base_prompts") if isinstance(old, dict) else None
        if not isinstance(old_prompts, list) and isinstance(old, dict):
            old_prompts = old.get("prompts")
        if old and isinstance(old_prompts, list) and int(old.get("num_prompts", -1)) == k:
            old_tr = (old.get("trigger_word") or "").strip()
            old_pp = (old.get("prefix_phrase") or "").strip()
            old_st = (old.get("strategy") or "").strip()
            if old_tr == (trigger_word or "").strip() and old_pp == (prefix_phrase or "").strip() and old_st == strategy:
                prompts = [str(x) for x in old_prompts if str(x).strip()]
                if len(prompts) == k:
                    return prompts
    rng = random.Random(int(prompt_seed) + _stable_model_offset(model_id))
    indices = _select_diverse_indices(all_prompts, k, rng)
    selected = [all_prompts[i] for i in indices]
    manifest = {
        "model_id": model_id,
        "num_prompts": k,
        "prompt_seed": int(prompt_seed),
        "trigger_word": (trigger_word or "").strip(),
        "prefix_phrase": (prefix_phrase or "").strip(),
        "strategy": strategy,
        "indices": indices,
        "prompts": selected,
        "base_prompts": selected,
    }
    _save_selected_prompts(eval_dir, manifest)
    return selected


def process_one_lora(
    lora_path: str,
    base_model: str,
    comfy_url: str,
    wf_json: str,
    meta_root: str,
    meta_index: Dict[str, List[str]],
    output_root: str,
    local_tmp_root: str,
    base_prompt_txt_local: str,
    num_prompts_to_gen: int,
    prompt_seed: int,
    gen_seed: Optional[int],
    overwrite: bool = False,
    strength_model: float = 1.0,
    strength_clip: float = 1.0,
    model_prompt_file: Optional[str] = None,
    output_subdir: Optional[str] = None,
    exec_timeout: Optional[float] = None,
    download_retry_rounds: int = 2,
    download_retry_wait: float = 2.0,
    prefix_phrase: str = "",
    negative_prompt: str = "",
    negative_node_id: int = 12,
    download_workers: int = 4,
):
    """处理单个 LoRA 的完整生成流程。"""
    if _is_stopping():
        return
    lora_name = os.path.basename(lora_path)
    stem, _ = os.path.splitext(lora_name)
    m = re.match(r"(\d+)", stem)
    if not m:
        print(f"[WARN] LoRA 文件名不以数字开头，无法解析 model_id: {lora_name}，跳过")
        return
    model_id = m.group(1)
    model_output_dir = join_path(output_root, model_id)
    if output_subdir:
        model_output_dir = join_path(model_output_dir, output_subdir)
    eval_dir = join_path(model_output_dir, "eval_images_with_negative_new")
    # 从meta中读取trigger word
    trigger_from_json, image_prompts = parse_trigger_from_meta(model_id, meta_root, meta_index)
    # 从model_id_img1.json中读取trigger word
    trigger_word = (_read_trigger_word_from_model_meta(model_id, meta_root) or trigger_from_json or "").strip()
    #这部分暂时不要
    # from_website = False
    # if trigger_word:
    #     from_website = True
    # else:
    #     #如果还是为空就从prompt示例里面找共同的短语
    #     trigger_word = compute_trigger_from_image_prompts(image_prompts).strip()
    #     from_website = False
    # # copy_meta_for_model(...) ：把模型相关 json/示例图拷到输出目录。
    # copy_meta_for_model(model_id, meta_root, model_output_dir, meta_index)
    # # add_trigger_to_img_jsons(...) ：把最终 trigger 回写到 model_id_img*.json ，便于后续追溯。
    # add_trigger_to_img_jsons(model_id, model_output_dir, trigger_word, from_website)

    all_prompts = load_prompts_from_txt(base_prompt_txt_local)
    # 先在原始 prompt 池去重并采样，再在采样结果前注入 trigger/prefix
    # all_prompts = _dedup_prompts(all_prompts)
    # if not all_prompts:
    #     print("no available prompts after dedup, skip")
    #     return
    # if len(all_prompts) < int(num_prompts_to_gen):
    #     print(
    #         f"[WARN] {model_id}: 去重后可用 prompt 数量不足，"
    #         f"unique={len(all_prompts)} < need={int(num_prompts_to_gen)}，跳过以保证不重复和数量要求"
    #     )
    #     return

    safe_makedirs(eval_dir)
    selected_base_prompts = select_diverse_prompts_for_model(
        model_id=model_id,
        all_prompts=all_prompts,
        num_prompts=num_prompts_to_gen,
        prompt_seed=prompt_seed,
        trigger_word=trigger_word,
        prefix_phrase=prefix_phrase,
        eval_dir=eval_dir,
        overwrite=overwrite,
    )
    selected_prompts = _attach_prefix_and_trigger_to_prompts(selected_base_prompts, trigger_word, prefix_phrase)
    old_manifest = _load_selected_prompts_if_any(eval_dir) or {}
    _save_selected_prompts(
        eval_dir,
        {
            "model_id": model_id,
            "num_prompts": len(selected_prompts),
            "prompt_seed": int(prompt_seed),
            "trigger_word": (trigger_word or "").strip(),
            "prefix_phrase": (prefix_phrase or "").strip(),
            "strategy": old_manifest.get("strategy") or "greedy_maxmin_jaccard",
            "indices": old_manifest.get("indices") or list(range(len(selected_prompts))),
            "prompts": selected_prompts,
            "base_prompts": selected_base_prompts,
        },
    )
    N = len(selected_prompts)
    done_indices = set(scan_done_prompt_indices(eval_dir))
    done_cnt = len(done_indices)
    if overwrite:
        indices_to_run = list(range(N))
    else:
        if done_cnt >= N:
            print(f"[SKIP] {model_id}: 已完成 {done_cnt}/{N}（按 prompt 索引计），跳过")
            return
        indices_to_run = [i for i in range(N) if i not in done_indices]
    prompts_to_run = [(i, selected_prompts[i]) for i in indices_to_run]
    if _is_stopping():
        return

    runtime_wf_json = wf_json
    created_runtime_wf = False
    remain_to_run = list(prompts_to_run)
    try:
        runtime_wf_json = _prepare_workflow_with_negative(
            wf_json=wf_json,
            negative_prompt=negative_prompt,
            negative_node_id=negative_node_id,
            local_tmp_root=local_tmp_root,
            model_id=model_id,
        )
        created_runtime_wf = runtime_wf_json != wf_json
        remain_to_run = asyncio.run(
            _run_with_download_queue(
                wf_json=runtime_wf_json,
                base_model=base_model,
                lora_name=lora_name,
                prompts_to_run=remain_to_run,
                selected_prompts=selected_prompts,
                indices_to_run=indices_to_run,
                eval_dir=eval_dir,
                strength_model=strength_model,
                strength_clip=strength_clip,
                comfy_url=comfy_url,
                gen_seed=gen_seed,
                exec_timeout=exec_timeout,
                download_retry_rounds=download_retry_rounds,
                download_retry_wait=download_retry_wait,
                download_workers=download_workers,
            )
        )
    finally:
        if created_runtime_wf and os.path.isfile(runtime_wf_json):
            try:
                os.remove(runtime_wf_json)
            except Exception:
                pass

    if not is_remote_path(eval_dir):
        ensure_only_images(eval_dir)


def read_model_id_txt(path: str) -> List[str]:
    """读取 model_id 文本并解析有效 id。"""
    ids: List[str] = []
    if not path:
        return ids
    with mopen(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().endswith(".safetensors"):
                s = os.path.splitext(s)[0]
            m = re.match(r"(\d+)", s)
            if m:
                ids.append(m.group(1))
            else:
                ids.append(s)
    return ids


def process_one(
    lora_path: str,
    comfy_url: str,
    base_model: str,
    wf_json: str,
    meta_root: str,
    meta_index: Dict[str, List[str]],
    output_root: str,
    local_tmp_root: str,
    base_prompt_txt_local: str,
    num_prompts_to_gen: int,
    prompt_seed: int,
    gen_seed: Optional[int],
    overwrite: bool,
    strength_model: float,
    strength_clip: float,
    model_prompt_file: Optional[str],
    output_subdir: Optional[str],
    exec_timeout: Optional[float],
    download_retry_rounds: int,
    download_retry_wait: float,
    prefix_phrase: str,
    negative_prompt: str,
    negative_node_id: int,
    download_workers: int,
) -> Tuple[str, Optional[bool]]:
    """线程 worker 的单任务包装函数。用于捕捉线程失败原因"""
    if _is_stopping():
        return lora_path, False
    try:
        process_one_lora(
            lora_path=lora_path,
            base_model=base_model,
            comfy_url=comfy_url,
            wf_json=wf_json,
            meta_root=meta_root,
            meta_index=meta_index,
            output_root=output_root,
            local_tmp_root=local_tmp_root,
            base_prompt_txt_local=base_prompt_txt_local,
            num_prompts_to_gen=num_prompts_to_gen,
            prompt_seed=prompt_seed,
            gen_seed=gen_seed,
            overwrite=overwrite,
            strength_model=strength_model,
            strength_clip=strength_clip,
            model_prompt_file=model_prompt_file,
            output_subdir=output_subdir,
            exec_timeout=exec_timeout,
            download_retry_rounds=download_retry_rounds,
            download_retry_wait=download_retry_wait,
            prefix_phrase=prefix_phrase,
            negative_prompt=negative_prompt,
            negative_node_id=negative_node_id,
            download_workers=download_workers,
        )
        return lora_path, False if _is_stopping() else True
    except Exception as e:
        print(f"[WORKER] 处理 LoRA 失败: {lora_path} ----> {e}")
        return lora_path, None


def main():
    """命令行入口，组织参数解析与并发调度。"""
    STOP_EVENT.clear()
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    parser = argparse.ArgumentParser(description="批量生成 eval_images（standalone）")
    parser.add_argument("--lora-root", required=True)
    parser.add_argument("--meta-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompt-txt", default="/data/LoraPipeline/assets/diverse_prompts_100.txt")
    parser.add_argument("--model-prompt-file", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--prefix-phrase", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--negative-node-id", type=int, default=12)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--workflow-json", default="/data/LoraPipeline/assets/illustrious_simple.json")
    parser.add_argument("--strength-model", type=float, default=1.0)
    parser.add_argument("--strength-clip", type=float, default=1.0)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--gen-seed", type=int, default=None)
    parser.add_argument("--comfy-host", action="append", default=[])
    parser.add_argument("--base-port", type=int, default=8188)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--local-tmp-root", default="/tmp/lora_comfy_cache")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--filter-model-id", default=None)
    parser.add_argument("--model-id-txt", default=None)
    parser.add_argument("--exec-timeout", type=float, default=0)
    parser.add_argument("--download-retry-rounds", type=int, default=2)
    parser.add_argument("--download-retry-wait", type=float, default=2.0)
    parser.add_argument("--download-workers", type=int, default=1, help="每个 LoRA 的下载协程数量（共享 keep-alive session）")
    args = parser.parse_args()

    lora_root = args.lora_root.rstrip("/")
    meta_root = args.meta_root.rstrip("/")
    output_root = args.output_root.rstrip("/")
    if args.num_workers <= 0:
        args.num_workers = 1
    os.makedirs(args.local_tmp_root, exist_ok=True)
    exec_timeout: Optional[float] = args.exec_timeout
    if exec_timeout is not None and exec_timeout <= 0:
        exec_timeout = None

    raw_hosts = args.comfy_host if isinstance(args.comfy_host, list) else [args.comfy_host]
    comfy_hosts: List[str] = []
    for h in raw_hosts:
        if not h:
            continue
        for part in h.split(","):
            part = part.strip()
            if part:
                comfy_hosts.append(part.rstrip("/"))
    comfy_hosts = list(dict.fromkeys(comfy_hosts)) or ["http://127.0.0.1"]

    endpoints = build_reachable_endpoints(
        comfy_hosts=comfy_hosts,
        base_port=args.base_port,
        num_workers=args.num_workers,
    )
    print("available endpoints: =====> ", len(endpoints))
    base_prompt_txt_local = prepare_prompt_txt(args.prompt_txt, args.local_tmp_root)
    meta_index = build_meta_index(meta_root)

    filter_ids: List[str] = []
    if args.model_id_txt:
        filter_ids.extend(read_model_id_txt(args.model_id_txt))
    if args.filter_model_id:
        if smart_exists(args.filter_model_id):
            filter_ids.extend(read_model_id_txt(args.filter_model_id))
        else:
            for part in str(args.filter_model_id).split(","):
                s = part.strip()
                if s:
                    m = re.match(r"(\d+)", s)
                    filter_ids.append(m.group(1) if m else s)
    filter_set = set(filter_ids)

    lora_paths: List[str] = []
    try:
        raw_names = smart_listdir(lora_root)
        by_id: Dict[str, List[str]] = {}
        for name in raw_names:
            if not str(name).lower().endswith(".safetensors"):
                continue
            stem = os.path.splitext(os.path.basename(str(name)))[0]
            m = re.match(r"(\d+)", stem)
            if not m:
                continue
            model_id = m.group(1)
            if filter_set and model_id not in filter_set:
                continue
            by_id.setdefault(model_id, []).append(str(name))
        final_names: List[str] = []
        for model_id, names in by_id.items():
            def sort_key(x: str):
                """函数用途说明：执行该步骤的核心逻辑。"""
                s = os.path.splitext(os.path.basename(x))[0]
                suffix = s[len(model_id):].lstrip("_")
                return suffix
            names.sort(key=sort_key, reverse=True)
            final_names.append(names[0])
        for name in final_names:
            lora_paths.append(join_path(lora_root, name))
    except FileNotFoundError:
        print(f"[ERROR] lora_root 不存在: {lora_root}")
        return

    lora_paths.sort()
    if not lora_paths:
        print("[INFO] 没有需要处理的 LoRA，退出。")
        return

    num_workers = min(len(endpoints), len(lora_paths))
    tasks: List[Tuple[str, str]] = []
    for idx, p in enumerate(lora_paths):
        ep_idx = idx % len(endpoints)
        tasks.append((p, endpoints[ep_idx]))

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    futures: Dict[concurrent.futures.Future, str] = {}
    try:
        futures = {
            executor.submit(
                process_one,
                lora_path,
                comfy_url,
                args.base_model,
                args.workflow_json,
                meta_root,
                meta_index,
                output_root,
                args.local_tmp_root,
                base_prompt_txt_local,
                args.num_prompts,
                args.prompt_seed,
                args.gen_seed,
                args.overwrite,
                args.strength_model,
                args.strength_clip,
                args.model_prompt_file,
                args.output_subdir,
                exec_timeout,
                args.download_retry_rounds,
                args.download_retry_wait,
                args.prefix_phrase,
                args.negative_prompt,
                args.negative_node_id,
                args.download_workers,
            ): lora_path
            for lora_path, comfy_url in tasks
        }
        pending = set(futures.keys())
        progress = tqdm(total=len(futures))
        try:
            while pending:
                if _is_stopping():
                    break
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    key = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"{key} generated an exception: {exc}")
                    finally:
                        progress.update(1)
        finally:
            progress.close()
        if _is_stopping():
            print("[STOP] 正在取消未完成任务...")
    finally:
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        if _is_stopping():
            raise SystemExit(130)


if __name__ == "__main__":
    main()
