import argparse
import asyncio
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import aiohttp

from comfykit import ComfyKit


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def detect_image_type(data: bytes) -> str:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if len(data) >= 2 and data[:2] == b"BM":
        return "bmp"
    return ""


def extract_image_filename(image_url: str) -> str:
    parsed = urlparse(image_url)
    query = parse_qs(parsed.query)
    return (query.get("filename") or [""])[0]


class DownloadSessionPool:
    def __init__(self, size: int, timeout_sec: int):
        self.size = max(1, int(size))
        self.timeout_sec = max(1, int(timeout_sec))
        self._queue: asyncio.Queue[aiohttp.ClientSession] = asyncio.Queue(maxsize=self.size)
        self._sessions: List[aiohttp.ClientSession] = []

    async def __aenter__(self):
        for _ in range(self.size):
            timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
            connector = aiohttp.TCPConnector(limit=0, enable_cleanup_closed=True)
            session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            self._sessions.append(session)
            self._queue.put_nowait(session)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await asyncio.gather(*(s.close() for s in self._sessions), return_exceptions=True)
        self._sessions.clear()

    async def acquire(self) -> aiohttp.ClientSession:
        return await self._queue.get()

    async def release(self, session: aiohttp.ClientSession):
        await self._queue.put(session)


async def run_one(
    semaphore: Optional[asyncio.Semaphore],
    download_pool: DownloadSessionPool,
    download_dir: Path,
    filename_prefix: str,
    host: str,
    port: int,
    index: int,
    workflow_file: str,
    params: Dict[str, Any],
    session_pool_size: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    url = f"http://{host}:{port}"
    started_at = iso_now()
    start = time.perf_counter()
    record: Dict[str, Any] = {
        "timestamp": started_at,
        "host": host,
        "port": port,
        "url": url,
        "iteration": index,
        "status": "failed",
        "duration_ms": 0.0,
        "error": "",
        "result_status": "",
        "image_count": 0,
        "prompt_id": "",
        "image_url": "",
        "image_local_path": "",
        "image_valid": False,
    }

    if semaphore is not None:
        await semaphore.acquire()
    try:
        try:
            async with ComfyKit(comfyui_url=url, session_pool_size=session_pool_size) as kit:
                result = await asyncio.wait_for(kit.execute(workflow_file, params), timeout=timeout_sec)
            record["result_status"] = result.status
            record["prompt_id"] = result.prompt_id or ""
            record["image_count"] = len(result.images or [])
            if result.status != "completed":
                record["status"] = "failed"
                record["error"] = result.msg or f"workflow status={result.status}"
            elif not result.images:
                record["status"] = "failed"
                record["error"] = "no image url in result"
            else:
                image_url = result.images[0]
                record["image_url"] = image_url
                #前缀匹配，避免下载到其他节点的图片
                filename = extract_image_filename(image_url)
                if filename_prefix and not filename.startswith(filename_prefix):
                    raise Exception(f"filename prefix mismatch: expected '{filename_prefix}', got '{filename}'")
                #下载session池子负责下载
                session = await download_pool.acquire()
                try:
                    async with session.get(image_url) as response:
                        if response.status != 200:
                            raise Exception(f"download image failed: HTTP {response.status}")
                        data = await response.read()
                finally:
                    await download_pool.release(session)
                if not data:
                    raise Exception("download image is empty")
                image_type = detect_image_type(data)
                if not image_type:
                    raise Exception("downloaded image is invalid")
                image_id = uuid.uuid4().hex
                local_path = download_dir / f"{image_id}.{image_type}"
                local_path.write_bytes(data)
                record["image_local_path"] = str(local_path)
                record["image_valid"] = True
                record["status"] = "success"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
        finally:
            record["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
    finally:
        if semaphore is not None:
            semaphore.release()
    return record


def build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in records:
        key = f"{item['host']}:{item['port']}"
        grouped[key].append(item)

    per_port = []
    for endpoint, items in sorted(grouped.items()):
        durations = [x["duration_ms"] for x in items]
        success = sum(1 for x in items if x["status"] == "success")
        fail = len(items) - success
        p50 = percentile(durations, 0.5)
        p95 = percentile(durations, 0.95)
        p99 = percentile(durations, 0.99)
        per_port.append(
            {
                "endpoint": endpoint,
                "total_requests": len(items),
                "success_count": success,
                "fail_count": fail,
                "success_rate": round(success / len(items), 4) if items else 0.0,
                "success_ratio": f"{success}/{len(items)}",
                "avg_latency_ms": round(mean(durations), 2) if durations else 0.0,
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "max_latency_ms": round(max(durations), 2) if durations else 0.0,
                "min_latency_ms": round(min(durations), 2) if durations else 0.0,
            }
        )

    total = len(records)
    total_success = sum(1 for x in records if x["status"] == "success")
    total_fail = total - total_success
    all_durations = [x["duration_ms"] for x in records]
    return {
        "total_requests": total,
        "success_count": total_success,
        "fail_count": total_fail,
        "success_rate": round(total_success / total, 4) if total else 0.0,
        "success_ratio": f"{total_success}/{total}",
        "avg_latency_ms": round(mean(all_durations), 2) if all_durations else 0.0,
        "p95_latency_ms": round(percentile(all_durations, 0.95), 2) if all_durations else 0.0,
        "p99_latency_ms": round(percentile(all_durations, 0.99), 2) if all_durations else 0.0,
        "per_port": per_port,
    }


async def main():
    parser = argparse.ArgumentParser(description="ComfyUI 多服务器多端口高并发压测脚本")
    parser.add_argument("--hosts", type=str, default="10.201.18.49", help="逗号分隔主机列表")
    parser.add_argument("--start-port", type=int, default=8188, help="起始端口")
    parser.add_argument("--port-count", type=int, default=8, help="连续端口数量")
    parser.add_argument("--requests-per-port", type=int, default=20, help="每个端口请求次数")
    parser.add_argument("--workflow-file", type=str, default="/data/LoraPipeline/assets/SDXL_illustrious_magic.json", help="工作流文件路径")
    parser.add_argument("--params-json", type=str, default="{}", help="workflow 参数 JSON 字符串")
    parser.add_argument("--concurrency", type=int, default=64, help="全局并发数，传0表示不限制")
    parser.add_argument("--session-pool-size", type=int, default=2, help="每个 ComfyKit 的 session 池大小")
    parser.add_argument("--timeout-sec", type=int, default=600, help="单请求超时（秒）")
    parser.add_argument("--download-session-count", type=int, default=1, help="下载图片会话数量")
    parser.add_argument("--download-filename-prefix", type=str, default="", help="只下载文件名以此前缀开头的图片")
    parser.add_argument("--download-dir", type=str, default="/data/benchmark_metrics/lora_pipeline/comfykit_stress_images", help="下载并校验图片落盘目录")
    parser.add_argument("--output-file", type=str, default="/data/benchmark_metrics/lora_pipeline/comfykit_stress_results.json", help="结果输出文件")
    args = parser.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    params = json.loads(args.params_json)
    ports = [args.start_port + i for i in range(args.port_count)]
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    started_at = iso_now()
    semaphore = None if args.concurrency == 0 else asyncio.Semaphore(max(1, args.concurrency))
    filename_prefix = args.download_filename_prefix.strip()
    async with DownloadSessionPool(size=max(1, args.download_session_count), timeout_sec=max(1, args.timeout_sec)) as download_pool:
        tasks = []
        for host in hosts:
            for port in ports:
                for i in range(args.requests_per_port):
                    tasks.append(
                        run_one(
                            semaphore=semaphore,
                            download_pool=download_pool,
                            download_dir=download_dir,
                            filename_prefix=filename_prefix,
                            host=host,
                            port=port,
                            index=i,
                            workflow_file=args.workflow_file,
                            params=params,
                            session_pool_size=max(1, args.session_pool_size),
                            timeout_sec=max(1, args.timeout_sec),
                        )
                    )
        records = await asyncio.gather(*tasks)
    summary = build_summary(records)
    ended_at = iso_now()

    output = {
        "started_at": started_at,
        "ended_at": ended_at,
        "hosts": hosts,
        "ports": ports,
        "requests_per_port": args.requests_per_port,
        "concurrency": args.concurrency,
        "session_pool_size": args.session_pool_size,
        "download_session_count": args.download_session_count,
        "download_filename_prefix": filename_prefix,
        "download_dir": str(download_dir),
        "overall_test_success_rate": summary["success_rate"],
        "overall_test_success_ratio": summary["success_ratio"],
        "summary": summary,
        "records": records,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{iso_now()}] done, output={output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
