"""
探测comfyui端口可达性
python /data/benchmark_metrics/lora_pipeline/probe_comfy_ports.py \
  --shell-file /data/benchmark_metrics/lora_pipeline/scripts/dual_lora_flux.sh \
  --start-port 8188 \
  --port-count 8 \
  --timeout-sec 2 \
  --concurrency 256 \
  --output-file /data/benchmark_metrics/logs/comfy_port_probe_results.json
python /data/benchmark_metrics/lora_pipeline/probe_comfy_ports.py \
  --shell-file /data/benchmark_metrics/lora_pipeline/scripts/dual_lora_illustrious.sh \
  --start-port 8188 \
  --port-count 8 \
  --timeout-sec 2 \
  --concurrency 256 \
  --output-file /data/benchmark_metrics/logs/comfy_port_probe_results.json
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List


def parse_hosts_from_shell(shell_path: str) -> List[str]:
    text = Path(shell_path).read_text(encoding="utf-8")
    hosts = []
    for line in text.splitlines():
        m = re.match(r"\s*ip\d+\s*=\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\s*$", line)
        if m:
            hosts.append(m.group(1))
    return list(dict.fromkeys(hosts))


async def probe_tcp(host: str, port: int, timeout_sec: float) -> Dict[str, object]:
    start = time.perf_counter()
    try:
        fut = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(fut, timeout=timeout_sec)
        writer.close()
        await writer.wait_closed()
        latency = round((time.perf_counter() - start) * 1000, 2)
        return {"host": host, "port": port, "reachable": True, "latency_ms": latency, "error": ""}
    except Exception as exc:
        latency = round((time.perf_counter() - start) * 1000, 2)
        return {"host": host, "port": port, "reachable": False, "latency_ms": latency, "error": str(exc)}


async def run_probe(hosts: List[str], start_port: int, port_count: int, timeout_sec: float, concurrency: int) -> List[Dict[str, object]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = []

    async def one(host: str, port: int):
        async with semaphore:
            return await probe_tcp(host, port, timeout_sec)

    for host in hosts:
        for p in range(start_port, start_port + port_count):
            tasks.append(one(host, p))
    return await asyncio.gather(*tasks)


def summarize(results: List[Dict[str, object]]) -> Dict[str, object]:
    by_host: Dict[str, Dict[str, object]] = {}
    for r in results:
        host = str(r["host"])
        port = int(r["port"])
        rec = by_host.setdefault(host, {"reachable_ports": [], "unreachable_ports": [], "errors": {}})
        if r["reachable"]:
            rec["reachable_ports"].append(port)
        else:
            rec["unreachable_ports"].append(port)
            rec["errors"][str(port)] = r["error"]
    total = len(results)
    ok = sum(1 for r in results if r["reachable"])
    return {
        "total_probes": total,
        "reachable_count": ok,
        "unreachable_count": total - ok,
        "reachable_rate": round(ok / total, 4) if total else 0.0,
        "by_host": by_host,
    }


async def main():
    parser = argparse.ArgumentParser(description="探测 ComfyUI 服务器端口可达性")
    parser.add_argument("--shell-file", default="/data/benchmark_metrics/lora_pipeline/illustrious_one_lora_diverse.sh")
    parser.add_argument("--hosts", default="", help="逗号分隔IP，传入后优先于 shell-file")
    parser.add_argument("--start-port", type=int, default=8188)
    parser.add_argument("--port-count", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--output-file", default="/data/benchmark_metrics/lora_pipeline/comfy_port_probe_results.json")
    args = parser.parse_args()

    if args.hosts.strip():
        hosts = [x.strip() for x in args.hosts.split(",") if x.strip()]
    else:
        hosts = parse_hosts_from_shell(args.shell_file)
    if not hosts:
        raise RuntimeError("未解析到任何 host")

    results = await run_probe(
        hosts=hosts,
        start_port=args.start_port,
        port_count=args.port_count,
        timeout_sec=args.timeout_sec,
        concurrency=args.concurrency,
    )
    summary = summarize(results)
    payload = {
        "hosts": hosts,
        "start_port": args.start_port,
        "port_count": args.port_count,
        "timeout_sec": args.timeout_sec,
        "concurrency": args.concurrency,
        "summary": summary,
        "results": results,
    }
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"hosts={len(hosts)} total={summary['total_probes']} reachable={summary['reachable_count']} unreachable={summary['unreachable_count']} rate={summary['reachable_rate']}")
    for host in hosts:
        item = summary["by_host"].get(host, {"reachable_ports": [], "unreachable_ports": []})
        print(f"{host} ok={item['reachable_ports']} fail={item['unreachable_ports']}")
    print(f"output={out}")


if __name__ == "__main__":
    asyncio.run(main())
