import gc
from collections import defaultdict
from dataclasses import dataclass

import torch
from loguru import logger


class MemoryLeakDetector:
    """
    Detects and reports GPU memory leaks during training.

    Features:
    - Track memory allocation over time
    - Detect memory growth patterns
    - Track tensor allocations with size information
    - Generate memory snapshots for debugging
    - Alert when memory usage exceeds thresholds
    """

    def __init__(
        self,
        device: torch.device,
        check_interval: int = 10,
        snapshot_interval: int = 100,
        alert_threshold_mb: float = 1000.0,
        enable_snapshot: bool = True,
        track_tensors: bool = True,
        track_references: bool = True,
    ):
        """
        Args:
            device: CUDA device to monitor
            check_interval: Check memory every N steps
            snapshot_interval: Save memory snapshot every N steps
            alert_threshold_mb: Alert when growth exceeds this (MB)
            enable_snapshot: Enable torch memory snapshots (may have overhead)
            track_tensors: Track individual tensor allocations
            track_references: Track tensor reference chains on leak detection
        """
        self.device = device
        self.check_interval = check_interval
        self.snapshot_interval = snapshot_interval
        self.alert_threshold_mb = alert_threshold_mb
        self.enable_snapshot = enable_snapshot
        self.track_tensors = track_tensors
        self.track_references = track_references

        # Memory tracking
        self.memory_history = []
        self.baseline_memory = None
        self.step_count = 0

        # Tensor tracking
        self.tensor_snapshots = []  # Track tensor info at each checkpoint
        self.last_tensor_info = None  # Track last tensor info for comparison

        # Initialize
        torch.cuda.reset_peak_memory_stats(device)
        if self.enable_snapshot:
            torch.cuda.memory._record_memory_history(enabled=True)

        logger.info(f"MemoryLeakDetector initialized for {device}")
        logger.info(f"  Check interval: {check_interval} steps")
        logger.info(f"  Snapshot interval: {snapshot_interval} steps")
        logger.info(f"  Alert threshold: {alert_threshold_mb} MB")
        logger.info(f"  Track tensors: {track_tensors}")
        logger.info(f"  Track references: {track_references}")

    def get_memory_stats(self):
        """Get current GPU memory statistics."""
        return {
            "allocated": torch.cuda.memory_allocated(self.device) / 1024**2,  # MB
            "reserved": torch.cuda.memory_reserved(self.device) / 1024**2,  # MB
            "peak_allocated": torch.cuda.max_memory_allocated(self.device) / 1024**2,  # MB
            "peak_reserved": torch.cuda.max_memory_reserved(self.device) / 1024**2,  # MB
        }

    def get_tensor_info(self, collect_tensors=False):  # noqa: C901
        """Get information about all live tensors on the GPU.

        Args:
            collect_tensors: If True, store tensor objects for reference tracking
        """
        if not self.track_tensors:
            return {}

        # First, force garbage collection to get accurate count
        gc.collect()
        torch.cuda.synchronize(self.device)

        tensor_info = defaultdict(lambda: {"count": 0, "total_size_mb": 0.0, "shapes": defaultdict(int)})

        # Optionally collect actual tensor objects for reference tracking
        collected_tensors = defaultdict(list) if collect_tensors else None

        # Get device index for comparison
        device_idx = self.device if isinstance(self.device, int) else self.device.index

        # Use gc to find all tensor objects
        tensor_count = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    # Check if tensor is on our device
                    obj_device = obj.device.index if hasattr(obj.device, "index") else obj.device
                    if obj_device == device_idx:
                        dtype_str = str(obj.dtype)
                        size_mb = obj.element_size() * obj.nelement() / 1024**2

                        # Convert shape to string to handle SymInt (from torch.compile)
                        try:
                            # Try to convert to tuple first (for normal tensors)
                            shape_key = tuple(int(s) for s in obj.shape)
                        except (TypeError, ValueError):
                            # If shape contains SymInt or other non-hashable types, use string
                            shape_key = str(tuple(obj.shape))

                        tensor_info[dtype_str]["count"] += 1  # type: ignore
                        tensor_info[dtype_str]["total_size_mb"] += size_mb  # type: ignore
                        tensor_info[dtype_str]["shapes"][shape_key] += 1  # type: ignore
                        tensor_count += 1

                        # Collect tensor reference if requested
                        if collect_tensors:
                            collected_tensors[dtype_str].append((obj, shape_key, size_mb))  # type: ignore
            except Exception as _e:
                logger.exception("Get Tensor Info Error")
                # Skip objects that can't be accessed
                continue

        # If we found no tensors but memory is allocated, log a warning
        if tensor_count == 0:
            allocated_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            if allocated_mb > 100:  # If more than 100MB allocated
                logger.warning(
                    f"[TensorTracking] gc.get_objects() found no tensors, but {allocated_mb:.2f} MB is allocated. "
                    "This may indicate the tensors are not tracked by Python GC (e.g., in C++ extensions)."
                )

        if collect_tensors:
            return dict(tensor_info), dict(collected_tensors)
        return dict(tensor_info)

    def track_step(self, step: int, force: bool = False):
        """
        Track memory usage at this step.

        Args:
            step: Current training step
            force: Force check even if not at interval
        """
        self.step_count = step

        # Check if we should record this step
        if not force and step % self.check_interval != 0:
            return

        stats = self.get_memory_stats()
        stats["step"] = step
        self.memory_history.append(stats)

        # Track tensor information
        if self.track_tensors:
            tensor_info = self.get_tensor_info()
            self.tensor_snapshots.append({"step": step, "tensors": tensor_info})

        # Set baseline if first check
        if self.baseline_memory is None:
            self.baseline_memory = stats["allocated"]
            logger.info(f"[MemoryMonitor] Baseline memory: {self.baseline_memory:.2f} MB")

        # Check for memory growth
        growth = stats["allocated"] - self.baseline_memory
        if growth > self.alert_threshold_mb:
            logger.warning(
                f"[MemoryAlert] Step {step}: Memory growth {growth:.2f} MB "
                f"(current: {stats['allocated']:.2f} MB, baseline: {self.baseline_memory:.2f} MB)"
            )
            self._log_memory_summary()
            self._log_tensor_summary()

        # Save snapshot if enabled
        if self.enable_snapshot and step % self.snapshot_interval == 0:
            self._save_memory_snapshot(step)

    def _get_referrer_info(self, obj, max_depth=2, current_depth=0):  # noqa: C901
        """Get information about what's referencing this object.

        Args:
            obj: Object to analyze
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            List of referrer descriptions
        """
        if current_depth >= max_depth:
            return []

        referrers = gc.get_referrers(obj)
        results = []

        for ref in referrers:
            try:
                # Skip frame objects and module dicts (too noisy)
                ref_type = type(ref).__name__
                if ref_type in ["frame", "cell", "function", "method", "module"]:
                    continue

                # Describe the referrer
                if isinstance(ref, dict):
                    # Try to find which key references our object
                    keys_referring = [k for k, v in ref.items() if v is obj]
                    if keys_referring:
                        results.append(f"{'  ' * current_depth}dict[{keys_referring[0]!r}]")
                    else:
                        results.append(f"{'  ' * current_depth}dict (value)")
                elif isinstance(ref, list):
                    try:
                        idx = ref.index(obj)
                        results.append(f"{'  ' * current_depth}list[{idx}]")
                    except ValueError:
                        results.append(f"{'  ' * current_depth}list (element)")
                elif isinstance(ref, tuple):
                    try:
                        idx = ref.index(obj)
                        results.append(f"{'  ' * current_depth}tuple[{idx}]")
                    except ValueError:
                        results.append(f"{'  ' * current_depth}tuple (element)")
                elif hasattr(ref, "__name__"):
                    results.append(f"{'  ' * current_depth}{ref_type}: {ref.__name__}")
                elif hasattr(ref, "__class__"):
                    results.append(f"{'  ' * current_depth}{ref_type} instance")
                else:
                    results.append(f"{'  ' * current_depth}{ref_type}")

                # Recursively check referrers (only for non-collection types)
                if ref_type not in ["dict", "list", "tuple"] and current_depth < max_depth - 1:
                    sub_refs = self._get_referrer_info(ref, max_depth, current_depth + 1)
                    results.extend(sub_refs)

            except Exception:
                continue

        return results

    def analyze_tensor_references(self, max_tensors_per_dtype=3):
        """Analyze and log reference chains for the largest tensors.

        Args:
            max_tensors_per_dtype: Number of largest tensors to analyze per dtype
        """
        logger.info("[TensorReferences] Analyzing tensor reference chains...")

        # Collect tensors
        result = self.get_tensor_info(collect_tensors=True)
        if isinstance(result, tuple):
            _tensor_info, collected_tensors = result
        else:
            logger.warning("[TensorReferences] Could not collect tensors for reference tracking")
            return

        # Analyze largest tensors for each dtype
        for dtype, tensors_list in collected_tensors.items():
            # Sort by size (descending)
            tensors_list.sort(key=lambda x: x[2], reverse=True)  # type: ignore

            logger.info(f"\n[TensorReferences] {dtype} - Top {max_tensors_per_dtype} largest tensors:")

            for idx, (tensor, shape, size_mb) in enumerate(tensors_list[:max_tensors_per_dtype]):  # type: ignore
                logger.info(f"  Tensor #{idx + 1}: shape={shape}, size={float(size_mb):.2f} MB")

                # Get reference chain
                referrers = self._get_referrer_info(tensor, max_depth=3)

                if referrers:
                    logger.info("    Referenced by:")
                    for ref_desc in referrers[:10]:  # Limit to top 10 referrers
                        logger.info(f"      {ref_desc}")
                else:
                    logger.info("    No Python referrers found (may be held by C++ code)")

    def analyze_shape_references(self, dtype_shape_list, max_samples=2):
        """Analyze reference chains for specific tensor shapes.

        Args:
            dtype_shape_list: List of (dtype, shape) tuples to analyze
            max_samples: Number of tensor samples to analyze per shape
        """
        if not dtype_shape_list:
            return

        # Collect tensors
        result = self.get_tensor_info(collect_tensors=True)
        if isinstance(result, tuple):
            _tensor_info, collected_tensors = result
        else:
            logger.warning("[ShapeReferences] Could not collect tensors for reference tracking")
            return

        logger.info("[ShapeReferences] Analyzing reference chains for changed shapes:")

        # Analyze specific shapes
        for dtype, target_shape in dtype_shape_list:
            if dtype not in collected_tensors:
                continue

            # Filter tensors matching the target shape
            matching_tensors = [
                (tensor, shape, size_mb)
                for tensor, shape, size_mb in collected_tensors[dtype]  # type: ignore
                if shape == target_shape
            ]

            if not matching_tensors:
                continue

            # Sort by size and take samples
            matching_tensors.sort(key=lambda x: x[2], reverse=True)
            samples = matching_tensors[:max_samples]

            logger.info(
                f"\n  {dtype} shape={target_shape} ({len(matching_tensors)} total, showing {len(samples)} samples):"
            )

            for idx, (tensor, _shape, size_mb) in enumerate(samples):
                logger.info(f"    Sample #{idx + 1}: size={float(size_mb):.2f} MB")

                # Get reference chain
                referrers = self._get_referrer_info(tensor, max_depth=3)

                if referrers:
                    logger.info("      Referenced by:")
                    for ref_desc in referrers[:8]:  # Limit to top 8 referrers
                        logger.info(f"        {ref_desc}")
                else:
                    logger.info("      No Python referrers found (may be held by C++ code)")

    def _log_memory_summary(self):
        """Log detailed memory summary."""
        stats = self.get_memory_stats()
        logger.info(
            f"[MemorySummary]\n"
            f"  Allocated: {stats['allocated']:.2f} MB\n"
            f"  Reserved:  {stats['reserved']:.2f} MB\n"
            f"  Peak Allocated: {stats['peak_allocated']:.2f} MB\n"
            f"  Peak Reserved:  {stats['peak_reserved']:.2f} MB"
        )

    def _log_tensor_summary(self):
        """Log summary of current tensors and changes since last check."""
        if not self.track_tensors:
            return

        tensor_info = self.get_tensor_info()
        if not tensor_info:
            logger.info("[TensorSummary] No tensors found")
            return

        logger.info("[TensorSummary] Current tensor allocations:")
        total_tensors = 0
        total_mb = 0.0

        for dtype, info in sorted(tensor_info.items(), key=lambda x: x[1]["total_size_mb"], reverse=True):  # type: ignore
            total_tensors += info["count"]
            total_mb += info["total_size_mb"]
            logger.info(f"  {dtype}: {info['count']} tensors, {float(info['total_size_mb']):.2f} MB")

            # Show top 5 shapes by count
            top_shapes = sorted(info["shapes"].items(), key=lambda x: x[1], reverse=True)[:5]
            for shape, count in top_shapes:
                logger.info(f"    Shape {shape}: {count} tensors")

        logger.info(f"  Total: {total_tensors} tensors, {float(total_mb):.2f} MB")

        # Compare with last check and show changes
        if self.last_tensor_info is not None:
            self._log_tensor_changes(tensor_info, self.last_tensor_info)

        # Update last tensor info
        self.last_tensor_info = tensor_info

    def _log_tensor_changes(self, current_info, last_info):
        """Log changes in tensor allocations since last check."""
        logger.info("[TensorChanges] Changes since last check:")

        # Calculate total changes
        current_total = sum(info["count"] for info in current_info.values())
        last_total = sum(info["count"] for info in last_info.values())
        total_count_diff = current_total - last_total

        current_total_mb = sum(info["total_size_mb"] for info in current_info.values())
        last_total_mb = sum(info["total_size_mb"] for info in last_info.values())
        total_mb_diff = current_total_mb - last_total_mb

        logger.info(f"  Overall: {total_count_diff:+d} tensors, {float(total_mb_diff):+.2f} MB")

        # Track changes by dtype and collect shapes with significant increases
        all_dtypes = set(list(current_info.keys()) + list(last_info.keys()))
        increased_shapes = []  # List of (dtype, shape, count_increase) for reference tracking

        for dtype in sorted(all_dtypes):
            current = current_info.get(dtype, {"count": 0, "total_size_mb": 0.0, "shapes": {}})
            last = last_info.get(dtype, {"count": 0, "total_size_mb": 0.0, "shapes": {}})

            count_diff = current["count"] - last["count"]
            mb_diff = current["total_size_mb"] - last["total_size_mb"]

            # Only log if there are changes
            if count_diff != 0 or abs(mb_diff) > 0.01:
                logger.info(f"  {dtype}: {count_diff:+d} tensors, {float(mb_diff):+.2f} MB")

                # Show shape changes for this dtype
                all_shapes = set(list(current["shapes"].keys()) + list(last["shapes"].keys()))

                # Sort shapes by the magnitude of change
                shape_changes = []
                for shape in all_shapes:
                    curr_count = current["shapes"].get(shape, 0)
                    last_count = last["shapes"].get(shape, 0)
                    shape_diff = curr_count - last_count
                    if shape_diff != 0:
                        shape_changes.append((shape, shape_diff, curr_count))
                        # Collect shapes with increases for reference tracking
                        if shape_diff > 0 and shape_diff >= 2:  # Only track significant increases (>=2 tensors)
                            increased_shapes.append((dtype, shape, shape_diff))

                # Show top 5 shapes with biggest changes
                shape_changes.sort(key=lambda x: abs(x[1]), reverse=True)
                for shape, diff, curr_count in shape_changes[:5]:
                    logger.info(f"    Shape {shape}: {diff:+d} tensors (now: {curr_count})")

        # If reference tracking is enabled and we have significant increases, analyze them
        if self.track_references and increased_shapes:
            # Sort by increase magnitude and take top 5
            increased_shapes.sort(key=lambda x: x[2], reverse=True)
            top_increased = [(dtype, shape) for dtype, shape, _ in increased_shapes[:5]]

            # Only analyze if total change is significant (e.g., >10 tensors or >50MB)
            if abs(total_count_diff) > 10 or abs(total_mb_diff) > 50.0:
                logger.info(f"\n[TensorChanges] Analyzing references for top {len(top_increased)} increased shapes...")
                self.analyze_shape_references(top_increased, max_samples=2)

    def _save_memory_snapshot(self, step: int):
        """Save memory snapshot for debugging."""
        try:
            snapshot_path = f"memory_snapshot_step_{step}.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_path)
            logger.info(f"[MemorySnapshot] Saved to {snapshot_path}")
        except Exception as e:
            logger.warning(f"[MemorySnapshot] Failed to save: {e}")

    def check_for_leaks(self):
        """Analyze memory history to detect leaks and report tensor growth."""
        if len(self.memory_history) < 5:
            return False

        # Check if memory is consistently growing
        recent_stats = self.memory_history[-5:]
        memory_values = [s["allocated"] for s in recent_stats]

        # Simple linear regression to detect trend
        n = len(memory_values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(memory_values) / n

        numerator = sum((x[i] - x_mean) * (memory_values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
            if slope > 10:  # Growing more than 10MB per check
                logger.warning(f"[MemoryLeak] Detected consistent memory growth: {slope:.2f} MB per check")

                # Analyze tensor growth if available
                if self.track_tensors and len(self.tensor_snapshots) >= 2:
                    self._analyze_tensor_growth()

                # Analyze reference chains for the largest tensors if enabled
                if self.track_references:
                    logger.info("[MemoryLeak] Analyzing reference chains for largest tensors...")
                    self.analyze_tensor_references(max_tensors_per_dtype=3)

                return True

        return False

    def _analyze_tensor_growth(self):
        """Analyze which tensors are growing over time."""
        if len(self.tensor_snapshots) < 2:
            return

        # Compare last two snapshots
        prev_snapshot = self.tensor_snapshots[-2]
        curr_snapshot = self.tensor_snapshots[-1]

        logger.info(f"[TensorGrowth] Analyzing growth from step {prev_snapshot['step']} to {curr_snapshot['step']}")

        for dtype in set(list(prev_snapshot["tensors"].keys()) + list(curr_snapshot["tensors"].keys())):
            prev_info = prev_snapshot["tensors"].get(dtype, {"count": 0, "total_size_mb": 0.0, "shapes": {}})
            curr_info = curr_snapshot["tensors"].get(dtype, {"count": 0, "total_size_mb": 0.0, "shapes": {}})

            count_diff = curr_info["count"] - prev_info["count"]
            size_diff = curr_info["total_size_mb"] - prev_info["total_size_mb"]

            if count_diff > 0 or size_diff > 1.0:  # Threshold: 1MB growth or new tensors
                logger.warning(f"  {dtype}: +{count_diff} tensors, +{float(size_diff):.2f} MB")

                # Show which shapes increased
                prev_shapes = prev_info["shapes"]
                curr_shapes = curr_info["shapes"]

                for shape in set(list(prev_shapes.keys()) + list(curr_shapes.keys())):
                    prev_count = prev_shapes.get(shape, 0)
                    curr_count = curr_shapes.get(shape, 0)
                    if curr_count > prev_count:
                        logger.warning(f"    Shape {shape}: +{curr_count - prev_count} tensors")

    def cleanup(self):
        """Clean up memory tracking."""
        if self.enable_snapshot:
            torch.cuda.memory._record_memory_history(enabled=False)


class TensorCleaner:
    """Helper class to ensure tensors are properly cleaned up."""

    @staticmethod
    def detach_and_clone(tensor: torch.Tensor) -> torch.Tensor:
        """Safely detach and clone a tensor, breaking the computation graph."""
        return tensor.detach().clone()

    @staticmethod
    def extract_scalar(tensor: torch.Tensor) -> float:
        """Extract scalar value and release the tensor."""
        value = tensor.detach().item()
        del tensor
        return value

    @staticmethod
    def cleanup_dict(d: dict):
        """Clean up tensors in a dictionary."""
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor):
                d[k] = v.detach().item() if v.numel() == 1 else v.detach().cpu()
            elif isinstance(v, dict):
                TensorCleaner.cleanup_dict(v)


@dataclass
class MemoryMonitorArgs:
    """Configuration for memory monitoring."""

    enable: bool = False
    check_interval: int = 10
    snapshot_interval: int = 100
    alert_threshold_mb: float = 1000.0
    enable_snapshot: bool = False  # Disabled by default due to overhead
    track_tensors: bool = True  # Track tensor sizes and shapes
    track_references: bool = True  # Track tensor references on leak detection
    aggressive_cleanup: bool = False  # Enable aggressive memory cleanup


__all__ = ["MemoryLeakDetector", "MemoryMonitorArgs", "TensorCleaner"]
