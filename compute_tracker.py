import json
import time
import platform
import os
from typing import Optional, Dict, Any, List

import torch

try:
    import pynvml

    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


class ComputeTracker:
    """Tracks compute cost metrics (time, memory, throughput, energy) during training.

    Lifecycle:
        tracker = ComputeTracker(log_dir, device)
        tracker.start_training()
        for epoch in ...:
            tracker.start_phase(epoch, "train", num_samples, num_batches)
            ... run training epoch ...
            tracker.end_phase(epoch, "train")
        tracker.end_training()
        tracker.save()   # writes compute_metrics.json
        tracker.close()
    """

    def __init__(self, log_dir: str, device: torch.device):
        self.log_dir = log_dir
        self.device = device
        self._hardware_info = self._collect_hardware_info()
        self._epoch_metrics: List[Dict] = []
        self._training_summary: Dict[str, Any] = {}
        self._phase_timers: Dict[str, Dict] = {}
        self._training_start_time: Optional[float] = None
        self._total_energy_joules: float = 0.0

        self._nvml_handle = None
        if _HAS_PYNVML and self.device.type == "cuda":
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                    self.device.index or 0
                )
            except pynvml.NVMLError:
                self._nvml_handle = None

    def _collect_hardware_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "device_type": self.device.type,
        }

        if self.device.type == "cuda":
            idx = self.device.index or 0
            props = torch.cuda.get_device_properties(idx)
            info["gpu"] = {
                "name": torch.cuda.get_device_name(idx),
                "count": torch.cuda.device_count(),
                "memory_total_mb": round(props.total_mem / 1e6, 2),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }

        info["cpu"] = {
            "model": platform.processor() or platform.machine(),
            "count": os.cpu_count(),
        }

        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            info["ram_total_gb"] = round(page_size * page_count / 1e9, 2)
        except (ValueError, OSError, AttributeError):
            pass

        info["python_version"] = platform.python_version()
        info["pytorch_version"] = torch.__version__
        info["os"] = platform.platform()

        return info

    def start_training(self) -> None:
        self._training_start_time = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def end_training(self) -> None:
        wall_clock_s = time.monotonic() - self._training_start_time
        summary: Dict[str, Any] = {
            "total_wall_clock_s": round(wall_clock_s, 2),
            "total_epochs": len(self._epoch_metrics),
        }

        if self.device.type == "cuda":
            summary["total_gpu_hours"] = round(wall_clock_s / 3600.0, 4)
            summary["peak_memory_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 1e6, 2
            )
            summary["peak_memory_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / 1e6, 2
            )

        if self._total_energy_joules > 0:
            summary["total_energy_joules"] = round(self._total_energy_joules, 2)
            summary["total_energy_kwh"] = round(self._total_energy_joules / 3.6e6, 6)

        self._training_summary = summary

    def start_phase(
        self, epoch: int, phase: str, num_samples: int, num_batches: int
    ) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        timer: Dict[str, Any] = {
            "start": time.monotonic(),
            "num_samples": num_samples,
            "num_batches": num_batches,
        }

        if self._nvml_handle is not None:
            try:
                timer["power_start_mw"] = pynvml.nvmlDeviceGetPowerUsage(
                    self._nvml_handle
                )
            except pynvml.NVMLError:
                pass

        self._phase_timers[phase] = timer

    def end_phase(self, epoch: int, phase: str) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        timer = self._phase_timers.pop(phase)
        elapsed = time.monotonic() - timer["start"]
        num_samples = timer["num_samples"]
        num_batches = timer["num_batches"]

        phase_data: Dict[str, Any] = {
            "wall_clock_s": round(elapsed, 3),
            "samples_per_second": round(num_samples / elapsed, 2) if elapsed > 0 else 0,
            "batches_per_second": round(num_batches / elapsed, 2) if elapsed > 0 else 0,
        }

        if self.device.type == "cuda":
            phase_data["peak_memory_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 1e6, 2
            )
            phase_data["peak_memory_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / 1e6, 2
            )
            torch.cuda.reset_peak_memory_stats()

        if self._nvml_handle is not None and "power_start_mw" in timer:
            try:
                power_end_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                avg_power_w = (timer["power_start_mw"] + power_end_mw) / 2.0 / 1000.0
                energy_j = avg_power_w * elapsed
                phase_data["avg_power_w"] = round(avg_power_w, 2)
                phase_data["energy_joules"] = round(energy_j, 2)
                self._total_energy_joules += energy_j
            except pynvml.NVMLError:
                pass

        epoch_entry = self._get_or_create_epoch(epoch)
        epoch_entry["phases"][phase] = phase_data
        epoch_entry["total_wall_clock_s"] = round(
            sum(p["wall_clock_s"] for p in epoch_entry["phases"].values()), 3
        )

    def _get_or_create_epoch(self, epoch: int) -> Dict:
        for entry in self._epoch_metrics:
            if entry["epoch"] == epoch:
                return entry
        entry: Dict[str, Any] = {
            "epoch": epoch,
            "phases": {},
            "total_wall_clock_s": 0.0,
        }
        self._epoch_metrics.append(entry)
        return entry

    def save(self) -> None:
        output = {
            "hardware": self._hardware_info,
            "epochs": self._epoch_metrics,
            "training_summary": self._training_summary,
        }
        path = os.path.join(self.log_dir, "compute_metrics.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    def close(self) -> None:
        if self._nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
