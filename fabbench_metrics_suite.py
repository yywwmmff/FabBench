#!/usr/bin/env python3
"""
FabBench metric suite for SEM-domain process-conditioned lithography models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.metric import Metric

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception as exc:  # pragma: no cover
    skimage_ssim = None
    _SSIM_IMPORT_ERROR = exc
else:
    _SSIM_IMPORT_ERROR = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    _FID_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    FrechetInceptionDistance = None
    _FID_IMPORT_ERROR = exc

try:
    from torchmetrics.image.kid import KernelInceptionDistance
    _KID_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    KernelInceptionDistance = None
    _KID_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fabbench_sam_loader import LoRA_SAM_Lightning, get_args


def get_boundary_region(mask: np.ndarray, d: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask, dtype=bool)
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, -1, 1, 1)
    dist = distance_transform_edt(1 - contour_mask)
    d_mask = dist <= d
    return (mask > 0) & d_mask


class FabBenchMetrics(Metric):
    def __init__(
        self,
        sam_path,
        save_dir,
        boundary_d=5,
        sam_type="vit_b",
        epe_spacing=20,
        min_edge_length=40,
        corner_rounding=40,
        max_epe=60,
        px_size: float = 1.0,
        kid_subset_size: int = 50,
        kid_subsets: int = 10,
        compute_image_metrics: bool = True,
    ):
        del sam_type
        super().__init__()

        for prefix in ["top", "bottom"]:
            self.add_state(f"{prefix}_signed_epe", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{prefix}_count", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"{prefix}_mask_dice", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{prefix}_boundary_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{prefix}_iou_count", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"{prefix}_cd_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{prefix}_cd_signed_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{prefix}_cd_count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("defect_status_tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("defect_status_fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("defect_status_tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("defect_status_fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("defect_status_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("defect_type_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("defect_type_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        self.boundary_d = boundary_d
        self.EPE_SPACING = int(epe_spacing)
        self.MIN_EDGE_LENGTH = int(min_edge_length)
        self.CORNER_ROUNDING = int(corner_rounding)
        self.MAX_EPE = int(max_epe)
        self.sam_path = sam_path
        self.save_dir = os.path.join(os.environ["PROJECT_ROOT"], str(save_dir))
        os.makedirs(self.save_dir, exist_ok=True)
        self.failure_case_dir = os.path.join(self.save_dir, "failure_cases")
        self.training = False
        self.trainer = None
        self.seg_model = None
        self.px_size = float(px_size)
        self.compute_image_metrics = bool(compute_image_metrics)
        self._warned_fid = False
        self._warned_kid = False
        self.failure_case_limit = 0
        self.failure_case_counts = {}
        self.debug_cd_batches = 3
        self.psnr_metric = PeakSignalNoiseRatio(data_range=255.0)
        self.kid_subset_size = max(2, int(kid_subset_size))
        self.kid_subsets = int(kid_subsets)
        self.fid_metric = None
        self.kid_metric = None
        self.sample_metrics_path = os.path.join(self.save_dir, "sample_metrics_rank0.jsonl")

    def _empty_failure_case_counts(self) -> Dict[str, int]:
        return {}

    def set_output_dir(self, save_dir: str, reset_failure_cases: bool = True) -> None:
        self.save_dir = str(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        rank = 0
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "global_rank"):
            rank = int(trainer.global_rank)
        self.sample_metrics_path = os.path.join(self.save_dir, f"sample_metrics_rank{rank}.jsonl")
        Path(self.sample_metrics_path).write_text("", encoding="utf-8")
        self.failure_case_dir = os.path.join(self.save_dir, "failure_cases")
        if reset_failure_cases:
            self.failure_case_counts = self._empty_failure_case_counts()

    def _ensure_distribution_metrics(self, device: torch.device) -> None:
        if self.fid_metric is None and FrechetInceptionDistance is not None:
            self.fid_metric = FrechetInceptionDistance(normalize=False).to(device)
        if self.kid_metric is None and KernelInceptionDistance is not None:
            self.kid_metric = KernelInceptionDistance(
                subset_size=self.kid_subset_size,
                subsets=self.kid_subsets,
                normalize=False,
            ).to(device)

    @staticmethod
    def _ensure_nchw(tensor: Tensor) -> Tensor:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected tensor with 4 dims [B,C,H,W], got {tensor.shape}")
        return tensor

    @staticmethod
    def _ensure_three_channel_uint8(tensor: Tensor) -> Tensor:
        tensor = FabBenchMetrics._ensure_nchw(tensor)
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] >= 3:
            tensor = tensor[:, :3, :, :]
        else:
            repeat_factor = (3 + tensor.shape[1] - 1) // tensor.shape[1]
            tensor = tensor.repeat(1, repeat_factor, 1, 1)[:, :3, :, :]
        return torch.clamp(tensor, 0, 255).round().to(torch.uint8)

    @staticmethod
    def _tensor_to_hwc_uint8(tensor: Tensor) -> np.ndarray:
        tensor = FabBenchMetrics._ensure_nchw(tensor.detach())[0]
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] > 3:
            tensor = tensor[:3]
        return torch.clamp(tensor, 0, 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().float().mean().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _json_safe_value(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 1:
                value = value.detach().cpu().item()
            else:
                value = value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (Path,)):
            value = str(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def _append_sample_metric_record(self, record: Dict[str, object]) -> None:
        serializable = {key: self._json_safe_value(value) for key, value in record.items()}
        with open(self.sample_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    def _sample_psnr(self, pred: Tensor, target: Tensor) -> float:
        pred_f = pred.detach().float()
        target_f = target.detach().float()
        mse = F.mse_loss(pred_f, target_f, reduction="mean").item()
        if mse <= 1e-12:
            return 99.0
        return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))

    def _collect_numeric_values(self, obj) -> List[float]:
        values: List[float] = []
        if isinstance(obj, dict):
            for value in obj.values():
                values.extend(self._collect_numeric_values(value))
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                values.extend(self._collect_numeric_values(value))
        else:
            try:
                value = float(obj)
            except (TypeError, ValueError):
                return values
            if np.isfinite(value):
                values.append(value)
        return values

    def _mean_numeric_value(self, obj) -> Optional[float]:
        values = self._collect_numeric_values(obj)
        if not values:
            return None
        return float(np.mean(values))

    def _gt_pred_epe_bias(
        self,
        offline_label: Optional[Dict],
        pred_top_signed_epe: Optional[float],
        pred_bottom_signed_epe: Optional[float],
    ) -> Optional[float]:
        if not isinstance(offline_label, dict):
            return None
        gt_top_signed_epe = self._mean_numeric_value(offline_label.get("epe_top", {}))
        gt_bottom_signed_epe = self._mean_numeric_value(offline_label.get("epe_bottom", {}))
        diffs = []
        if gt_top_signed_epe is not None and pred_top_signed_epe is not None:
            diffs.append(abs(float(pred_top_signed_epe) - float(gt_top_signed_epe)))
        if gt_bottom_signed_epe is not None and pred_bottom_signed_epe is not None:
            diffs.append(abs(float(pred_bottom_signed_epe) - float(gt_bottom_signed_epe)))
        if not diffs:
            return None
        return float(np.mean(diffs))

    @torch.no_grad()
    def update(self, images: torch.Tensor, targets_gt: torch.Tensor, mask, batch_idx, gauge_info=None):
        def _is_trivial_mask(img_arr, threshold=0.95):
            total_pixels = img_arr.size
            ratio_black = np.count_nonzero(img_arr == 0) / total_pixels
            ratio_white = np.count_nonzero(img_arr == 255) / total_pixels
            return (ratio_black > threshold) or (ratio_white > threshold)

        images = self._ensure_nchw(images.detach())
        targets_gt = self._ensure_nchw(targets_gt.detach())
        assert images.shape == targets_gt.shape, f"Input shape mismatch: images {images.shape} vs targets {targets_gt.shape}"

        if self.compute_image_metrics:
            images_for_quality = images.float()
            targets_for_quality = targets_gt.float()
            self.psnr_metric.update(images_for_quality, targets_for_quality)
            self._ensure_distribution_metrics(images.device)
            fake_uint8 = self._ensure_three_channel_uint8(images_for_quality)
            real_uint8 = self._ensure_three_channel_uint8(targets_for_quality)
            if self.fid_metric is not None:
                self.fid_metric.update(real_uint8, real=True)
                self.fid_metric.update(fake_uint8, real=False)
            elif not self._warned_fid and _FID_IMPORT_ERROR is not None:
                print(f"[fabbench_metrics] FID unavailable: {_FID_IMPORT_ERROR}", flush=True)
                self._warned_fid = True
            if self.kid_metric is not None:
                self.kid_metric.update(real_uint8, real=True)
                self.kid_metric.update(fake_uint8, real=False)
            elif not self._warned_kid and _KID_IMPORT_ERROR is not None:
                print(f"[fabbench_metrics] KID unavailable: {_KID_IMPORT_ERROR}", flush=True)
                self._warned_kid = True
            self.num_samples += images.shape[0]

        if torch.is_tensor(mask):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.array(mask)
        if mask_np.shape[1] == 3:
            mask_np = mask_np[:, 0, :, :]
        elif mask_np.shape[-1] == 3:
            mask_np = mask_np[:, :, :, 0]

        batch_size = images.shape[0]
        combined_batch = torch.cat([images, targets_gt], dim=0)
        all_bin_top, all_bin_bottom = self.process_batch_with_mask_prompt(combined_batch)

        pred_top_np = (all_bin_top[:batch_size].detach().cpu().numpy() * 255).astype(np.uint8)
        target_top_np = (all_bin_top[batch_size:].detach().cpu().numpy() * 255).astype(np.uint8)
        pred_bottom_np = (all_bin_bottom[:batch_size].detach().cpu().numpy() * 255).astype(np.uint8)
        target_bottom_np = (all_bin_bottom[batch_size:].detach().cpu().numpy() * 255).astype(np.uint8)
        if gauge_info is None:
            gauge_info_list = [None] * batch_size
        elif isinstance(gauge_info, list):
            gauge_info_list = gauge_info
        else:
            gauge_info_list = [gauge_info] * batch_size

        for i in range(batch_size):
            sample_gauge_info = gauge_info_list[i] if i < len(gauge_info_list) else None

            if _is_trivial_mask(pred_bottom_np[i, 0]) and _is_trivial_mask(target_bottom_np[i, 0]):
                m_iou_b, b_iou_b = 1.0, 1.0
                res_bottom = (0.0, 0.0, 0.0)
            else:
                m_iou_b, b_iou_b = self._calculate_all_iou(pred_bottom_np[i, 0], target_bottom_np[i, 0])
                res_bottom = self._calculate_epe(pred_bottom_np[i, 0], target_bottom_np[i, 0], mask_np[i], "bottom", batch_idx, i)
            self._update_iou_states("bottom", m_iou_b, b_iou_b)
            self._update_state_values("bottom", res_bottom)
            self._update_cd_from_gauge_info("bottom", pred_bottom_np[i, 0], mask_np[i], sample_gauge_info)

            if _is_trivial_mask(pred_top_np[i, 0]) and _is_trivial_mask(target_top_np[i, 0]):
                m_iou_t, b_iou_t = 1.0, 1.0
                res_top = (0.0, 0.0, 0.0)
            else:
                m_iou_t, b_iou_t = self._calculate_all_iou(pred_top_np[i, 0], target_top_np[i, 0])
                res_top = self._calculate_epe(pred_top_np[i, 0], target_top_np[i, 0], mask_np[i], "top", batch_idx, i)
            self._update_iou_states("top", m_iou_t, b_iou_t)
            self._update_state_values("top", res_top)
            self._update_cd_from_gauge_info("top", pred_top_np[i, 0], mask_np[i], sample_gauge_info)

            gt_type = self.get_defect_type(target_top_np[i, 0], target_bottom_np[i, 0], mask_np[i])
            pred_type = self.get_defect_type(pred_top_np[i, 0], pred_bottom_np[i, 0], mask_np[i])
            self._update_defect_metrics(
                pred_top_np[i, 0],
                pred_bottom_np[i, 0],
                target_top_np[i, 0],
                target_bottom_np[i, 0],
                mask_np[i],
                sample_gauge_info,
            )

            gt_types = set([gt_type]) if gt_type is not None else set()
            pred_status = pred_type is not None
            gt_status = gt_type is not None
            top_cd_err = None
            bottom_cd_err = None

            if isinstance(sample_gauge_info, dict):
                top_gt_cd_values = self._extract_cd_values_from_gauge_info(sample_gauge_info, "top")
                bottom_gt_cd_values = self._extract_cd_values_from_gauge_info(sample_gauge_info, "bottom")
                top_gauges = sample_gauge_info.get("gauges", [])
                gauge_map = {str(g["gauge_id"]): g for g in top_gauges if isinstance(g, dict) and "gauge_id" in g}

                top_errs = []
                for gauge_id, gt_cd in top_gt_cd_values.items():
                    gauge = gauge_map.get(gauge_id)
                    if gauge is None:
                        continue
                    pred_cd_px = self._measure_cd_on_gauge((pred_top_np[i, 0] > 0).astype(np.uint8), gauge)
                    if pred_cd_px is not None:
                        top_errs.append(abs(float(pred_cd_px) * self.px_size - gt_cd))
                if top_errs:
                    top_cd_err = float(np.mean(top_errs))

                bottom_errs = []
                for gauge_id, gt_cd in bottom_gt_cd_values.items():
                    gauge = gauge_map.get(gauge_id)
                    if gauge is None:
                        continue
                    pred_cd_px = self._measure_cd_on_gauge((pred_bottom_np[i, 0] > 0).astype(np.uint8), gauge)
                    if pred_cd_px is not None:
                        bottom_errs.append(abs(float(pred_cd_px) * self.px_size - gt_cd))
                if bottom_errs:
                    bottom_cd_err = float(np.mean(bottom_errs))

            sample_record = {
                "batch_idx": int(batch_idx),
                "sample_idx": int(i),
                "sample_id": f"batch_{batch_idx:04d}_sample_{i:04d}",
                "structure_defect_gt": bool(gt_status),
                "pred_defect_type": pred_type,
                "gt_defect_types": sorted(gt_types) if gt_types else [],
                "pred_defect_status": bool(pred_status),
                "gt_defect_status": bool(gt_status),
                "sample_psnr": self._sample_psnr(images[i], targets_gt[i]),
                "top_boundary_iou": self._safe_float(b_iou_t),
                "top_mask_dice": self._safe_float((2 * m_iou_t) / (m_iou_t + 1)),
                "top_signed_epe": self._safe_float(res_top[1] if res_top is not None else None),
                "top_cd_abs_error": self._safe_float(top_cd_err),
                "bottom_boundary_iou": self._safe_float(b_iou_b),
                "bottom_mask_dice": self._safe_float((2 * m_iou_b) / (m_iou_b + 1)),
                "bottom_signed_epe": self._safe_float(res_bottom[1] if res_bottom is not None else None),
                "bottom_cd_abs_error": self._safe_float(bottom_cd_err),
            }
            self._append_sample_metric_record(sample_record)

    def _calculate_all_iou(self, pred, target, smooth=1e-5):
        intersection = (pred & target).sum()
        union = (pred | target).sum()
        mask_iou = (intersection + smooth) / (union + smooth)
        b_pred = get_boundary_region(pred, self.boundary_d)
        b_target = get_boundary_region(target, self.boundary_d)
        b_intersection = (b_pred & b_target).sum()
        b_union = (b_pred | b_target).sum()
        boundary_iou = (b_intersection + smooth) / (b_union + smooth)
        return mask_iou, boundary_iou

    def _update_iou_states(self, prefix, m_iou, b_iou):
        mask_dice = 2 * m_iou / (m_iou + 1)
        getattr(self, f"{prefix}_mask_dice").add_(torch.tensor(mask_dice, device=self.device))
        getattr(self, f"{prefix}_boundary_iou").add_(torch.tensor(b_iou, device=self.device))
        getattr(self, f"{prefix}_iou_count").add_(1)

    def _update_state_values(self, prefix, res):
        if res is None:
            return
        _, mean_signed = res[:2]
        if mean_signed is not None:
            getattr(self, f"{prefix}_signed_epe").add_(torch.tensor(mean_signed, device=self.device))
            getattr(self, f"{prefix}_count").add_(1)

    def _normalize_defect_type(self, defect_type: str) -> str:
        value = str(defect_type).strip().lower().replace("-", "_").replace(" ", "_")
        alias_map = {
            "missing_pattern": "missing",
            "loss": "missing",
            "open": "break",
            "open_circuit": "break",
            "broken": "break",
            "breakage": "break",
            "short": "bridge",
            "short_circuit": "bridge",
            "merge": "bridge",
            "bridging": "bridge",
            "extra": "extra",
            "protrusion": "extra",
            "bulge": "extra",
        }
        return alias_map.get(value, value)

    def _extract_cd_values_from_gauge_info(self, gauge_info: Optional[Dict], prefix: str) -> Dict[str, float]:
        if not isinstance(gauge_info, dict):
            return {}
        cd_values = gauge_info.get("cd_values", {})
        if not isinstance(cd_values, dict):
            return {}
        layer_values = cd_values.get(prefix, {})
        if not isinstance(layer_values, dict) or len(layer_values) == 0:
            return {}
        return {str(k): float(v) for k, v in layer_values.items()}

    def _measure_cd_at_point(self, binary_img_u8, pt_xy, scan_axis: str):
        rows, cols = binary_img_u8.shape
        x0, y0 = int(pt_xy[0]), int(pt_xy[1])
        if not (0 <= x0 < cols and 0 <= y0 < rows):
            return None
        cur = binary_img_u8[y0, x0]
        is_expansion = cur > 127
        dirs = [(1, 0), (-1, 0)] if scan_axis == "horizontal" else [(0, 1), (0, -1)]
        dists = []
        for dx, dy in dirs:
            cx, cy, dist = x0, y0, 0
            found = False
            while 0 <= cx < cols and 0 <= cy < rows and dist < self.MAX_EPE:
                pv = binary_img_u8[cy, cx]
                if (is_expansion and pv < 127) or ((not is_expansion) and pv > 127):
                    found = True
                    break
                cx += dx
                cy += dy
                dist += 1
            if found:
                dists.append(dist)
        if len(dists) != 2:
            return None
        return float(dists[0] + dists[1])

    def _extract_metric_contours(self, mask01: np.ndarray) -> List[np.ndarray]:
        img = (mask01 > 0).astype(np.uint8) * 255
        padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = []
        for contour in contours:
            if contour.shape[0] < 2:
                continue
            pts = contour[:, 0, :].astype(np.float32)
            pts[:, 0] -= 1.0
            pts[:, 1] -= 1.0
            out.append(pts)
        return out

    def _line_segment_intersection_t(self, origin: np.ndarray, direction: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> Optional[float]:
        seg = p1 - p0
        denom = float(direction[0] * seg[1] - direction[1] * seg[0])
        if abs(denom) <= 1e-8:
            return None
        delta = p0 - origin
        t = float((delta[0] * seg[1] - delta[1] * seg[0]) / denom)
        u = float((delta[0] * direction[1] - delta[1] * direction[0]) / denom)
        if -1e-6 <= u <= 1.0 + 1e-6:
            return t
        return None

    def _contour_intersections_along_line(self, contours: List[np.ndarray], origin: np.ndarray, direction: np.ndarray) -> List[float]:
        ts = []
        for pts in contours:
            n = pts.shape[0]
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                t = self._line_segment_intersection_t(origin, direction, p0, p1)
                if t is not None and np.isfinite(t):
                    ts.append(float(t))
        if not ts:
            return []
        ts.sort()
        dedup = []
        for t in ts:
            if not dedup or abs(t - dedup[-1]) > 0.25:
                dedup.append(t)
        return dedup

    def _measure_cd_on_gauge(self, binary_mask: np.ndarray, gauge: Dict) -> Optional[float]:
        contours = self._extract_metric_contours((binary_mask > 0).astype(np.uint8))
        if not contours:
            return None
        direction = np.array([float(gauge["normal_x"]), float(gauge["normal_y"])], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            return None
        direction /= norm
        tangent = np.array([float(gauge["tangent_x"]), float(gauge["tangent_y"])], dtype=np.float32)
        values = []
        for offset in np.linspace(
            -float(gauge["width"]) / 2.0,
            float(gauge["width"]) / 2.0,
            max(1, int(gauge["sample_count"])),
            dtype=np.float32,
        ):
            origin = np.array([float(gauge["center_x"]), float(gauge["center_y"])], dtype=np.float32) + tangent * float(offset)
            ts = self._contour_intersections_along_line(contours, origin, direction)
            if len(ts) < 2:
                continue
            neg = [t for t in ts if t <= 0.0]
            pos = [t for t in ts if t >= 0.0]
            if not neg or not pos:
                continue
            width = float(min(pos) - max(neg))
            if width > 0:
                values.append(width)
        if not values:
            return None
        return float(np.mean(values))

    def _update_cd_from_gauge_info(self, prefix: str, pred_bin, mask, gauge_info: Optional[Dict]) -> None:
        del mask
        if not isinstance(gauge_info, dict):
            return
        gt_cd_values = self._extract_cd_values_from_gauge_info(gauge_info, prefix)
        gauges = gauge_info.get("gauges", [])
        if not gt_cd_values or not isinstance(gauges, list) or len(gauges) == 0:
            return
        gauge_map = {str(g["gauge_id"]): g for g in gauges if isinstance(g, dict) and "gauge_id" in g}
        abs_errors = []
        bias_errors = []
        for gauge_id, gt_cd in gt_cd_values.items():
            gauge = gauge_map.get(gauge_id)
            if gauge is None:
                continue
            pred_cd_px = self._measure_cd_on_gauge((pred_bin > 0).astype(np.uint8), gauge)
            if pred_cd_px is None:
                continue
            pred_cd = float(pred_cd_px) * self.px_size
            abs_error = abs(float(pred_cd - gt_cd))
            bias_errors.append(abs_error)
            abs_errors.append(abs_error)
        if not abs_errors or bool(gauge_info.get("structure_defect", False)):
            return
        mean_abs_error = float(np.mean(abs_errors))
        mean_bias_error = float(np.mean(bias_errors))
        getattr(self, f"{prefix}_cd_abs_error").add_(torch.tensor(mean_abs_error, device=self.device))
        getattr(self, f"{prefix}_cd_signed_error").add_(torch.tensor(mean_bias_error, device=self.device))
        getattr(self, f"{prefix}_cd_count").add_(1)

    def _ensure_single_channel_mask(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return (arr > 0).astype(np.uint8)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                arr = arr[0]
            elif arr.shape[-1] in (1, 3):
                arr = arr[..., 0]
            else:
                raise ValueError(f"Unsupported mask shape for single-channel conversion: {arr.shape}")
            return (arr > 0).astype(np.uint8)
        raise ValueError(f"Unsupported mask ndim for single-channel conversion: {arr.shape}")

    def _has_large_top_component_like_full_image(self, top01: np.ndarray, area_ratio_th: float = 0.90) -> bool:
        top_bin = (top01 > 0).astype(np.uint8)
        n_cc, _, stats, _ = cv2.connectedComponentsWithStats(top_bin, connectivity=8)
        if n_cc <= 1:
            return False
        image_area = float(top_bin.shape[0] * top_bin.shape[1])
        areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, int(n_cc))), reverse=True)
        if not areas:
            return False
        largest = float(areas[0]) / max(1.0, image_area)
        if largest >= float(area_ratio_th):
            return True
        if len(areas) >= 2:
            second = float(areas[1]) / max(1.0, image_area)
            if second >= float(area_ratio_th):
                return True
        return False

    def _is_missing_by_top_bottom_similarity(self, top01: np.ndarray, down01: np.ndarray, diff_ratio_th: float = 0.02) -> bool:
        top = top01 > 0
        down = down01 > 0
        inter = int(np.count_nonzero(top & down))
        union = int(np.count_nonzero(top | down))
        gap_ratio = float((union - inter) / max(1, union))
        image_ratio = float(union / max(1, top.size))
        return gap_ratio < float(diff_ratio_th) and image_ratio >= 0.90

    def _has_bottom_without_top_overlap_and_with_mask(self, top01: np.ndarray, down01: np.ndarray, mask01: np.ndarray) -> bool:
        down_bin = self._ensure_single_channel_mask(down01)
        top_bin = self._ensure_single_channel_mask(top01)
        mask_bin = self._ensure_single_channel_mask(mask01)
        if int(np.count_nonzero(mask_bin)) > 0 and int(np.count_nonzero(top_bin)) == 0 and int(np.count_nonzero(down_bin)) == 0:
            return True
        n_down, down_labels = cv2.connectedComponents(down_bin, connectivity=8)
        for down_id in range(1, int(n_down)):
            region = down_labels == down_id
            if int(np.count_nonzero(region)) <= 0:
                continue
            if int(np.count_nonzero(top_bin[region])) == 0 and int(np.count_nonzero(mask_bin[region])) > 0:
                return True
        return False

    def _bridge_pinch(self, mask01: np.ndarray, top01: np.ndarray, down01: np.ndarray) -> Dict[str, bool]:
        mask_bin = self._ensure_single_channel_mask(mask01)
        top_bin = self._ensure_single_channel_mask(top01)
        down_bin = self._ensure_single_channel_mask(down01)
        n_mask, mask_labels, _, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        n_top, top_labels, _, _ = cv2.connectedComponentsWithStats(top_bin, connectivity=8)
        large_top_like_full_image = self._has_large_top_component_like_full_image(top01, area_ratio_th=0.90)

        bridge = False
        if not large_top_like_full_image:
            for top_id in range(1, int(n_top)):
                region = top_labels == top_id
                overlap_mask = np.unique(mask_labels[region])
                overlap_mask = overlap_mask[overlap_mask > 0]
                if overlap_mask.size >= 2:
                    bridge = True
                    break

        bottom_minus_top_bin = np.logical_and(down_bin > 0, top_bin == 0).astype(np.uint8)
        n_bt, bt_labels, _, _ = cv2.connectedComponentsWithStats(bottom_minus_top_bin, connectivity=8)
        scumming = False
        for bt_id in range(1, int(n_bt)):
            region = bt_labels == bt_id
            overlap_mask = np.unique(mask_labels[region])
            overlap_mask = overlap_mask[overlap_mask > 0]
            if overlap_mask.size >= 2:
                scumming = True
                break

        pinch = False
        for mask_id in range(1, int(n_mask)):
            region = mask_labels == mask_id
            overlap_top = np.unique(top_labels[region])
            overlap_top = overlap_top[overlap_top > 0]
            if overlap_top.size >= 2:
                pinch = True
                break

        return {
            "bridge": bool(bridge),
            "pinch": bool(pinch),
            "scumming": bool(scumming),
            "large_top_like_full_image": bool(large_top_like_full_image),
        }

    def get_defect_type(self, top_mask: np.ndarray, bottom_mask: np.ndarray, mask01: np.ndarray) -> Optional[str]:
        top01 = self._ensure_single_channel_mask(top_mask)
        down01 = self._ensure_single_channel_mask(bottom_mask)
        mask01 = self._ensure_single_channel_mask(mask01)
        bpm = self._bridge_pinch(mask01, top01, down01)
        is_missing = bool(bpm.get("large_top_like_full_image", False)) or self._is_missing_by_top_bottom_similarity(top01, down01, diff_ratio_th=0.02)
        washout = self._has_bottom_without_top_overlap_and_with_mask(top01, down01, mask01)
        for defect_name, defect_flag in [
            ("missing", bool(is_missing)),
            ("washout", bool(washout)),
            ("bridge", bool(bpm.get("bridge", False))),
            ("pinch", bool(bpm.get("pinch", False))),
            ("scumming", bool(bpm.get("scumming", False))),
        ]:
            if defect_flag:
                return defect_name
        return None

    def _update_defect_metrics(self, pred_top, pred_bottom, gt_top, gt_bottom, mask, gauge_info: Optional[Dict]) -> None:
        del gauge_info
        gt_type = self.get_defect_type(gt_top, gt_bottom, mask)
        gt_types = set([gt_type]) if gt_type is not None else set()
        gt_status = gt_type is not None
        pred_type = self.get_defect_type(pred_top, pred_bottom, mask)
        pred_types = set([pred_type]) if pred_type is not None else set()
        pred_status = pred_type is not None
        if pred_status and gt_status:
            self.defect_status_tp.add_(1)
        elif pred_status and not gt_status:
            self.defect_status_fp.add_(1)
        elif (not pred_status) and gt_status:
            self.defect_status_fn.add_(1)
        else:
            self.defect_status_tn.add_(1)
        self.defect_status_count.add_(1)
        inter = len(pred_types & gt_types)
        precision = inter / max(1, len(pred_types))
        recall = inter / max(1, len(gt_types))
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        self.defect_type_f1.add_(torch.tensor(float(f1), device=self.device))
        self.defect_type_count.add_(1)

    def compute(self):
        results = {}
        for prefix in ["top", "bottom"]:
            count = getattr(self, f"{prefix}_iou_count")
            if count > 0:
                results[f"{prefix}_Boundary_IoU"] = getattr(self, f"{prefix}_boundary_iou") / count
                results[f"{prefix}_Mask_Dice"] = getattr(self, f"{prefix}_mask_dice") / count
            count = getattr(self, f"{prefix}_count")
            if count > 0:
                results[f"{prefix}_EPE"] = getattr(self, f"{prefix}_signed_epe") / count
            else:
                results[f"{prefix}_EPE"] = torch.tensor(0.0, device=self.device)
            cd_count = getattr(self, f"{prefix}_cd_count")
            if cd_count > 0:
                results[f"{prefix}_CD_MAE"] = getattr(self, f"{prefix}_cd_abs_error") / cd_count
                results[f"{prefix}_CD bias"] = getattr(self, f"{prefix}_cd_signed_error") / cd_count
            else:
                results[f"{prefix}_CD_MAE"] = torch.tensor(0.0, device=self.device)
                results[f"{prefix}_CD bias"] = torch.tensor(0.0, device=self.device)
        if self.compute_image_metrics:
            results["PSNR"] = self.psnr_metric.compute()
            if self.fid_metric is not None and self.num_samples > 0:
                results["FID"] = self.fid_metric.compute()
            else:
                results["FID"] = torch.tensor(float("nan"), device=self.device)
            if self.kid_metric is not None and self.num_samples > 0:
                kid_mean, kid_std = self.kid_metric.compute()
                results["KID_Mean"] = kid_mean
                results["KID_Std"] = kid_std
            else:
                results["KID_Mean"] = torch.tensor(float("nan"), device=self.device)
                results["KID_Std"] = torch.tensor(float("nan"), device=self.device)
        if self.defect_status_count > 0:
            tp = self.defect_status_tp.float()
            fp = self.defect_status_fp.float()
            fn = self.defect_status_fn.float()
            results["Defect_Status_F1"] = 2 * tp / torch.clamp(2 * tp + fp + fn, min=1.0)
        if self.defect_type_count > 0:
            results["Defect_Type_F1"] = self.defect_type_f1 / self.defect_type_count
        return results

    def reset(self) -> None:
        super().reset()
        if self.compute_image_metrics:
            self.psnr_metric.reset()
            if self.fid_metric is not None:
                self.fid_metric.reset()
            if self.kid_metric is not None:
                self.kid_metric.reset()

    def load_sam_model(self, path):
        if LoRA_SAM_Lightning is None:
            raise ImportError(
                "LoRA_SAM_Lightning is unavailable. Please check "
                "`release_fabbench/fabbench_sam_loader.py` and install the official SAM dependencies."
            )
        args = get_args()
        if "CKPT_ROOT" not in os.environ:
            raise EnvironmentError("Missing environment variable CKPT_ROOT for the base SAM checkpoint.")
        args.checkpoint = os.environ["CKPT_ROOT"] + "/sam/sam_vit_b_01ec64.pth"
        args.sam_type = "vit_b"
        model = LoRA_SAM_Lightning.load_from_checkpoint(path, args=args)
        model.eval()
        return model

    @torch.no_grad()
    def get_image_embedding(self, img_3c):
        h, w = img_3c.shape[-2:]
        img_1024 = F.interpolate(img_3c, size=(1024, 1024), mode="bilinear", align_corners=False)
        preprocessed_img = self.seg_model.lora_sam_model.sam.preprocess(img_1024)
        embedding = self.seg_model.lora_sam_model.sam.image_encoder(preprocessed_img)
        return embedding, h, w

    @torch.no_grad()
    def process_batch_with_mask_prompt(self, image):
        batch_size = image.shape[0]
        img_embed, h, w = self.get_image_embedding(image)
        device = image.device
        prompt_encoder = self.seg_model.lora_sam_model.sam.prompt_encoder
        sparse_embeddings = torch.zeros(batch_size, 0, prompt_encoder.embed_dim, device=device)
        feat_size = prompt_encoder.image_embedding_size
        dense_top = self.seg_model.no_mask_embed_top.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, feat_size[0], feat_size[1]
        )
        dense_bottom = self.seg_model.no_mask_embed_down.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, feat_size[0], feat_size[1]
        )
        low_res_top, _ = self.seg_model.lora_sam_model.sam.mask_decoder(
            image_embeddings=img_embed,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_top,
            multimask_output=False,
        )
        low_res_bottom, _ = self.seg_model.lora_sam_model.sam.mask_decoder(
            image_embeddings=img_embed,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_bottom,
            multimask_output=False,
        )
        upscaled_top = self.seg_model.lora_sam_model.sam.postprocess_masks(low_res_top, (1024, 1024), (h, w))
        upscaled_bottom = self.seg_model.lora_sam_model.sam.postprocess_masks(low_res_bottom, (1024, 1024), (h, w))
        bin_top = (upscaled_top > self.seg_model.lora_sam_model.sam.mask_threshold).float()
        bin_bottom = (upscaled_bottom > self.seg_model.lora_sam_model.sam.mask_threshold).float()
        if self.sam_path.split("/")[-1] == "lora_sam_best_final_version.ckpt":
            bin_bottom = torch.maximum(bin_top, bin_bottom)
        return bin_top, bin_bottom

    def _get_signed_distance(self, pred_img, start_point, scan_axis):
        rows, cols = pred_img.shape
        x, y = start_point
        try:
            current_val = pred_img[y, x]
        except IndexError:
            return False, 0, start_point
        is_expansion = current_val > 127
        directions = [(1, 0), (-1, 0)] if scan_axis == "horizontal" else [(0, 1), (0, -1)]
        min_dist = self.MAX_EPE
        best_end_point = start_point
        found = False
        for dx, dy in directions:
            curr_x, curr_y = x, y
            dist = 0
            while 0 <= curr_x < cols and 0 <= curr_y < rows and dist < self.MAX_EPE:
                pixel_val = pred_img[curr_y, curr_x]
                target_found = pixel_val < 127 if is_expansion else pixel_val > 127
                if target_found:
                    if dist < min_dist:
                        min_dist = dist
                        best_end_point = (curr_x, curr_y)
                        found = True
                    break
                curr_x += dx
                curr_y += dy
                dist += 1
        if not found:
            return found, 0, start_point
        final_epe = min_dist if is_expansion else -min_dist
        return found, final_epe, best_end_point

    def _sample_points_on_contour(self, contour):
        points = []
        axes = []
        for i in range(len(contour)):
            p_start = contour[i][0]
            p_end = contour[(i + 1) % len(contour)][0]
            vec = p_end - p_start
            length = np.linalg.norm(vec)
            if length < 1e-3:
                continue
            scan_axis = "vertical" if abs(vec[0]) > abs(vec[1]) else "horizontal"
            if length < self.MIN_EDGE_LENGTH:
                mid_x = int((p_start[0] + p_end[0]) / 2)
                mid_y = int((p_start[1] + p_end[1]) / 2)
                points.append((mid_x, mid_y))
                axes.append(scan_axis)
            else:
                unit_vec = vec / length
                eff_len = length - 2 * (self.CORNER_ROUNDING / 2)
                if eff_len <= 0:
                    continue
                start_offset = self.CORNER_ROUNDING / 2
                num_points = int(eff_len / self.EPE_SPACING)
                for k in range(num_points + 1):
                    dist_from_start = start_offset + k * self.EPE_SPACING
                    if dist_from_start > length - start_offset:
                        break
                    px = int(p_start[0] + unit_vec[0] * dist_from_start)
                    py = int(p_start[1] + unit_vec[1] * dist_from_start)
                    points.append((px, py))
                    axes.append(scan_axis)
        return points, axes

    def _calculate_epe(self, target_bin, pred_bin, mask, mode, batch_idx, sample_idx):
        del mode
        del batch_idx
        del sample_idx

        def ensure_cv_img(img):
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = (img * 255 if img.max() <= 1.0 else img).astype(np.uint8)
            return np.squeeze(img)

        mask = ensure_cv_img(mask)
        target_bin = ensure_cv_img(target_bin)
        pred_bin = ensure_cv_img(pred_bin)
        mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        epe_data = []
        cd_vals_px = []
        cd_vals_tx = []

        for cnt in mask_contours:
            pts, axes = self._sample_points_on_contour(cnt)
            for idx, pt in enumerate(pts):
                is_found_target, dist_t, end_pt_target = self._get_signed_distance(target_bin, pt, axes[idx])
                is_found_pred, dist_p, end_pt_pred = self._get_signed_distance(pred_bin, pt, axes[idx])
                if is_found_target and is_found_pred:
                    relative_epe = dist_p - dist_t
                    epe_data.append({"start": end_pt_target, "end": end_pt_pred, "val": relative_epe})
                    cd_px = self._measure_cd_at_point(pred_bin, pt, axes[idx])
                    if cd_px is not None:
                        cd_vals_px.append(cd_px)
                    cd_tx = self._measure_cd_at_point(target_bin, pt, axes[idx])
                    if cd_tx is not None:
                        cd_vals_tx.append(cd_tx)
                elif not is_found_target and not is_found_pred:
                    continue
                else:
                    epe_data.append({"start": pt, "end": pt, "val": self.MAX_EPE})

        if not epe_data:
            return 0.0, 0.0, 0.0
        total_abs_epe = 0.0
        total_signed_epe = 0.0
        for item in epe_data:
            val = float(item["val"])
            total_abs_epe += abs(val)
            total_signed_epe += val
        mean_abs_epe = total_abs_epe / len(epe_data)
        mean_signed_epe = total_signed_epe / len(epe_data)

        return mean_abs_epe, mean_signed_epe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image FabBench evaluation suite for reviewer release.")
    parser.add_argument("--test_image", type=Path, required=True, help="Predicted SEM image to be evaluated.")
    parser.add_argument("--gt_image", type=Path, default=None, help="Ground-truth SEM image. If omitted, try <test_image_dir>/gt_sem.png.")
    parser.add_argument("--mask_image", type=Path, default=None, help="Binary layout mask image. If omitted, try <test_image_dir>/mask.png.")
    parser.add_argument("--gauge_info", type=Path, default=None, help="Optional JSON file containing gauges and cd_values for the current mask.")
    parser.add_argument("--sam_ckpt", type=str, default="", help="Optional private LoRA-SAM checkpoint. Empty means geometry metrics are skipped.")
    parser.add_argument("--ckpt_root", type=Path, default=None, help="Optional root containing the base SAM checkpoint.")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "fabbench_metrics_suite", help="Directory for JSON outputs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Execution device.")
    parser.add_argument("--px_size", type=float, default=1.0, help="Pixel size in nm for CD-related FabBench metrics.")
    return parser.parse_args()


def ensure_file(path: Path | None, name: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"{name} is required but was not provided.")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def infer_side_input(test_image: Path, filename: str) -> Path | None:
    candidate = test_image.resolve().parent / filename
    return candidate if candidate.is_file() else None


def load_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read grayscale image: {path}")
    return image


def load_json_file(path: Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    path = ensure_file(path, "gauge_info")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_3ch(image_hw: np.ndarray) -> np.ndarray:
    if image_hw.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image_hw.shape}")
    return np.repeat(image_hw[:, :, None], 3, axis=2)


def np_to_nchw_uint8(image_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    if image_hwc.ndim != 3 or image_hwc.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image_hwc.shape}")
    tensor = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=torch.uint8)


def mask_to_nchw_uint8(mask_hw: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(mask_hw).unsqueeze(0).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=torch.uint8)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 1:
            value = value.detach().cpu().item()
        else:
            value = value.detach().float().mean().cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def compute_entropy(image: np.ndarray) -> float:
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(1.0, float(hist.sum()))
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def compute_snr_db(image: np.ndarray) -> float | None:
    image_f = image.astype(np.float32)
    smooth = cv2.GaussianBlur(image_f, (5, 5), 0)
    noise = image_f - smooth
    signal_power = float(np.mean(smooth ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power <= 1e-12:
        return None
    return float(10.0 * math.log10(signal_power / noise_power))


def compute_no_reference_image_metrics(image: np.ndarray) -> dict[str, float | None]:
    image_f = image.astype(np.float32)
    lap = cv2.Laplacian(image_f, cv2.CV_32F)
    gx = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_map = cv2.Canny(image, 100, 200)
    return {
        "mean_intensity": float(np.mean(image_f)),
        "std_intensity": float(np.std(image_f)),
        "dynamic_range": float(np.max(image_f) - np.min(image_f)),
        "entropy": compute_entropy(image),
        "sharpness_laplacian_var": float(lap.var()),
        "mean_gradient": float(np.mean(grad_mag)),
        "edge_density": float(np.count_nonzero(edge_map)) / float(edge_map.size),
        "snr_db_estimate": compute_snr_db(image),
    }


def compute_basic_image_metrics(pred_gray: np.ndarray, gt_gray: np.ndarray) -> dict[str, float | None]:
    pred = pred_gray.astype(np.float32)
    gt = gt_gray.astype(np.float32)
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    psnr = 99.0 if mse <= 1e-12 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))
    ssim = None
    if skimage_ssim is not None:
        ssim = float(skimage_ssim(pred_gray, gt_gray, data_range=255))
    return {"PSNR": psnr, "SSIM": ssim}


def compute_fabbench_geometry_metrics(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    mask_gray: np.ndarray,
    gauge_info: Dict[str, Any] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float | None], list[str], str | None]:
    skipped: list[str] = []
    if not args.sam_ckpt:
        skipped.extend(
            [
                "top/bottom Boundary IoU",
                "top/bottom Mask Dice",
                "top/bottom EPE",
                "top/bottom CD_MAE",
                "top/bottom CD bias",
                "Defect_Status_F1",
                "Defect_Type_F1",
            ]
        )
        return {}, skipped, "SAM checkpoint not provided. Geometry metrics were intentionally skipped."

    if args.ckpt_root is not None:
        os.environ["CKPT_ROOT"] = str(args.ckpt_root.expanduser().resolve())

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = FabBenchMetrics(
        sam_path=args.sam_ckpt,
        save_dir=str(output_dir),
        epe_spacing=50,
        min_edge_length=40,
        max_epe=20,
        px_size=args.px_size,
        compute_image_metrics=False,
    )
    try:
        metric.seg_model = metric.load_sam_model(metric.sam_path)
    except Exception as exc:
        return {}, skipped, f"Failed to initialize SAM-backed FabBenchMetrics: {exc}"
    metric.to(device)
    metric.eval()
    metric.set_output_dir(str(output_dir))

    pred_t = np_to_nchw_uint8(pred_rgb, device)
    gt_t = np_to_nchw_uint8(gt_rgb, device)
    mask_t = mask_to_nchw_uint8(mask_gray, device)
    with torch.no_grad():
        metric.update(pred_t, gt_t, mask_t, batch_idx=0, gauge_info=gauge_info)
        raw = metric.compute()
    return {k: safe_float(v) for k, v in raw.items()}, skipped, None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    test_image = ensure_file(args.test_image, "test_image")
    pred_gray = load_gray_image(test_image)
    gt_image = args.gt_image or infer_side_input(test_image, "gt_sem.png")
    mask_image = args.mask_image or infer_side_input(test_image, "mask.png")
    gauge_info = load_json_file(args.gauge_info)

    gt_gray = None
    mask_gray = None
    image_metrics: dict[str, float | None] = {}
    image_metrics.update(compute_no_reference_image_metrics(pred_gray))

    if gt_image is not None and Path(gt_image).is_file():
        gt_image = ensure_file(gt_image, "gt_image")
        gt_gray = load_gray_image(gt_image)
        if pred_gray.shape != gt_gray.shape:
            pred_gray = cv2.resize(pred_gray, (gt_gray.shape[1], gt_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        image_metrics.update(compute_basic_image_metrics(pred_gray, gt_gray))

    fabbench_metrics: dict[str, float | None] = {}
    skipped_metrics: list[str] = []
    geometry_message = "Geometry metrics were not requested."

    if args.sam_ckpt:
        if gt_gray is None:
            raise FileNotFoundError("gt_image is required when sam_ckpt is provided.")
        if mask_image is None or not Path(mask_image).is_file():
            raise FileNotFoundError("mask_image is required when sam_ckpt is provided.")
        mask_image = ensure_file(mask_image, "mask_image")
        mask_gray = load_gray_image(mask_image)
        if mask_gray.shape != gt_gray.shape:
            mask_gray = cv2.resize(mask_gray, (gt_gray.shape[1], gt_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_rgb = to_3ch(pred_gray)
        gt_rgb = to_3ch(gt_gray)
        fabbench_metrics, skipped_metrics, geometry_message = compute_fabbench_geometry_metrics(
            pred_rgb=pred_rgb,
            gt_rgb=gt_rgb,
            mask_gray=mask_gray,
            gauge_info=gauge_info,
            args=args,
            device=device,
        )

    result = {
        "test_image": str(test_image),
        "gt_image": str(gt_image) if gt_image is not None and Path(gt_image).is_file() else None,
        "mask_image": str(mask_image) if mask_image is not None and Path(mask_image).is_file() else None,
        "gauge_info": str(args.gauge_info) if args.gauge_info is not None else None,
        "device": str(device),
        "sam_ckpt_provided": bool(args.sam_ckpt),
        "image_metrics": image_metrics,
        "fabbench_geometry_metrics": fabbench_metrics,
        "skipped_metrics": skipped_metrics,
        "notes": {
            "geometry_metrics": geometry_message,
        },
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fabbench_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()
