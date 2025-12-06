from __future__ import annotations

from enum import Enum
import numpy as np
import torch

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.geometry import affine_inverse_np
from depth_anything_3.utils.pose_align import transform_points_sim3
from depth_anything_3.utils.logger import logger


class GaussianScalingMethod(str, Enum):
    UMEYAMA_POINTS = "umeyama_points"
    MEDIAN_DISTANCE = "median_distance"
    BBOX_SCALING = "bbox_scaling"


def _reshape_points_to_2d(points: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Reshape points array to 2D (N, 3) format, handling batch dimensions.
    
    Args:
        points: Points array with shape (N, 3), (B, N, 3), or (B, N, D)
        
    Returns:
        Reshaped points (M, 3) and original shape for reshaping back
    """
    original_shape = points.shape
    if len(points.shape) == 2:
        # Already (N, 3)
        return points, original_shape
    elif len(points.shape) == 3:
        # (B, N, 3) -> (B*N, 3)
        return points.reshape(-1, points.shape[-1]), original_shape
    else:
        raise ValueError(f"Unexpected points shape: {points.shape}, expected (N, 3) or (B, N, 3)")


def _reshape_points_back(points: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Reshape points back to original shape.
    
    Args:
        points: Points array (M, 3)
        original_shape: Original shape tuple
        
    Returns:
        Reshaped points with original_shape
    """
    if len(original_shape) == 2:
        return points
    elif len(original_shape) == 3:
        return points.reshape(original_shape)
    else:
        raise ValueError(f"Unexpected original shape: {original_shape}")


def _extrinsics_to_4x4(extrinsics: np.ndarray) -> np.ndarray:
    """
    Convert extrinsics to 4x4 format if needed.
    
    Args:
        extrinsics: Camera extrinsics w2c (N, 3, 4) or (N, 4, 4)
        
    Returns:
        Extrinsics in 4x4 format (N, 4, 4)
    """
    if extrinsics.shape[1] == 3:
        ext_4x4 = np.zeros((extrinsics.shape[0], 4, 4), dtype=extrinsics.dtype)
        ext_4x4[:, :3, :] = extrinsics
        ext_4x4[:, 3, 3] = 1.0
        return ext_4x4
    return extrinsics


def umeyama_points(
    points_src: np.ndarray,  # (N, 3) - source points (Gaussians)
    points_tgt: np.ndarray,  # (M, 3) - target points (point cloud)
    ransac: bool = True,
    ransac_max_iters: int = 50,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Align point clouds using Umeyama Sim(3) algorithm with optional RANSAC.
    
    Args:
        points_src: Source point cloud (N, 3)
        points_tgt: Target point cloud (M, 3)
        ransac: Whether to use RANSAC for robust alignment
        ransac_max_iters: Maximum RANSAC iterations
        random_state: Random seed for RANSAC
        
    Returns:
        Rotation (3, 3), translation (3,), scale (float)
    """
    from evo.core.trajectory import PosePath3D
    
    # Create dummy poses from point clouds (identity rotation, points as translation)
    # This allows us to reuse the existing pose alignment code
    n_src = points_src.shape[0]
    n_tgt = points_tgt.shape[0]
    
    # Create identity rotations
    R_identity = np.eye(3)
    
    # Create poses: identity rotation, points as translations
    poses_src = np.zeros((n_src, 4, 4))
    poses_src[:, :3, :3] = R_identity
    poses_src[:, :3, 3] = points_src
    poses_src[:, 3, 3] = 1.0
    
    poses_tgt = np.zeros((n_tgt, 4, 4))
    poses_tgt[:, :3, :3] = R_identity
    poses_tgt[:, :3, 3] = points_tgt
    poses_tgt[:, 3, 3] = 1.0
    
    # Use Umeyama alignment via PosePath3D
    if not ransac:
        path_src = PosePath3D(poses_se3=poses_src.copy())
        path_tgt = PosePath3D(poses_se3=poses_tgt.copy())
        r, t, s = path_src.align(path_tgt, correct_scale=True)
    else:
        # RANSAC alignment
        rng = np.random.default_rng(random_state)
        n_min = min(n_src, n_tgt)
        sub_n = max(3, min(n_min // 2, 100))  # Sample subset for RANSAC
        
        best_model = None
        best_inliers = None
        best_score = (-1, np.inf)
        
        # Pre-alignment to get threshold
        path_src_pre = PosePath3D(poses_se3=poses_src.copy())
        path_tgt_pre = PosePath3D(poses_se3=poses_tgt.copy())
        r0, t0, s0 = path_src_pre.align(path_tgt_pre, correct_scale=True)
        
        # Apply pre-alignment and compute threshold
        poses_src_aligned = poses_src.copy()
        poses_src_aligned[:, :3, :3] = r0 @ poses_src_aligned[:, :3, :3]
        poses_src_aligned[:, :3, 3] = (r0 @ (s0 * poses_src_aligned[:, :3, 3].T)).T + t0
        
        # Compute distances to nearest neighbors for threshold
        tgt_positions = poses_tgt[:, :3, 3]
        src_positions_aligned = poses_src_aligned[:, :3, 3]
        dists = []
        for pos in src_positions_aligned:
            dd = np.linalg.norm(tgt_positions - pos[None, :], axis=1)
            dists.append(dd.min())
        inlier_thresh = float(np.median(dists)) if dists else 1.0
        
        # Initialize best_inliers with pre-alignment result
        poses_src_init = poses_src.copy()
        poses_src_init[:, :3, :3] = r0 @ poses_src_init[:, :3, :3]
        poses_src_init[:, :3, 3] = (r0 @ (s0 * poses_src_init[:, :3, 3].T)).T + t0
        src_positions_init = poses_src_init[:, :3, 3]
        errs_init = []
        for pos in src_positions_init:
            dd = np.linalg.norm(tgt_positions - pos[None, :], axis=1)
            errs_init.append(dd.min())
        errs_init = np.array(errs_init)
        inliers_init = errs_init <= inlier_thresh
        k_init = int(inliers_init.sum())
        mean_err_init = float(errs_init[inliers_init].mean()) if k_init > 0 else np.inf
        best_score = (k_init, mean_err_init)
        best_model = (r0, t0, s0)
        best_inliers = inliers_init
        
        # RANSAC loop
        for _ in range(ransac_max_iters):
            # Sample corresponding indices
            if n_src <= sub_n:
                src_idx = np.arange(n_src)
            else:
                src_idx = rng.choice(n_src, sub_n, replace=False)
            
            if n_tgt <= sub_n:
                tgt_idx = np.arange(n_tgt)
            else:
                tgt_idx = rng.choice(n_tgt, sub_n, replace=False)
            
            try:
                path_src_sample = PosePath3D(poses_se3=poses_src[src_idx].copy())
                path_tgt_sample = PosePath3D(poses_se3=poses_tgt[tgt_idx].copy())
                r, t, s = path_src_sample.align(path_tgt_sample, correct_scale=True)
            except Exception:
                continue
            
            # Apply transformation and compute inliers
            poses_src_test = poses_src.copy()
            poses_src_test[:, :3, :3] = r @ poses_src_test[:, :3, :3]
            poses_src_test[:, :3, 3] = (r @ (s * poses_src_test[:, :3, 3].T)).T + t
            
            src_positions_test = poses_src_test[:, :3, 3]
            errs = []
            for pos in src_positions_test:
                dd = np.linalg.norm(tgt_positions - pos[None, :], axis=1)
                errs.append(dd.min())
            errs = np.array(errs)
            
            inliers = errs <= inlier_thresh
            k = int(inliers.sum())
            mean_err = float(errs[inliers].mean()) if k > 0 else np.inf
            
            if (k > best_score[0]) or (k == best_score[0] and mean_err < best_score[1]):
                best_score = (k, mean_err)
                best_model = (r, t, s)
                best_inliers = inliers
        
        # Refit with best inliers
        if best_inliers is not None and best_inliers.sum() >= 3:
            try:
                path_src_best = PosePath3D(poses_se3=poses_src[best_inliers].copy())
                # Find corresponding target points (nearest neighbors)
                src_positions_best = poses_src[best_inliers, :3, 3]
                tgt_indices = []
                for pos in src_positions_best:
                    dd = np.linalg.norm(tgt_positions - pos[None, :], axis=1)
                    tgt_indices.append(dd.argmin())
                tgt_indices = np.array(tgt_indices)
                path_tgt_best = PosePath3D(poses_se3=poses_tgt[tgt_indices].copy())
                r, t, s = path_src_best.align(path_tgt_best, correct_scale=True)
            except Exception:
                r, t, s = best_model
        else:
            r, t, s = best_model
    
    return r, t, s


def median_distance_scaling(
    extrinsics: np.ndarray,  # (N, 3, 4) or (N, 4, 4) - w2c extrinsics
) -> float:
    """
    Compute scale factor using median distance of camera positions.
    
    Args:
        extrinsics: Camera extrinsics w2c (N, 3, 4) or (N, 4, 4)
        
    Returns:
        Scale factor (float)
    """
    extrinsics = _extrinsics_to_4x4(extrinsics)
    
    # Extract camera positions
    c2w = affine_inverse_np(extrinsics)
    camera_positions = c2w[:, :3, 3]  # (N, 3)
    
    # Compute median distance
    distances = np.linalg.norm(camera_positions, axis=1)
    median_dist = np.median(distances)
    return max(median_dist, 1e-1)  # Clamp minimum


def filter_points_by_percentile(
    points: np.ndarray,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    """
    Filter points by removing those outside the percentile range in each dimension.
    
    Args:
        points: Point cloud (N, 3)
        percentile_low: Lower percentile threshold (e.g., 5.0)
        percentile_high: Upper percentile threshold (e.g., 95.0)
        
    Returns:
        Filtered points (M, 3) where M <= N
    """
    filtered_points = points.copy()
    
    for dim in range(3):  # x, y, z dimensions
        # Use filtered_points (not original points) to get values for current dimension
        values = filtered_points[:, dim]
        low_threshold = np.percentile(values, percentile_low)
        high_threshold = np.percentile(values, percentile_high)
        
        # Keep points within percentile range for this dimension
        mask = (values >= low_threshold) & (values <= high_threshold)
        filtered_points = filtered_points[mask]
    
    return filtered_points


def bbox_scaling(
    gaussian_points: np.ndarray,  # (M, 3) - Gaussian means
    point_cloud_points: np.ndarray,  # (N, 3) - Point cloud points
    percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute scale factor using bounding box comparison with outlier rejection.
    
    Filters points by removing outliers using percentile-based filtering in each
    dimension, then computes bounding boxes on the filtered points and derives
    scale from the ratio of bbox diagonals.
    
    Args:
        gaussian_points: Gaussian means (M, 3)
        point_cloud_points: Point cloud points (N, 3)
        percentile: Percentile to use for filtering (default: 95.0)
                   Uses (100-percentile)/2 and (100+percentile)/2 as bounds
                   (e.g., 95% -> 5% and 95%)
        
    Returns:
        Rotation (3, 3), translation (3,), scale (float)
    """
    if gaussian_points.shape[0] < 3 or point_cloud_points.shape[0] < 3:
        raise ValueError("Need at least 3 points for bbox scaling")
    
    # Compute percentile bounds (e.g., for 95%: use 5% and 95%)
    percentile_low = (100.0 - percentile) / 2.0
    percentile_high = 100.0 - percentile_low
    
    # Filter outliers from both point clouds
    gaussian_filtered = filter_points_by_percentile(
        gaussian_points,
        percentile_low,
        percentile_high,
    )
    
    point_cloud_filtered = filter_points_by_percentile(
        point_cloud_points,
        percentile_low,
        percentile_high,
    )
    
    # Check we still have enough points after filtering
    if gaussian_filtered.shape[0] < 3 or point_cloud_filtered.shape[0] < 3:
        raise ValueError(
            f"Not enough points after filtering: "
            f"gaussian={gaussian_filtered.shape[0]}, "
            f"point_cloud={point_cloud_filtered.shape[0]}"
        )
    
    # Compute bounding boxes on filtered points
    gaussian_min = gaussian_filtered.min(axis=0)  # (3,)
    gaussian_max = gaussian_filtered.max(axis=0)  # (3,)
    gaussian_size = gaussian_max - gaussian_min  # (3,)
    gaussian_diagonal = np.linalg.norm(gaussian_size)  # scalar
    
    pc_min = point_cloud_filtered.min(axis=0)  # (3,)
    pc_max = point_cloud_filtered.max(axis=0)  # (3,)
    pc_size = pc_max - pc_min  # (3,)
    pc_diagonal = np.linalg.norm(pc_size)  # scalar
    
    # Compute scale from ratio of diagonals
    if gaussian_diagonal < 1e-6:
        raise ValueError("Gaussian bbox is too small, cannot compute scale")
    
    scale = pc_diagonal / gaussian_diagonal
    
    # Compute translation to align centers
    gaussian_center = (gaussian_min + gaussian_max) / 2.0  # (3,)
    pc_center = (pc_min + pc_max) / 2.0  # (3,)
    
    # Translation: align centers after scaling
    # We want: scale * gaussian_center + t = pc_center
    # So: t = pc_center - scale * gaussian_center
    t = pc_center - scale * gaussian_center
    
    # Rotation: identity (bbox scaling doesn't account for rotation)
    R = np.eye(3)
    
    return R, t, scale


def sample_points_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    conf: np.ndarray | None = None,
    max_samples: int = 10000,
    conf_thresh: float = 0.5,
) -> np.ndarray:
    """
    Sample points from depth maps by unprojecting to world coordinates.
    
    Args:
        depth: Depth maps (N, H, W)
        intrinsics: Camera intrinsics (N, 3, 3)
        extrinsics: Camera extrinsics w2c (N, 3, 4) or (N, 4, 4)
        conf: Confidence maps (N, H, W) or None
        max_samples: Maximum number of points to sample
        conf_thresh: Confidence threshold for filtering
        
    Returns:
        Sampled world points (M, 3) where M <= max_samples
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W, 3)
    
    pts_all = []
    
    for i in range(N):
        d = depth[i]  # (H, W)
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thresh
        
        if not np.any(valid):
            continue
        
        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))
        
        # Sample if too many points
        if len(vidx) > max_samples // N:
            vidx = np.random.choice(vidx, max_samples // N, replace=False)
        
        K_inv = np.linalg.inv(intrinsics[i])  # (3, 3)
        
        # Convert extrinsics to 4x4 if needed
        ext_w2c = extrinsics[i]
        if ext_w2c.shape == (3, 4):
            ext_4x4 = np.eye(4, dtype=ext_w2c.dtype)
            ext_4x4[:3, :] = ext_w2c
            ext_w2c = ext_4x4
        elif ext_w2c.shape != (4, 4):
            continue
        
        c2w = np.linalg.inv(ext_w2c)  # (4, 4)
        
        rays = K_inv @ pix[vidx].T  # (3, M)
        Xc = rays * d_flat[vidx][None, :]  # (3, M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M, 3)
        
        pts_all.append(Xw)
    
    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    points = np.concatenate(pts_all, 0)
    
    # Final sampling if still too many
    if points.shape[0] > max_samples:
        idx = np.random.choice(points.shape[0], max_samples, replace=False)
        points = points[idx]
    
    return points


def align_gaussians(
    method: GaussianScalingMethod,
    gaussian_points: np.ndarray,  # (M, 3) - Gaussian means
    extrinsics: np.ndarray,  # (N, 3, 4) or (N, 4, 4) - w2c extrinsics
    point_cloud_points: np.ndarray | None = None,  # (K, 3) - optional point cloud points
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Align Gaussians using the specified method.
    
    Args:
        method: Scaling method to use
        gaussian_points: Gaussian means (M, 3)
        extrinsics: Camera extrinsics w2c (N, 3, 4) or (N, 4, 4)
        point_cloud_points: Optional point cloud points for umeyama_points method
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Rotation (3, 3), translation (3,), scale (float)
    """
    if method == GaussianScalingMethod.UMEYAMA_POINTS:
        if point_cloud_points is None:
            raise ValueError("point_cloud_points required for umeyama_points method")
        return umeyama_points(
            gaussian_points,
            point_cloud_points,
            ransac=kwargs.get("ransac", True),
            ransac_max_iters=kwargs.get("ransac_max_iters", 50),
            random_state=kwargs.get("random_state", 42),
        )
    
    elif method == GaussianScalingMethod.MEDIAN_DISTANCE:
        scale = median_distance_scaling(extrinsics)
        # Return identity rotation and zero translation, only scale
        return np.eye(3), np.zeros(3), scale
    
    elif method == GaussianScalingMethod.BBOX_SCALING:
        if point_cloud_points is None:
            raise ValueError("point_cloud_points required for bbox_scaling method")
        return bbox_scaling(
            gaussian_points,
            point_cloud_points,
            percentile=kwargs.get("percentile", 95.0),
        )
    
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def align_gaussians_to_point_cloud(
    prediction: Prediction,
    method: GaussianScalingMethod | str = GaussianScalingMethod.UMEYAMA_POINTS,
    max_samples: int = 10000,
    n_samples_for_alignment: int = 5000,
    ransac: bool = True,
    ransac_max_iters: int = 50,
    random_state: int | None = 42,
) -> Prediction:
    """
    Align Gaussians in prediction to match point cloud coordinate space.
    
    This is a high-level function that takes a Prediction object and aligns
    the Gaussians to match the point cloud coordinate frame.
    
    Args:
        prediction: Prediction object containing Gaussians, depth, extrinsics, etc.
        method: Scaling method to use
        max_samples: Maximum number of points to sample from depth for alignment
        n_samples_for_alignment: Number of points to use for alignment computation
        ransac: Whether to use RANSAC for robust alignment
        ransac_max_iters: Maximum RANSAC iterations
        random_state: Random seed for RANSAC
        
    Returns:
        Prediction with aligned Gaussians (modified in-place)
    """
    if prediction.gaussians is None or prediction.extrinsics is None:
        logger.warn("Cannot align Gaussians: missing gaussians or extrinsics")
        return prediction
    
    # Convert method string to enum if needed
    if isinstance(method, str):
        method = GaussianScalingMethod(method)
    
    # Convert tensors to numpy if needed
    gaussian_points = prediction.gaussians.means  # (B, N, 3) or (N, 3)
    if isinstance(gaussian_points, torch.Tensor):
        gaussian_points = gaussian_points.detach().cpu().numpy()
    
    # Reshape to handle batch dimensions
    gaussian_points_2d, gaussian_original_shape = _reshape_points_to_2d(gaussian_points)
    
    # Ensure extrinsics are numpy arrays
    extrinsics_np = prediction.extrinsics
    if isinstance(extrinsics_np, torch.Tensor):
        extrinsics_np = extrinsics_np.detach().cpu().numpy()
    
    try:
        if method == GaussianScalingMethod.UMEYAMA_POINTS:
            # Sample corresponding points from point cloud (by unprojecting depth)
            point_cloud_points = sample_points_from_depth(
                prediction.depth,
                prediction.intrinsics,
                extrinsics_np,
                prediction.conf,
                max_samples=max_samples,
            )
            
            if point_cloud_points.shape[0] < 3 or gaussian_points_2d.shape[0] < 3:
                logger.warn("Not enough points for alignment. Skipping.")
                return prediction
            
            # Sample a subset for faster alignment if we have too many points
            n_samples = min(n_samples_for_alignment, point_cloud_points.shape[0], gaussian_points_2d.shape[0])
            if point_cloud_points.shape[0] > n_samples:
                idx = np.random.choice(point_cloud_points.shape[0], n_samples, replace=False)
                point_cloud_samples = point_cloud_points[idx]
            else:
                point_cloud_samples = point_cloud_points
            
            if gaussian_points_2d.shape[0] > n_samples:
                idx = np.random.choice(gaussian_points_2d.shape[0], n_samples, replace=False)
                gaussian_samples = gaussian_points_2d[idx]
            else:
                gaussian_samples = gaussian_points_2d
            
            rot, trans, scale = align_gaussians(
                method=method,
                gaussian_points=gaussian_samples,
                extrinsics=extrinsics_np,
                point_cloud_points=point_cloud_samples,
                ransac=ransac,
                ransac_max_iters=ransac_max_iters,
                random_state=random_state,
            )
        
        elif method == GaussianScalingMethod.MEDIAN_DISTANCE:
            rot, trans, scale = align_gaussians(
                method=method,
                gaussian_points=gaussian_points_2d,
                extrinsics=extrinsics_np,
            )
        
        elif method == GaussianScalingMethod.BBOX_SCALING:
            # Sample corresponding points from point cloud
            point_cloud_points = sample_points_from_depth(
                prediction.depth,
                prediction.intrinsics,
                extrinsics_np,
                prediction.conf,
                max_samples=10000,
            )
            
            if point_cloud_points.shape[0] < 3 or gaussian_points_2d.shape[0] < 3:
                logger.warn("Not enough points for bbox scaling. Skipping.")
                return prediction
            
            rot, trans, scale = align_gaussians(
                method=method,
                gaussian_points=gaussian_points_2d,
                extrinsics=extrinsics_np,
                point_cloud_points=point_cloud_points,
                percentile=95.0,  # Hardcoded for now
            )
        
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Apply transformation to all Gaussian points (using 2D reshaped version)
        gaussian_means_np_2d = transform_points_sim3(
            gaussian_points_2d, rot, trans, scale, inverse=False
        )
        
        # Reshape back to original shape
        gaussian_means_np = _reshape_points_back(gaussian_means_np_2d, gaussian_original_shape)
        
        # Convert back to tensor if original was tensor
        if isinstance(prediction.gaussians.means, torch.Tensor):
            prediction.gaussians.means = torch.from_numpy(gaussian_means_np).to(
                prediction.gaussians.means.device
            )
        else:
            prediction.gaussians.means = gaussian_means_np
        
        # Scale the Gaussian scales
        if isinstance(prediction.gaussians.scales, torch.Tensor):
            scale_tensor = torch.tensor(
                scale,
                dtype=prediction.gaussians.scales.dtype,
                device=prediction.gaussians.scales.device,
            )
            prediction.gaussians.scales = prediction.gaussians.scales * scale_tensor
        else:
            prediction.gaussians.scales = prediction.gaussians.scales * scale
        
        logger.info(
            f"Aligned Gaussians using {method.value}: "
            f"scale={scale:.4f}, translation norm={np.linalg.norm(trans):.4f}"
        )
    except Exception as e:
        logger.warn(f"Failed to align Gaussians using {method.value}: {e}. Skipping alignment.")
    
    return prediction

