# checks/duplicate.py
# ─────────────────────────────────────────────
# Near-duplicate detection using two-stage pipeline:
#
# Stage 1 — Optical Flow (fast pre-filter)
#   Farneback dense flow on downsampled masked frames.
#   Low mean flow magnitude → candidate duplicate.
#   Person movement masked out before computing flow.
#
# Stage 2 — Embedding Similarity (confirmation)
#   DINOv2 self-supervised embeddings for scene understanding.
#   High cosine similarity → confirmed duplicate.
#
# Cluster-based deduplication:
#   Groups consecutive duplicates into clusters.
#   Retains the SHARPEST frame per cluster (reuses blur scores).
#
# DINOv2 Choice: Self-supervised learning captures construction
# scene structure better than supervised models, enabling both
# duplicate detection and preparation for Task 2 progress analysis.
# ─────────────────────────────────────────────

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pathlib import Path
from config import CONFIG
from utils.regions import get_analysis_region, get_analysis_region_gray


# ── Embedding Model ───────────────────────────

class EmbeddingModel:
    """
    DINOv2 self-supervised feature extractor.
    Learns construction scene structure without labels.
    
    Advantages for construction sites:
    - Self-supervised → learns visual patterns, not labels
    - Better for structure detection (walls, floors, scaffolds)
    - Enables Task 2 progress monitoring via embeddings
    - 768-dim embeddings (~86MB model)
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load DINOv2 small model (lightweight)
        model_name = "facebook/dinov2-small"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract embedding from the analysis region of a single frame.
        Returns normalized 768-dim vector (DINOv2 feature dimension).
        """
        region = get_analysis_region(frame)
        
        # Convert BGR to RGB for PIL
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(region_rgb)
        
        # Preprocess and extract features
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        # Get pooled output (taking mean of patch tokens)
        feat = outputs.last_hidden_state.mean(dim=1).squeeze()  # (768,)
        feat = feat.cpu().numpy()
        
        # L2 normalize
        return feat / (np.linalg.norm(feat) + 1e-8)

    @torch.no_grad()
    def get_embeddings_batch(
        self,
        frames: list[np.ndarray],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Batched embedding extraction — much faster than one-by-one.
        Returns (N, 768) normalized array.
        Note: Smaller batch size than MobileNetV2 due to larger DINOv2 model.
        """
        all_embeddings = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            
            # Convert to PIL images in RGB
            pil_images = []
            for frame in batch_frames:
                region = get_analysis_region(frame)
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(region_rgb))
            
            # Batch preprocess
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            # Extract features (mean pooling over patch tokens)
            feats = outputs.last_hidden_state.mean(dim=1)  # (B, 768)
            feats = feats.cpu().numpy()
            
            # L2 normalize each embedding
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            feats = feats / norms
            all_embeddings.append(feats)
        
        return np.vstack(all_embeddings)  # (N, 768)


# ── Stage 1 — Optical Flow ────────────────────

def compute_optical_flow_magnitude(
    frame1: np.ndarray,
    frame2: np.ndarray
) -> float:
    """
    Compute mean optical flow magnitude between two consecutive frames.
    Low magnitude → camera barely moved → candidate duplicate.

    Downsamples before computing flow for speed.
    Uses analysis region only — excludes person at bottom.
    """
    scale = CONFIG["flow_downsample"]   # e.g. 0.25

    gray1 = get_analysis_region_gray(frame1)
    gray2 = get_analysis_region_gray(frame2)

    # Downsample for speed
    h, w  = gray1.shape
    small1 = cv2.resize(gray1, (int(w * scale), int(h * scale)))
    small2 = cv2.resize(gray2, (int(w * scale), int(h * scale)))

    flow  = cv2.calcOpticalFlowFarneback(
        small1, small2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Magnitude of flow vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(magnitude.mean())


def is_flow_candidate(flow_magnitude: float) -> bool:
    """Stage 1 gate — is this a candidate duplicate based on flow?"""
    return flow_magnitude < CONFIG["flow_threshold"]


# ── Stage 2 — Embedding Similarity ───────────

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings."""
    return float(np.dot(emb1, emb2))


def is_confirmed_duplicate(emb1: np.ndarray, emb2: np.ndarray) -> bool:
    """Stage 2 confirmation — is cosine similarity above threshold?"""
    sim = cosine_similarity(emb1, emb2)
    return sim > CONFIG["cosine_threshold"]


# ── Cluster-Based Deduplication ───────────────

def find_duplicate_clusters(
    frame_paths  : list[Path],
    embeddings   : np.ndarray,
    blur_scores  : list[float],
    load_frame_fn,
) -> list[dict]:
    """
    Main deduplication function.

    Algorithm:
      1. Compare each frame to the previous one using optical flow
      2. If flow is low → run embedding similarity check
      3. Group consecutive duplicates into clusters
      4. Within each cluster → keep the sharpest frame

    Args:
        frame_paths   : ordered list of frame file paths
        embeddings    : (N, 1280) precomputed embeddings
        blur_scores   : list of blur mean scores per frame (from Pass 1)
        load_frame_fn : callable(Path) → np.ndarray

    Returns:
        List of dicts: {frame, status, reason, duplicate_of}
    """
    n       = len(frame_paths)
    results = [
        {
            "frame"       : p.name,
            "status"      : "KEEP",
            "reason"      : "",
            "duplicate_of": "",
        }
        for p in frame_paths
    ]

    # Build clusters of consecutive near-duplicates
    # cluster_id[i] = which cluster frame i belongs to
    cluster_id     = list(range(n))   # start: each frame is its own cluster
    cluster_rep    = list(range(n))   # representative (sharpest) per cluster

    prev_frame = load_frame_fn(frame_paths[0])

    for i in range(1, n):
        curr_frame = load_frame_fn(frame_paths[i])

        # Stage 1 — optical flow
        flow_mag = compute_optical_flow_magnitude(prev_frame, curr_frame)

        if is_flow_candidate(flow_mag):
            # Stage 2 — embedding similarity
            if is_confirmed_duplicate(embeddings[i-1], embeddings[i]):
                sim = cosine_similarity(embeddings[i-1], embeddings[i])
                # Merge into same cluster as previous
                prev_cluster          = cluster_id[i - 1]
                cluster_id[i]         = prev_cluster

                # Update representative to sharpest in cluster
                rep_idx               = cluster_rep[prev_cluster]
                if blur_scores[i] > blur_scores[rep_idx]:
                    cluster_rep[prev_cluster] = i

                results[i]["reason"]        = (
                    f"duplicate_of_{frame_paths[cluster_rep[prev_cluster]].name}"
                    f" (flow={flow_mag:.2f}, sim={sim:.3f})"
                )

        prev_frame = curr_frame

    # Mark non-representatives as duplicates
    for i in range(n):
        c   = cluster_id[i]
        rep = cluster_rep[c]
        if i != rep and cluster_id[i] == cluster_id[rep]:
            results[i]["status"]       = "DISCARD"
            # Update reason to point to actual representative
            results[i]["duplicate_of"] = frame_paths[rep].name
            if not results[i]["reason"]:
                results[i]["reason"]   = f"duplicate_of_{frame_paths[rep].name}"

    return results
