"""Memorization + fidelity assessment for CIFAR-10 samples.

Memorization: unlike qm9's aligned-RMSD (molecules need rotation+permutation
invariance), CIFAR-10 images are already canonically oriented, so "distance to
a training point" is plain pixel-space L2 -- brute-force torch.cdist against
the exact N_TRAIN-image subset finetune_cifar10_ddpm.py trained on (loaded via
the same checkpoints/train_idx.npy it saved) is exact and cheap (thousands of
images x 3072 dims, not millions). d1/d2 (nearest / second-nearest distance)
and the memorized-ratio-threshold classification are the direct CIFAR-10
analogue of assess_memorization_qm9.py's classify_molecules.

NOTE ON PIXEL SPACE: this model (google/ddpm-cifar10-32, finetuned) denoises
RGB pixels directly -- there is no VAE/latent step. That matters for the d1/d2
read: raw pixel-space L2 is sensitive to small spatial shifts/color jitter in
a way a latent-space or perceptual distance would not be, so a "good" (not
memorized) sample that's merely stylistically similar to some training image
can still land at a deceptively low L2 -- worth eyeballing checkpoints/
nearest_neighbor_grid.png rather than trusting the ratio alone.

Fidelity: FID between generated samples and the CIFAR-10 held-out test set,
via torchvision's ImageNet-pretrained InceptionV3 pool features (standard
practice for CIFAR-10 FID despite the native 32x32 resolution -- images are
upsampled to 299x299 for the Inception forward pass).
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from cifar10_data import load_cifar10

SAMPLES_PATH = os.environ.get("SAMPLES_PATH", "checkpoints/samples.pt")
TRAIN_IDX_PATH = os.environ.get("TRAIN_IDX_PATH", "checkpoints/train_idx.npy")
OUT_PATH = os.environ.get("OUT_PATH", "checkpoints/memorization_results.pt")
PLOT_PATH = os.environ.get("PLOT_PATH", "checkpoints/memorization_plot.png")
NN_GRID_PATH = os.environ.get("NN_GRID_PATH", "checkpoints/nearest_neighbor_grid.png")
MEMORIZED_RATIO_THRESHOLD = float(os.environ.get("MEMORIZED_RATIO_THRESHOLD", 1.0 / 3.0))
N_HELDOUT = int(os.environ.get("N_HELDOUT", 500))
COMPUTE_FID = os.environ.get("COMPUTE_FID", "1") == "1"
DEVICE = os.environ.get("DEVICE", "cpu")


def nearest_neighbor_d1_d2(query, bank):
    """query: [B, D], bank: [N, D] (both flat pixel vectors). Returns
    (d1 [B], d2 [B], nn_idx [B]) -- L2 distance to nearest and 2nd-nearest
    bank point, chunked over the query batch to bound memory."""
    d1 = torch.empty(query.shape[0])
    d2 = torch.empty(query.shape[0])
    nn_idx = torch.empty(query.shape[0], dtype=torch.long)
    chunk = 256
    for start in range(0, query.shape[0], chunk):
        q = query[start:start + chunk]
        d = torch.cdist(q, bank)
        top2 = torch.topk(d, 2, dim=1, largest=False)
        d1[start:start + chunk] = top2.values[:, 0]
        d2[start:start + chunk] = top2.values[:, 1]
        nn_idx[start:start + chunk] = top2.indices[:, 0]
    return d1, d2, nn_idx


def classify_images(n_total, d1, d2, ratio_threshold=MEMORIZED_RATIO_THRESHOLD):
    """Direct CIFAR-10 analogue of qm9's classify_molecules -- no 'broken'
    bucket here (no stability filter for images), just memorized vs good."""
    ratio = d1 / d2
    is_memorized = ratio < ratio_threshold
    n_memorized = int(is_memorized.sum())
    return {
        "n_total": n_total, "n_memorized": n_memorized, "n_good": n_total - n_memorized,
        "frac_memorized": n_memorized / n_total, "frac_good": (n_total - n_memorized) / n_total,
        "is_memorized": is_memorized,
    }


def select_diverse_indices(order, nn_idx, n_show):
    """Walk `order` (a ranking of generated-sample indices, most-relevant
    first) and keep the first n_show whose nearest-training-image (nn_idx)
    hasn't been shown yet in this group -- otherwise the grid just shows the
    same few training images repeated, since many generated samples collapse
    onto the same handful of heavily-memorized training points."""
    seen_targets = set()
    picked = []
    for i in order:
        idx = int(nn_idx[i])
        if idx in seen_targets:
            continue
        seen_targets.add(idx)
        picked.append(int(i))
        if len(picked) == n_show:
            break
    return picked


def save_nn_grid(x_gen, X_train, nn_idx, indices, path):
    if len(indices) == 0:
        print(f"skipping {path} -- no candidates in this group", flush=True)
        return
    pairs = []
    for i in indices:
        pairs.append((x_gen[i] + 1) / 2)
        pairs.append((X_train[nn_idx[i]] + 1) / 2)
    save_image(torch.stack(pairs), path, nrow=2)
    print(f"saved {len(indices)} generated/nearest-train pairs to {path}", flush=True)


def grid_path_with_suffix(path, suffix):
    root, ext = os.path.splitext(path)
    return f"{root}_{suffix}{ext}"


def compute_fid(gen_01, real_01, device):
    """gen_01, real_01: [N, 3, 32, 32] in [0, 1]. Returns FID (float) or None
    if torchvision's pretrained Inception weights aren't available (e.g. no
    internet on this node -- download them locally and stage alongside the
    DDPM checkpoint, same as download_model.py, if this fails on the cluster)."""
    from scipy import linalg
    import torch.nn.functional as Fn
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    except Exception as e:
        print(f"skipping FID -- could not load pretrained InceptionV3: {e}", flush=True)
        return None
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()

    def features(x):
        # gen_01/real_01 come in on CPU (torch.load(..., map_location="cpu")) -- move to
        # device before the mean/std normalization below, which builds its tensors directly
        # on device (verified: skipping this raises "Expected all tensors to be on the same
        # device, but found at least two devices, cuda:0 and cpu").
        x = Fn.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
        feats = []
        with torch.no_grad():
            for start in range(0, x.shape[0], 128):
                feats.append(net(x[start:start + 128]).cpu())
        return torch.cat(feats).numpy()

    f_gen = features(gen_01)
    f_real = features(real_01)
    mu_gen, mu_real = f_gen.mean(0), f_real.mean(0)
    cov_gen, cov_real = np.cov(f_gen, rowvar=False), np.cov(f_real, rowvar=False)
    diff = mu_gen - mu_real
    covmean, _ = linalg.sqrtm(cov_gen @ cov_real, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(cov_gen + cov_real - 2 * covmean))
    return fid


if __name__ == "__main__":
    samples = torch.load(SAMPLES_PATH, map_location="cpu")
    x_gen = samples["x_gen"]  # [-1, 1]
    n_gen = x_gen.shape[0]
    print(f"loaded {n_gen} generated images from {SAMPLES_PATH}", flush=True)

    images, _ = load_cifar10(split="train")
    train_idx = np.load(TRAIN_IDX_PATH)
    X_train = images[train_idx]
    print(f"comparing against {len(X_train)} training images (checkpoints/train_idx.npy)", flush=True)

    test_images, _ = load_cifar10(split="test")
    rng = np.random.default_rng(0)
    heldout_idx = rng.choice(len(test_images), size=min(N_HELDOUT, len(test_images)), replace=False)
    X_heldout = test_images[heldout_idx]

    gen_flat = x_gen.flatten(1)
    train_flat = X_train.flatten(1)
    heldout_flat = X_heldout.flatten(1)

    gen_d1, gen_d2, gen_nn_idx = nearest_neighbor_d1_d2(gen_flat, train_flat)
    heldout_d1, heldout_d2, _ = nearest_neighbor_d1_d2(heldout_flat, train_flat)

    gen_class = classify_images(n_gen, gen_d1, gen_d2)
    heldout_class = classify_images(len(X_heldout), heldout_d1, heldout_d2)
    print(f"[generated]  memorized={gen_class['n_memorized']} ({gen_class['frac_memorized']:.1%})   "
          f"good={gen_class['n_good']} ({gen_class['frac_good']:.1%})  "
          f"(ratio threshold={MEMORIZED_RATIO_THRESHOLD:.3f})", flush=True)
    print(f"[held-out]   memorized={heldout_class['n_memorized']} ({heldout_class['frac_memorized']:.1%})   "
          f"good={heldout_class['n_good']} ({heldout_class['frac_good']:.1%})  "
          f"(ratio threshold={MEMORIZED_RATIO_THRESHOLD:.3f})", flush=True)

    # frac_memorized above is "what share of generated samples are memorized copies" --
    # a separate and often much smaller number is "what share of the *training set* those
    # copies actually cover" (many memorized samples can collapse onto the same few
    # training images). n_train_memorized/frac_train_memorized answers that.
    n_train = len(X_train)
    train_memorized_idx = torch.unique(gen_nn_idx[gen_class["is_memorized"]])
    n_train_memorized = int(train_memorized_idx.numel())
    frac_train_memorized = n_train_memorized / n_train
    print(f"[training set]  {n_train_memorized}/{n_train} unique training images "
          f"were memorized ({frac_train_memorized:.1%} of the training set)", flush=True)

    fid = None
    if COMPUTE_FID:
        fid = compute_fid((x_gen + 1) / 2, (X_heldout + 1) / 2, DEVICE)
        if fid is not None:
            print(f"FID (generated vs. held-out test set) = {fid:.3f}", flush=True)

    torch.save({
        "gen_d1": gen_d1, "gen_d2": gen_d2, "gen_nn_idx": gen_nn_idx,
        "heldout_d1": heldout_d1, "heldout_d2": heldout_d2,
        "gen_classification": gen_class, "heldout_classification": heldout_class,
        "memorized_ratio_threshold": MEMORIZED_RATIO_THRESHOLD, "fid": fid,
        "n_train": n_train, "n_train_memorized": n_train_memorized,
        "frac_train_memorized": frac_train_memorized,
    }, OUT_PATH)
    print(f"saved results to {OUT_PATH}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(gen_d1.numpy(), bins=30, alpha=0.6, density=True, color="steelblue", label="generated")
    axes[0].hist(heldout_d1.numpy(), bins=30, alpha=0.6, density=True, color="darkorange", label="held-out (real)")
    axes[0].set(xlabel="pixel-space L2 distance to nearest training image", ylabel="density",
                title="Memorization (pixel-space NN distance)")
    axes[0].legend()

    ratio_gen = (gen_d1 / gen_d2).numpy()
    ratio_heldout = (heldout_d1 / heldout_d2).numpy()
    axes[1].hist(ratio_gen, bins=30, alpha=0.6, density=True, color="steelblue", label="generated")
    axes[1].hist(ratio_heldout, bins=30, alpha=0.6, density=True, color="darkorange", label="held-out (real)")
    axes[1].axvline(MEMORIZED_RATIO_THRESHOLD, color="red", linestyle="--", linewidth=1,
                     label=f"memorized threshold ({MEMORIZED_RATIO_THRESHOLD:.2f})")
    axes[1].set(xlabel="d(1st NN) / d(2nd NN)", ylabel="density", title="Nearest-neighbor ratio")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"saved plot to {PLOT_PATH}", flush=True)

    # Three grids spanning the full memorization spectrum, not just the closest matches --
    # "low_dist"/"mid_dist"/"high_dist" refer to nearest-neighbor distance (d1), i.e.
    # most-memorized / borderline / least-memorized. Within each group, select_diverse_indices
    # skips generated samples whose nearest training image was already shown, since without
    # that the low-distance grid tends to be dominated by a handful of training images that
    # get reproduced many times over.
    n_show = 8
    order_low = torch.argsort(gen_d1).tolist()
    order_high = torch.argsort(gen_d1, descending=True).tolist()
    order_mid = torch.argsort((gen_d1 - gen_d1.median()).abs()).tolist()

    low_idx = select_diverse_indices(order_low, gen_nn_idx, n_show)
    mid_idx = select_diverse_indices(order_mid, gen_nn_idx, n_show)
    high_idx = select_diverse_indices(order_high, gen_nn_idx, n_show)

    save_nn_grid(x_gen, X_train, gen_nn_idx, low_idx, grid_path_with_suffix(NN_GRID_PATH, "low_dist"))
    save_nn_grid(x_gen, X_train, gen_nn_idx, mid_idx, grid_path_with_suffix(NN_GRID_PATH, "mid_dist"))
    save_nn_grid(x_gen, X_train, gen_nn_idx, high_idx, grid_path_with_suffix(NN_GRID_PATH, "high_dist"))
