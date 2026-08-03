"""Run LOCALLY (not on the cluster) to fetch the pretrained pixel-space DDPM
CIFAR-10 checkpoint from Hugging Face and stage it for a manual scp/rsync to
the cluster -- compute nodes here may not have internet, so this avoids the
same fallback-download-on-node trickery the qm9 .sh scripts use.

google/ddpm-cifar10-32: unconditional DDPM, PIXEL SPACE (no VAE/latent step --
the UNet denoises 32x32x3 RGB directly), trained on the full 50k-image CIFAR-10
train split with standard practices. FID 3.17 / IS 9.46 -- good fidelity, but
NOT expected to be highly memorized (the memorization literature -- Carlini et
al. 2023, Gu et al. "On Memorization in Diffusion Models" -- shows strong
memorization needs either a much smaller training subset, ~2-5k images, or
pathological label-conditioning; full-50k runs like this one are close to 0%
memorized). This script only stages the pretrained WEIGHTS as the starting
point -- finetune_cifar10_ddpm.py does the actual small-subset finetuning on
the cluster that induces memorization.
"""
import os
from huggingface_hub import snapshot_download

REPO_ID = "google/ddpm-cifar10-32"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "pretrained", "ddpm-cifar10-32")

if __name__ == "__main__":
    path = snapshot_download(repo_id=REPO_ID, local_dir=LOCAL_DIR)
    print(f"downloaded {REPO_ID} to {path}")
    print(f"scp/rsync this whole directory to the cluster, e.g.:\n"
          f"  rsync -avz {LOCAL_DIR} <cluster>:~/diffusion_so3/cluster-codes/cifar10/pretrained/")
