import os
import sys
import csv
import json
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FLAGS_ALL = ["white_bg", "exp_lr", "mse_loss", "tight_bounds",
             "pe_input", "anneal_pe", "deeper"]


# =============================================================================
# Arguments
# =============================================================================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=os.environ.get(
        "NERF_DATA_ROOT",
        r"C:\Users\khoac\Downloads\pytorch3d_env-20261005T170027Z-1-001"
        r"\pytorch3d_env\oblig2\acit4030-data\data"))
    p.add_argument("--dataset", default="poster", choices=["lego", "poster"])
    p.add_argument("--variant", default="improved", choices=["baseline", "improved"])
    p.add_argument("--enable", nargs="*", default=[], choices=FLAGS_ALL)
    p.add_argument("--disable", nargs="*", default=[], choices=FLAGS_ALL)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--n_iter", type=int, default=40000) #change accrodingly
    p.add_argument("--batch_size", type=int, default=4) #reduce if memory doesnt allow
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rays_per_img", type=int, default=1024)#reduce if memory doesnt allow
    p.add_argument("--pts_per_ray", type=int, default=192)
    p.add_argument("--viz_size", type=int, default=200)
    p.add_argument("--viz_pts", type=int, default=128)
    p.add_argument("--eval_size", type=int, default=200)
    p.add_argument("--eval_pts", type=int, default=128)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--n_val_images", type=int, default=8)
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--n_harmonic_pos", type=int, default=10)
    p.add_argument("--n_harmonic_dir", type=int, default=4)
    p.add_argument("--anneal_frac", type=float, default=0.3,
                   help="fraction of training over which PE bands are enabled")
    p.add_argument("--convention", default=None, choices=["opengl", "opencv"],
                   help="camera convention of transform_matrix (default: opengl)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--compare", nargs="+", default=None,
                   help="result dirs to compare, then exit")
    return p.parse_known_args()[0]


def compare_runs(dirs):
    keys = ["psnr", "ssim", "ms_ssim", "lpips_vgg", "lpips_alex", "mip_avg",
            "sil_iou", "render_s"]
    print(f"{'run':32s}" + "".join(f"{k:>16s}" for k in keys))
    for d in dirs:
        with open(os.path.join(d, "metrics.json")) as f:
            m = json.load(f)
        s = m["test"]["summary"]
        row = f"{os.path.basename(os.path.normpath(d)):32s}"
        for k in keys:
            row += (f"{s[k]['mean']:9.4f}±{s[k]['std']:.3f}" if k in s else f"{'-':>16s}")
        print(row)


args = get_args()
if args.compare:
    compare_runs(args.compare)
    sys.exit(0)

# Heavy imports after the --compare early exit
from utils.plot_image_grid import image_grid
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
from utils.helper_functions import (
    generate_rotating_nerf,
    huber,
    show_full_render,
    sample_images_at_mc_locs,
)
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure,
)
import lpips

# =============================================================================
# Config
# =============================================================================
cfg = {k: (args.variant == "improved") for k in FLAGS_ALL}
for k in args.enable:
    cfg[k] = True
for k in args.disable:
    cfg[k] = False

DATASET = args.dataset
IS_LEGO = DATASET == "lego"
RUN_NAME = f"{DATASET}_{args.variant}" + (
    "_" + "-".join(f"no_{k}" for k in args.disable) if args.disable else "") + (
    "_" + "-".join(f"with_{k}" for k in args.enable) if args.enable else "")
OUT_DIR = args.out_dir or os.path.join("results", RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_BG = 1.0 if (cfg["white_bg"] and IS_LEGO) else 0.0   # background used while training
EVAL_BG = 1.0 if IS_LEGO else 0.0                          # NeRF-synthetic protocol: white
N_LAYERS = 6 if cfg["deeper"] else 4

print("Run:", RUN_NAME)
print("Flags:", cfg)

# =============================================================================
# Device / seeds
# =============================================================================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def free_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =============================================================================
# Model
# =============================================================================
class HarmonicEmbedding(nn.Module):
    """Sin/cos positional encoding [1] with optional raw input and BARF-style
    coarse-to-fine band weighting [2]."""

    def __init__(self, n_harmonic_functions=10, include_input=True, omega0=1.0):
        super().__init__()
        self.n = n_harmonic_functions
        self.include_input = include_input
        self.register_buffer(
            "frequencies", omega0 * (2.0 ** torch.arange(n_harmonic_functions)))
        self.register_buffer("band_weights", torch.ones(n_harmonic_functions))

    @property
    def out_dim(self):
        return 3 * 2 * self.n + (3 if self.include_input else 0)

    def set_anneal(self, alpha):
        """alpha in [0, n]: band k is weighted (1 - cos(pi * clamp(alpha - k, 0, 1))) / 2
        (BARF [2], Eq. 6). alpha=None -> all bands fully on."""
        if alpha is None:
            self.band_weights.fill_(1.0)
            return
        k = torch.arange(self.n, device=self.band_weights.device, dtype=torch.float32)
        w = 0.5 * (1.0 - torch.cos(torch.clamp(alpha - k, 0.0, 1.0) * math.pi))
        self.band_weights.copy_(w)

    def forward(self, x):
        embed = (x[..., None] * self.frequencies).reshape(*x.shape[:-1], -1)  # [..., 3n]
        w = self.band_weights.repeat(x.shape[-1])                              # matches c*n+k
        parts = [embed.sin() * w, embed.cos() * w]
        if self.include_input:
            parts = [x] + parts
        return torch.cat(parts, dim=-1)


class NeuralRadianceField(nn.Module):
    def __init__(self, n_harmonic_pos=10, n_harmonic_dir=4, n_hidden=256,
                 n_layers=4, include_input=True):
        super().__init__()
        self.emb_pos = HarmonicEmbedding(n_harmonic_pos, include_input)
        self.emb_dir = HarmonicEmbedding(n_harmonic_dir, include_input)
        pd, dd = self.emb_pos.out_dim, self.emb_dir.out_dim

        self.skips = {n_layers // 2 - 1}
        layers = [nn.Linear(pd, n_hidden)]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden + (pd if i in self.skips else 0), n_hidden))
        self.layers = nn.ModuleList(layers)

        # raw density through softplus (as in the PyTorch3D tutorial) -> always >= 0
        self.density_head = nn.Sequential(nn.Linear(n_hidden, 1), nn.Softplus(beta=10.0))
        self.density_head[0].bias.data[0] = -1.5

        self.color_head = nn.Sequential(
            nn.Linear(n_hidden + dd, n_hidden),
            nn.Softplus(beta=10.0),
            nn.Linear(n_hidden, 3),
            nn.Sigmoid(),
        )

    def set_pe_anneal(self, alpha):
        self.emb_pos.set_anneal(alpha)

    @staticmethod
    def _get_densities(raw):
        return 1.0 - (-raw).exp()

    def _get_colors(self, features, rays_directions):
        dirs_n = F.normalize(rays_directions, dim=-1)
        d = self.emb_dir(dirs_n)
        d = d[..., None, :].expand(*features.shape[:-1], d.shape[-1])
        return self.color_head(torch.cat((features, d), dim=-1))

    def forward(self, ray_bundle: RayBundle, **kwargs):
        pts = ray_bundle_to_ray_points(ray_bundle)
        emb = self.emb_pos(pts)
        h = emb
        for i, layer in enumerate(self.layers):
            h = F.softplus(layer(h), beta=10.0)
            if i in self.skips:
                h = torch.cat((h, emb), dim=-1)
        densities = self._get_densities(self.density_head(h))
        colors = self._get_colors(h, ray_bundle.directions)
        return densities, colors

    def batched_forward(self, ray_bundle: RayBundle, n_batches: int = 16, **kwargs):
        n_pts = ray_bundle.lengths.shape[-1]
        spatial = [*ray_bundle.origins.shape[:-1], n_pts]
        tot = ray_bundle.origins.shape[:-1].numel()
        outs = []
        for idx in torch.chunk(torch.arange(tot), n_batches):
            outs.append(self.forward(RayBundle(
                origins=ray_bundle.origins.reshape(-1, 3)[idx],
                directions=ray_bundle.directions.reshape(-1, 3)[idx],
                lengths=ray_bundle.lengths.reshape(-1, n_pts)[idx],
                xys=None,
            )))
        d, c = [torch.cat([b[i] for b in outs], dim=0).view(*spatial, -1) for i in (0, 1)]
        return d, c


class BatchedField(nn.Module):
    """Wrap a NeRF so ImplicitRenderer evaluates it in chunks (low memory at eval)."""

    def __init__(self, nerf, n_batches=32):
        super().__init__()
        self.nerf, self.n_batches = nerf, n_batches

    def forward(self, ray_bundle, **kwargs):
        return self.nerf.batched_forward(ray_bundle, n_batches=self.n_batches)


# =============================================================================
# Data loading
# =============================================================================
def _load_image(path, target_size=None, keep_alpha=False):
    img = Image.open(path).convert("RGBA" if keep_alpha else "RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)


def _resolve(root, rel):
    p = os.path.join(root, rel)
    return p if os.path.exists(p) else p + ".png"


def load_lego_split(data_path, split, downsample):
    with open(os.path.join(data_path, f"transforms_{split}.json")) as f:
        meta = json.load(f)
    images, poses, size = [], [], None
    for fr in meta["frames"]:
        p = _resolve(data_path, fr["file_path"])
        if size is None:
            w, h = Image.open(p).size
            size = (w // downsample, h // downsample)
        images.append(_load_image(p, size, keep_alpha=True))
        poses.append(torch.from_numpy(np.array(fr["transform_matrix"], dtype=np.float32)))
    return torch.stack(images), torch.stack(poses), meta["camera_angle_x"]


def load_poster(data_path, downsample, holdout_every=8):
    with open(os.path.join(data_path, "transforms.json")) as f:
        meta = json.load(f)
    W, H = meta["w"] // downsample, meta["h"] // downsample
    fl_y = meta["fl_y"] / downsample
    images, poses = [], []
    for fr in meta["frames"]:
        images.append(_load_image(_resolve(data_path, fr["file_path"]), (W, H)))
        poses.append(torch.from_numpy(np.array(fr["transform_matrix"], dtype=np.float32)))
    images, poses = torch.stack(images), torch.stack(poses)
    N = images.shape[0]
    test_idx = torch.arange(0, N, holdout_every)
    val_idx = torch.arange(holdout_every // 2, N, holdout_every)
    mask = torch.ones(N, dtype=torch.bool)
    mask[test_idx] = False
    mask[val_idx] = False
    return ((images[mask], poses[mask]), (images[val_idx], poses[val_idx]),
            (images[test_idx], poses[test_idx]), W, H, fl_y)


# =============================================================================
# Cameras
# =============================================================================
def nerf_poses_to_p3d_cameras(poses, fov, aspect, znear, zfar, convention):
    """NeRF/instant-NGP c2w poses -> PyTorch3D cameras.
    PyTorch3D camera frame: +X left, +Y up, +Z forward, row-vector convention
    X_cam = X_world @ R + T, hence R = (w2c rotation)^T."""
    flips = {"opengl": [-1., 1., -1., 1.],   # x right, y up,   z backward
             "opencv": [-1., -1., 1., 1.]}   # x right, y down, z forward
    flip = torch.diag(torch.tensor(flips[convention], dtype=poses.dtype))
    c2w = poses @ flip
    w2c = torch.linalg.inv(c2w)
    R = w2c[:, :3, :3].transpose(1, 2)
    T = w2c[:, :3, 3]
    N = poses.shape[0]
    full = lambda v: torch.full((N,), float(v), device=device)
    return FoVPerspectiveCameras(
        R=R.to(device), T=T.to(device),
        fov=full(fov), aspect_ratio=full(aspect),
        znear=full(znear), zfar=full(zfar),
        degrees=False, device=device)


def _camera_slice(cameras, i):
    return FoVPerspectiveCameras(
        R=cameras.R[i:i + 1], T=cameras.T[i:i + 1],
        fov=cameras.fov[i:i + 1], aspect_ratio=cameras.aspect_ratio[i:i + 1],
        znear=cameras.znear[i:i + 1], zfar=cameras.zfar[i:i + 1],
        degrees=False, device=device)


def sanity_check_cameras(cams, poses, name, check_origin):
    C = cams.get_camera_center()
    err = (C - poses[:, :3, 3].to(C.device)).abs().max().item()
    msg = f"[camera check:{name}] max |center - pose translation| = {err:.2e}"
    if err > 1e-3:
        msg += "  <-- WARNING: camera conversion looks wrong"
    print(msg)
    if check_origin:
        try:
            ndc = cams.transform_points_ndc(torch.zeros(len(cams), 1, 3, device=device))
            off = ndc[..., :2].abs().max().item()
            print(f"[camera check:{name}] scene origin max |NDC xy| = {off:.3f} "
                  f"(should be ~0)" + ("  <-- WARNING" if off > 0.15 else ""))
        except Exception as e:  # noqa
            print(f"[camera check:{name}] origin projection check skipped: {e}")


# =============================================================================
# Load data
# =============================================================================
if IS_LEGO:
    root = os.path.join(args.data_root, "lego")
    tr_img, tr_pose, fov_x = load_lego_split(root, "train", args.downsample)
    te_img, te_pose, _ = load_lego_split(root, "test", args.downsample)
    if os.path.exists(os.path.join(root, "transforms_val.json")):
        va_img, va_pose, _ = load_lego_split(root, "val", args.downsample)
    else:
        va_img, va_pose = te_img, te_pose
        print("No val split found: using test views for checkpoint selection.")
    H, W = tr_img.shape[1:3]
    aspect = W / H
    fov = fov_x
    convention = args.convention or "opengl"
    r_mean = tr_pose[:, :3, 3].norm(dim=-1).mean().item()
    if cfg["tight_bounds"]:
        znear, zfar = max(0.05, r_mean - 2.0), r_mean + 2.0       # NeRF: near=2, far=6
    else:
        znear, zfar = 0.05, r_mean + 3.0
else:
    (tr_img, tr_pose), (va_img, va_pose), (te_img, te_pose), W, H, fl_y = load_poster(
        os.path.join(args.data_root, "poster"), args.downsample)
    aspect = W / H
    fov = 2.0 * math.atan(H / (2.0 * fl_y))
    convention = args.convention or "opengl"   # nerfstudio / instant-NGP format
    r_mean = tr_pose[:, :3, 3].norm(dim=-1).mean().item()
    znear, zfar = max(0.05, r_mean - 1.5), r_mean + 1.5

print(f"{DATASET}: train={tuple(tr_img.shape)} val={tuple(va_img.shape)} "
      f"test={tuple(te_img.shape)} fov={math.degrees(fov):.2f}deg aspect={aspect:.3f} "
      f"znear={znear:.2f} zfar={zfar:.2f} convention={convention}")

train_cameras = nerf_poses_to_p3d_cameras(tr_pose, fov, aspect, znear, zfar, convention)
val_cameras = nerf_poses_to_p3d_cameras(va_pose, fov, aspect, znear, zfar, convention)
test_cameras = nerf_poses_to_p3d_cameras(te_pose, fov, aspect, znear, zfar, convention)
sanity_check_cameras(train_cameras, tr_pose, "train", check_origin=IS_LEGO)


def split_rgba(img):
    if img.shape[-1] == 4:
        return img[..., :3].to(device), img[..., 3:4].to(device)
    return img.to(device), torch.ones_like(img[..., :1]).to(device)


def composite(rgb, a, bg):
    return rgb * a + (1.0 - a) * bg


tr_rgb, train_alpha = split_rgba(tr_img)
va_rgb, va_alpha = split_rgba(va_img)
te_rgb, te_alpha = split_rgba(te_img)
train_gt = composite(tr_rgb, train_alpha, TRAIN_BG)          # training targets


def make_eval_targets(rgb, alpha, size):
    """Composite on EVAL_BG at full-res, then resize to the eval resolution."""
    gt = composite(rgb, alpha, EVAL_BG).permute(0, 3, 1, 2)
    al = alpha.permute(0, 3, 1, 2)
    gt = F.interpolate(gt, size=(size, size), mode="bilinear",
                       align_corners=False, antialias=True)
    al = F.interpolate(al, size=(size, size), mode="bilinear",
                       align_corners=False, antialias=True)
    return gt.permute(0, 2, 3, 1).clamp(0, 1), al.permute(0, 2, 3, 1).clamp(0, 1)


val_gt, val_gt_alpha = make_eval_targets(va_rgb, va_alpha, args.eval_size)
test_gt, test_gt_alpha = make_eval_targets(te_rgb, te_alpha, args.eval_size)
trsub_gt, trsub_alpha = make_eval_targets(tr_rgb, train_alpha, args.eval_size)

# =============================================================================
# Renderers
# =============================================================================
def mc_raysampler(n_rays, n_pts):
    return MonteCarloRaysampler(
        min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0,
        n_rays_per_image=n_rays, n_pts_per_ray=n_pts,
        min_depth=znear, max_depth=zfar, stratified_sampling=True)


def grid_raysampler(h, w, n_pts):
    return NDCMultinomialRaysampler(
        image_height=h, image_width=w, n_pts_per_ray=n_pts,
        min_depth=znear, max_depth=zfar, stratified_sampling=False)


raymarcher = EmissionAbsorptionRaymarcher()
renderer_mc = ImplicitRenderer(
    raysampler=mc_raysampler(args.rays_per_img, args.pts_per_ray), raymarcher=raymarcher).to(device)
renderer_viz = ImplicitRenderer(
    raysampler=grid_raysampler(args.viz_size, args.viz_size, args.viz_pts), raymarcher=raymarcher).to(device)
renderer_eval = ImplicitRenderer(
    raysampler=grid_raysampler(args.eval_size, args.eval_size, args.eval_pts), raymarcher=raymarcher).to(device)

# =============================================================================
# Model + optimiser
# =============================================================================
nerf = NeuralRadianceField(
    n_harmonic_pos=args.n_harmonic_pos, n_harmonic_dir=args.n_harmonic_dir,
    n_hidden=args.n_hidden, n_layers=N_LAYERS, include_input=cfg["pe_input"]).to(device)
n_params = sum(p.numel() for p in nerf.parameters())
print(f"NeRF parameters: {n_params:,}")
optimizer = torch.optim.Adam(nerf.parameters(), lr=args.lr)
field_eval = BatchedField(nerf, n_batches=32)

# =============================================================================
# Metrics
# =============================================================================
class MetricSuite:
    """Per-image image-quality metrics. Inputs are [H, W, 3] tensors in [0, 1]."""

    def __init__(self, nets=("vgg", "alex")):
        self.lp = {}
        for n in nets:
            try:
                self.lp[n] = lpips.LPIPS(net=n, verbose=False).to(device).eval()
            except Exception as e:  # noqa
                print(f"LPIPS-{n} unavailable: {e}")

    @torch.no_grad()
    def __call__(self, pred, gt, pred_sil=None, gt_alpha=None, full=True):
        p, g = pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
        out = {}
        mse = F.mse_loss(pred, gt).clamp_min(1e-10)
        out["psnr"] = float(-10.0 * torch.log10(mse))
        out["ssim"] = float(structural_similarity_index_measure(p, g, data_range=1.0))
        if not full:
            return out
        try:
            out["ms_ssim"] = float(multiscale_structural_similarity_index_measure(
                p, g, data_range=1.0, normalize="relu"))
        except Exception:
            out["ms_ssim"] = float("nan")
        for n, net in self.lp.items():
            out[f"lpips_{n}"] = float(net(p * 2 - 1, g * 2 - 1))
        l = out.get("lpips_alex", out.get("lpips_vgg"))
        if l is not None:   # Mip-NeRF "average" metric [6] (lower is better)
            out["mip_avg"] = float((mse.item() * math.sqrt(max(1 - out["ssim"], 1e-12)) * l) ** (1 / 3))
        if pred_sil is not None and gt_alpha is not None:
            pm, gm = pred_sil > 0.5, gt_alpha > 0.5
            union = (pm | gm).sum().item()
            out["sil_iou"] = (pm & gm).sum().item() / union if union > 0 else 1.0
            out["alpha_mae"] = float((pred_sil - gt_alpha).abs().mean())
        return out


suite = MetricSuite()


def summarize(per_image):
    keys = [k for k in per_image[0].keys()]
    return {k: {"mean": float(np.nanmean([d[k] for d in per_image])),
                "std": float(np.nanstd([d[k] for d in per_image]))} for k in keys}


@torch.no_grad()
def evaluate(field, cameras, gt_rgb, gt_alpha, full=True, max_images=None, keep=0):
    n_total = len(cameras)
    n = n_total if max_images is None else min(n_total, max_images)
    idxs = np.linspace(0, n_total - 1, n).round().astype(int)
    per_image, renders = [], []
    for i in idxs:
        sync(); t0 = time.time()
        rgba, _ = renderer_eval(cameras=_camera_slice(cameras, int(i)), volumetric_function=field)
        sync(); dt = time.time() - t0
        rgb, sil = rgba[0, ..., :3], rgba[0, ..., 3:4]
        pred = (rgb + (1.0 - sil) * EVAL_BG).clamp(0, 1)
        m = suite(pred, gt_rgb[i], sil if IS_LEGO else None,
                  gt_alpha[i] if IS_LEGO else None, full=full)
        m["render_s"] = dt
        m["index"] = int(i)
        per_image.append(m)
        if len(renders) < keep:
            renders.append((pred.cpu(), gt_rgb[i].cpu()))
        del rgba, rgb, sil, pred
        free_mem()
    summ = summarize([{k: v for k, v in d.items() if k != "index"} for d in per_image])
    return {"summary": summ, "per_image": per_image}, renders


def print_summary(title, res):
    s = res["summary"]
    parts = [f"{k}={s[k]['mean']:.4f}" for k in
             ["psnr", "ssim", "ms_ssim", "lpips_vgg", "lpips_alex", "mip_avg", "sil_iou"] if k in s]
    print(f"{title}: " + "  ".join(parts))


# =============================================================================
# Plot helpers
# =============================================================================
def save_curves(path, loss_color, loss_sil, val_hist):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    k = 50
    lc = np.convolve(loss_color, np.ones(k) / k, mode="valid") if len(loss_color) > k else loss_color
    ax[0].plot(lc, label="photometric")
    if IS_LEGO:
        ls = np.convolve(loss_sil, np.ones(k) / k, mode="valid") if len(loss_sil) > k else loss_sil
        ax[0].plot(ls, label="silhouette")
    ax[0].set_yscale("log"); ax[0].set_xlabel("iteration"); ax[0].set_title("Training loss (smoothed)")
    ax[0].legend()
    if val_hist:
        its = [h["iter"] for h in val_hist]
        ax[1].plot(its, [h["psnr"] for h in val_hist], "o-", label="val PSNR")
        ax2 = ax[1].twinx()
        ax2.plot(its, [h["ssim"] for h in val_hist], "s--", color="C1", label="val SSIM")
        ax[1].set_xlabel("iteration"); ax[1].set_ylabel("PSNR [dB]"); ax2.set_ylabel("SSIM")
        ax[1].set_title("Validation metrics")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_qualitative(path, renders):
    k = len(renders)
    fig, axes = plt.subplots(k, 3, figsize=(9, 3 * k))
    axes = np.atleast_2d(axes)
    for r, (p, g) in enumerate(renders):
        axes[r, 0].imshow(g.numpy()); axes[r, 1].imshow(p.numpy())
        axes[r, 2].imshow((p - g).abs().mean(-1).numpy(), cmap="inferno", vmin=0, vmax=0.5)
        for c, t in enumerate(["Ground truth", "NeRF", "|error|"]):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(t)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
# Training
# =============================================================================
loss_color, loss_sil, val_hist = [], [], []
best_psnr, best_path = -1.0, os.path.join(OUT_DIR, "best.pt")
lr_drop_iters = {int(args.n_iter * 0.66), int(args.n_iter * 0.90)}
anneal_end = max(1, int(args.anneal_frac * args.n_iter))
t_train0 = time.time()

for it in range(args.n_iter):
    # --- schedules ---------------------------------------------------------
    if cfg["exp_lr"]:
        lr = args.lr * (0.1 ** (it / args.n_iter))                  # [1]
        for g in optimizer.param_groups:
            g["lr"] = lr
    elif it in lr_drop_iters:
        for g in optimizer.param_groups:
            g["lr"] *= 0.1
        print(f"  [iter {it}] LR -> {optimizer.param_groups[0]['lr']:.2e}")
    if cfg["anneal_pe"]:
        nerf.set_pe_anneal(torch.tensor(args.n_harmonic_pos * min(1.0, it / anneal_end),
                                        device=device))             # [2]
    optimizer.zero_grad()

    # --- batch -------------------------------------------------------------
    bi = torch.randperm(len(train_cameras))[:args.batch_size]
    batch_cameras = FoVPerspectiveCameras(
        R=train_cameras.R[bi], T=train_cameras.T[bi],
        znear=train_cameras.znear[bi], zfar=train_cameras.zfar[bi],
        aspect_ratio=train_cameras.aspect_ratio[bi], fov=train_cameras.fov[bi],
        degrees=False, device=device)

    rgba, rays = renderer_mc(cameras=batch_cameras, volumetric_function=nerf)
    rgb, sil = rgba[..., :3], rgba[..., 3:4]
    pred = rgb + (1.0 - sil) * TRAIN_BG
    gt = sample_images_at_mc_locs(train_gt[bi], rays.xys)

    if cfg["mse_loss"]:
        color_err = F.mse_loss(pred, gt)
    else:
        color_err = huber(pred, gt).abs().mean()

    if IS_LEGO:
        a_gt = sample_images_at_mc_locs(train_alpha[bi], rays.xys)
        sil_err = huber(sil, a_gt).abs().mean()
        loss = color_err + 0.1 * sil_err
        loss_sil.append(float(sil_err))
    else:
        loss = color_err
        loss_sil.append(0.0)
    loss_color.append(float(color_err))

    loss.backward()
    optimizer.step()
    del rgba, rgb, sil, pred, gt, rays, batch_cameras

    if it % 100 == 0:
        print(f"[iter {it:5d}] color_err={loss_color[-1]:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"elapsed={time.time() - t_train0:.0f}s")

    # --- validation + best checkpoint ---------------------------------------
    if it > 0 and it % args.eval_every == 0:
        free_mem()
        res, _ = evaluate(field_eval, val_cameras, val_gt, val_gt_alpha,
                          full=False, max_images=args.n_val_images)
        v = {"iter": it, "psnr": res["summary"]["psnr"]["mean"],
             "ssim": res["summary"]["ssim"]["mean"]}
        val_hist.append(v)
        print(f"  >> val: PSNR={v['psnr']:.2f}  SSIM={v['ssim']:.4f}")
        if v["psnr"] > best_psnr:
            best_psnr = v["psnr"]
            torch.save({"model": nerf.state_dict(), "iter": it, "cfg": cfg}, best_path)
        free_mem()

    # --- intermediate figure -------------------------------------------------
    if it % args.eval_every == 0:
        free_mem()
        try:
            with torch.no_grad():
                si = int(torch.randperm(len(train_cameras))[0])
                fig = show_full_render(
                    nerf, _camera_slice(train_cameras, si),
                    train_gt[si], train_alpha[si], renderer_viz,
                    loss_color, loss_sil)
                fig.savefig(os.path.join(OUT_DIR, f"intermediate_{it}.png"))
                plt.close(fig)
        except Exception as e:  # noqa
            print(f"  (intermediate figure skipped: {e})")
        free_mem()

train_time = time.time() - t_train0
print(f"Training finished in {train_time / 60:.1f} min")

# =============================================================================
# Final quantitative evaluation (test set touched only here)
# =============================================================================
nerf.set_pe_anneal(None)
if os.path.exists(best_path):
    ck = torch.load(best_path, map_location=device)
    nerf.load_state_dict(ck["model"])
    print(f"Loaded best checkpoint (iter {ck['iter']}, val PSNR {best_psnr:.2f})")
else:  # n_iter < eval_every: keep final weights
    torch.save({"model": nerf.state_dict(), "iter": args.n_iter, "cfg": cfg}, best_path)

free_mem()
test_res, renders = evaluate(field_eval, test_cameras, test_gt, test_gt_alpha, full=True, keep=4)
train_res, _ = evaluate(field_eval, train_cameras, trsub_gt, trsub_alpha,
                        full=False, max_images=args.n_val_images)
print_summary(f"\nFINAL TEST [{RUN_NAME}]", test_res)
print(f"train-view PSNR (generalisation gap check): "
      f"{train_res['summary']['psnr']['mean']:.2f} dB")
print(f"render time/image @ {args.eval_size}px: "
      f"{test_res['summary']['render_s']['mean'] * 1000:.0f} ms")

metrics = {
    "run": RUN_NAME, "flags": cfg, "args": vars(args), "convention": convention,
    "znear": znear, "zfar": zfar, "n_params": n_params,
    "train_time_s": train_time, "best_val_psnr": best_psnr,
    "eval_protocol": {"size": args.eval_size, "pts_per_ray": args.eval_pts,
                      "background": "white" if EVAL_BG == 1.0 else "black"},
    "test": test_res, "train_subset": train_res["summary"], "val_history": val_hist,
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(OUT_DIR, "per_image.csv"), "w", newline="") as f:
    cols = list(test_res["per_image"][0].keys())
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(test_res["per_image"])

save_curves(os.path.join(OUT_DIR, "curves.png"), loss_color, loss_sil, val_hist)
save_qualitative(os.path.join(OUT_DIR, "qualitative.png"), renders)

# =============================================================================
# Rotating visualisation
# =============================================================================
free_mem()
try:
    with torch.no_grad():
        frames = generate_rotating_nerf(
            nerf, train_cameras, renderer_viz, n_frames=3 * 5, device=device)
    image_grid(frames.clamp(0., 1.).cpu().numpy(), rows=3, cols=5, rgb=True, fill=True)
    plt.savefig(os.path.join(OUT_DIR, "rotating.png"))
    plt.close("all")
except Exception as e:  # noqa
    print(f"(rotating visualisation skipped: {e})")

print(f"\nAll outputs written to: {OUT_DIR}")