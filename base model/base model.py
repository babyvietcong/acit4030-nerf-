# =============================================================================
# trainnerf.py — NeRF baseline for ACIT4030 (lego & poster)
# =============================================================================
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

from utils.plot_image_grid import image_grid
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)
from utils.helper_functions import (
    generate_rotating_nerf,
    huber,
    show_full_render,
    sample_images_at_mc_locs,
)
from nerfmodel import NeuralRadianceField

from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips

# Windows OpenMP workaround (keep only if you still hit the OMP error)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT  = r"C:\Users\khoac\Downloads\pytorch3d_env-20261005T170027Z-1-001\pytorch3d_env\oblig2\acit4030-data\data"
DATASET    = "lego"      # "lego" | "poster"
DOWNSAMPLE = 2
N_ITER     = 50000
BATCH_SIZE = 4
LR         = 1e-3

VIZ_SIZE  = 100     # intermediate figure resolution
VIZ_PTS   = 96      # samples per ray for the intermediate figure
EVAL_SIZE = 100     # eval render resolution
EVAL_PTS  = 64      # samples per ray for eval


# =============================================================================
# Device
# =============================================================================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# =============================================================================
# Data loaders
# =============================================================================
def _load_image(path, target_size=None):
    """Load an image as a float tensor in [0,1] with shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)


def load_lego_data(data_path, split="train", downsample=2):
    """Synthetic lego (Blender/NeRF format). Returns images, poses, fov_x(rad)."""
    with open(os.path.join(data_path, f"transforms_{split}.json")) as f:
        meta = json.load(f)
    fov_x = meta["camera_angle_x"]  # horizontal FOV in radians

    images, poses, target_size = [], [], None
    for frame in meta["frames"]:
        img_path = os.path.join(data_path, frame["file_path"])
        if not os.path.exists(img_path):
            img_path += ".png"
        img = _load_image(img_path)
        if target_size is None:
            H, W = img.shape[:2]
            target_size = (W // downsample, H // downsample)
        images.append(_load_image(img_path, target_size=target_size))
        poses.append(torch.from_numpy(np.array(frame["transform_matrix"], dtype=np.float32)))
    return torch.stack(images), torch.stack(poses), fov_x


def load_poster_data(data_path, downsample=2, holdout_every=8):
    """Nerfstudio poster. Returns train_imgs, train_poses, test_imgs, test_poses, intrinsics."""
    with open(os.path.join(data_path, "transforms.json")) as f:
        meta = json.load(f)

    W    = meta["w"] // downsample
    H    = meta["h"] // downsample
    fl_x = meta["fl_x"] / downsample
    fl_y = meta["fl_y"] / downsample

    images, poses = [], []
    for frame in meta["frames"]:
        img_path = os.path.join(data_path, frame["file_path"])
        if not os.path.exists(img_path):
            img_path += ".png"
        images.append(_load_image(img_path, target_size=(W, H)))
        poses.append(torch.from_numpy(np.array(frame["transform_matrix"], dtype=np.float32)))

    images = torch.stack(images)
    poses  = torch.stack(poses)

    N = images.shape[0]
    test_idx = torch.arange(0, N, holdout_every)
    mask = torch.ones(N, dtype=torch.bool)
    mask[test_idx] = False

    return (images[mask], poses[mask],
            images[test_idx], poses[test_idx],
            {"fl_x": fl_x, "fl_y": fl_y, "W": W, "H": H})


# =============================================================================
# Pose conversion — NOW HANDLES BOTH CONVENTIONS
# =============================================================================
def nerf_poses_to_p3d_cameras(poses, fov_y, aspect_ratio, device,
                              znear, zfar, convention="opengl"):
    """
    Convert c2w poses to PyTorch3D R, T.

    convention="opengl"  : NeRF/Blender lego  — camera looks down -Z (Y up)
    convention="opencv"  : nerfstudio poster  — camera looks down +Z (Y down)

    PyTorch3D FoVPerspectiveCameras expects to look down +Z with Y up,
    so:
      - OpenGL input: flip Z column.
      - OpenCV input: flip Y and Z columns.
    """
    if convention == "opengl":
        flip = torch.diag(torch.tensor([1., 1., -1., 1.], device=poses.device))
    elif convention == "opencv":
        flip = torch.diag(torch.tensor([1., -1., -1., 1.], device=poses.device))
    else:
        raise ValueError(f"Unknown convention: {convention}")

    c2w_p3d = poses @ flip
    w2c     = torch.inverse(c2w_p3d)

    R = w2c[:, :3, :3]
    T = w2c[:, :3, 3]
    N = poses.shape[0]

    fov_t   = fov_y if isinstance(fov_y, torch.Tensor) else torch.full((N,), fov_y, device=device)
    asp_t   = torch.full((N,), aspect_ratio, device=device)
    znear_t = torch.full((N,), znear,        device=device)
    zfar_t  = torch.full((N,), zfar,         device=device)

    return FoVPerspectiveCameras(
        R=R.to(device), T=T.to(device),
        fov=fov_t, aspect_ratio=asp_t,
        znear=znear_t, zfar=zfar_t,
        degrees=False, device=device,
    )


# =============================================================================
# Load data + build cameras (with auto depth range)
# =============================================================================
if DATASET == "lego":
    train_images, train_poses, fov_x = load_lego_data(
        os.path.join(DATA_ROOT, "lego"), "train", DOWNSAMPLE)
    test_images, test_poses, _ = load_lego_data(
        os.path.join(DATA_ROOT, "lego"), "test", DOWNSAMPLE)

    H, W = train_images.shape[1:3]
    aspect = W / H
    fov = torch.tensor(fov_x)
    convention = "opengl"

    # Auto depth range from camera radii
    cc_norms = train_poses[:, :3, 3].norm(dim=-1)
    r_mean   = cc_norms.mean().item()
    znear = 0.05
    zfar  = r_mean + 3.0

else:  # poster
    train_images, train_poses, test_images, test_poses, intr = load_poster_data(
        os.path.join(DATA_ROOT, "poster"), DOWNSAMPLE)

    W, H = intr["W"], intr["H"]
    aspect = W / H
    fov = torch.tensor(2.0 * np.arctan(H / (2.0 * intr["fl_y"])))
    convention = "opencv"

    cc_norms = train_poses[:, :3, 3].norm(dim=-1)
    r_mean   = cc_norms.mean().item()
    znear = max(0.05, r_mean - 1.5)
    zfar  = r_mean + 1.5

print(f"{DATASET}: train={tuple(train_images.shape)} test={tuple(test_images.shape)} "
      f"fov={np.degrees(fov.item()):.2f}° aspect={aspect:.3f} "
      f"znear={znear:.2f} zfar={zfar:.2f} convention={convention}")

train_cameras = nerf_poses_to_p3d_cameras(
    train_poses, fov, aspect, device, znear, zfar, convention)
test_cameras  = nerf_poses_to_p3d_cameras(
    test_poses,  fov, aspect, device, znear, zfar, convention)

train_images = train_images.to(device)
test_images  = test_images.to(device)

# Sanity: camera radii should be roughly constant
cc = train_cameras.get_camera_center().norm(dim=-1)
print(f"camera radii: min={cc.min().item():.2f} max={cc.max().item():.2f} "
      f"mean={cc.mean().item():.2f}")


# =============================================================================
# Renderers
# =============================================================================
raysampler_mc = MonteCarloRaysampler(
    min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0,
    n_rays_per_image=1024,
    n_pts_per_ray=128,
    min_depth=znear, max_depth=zfar,
)
raymarcher = EmissionAbsorptionRaymarcher(surface_thickness=1e-4)
renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)

# Full-res grid — kept for reference only; avoid using it in memory-tight paths
raysampler_grid = NDCMultinomialRaysampler(
    image_height=H, image_width=W,
    n_pts_per_ray=128,
    min_depth=znear, max_depth=zfar,
)
renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)

# Small renderer for intermediate figures
raysampler_viz = NDCMultinomialRaysampler(
    image_height=VIZ_SIZE, image_width=VIZ_SIZE,
    n_pts_per_ray=VIZ_PTS,
    min_depth=znear, max_depth=zfar,
)
renderer_viz = ImplicitRenderer(raysampler=raysampler_viz, raymarcher=raymarcher)

# Small renderer for eval
raysampler_eval = NDCMultinomialRaysampler(
    image_height=EVAL_SIZE, image_width=EVAL_SIZE,
    n_pts_per_ray=EVAL_PTS,
    min_depth=znear, max_depth=zfar,
)
renderer_eval = ImplicitRenderer(raysampler=raysampler_eval, raymarcher=raymarcher)


# =============================================================================
# Model
# =============================================================================
neural_radiance_field = NeuralRadianceField().to(device)
torch.manual_seed(1)

renderer_mc   = renderer_mc.to(device)
renderer_grid = renderer_grid.to(device)
renderer_viz  = renderer_viz.to(device)
renderer_eval = renderer_eval.to(device)

lr = LR
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr)


# =============================================================================
# Evaluation metrics
# =============================================================================
_ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
_lpips_fn = lpips.LPIPS(net="vgg").to(device)


def psnr_fn(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    return float("inf") if mse == 0 else (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


@torch.no_grad()
def evaluate(nerf, cameras, target_images, eval_size=EVAL_SIZE):
    """Full-image PSNR / SSIM / LPIPS on the test set."""
    nerf.eval()
    psnrs, ssims, lpipss = [], [], []

    for i in range(len(cameras)):
        cam_i = FoVPerspectiveCameras(
            R=cameras.R[i:i+1], T=cameras.T[i:i+1],
            fov=cameras.fov[i:i+1], aspect_ratio=cameras.aspect_ratio[i:i+1],
            znear=cameras.znear[i:i+1], zfar=cameras.zfar[i:i+1],
            degrees=False, device=device,
        )

        rgb_sil, _ = renderer_eval(cameras=cam_i, volumetric_function=nerf)
        rgb = rgb_sil[..., :3][0].clamp(0, 1)

        gt = target_images[i].permute(2, 0, 1).unsqueeze(0)
        gt = F.interpolate(gt, size=(eval_size, eval_size),
                           mode="bilinear", align_corners=False)
        gt = gt[0].permute(1, 2, 0)

        rgb_chw = rgb.permute(2, 0, 1).unsqueeze(0)
        gt_chw  = gt.permute(2, 0, 1).unsqueeze(0)

        psnrs.append(psnr_fn(rgb, gt))
        ssims.append(_ssim_fn(rgb_chw, gt_chw).item())
        lpipss.append(_lpips_fn(rgb_chw * 2 - 1, gt_chw * 2 - 1).item())

        del rgb_sil, rgb, gt, rgb_chw, gt_chw, cam_i
        if i % 4 == 0:
            torch.cuda.empty_cache()

    nerf.train()
    return {"PSNR":  float(np.mean(psnrs)),
            "SSIM":  float(np.mean(ssims)),
            "LPIPS": float(np.mean(lpipss))}


# =============================================================================
# Training loop
# =============================================================================
loss_history_color, loss_history_sil = [], []

for iteration in range(N_ITER):
    if iteration == round(N_ITER * 0.75):
        print("Decreasing LR 10-fold ...")
        optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr * 0.1)

    optimizer.zero_grad()

    batch_idx = torch.randperm(len(train_cameras))[:BATCH_SIZE]
    batch_cameras = FoVPerspectiveCameras(
        R=train_cameras.R[batch_idx],
        T=train_cameras.T[batch_idx],
        znear=train_cameras.znear[batch_idx],
        zfar=train_cameras.zfar[batch_idx],
        aspect_ratio=train_cameras.aspect_ratio[batch_idx],
        fov=train_cameras.fov[batch_idx],
        degrees=False,
        device=device,
    )

    rendered_images_silhouettes, sampled_rays = renderer_mc(
        cameras=batch_cameras,
        volumetric_function=neural_radiance_field,
    )
    rendered_images = rendered_images_silhouettes[..., :3]

    colors_at_rays = sample_images_at_mc_locs(
        train_images[batch_idx],
        sampled_rays.xys,
    )
    color_err = huber(rendered_images, colors_at_rays).abs().mean()
    loss = color_err

    loss_history_color.append(float(color_err))
    loss_history_sil.append(0.0)

    loss.backward()
    optimizer.step()

    del rendered_images_silhouettes, rendered_images, colors_at_rays, sampled_rays
    del batch_cameras, batch_idx

    if iteration % 100 == 0:
        print(f"[iter {iteration:5d}] color_err = {color_err.item():.5f}")

    # --- Periodic eval ---
    if iteration > 0 and iteration % 500 == 0:
        torch.cuda.empty_cache()
        m = evaluate(neural_radiance_field, test_cameras, test_images)
        torch.cuda.empty_cache()
        print(f"  >> eval: PSNR={m['PSNR']:.2f}  "
              f"SSIM={m['SSIM']:.4f}  LPIPS={m['LPIPS']:.4f}")

    # --- Cheap intermediate figure (no-grad, low-res, small renderer) ---
    if iteration % 500 == 0:
        torch.cuda.empty_cache()
        with torch.no_grad():
            show_idx = torch.randperm(len(train_cameras))[:1]
            fig = show_full_render(
                neural_radiance_field,
                FoVPerspectiveCameras(
                    R=train_cameras.R[show_idx],
                    T=train_cameras.T[show_idx],
                    znear=train_cameras.znear[show_idx],
                    zfar=train_cameras.zfar[show_idx],
                    aspect_ratio=train_cameras.aspect_ratio[show_idx],
                    fov=train_cameras.fov[show_idx],
                    degrees=False,
                    device=device,
                ),
                train_images[show_idx][0],
                torch.ones_like(train_images[show_idx][0, ..., :1]),
                renderer_viz,           # <-- small renderer, not renderer_grid
                loss_history_color,
                loss_history_sil,
            )
            fig.savefig(f"intermediate_{iteration}.png")
            plt.close(fig)
            del fig
        torch.cuda.empty_cache()


# =============================================================================
# Final evaluation + rotating visualization
# =============================================================================
torch.cuda.empty_cache()
final = evaluate(neural_radiance_field, test_cameras, test_images)
print(f"\nFINAL[{DATASET}]:  PSNR={final['PSNR']:.2f}  "
      f"SSIM={final['SSIM']:.4f}  LPIPS={final['LPIPS']:.4f}")

torch.cuda.empty_cache()
with torch.no_grad():
    rotating_nerf_frames = generate_rotating_nerf(
        neural_radiance_field,
        train_cameras,
        renderer_viz,          # <-- was renderer_grid — this is the OOM fix
        n_frames=3 * 5,
        device=device,
    )

image_grid(rotating_nerf_frames.clamp(0., 1.).cpu().numpy(),
           rows=3, cols=5, rgb=True, fill=True)
plt.show()