import torch
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    PerspectiveCameras,
    RayBundle,
    ray_bundle_to_ray_points,
)


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=10, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=10, n_hidden_neurons=256, omega0=0.1):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions, omega0=omega0)
        # The dimension of the harmonic embedding.
        embedding_dim = n_harmonic_functions * 2 * 3

        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.linear1 = torch.nn.Linear(embedding_dim, n_hidden_neurons)
        self.linear2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.linear3 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.linear4 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.linear5 = torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons)  # skip
        self.linear6 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.linear7 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.linear8 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)

        # Xavier init on the trunk (helps the shallow layers get usable gradients).
        for m in (self.linear1, self.linear2, self.linear3, self.linear4,
                self.linear5, self.linear6, self.linear7, self.linear8):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=1.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )

        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=1.0),
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )

        # We set the bias of the density layer to -1.5
        # in order to initialize the opacities of the
        # ray points to values close to 0.
        # This is a crucial detail for ensuring convergence
        # of the model.
        self.density_layer[0].bias.data[0] = 0
    def _mlp_forward(self, embeds):
        """8-layer MLP trunk with a skip connection at layer 4."""
        h = torch.nn.functional.leaky_relu(self.linear1(embeds), 0.01)
        h = torch.nn.functional.leaky_relu(self.linear2(h), 0.01)
        h = torch.nn.functional.leaky_relu(self.linear3(h), 0.01)
        h = torch.nn.functional.leaky_relu(self.linear4(h), 0.01)
        h = torch.cat([h, embeds], dim=-1)                          # <-- skip
        h = torch.nn.functional.leaky_relu(self.linear5(h), 0.01)
        h = torch.nn.functional.leaky_relu(self.linear6(h), 0.01)
        h = torch.nn.functional.leaky_relu(self.linear7(h), 0.01)
        h = self.linear8(h)                                         # no activation
        return h
    def _get_densities(self, features):
        """
        Unbounded density (raw Softplus output). The PyTorch3D tutorial's
        `1 - exp(-Softplus(...))` squash caps density at 1.0, which cannot
        represent an opaque surface. Returning the raw Softplus output lets
        density reach 5-20+ inside objects, which the importance sampler
        needs in order to peak sharply and the volume renderer needs to
        fully occlude background.
        """
        return self.density_layer(features)

    def _get_colors(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.

        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        spatial_size = features.shape[:-1]

        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding(
            rays_directions_normed
        )

        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding_expand),
            dim=-1
        )
        return self.color_layer(color_layer_input)

    def forward(
        self,
        ray_bundle: RayBundle,
        **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding(
            rays_points_world
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self._mlp_forward(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.

        # Eval path uses EmissionAbsorptionRaymarcher, which expects densities in [0, 1].
# Apply the tutorial's squash ONLY here, not in forward_points (which is used
# by the hierarchical renderer's own volume_render).
        rays_densities = 1 - torch.exp(-self._get_densities(features))
        # rays_densities.shape = [minibatch x ... x 1]

        rays_colors = self._get_colors(features, ray_bundle.directions)
        # rays_colors.shape = [minibatch x ... x 3]

        return rays_densities, rays_colors

    def forward_points(self, points, view_dirs):
        """
        Point-based entry point used by the hierarchical renderer.
        Args:
            points:    (P, 3)
            view_dirs: (P, 3)
        Returns:
            sigma: (P,)
            rgb:   (P, 3)
        """
        # Harmonic embedding of world positions.
        embeds = self.harmonic_embedding(points)                 # (P, F_xyz)

        # Shared trunk.
        features = self._mlp_forward(embeds)                               # (P, n_hidden)

        # Density branch.
        sigma = self._get_densities(features).squeeze(-1)        # (P,)

        # Color branch — inline, since _get_colors assumes a spatial axis.
        dirs_normed = torch.nn.functional.normalize(view_dirs, dim=-1)   # (P, 3)
        dir_embed   = self.harmonic_embedding(dirs_normed)               # (P, F_dir)
        color_in    = torch.cat((features, dir_embed), dim=-1)           # (P, n_hidden + F_dir)
        rgb         = self.color_layer(color_in)                         # (P, 3)

        return sigma, rgb

    def batched_forward(
        self,
        ray_bundle: RayBundle,
        n_batches: int = 16,
        **kwargs,
    ):
        """
        This function is used to allow for memory efficient processing
        of input rays. The input rays are first split to `n_batches`
        chunks and passed through the `self.forward` function one at a time
        in a for loop. Combined with disabling PyTorch gradient caching
        (`torch.no_grad()`), this allows for rendering large batches
        of rays that do not all fit into GPU memory in a single forward pass.
        In our case, batched_forward is used to export a fully-sized render
        of the radiance field for visualization purposes.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            n_batches: Specifies the number of batches the input rays are split into.
                The larger the number of batches, the smaller the memory footprint
                and the lower the processing speed.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.

        """

        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]

        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors


def _load_image(path, target_size=None):
    """Load image as float tensor in [0,1], shape (H,W,3)."""
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def load_lego_data(data_path, split="train", downsample=2):
    """
    Load the synthetic lego dataset (original NeRF format).
    Returns: images (N,H,W,3), poses (N,4,4), focal (float)
    """
    with open(os.path.join(data_path, f"transforms_{split}.json")) as f:
        meta = json.load(f)

    focal = meta["focal_length"]
    images, poses = [], []
    target_size = None  # set later from first image, then downsample

    for frame in meta["frames"]:
        # file_path often like 'train/r_0' -> add .png if needed
        rel = frame["file_path"]
        img_path = os.path.join(data_path, rel)
        if not os.path.exists(img_path):
            img_path = img_path + ".png"
        img = _load_image(img_path)

        if target_size is None:
            H, W = img.shape[:2]
            target_size = (W // downsample, H // downsample)
            focal = focal / downsample

        img = _load_image(img_path, target_size=target_size)
        images.append(img)
        poses.append(torch.from_numpy(np.array(frame["transform_matrix"], dtype=np.float32)))

    images = torch.stack(images)   # (N,H,W,3)
    poses  = torch.stack(poses)    # (N,4,4)
    return images, poses, float(focal)


def load_poster_data(data_path, split=None, downsample=2):
    """
    Load the nerfstudio poster dataset.
    If split is None, uses 'transforms.json'; otherwise transforms_{split}.json.
    Returns: images (N,H,W,3), poses (N,4,4), intrinsics (dict)
    """
    fname = "transforms.json" if split is None else f"transforms_{split}.json"
    with open(os.path.join(data_path, fname)) as f:
        meta = json.load(f)

    # Nerfstudio may scale intrinsics for downsampling via 'k1', etc.
    fl_x = meta["fl_x"] / downsample
    fl_y = meta["fl_y"] / downsample
    cx   = meta["cx"]   / downsample
    cy   = meta["cy"]   / downsample
    W    = meta["w"] // downsample
    H    = meta["h"] // downsample

    images, poses = [], []
    target_size = (W, H)

    for frame in meta["frames"]:
        img_path = os.path.join(data_path, frame["file_path"])
        if not os.path.exists(img_path):
            img_path = img_path + ".png"
        img = _load_image(img_path, target_size=target_size)
        images.append(img)
        poses.append(torch.from_numpy(np.array(frame["transform_matrix"], dtype=np.float32)))

    images = torch.stack(images)
    poses  = torch.stack(poses)
    intrinsics = {"fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy, "w": W, "h": H}
    return images, poses, intrinsics


def cameras_from_lego(poses, focal, image_size):
    """FoVPerspectiveCameras for the lego dataset."""
    H, W = image_size
    # fov in radians from focal (assuming centered principal point)
    fov = 2 * np.arctan(W / (2 * focal))
    R, T = [], []
    for c2w in poses:
        # NeRF c2w is OpenGL; PyTorch3D wants R, T for world->view (i.e., view->world inverse)
        w2c = torch.inverse(c2w)
        R.append(w2c[:3, :3].T)          # PyTorch3D uses row-major R
        T.append(w2c[:3, 3])
    R = torch.stack(R)
    T = torch.stack(T)
    return FoVPerspectiveCameras(fov=fov, R=R, T=T, degrees=False)


def cameras_from_poster(poses, intrinsics, device="cpu"):
    """PerspectiveCameras with per-image intrinsics for poster dataset."""
    W, H = intrinsics["w"], intrinsics["h"]
    focal_length = torch.tensor([[intrinsics["fl_x"], intrinsics["fl_y"]]], device=device)
    principal_point = torch.tensor([[intrinsics["cx"], intrinsics["cy"]]], device=device)
    image_size = torch.tensor([[W, H]], device=device)

    R, T = [], []
    for c2w in poses:
        w2c = torch.inverse(c2w)
        R.append(w2c[:3, :3].T)
        T.append(w2c[:3, 3])
    R = torch.stack(R)
    T = torch.stack(T)

    # Broadcast to N cameras
    N = len(poses)
    return PerspectiveCameras(
        focal_length=focal_length.repeat(N, 1),
        principal_point=principal_point.repeat(N, 1),
        image_size=image_size.repeat(N, 1),
        R=R, T=T, in_ndc=False, device=device,
    )


def get_rays(H, W, focal, c2w):
    """Standard NeRF ray generation (assumes principal point at center)."""
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy",
    )
    dirs = torch.stack([(i - W * 0.5) / focal,
                        -(j - H * 0.5) / focal,
                        -torch.ones_like(i)], dim=-1)
    rays_d = (dirs @ c2w[:3, :3].T)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_poster(H, W, fl_x, fl_y, cx, cy, c2w):
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy",
    )
    dirs = torch.stack([(i - cx) / fl_x,
                        -(j - cy) / fl_y,
                        -torch.ones_like(i)], dim=-1)
    rays_d = dirs @ c2w[:3, :3].T
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d