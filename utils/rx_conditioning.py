import torch
import torch.nn as nn


# ---- Learnable Fourier positional encoding ----

class LearnableFourierEncoding(nn.Module):
    """Fourier positional encoding with learnable frequencies."""

    def __init__(self, input_dim=3, num_freqs=6, init_max_freq_log2=5.0):
        super().__init__()
        self.out_dim = input_dim + input_dim * 2 * num_freqs

        init_freqs = 2.0 ** torch.linspace(0.0, init_max_freq_log2, steps=num_freqs)
        init_freqs = init_freqs.unsqueeze(0).expand(input_dim, -1).clone()
        self.freqs = nn.Parameter(init_freqs)  # (input_dim, num_freqs)

    def forward(self, x):
        scaled = x.unsqueeze(-1) * self.freqs
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)

        shape = x.shape[:-1]
        sin_feat = sin_feat.reshape(*shape, -1)
        cos_feat = cos_feat.reshape(*shape, -1)

        return torch.cat([x, sin_feat, cos_feat], dim=-1)


# ---- FLE component (l, m) features ----

def build_fle_component_features(max_degree):

    features = []
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            l_norm = l / max(max_degree, 1)
            m_norm = m / max(max_degree, 1)
            degree_ratio = l / max(max_degree, 1)
            order_ratio = abs(m) / max(l, 1) if l > 0 else 0.0
            features.append([l_norm, m_norm, degree_ratio, order_ratio])
    return torch.tensor(features, dtype=torch.float32)


# ---- Occupancy grid for visibility probing ----

class OccupancyGrid:
    """3D voxel grid of Gaussian density for fast ray transmittance queries."""

    def __init__(
        self,
        gaussian_xyz,
        gaussian_opacity,
        gaussian_scaling,
        resolution=128,
        padding=2.0,
    ):
        device = gaussian_xyz.device
        N = gaussian_xyz.shape[0]

        xyz_min = gaussian_xyz.min(dim=0)[0] - padding
        xyz_max = gaussian_xyz.max(dim=0)[0] + padding
        self.bounds_min = xyz_min
        self.bounds_max = xyz_max
        self.voxel_size = (xyz_max - xyz_min) / resolution

        grid = torch.zeros(resolution, resolution, resolution, device=device)

        scales = torch.exp(gaussian_scaling)  # (N, 3) — actual scale in world units
        opacity = gaussian_opacity.squeeze(-1)  # (N,)

        for i in range(N):
            pos = gaussian_xyz[i]
            scale = scales[i]
            opa = opacity[i].item()

            radius = 2.0 * scale  # 2 sigma covers ~95%

            lo = ((pos - radius - xyz_min) / self.voxel_size).long().clamp(0, resolution - 1)
            hi = ((pos + radius - xyz_min) / self.voxel_size).long().clamp(0, resolution - 1)

            # very large Gaussians: just place at center for efficiency
            span = hi - lo + 1
            if span.max() > 10:
                ci = ((pos - xyz_min) / self.voxel_size).long().clamp(0, resolution - 1)
                grid[ci[0], ci[1], ci[2]] += opa
                continue

            ix = torch.arange(lo[0], hi[0] + 1, device=device)
            iy = torch.arange(lo[1], hi[1] + 1, device=device)
            iz = torch.arange(lo[2], hi[2] + 1, device=device)
            gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing='ij')
            voxel_centers = torch.stack([gx, gy, gz], dim=-1).float()
            voxel_world = voxel_centers * self.voxel_size + xyz_min + self.voxel_size / 2

            diff = (voxel_world - pos) / scale.clamp(min=1e-6)
            weight = opa * torch.exp(-0.5 * (diff ** 2).sum(dim=-1))

            grid[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] += weight

        self.grid = grid.clamp(0, 1)

    def ray_transmittance(self, origins, targets, n_samples=16):

        device = origins.device
        N = origins.shape[0]

        if targets.dim() == 1:
            targets = targets.unsqueeze(0).expand(N, -1)

        t_vals = torch.linspace(0.05, 0.95, n_samples, device=device)
        ray_dirs = targets - origins
        sample_points = origins.unsqueeze(1) + t_vals.view(1, -1, 1) * ray_dirs.unsqueeze(1)

        flat_points = sample_points.reshape(-1, 3)

        # normalize to [-1, 1] for grid_sample
        normed = (flat_points - self.bounds_min) / (self.bounds_max - self.bounds_min)
        grid_coords = normed * 2 - 1
        # grid_sample coordinate order: (W=z, H=y, D=x), so flip (x,y,z) -> (z,y,x)
        grid_coords_flipped = grid_coords.flip(-1)
        NS = flat_points.shape[0]
        sample_grid = grid_coords_flipped.view(1, 1, 1, NS, 3)
        grid_input = self.grid.unsqueeze(0).unsqueeze(0)

        densities = torch.nn.functional.grid_sample(
            grid_input, sample_grid,
            mode='nearest', padding_mode='zeros', align_corners=True
        ).view(N, n_samples)

        transmittance = torch.prod(1 - densities.clamp(0, 1), dim=1)
        mean_density = densities.mean(dim=1)

        return transmittance, mean_density


# ---- Propagation-Aware: global + visibility-aware local ----

class RxConditionedFLE_PropAware(nn.Module):
    """Propagation-aware RX conditioning: global + visibility-aware local."""

    def __init__(
        self,
        n_fle_components,
        max_fle_degree=9,
        d_model=64,
        n_freqs=6,
        max_freq_log2=5.0,
        num_channels=2,
        local_hidden=None,
        comp_dim=16,
        n_probe_samples=16,
    ):
        super().__init__()
        self.n_fle_components = n_fle_components
        self.num_channels = num_channels
        self.n_probe_samples = n_probe_samples

        if local_hidden is None:
            local_hidden = max(32, 2 * num_channels)

        # ---- Global branch (per-component) ----
        self.rx_pos_enc = LearnableFourierEncoding(
            input_dim=3, num_freqs=n_freqs, init_max_freq_log2=max_freq_log2
        )
        rx_enc_dim = self.rx_pos_enc.out_dim

        self.register_buffer('struct_features',
                             build_fle_component_features(max_degree=max_fle_degree))

        self.component_embed = nn.Parameter(torch.randn(n_fle_components, comp_dim) * 0.02)

        film_mlp_in = rx_enc_dim + 4 + comp_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(film_mlp_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * num_channels),
        )

        # ---- Local branch (per-Gaussian, visibility-aware) ----
        # input: direction (3) + distance (1) + transmittance (1) + mean_density (1) = 6
        local_input_dim = 6
        self.local_mlp = nn.Sequential(
            nn.Linear(local_input_dim, local_hidden),
            nn.ReLU(),
            nn.Linear(local_hidden, local_hidden),
            nn.ReLU(),
            nn.Linear(local_hidden, 2 * num_channels),
        )

        self.occupancy_grid = None

        self._init_weights()

    def _init_weights(self):
        # zero-init last layers → identity at start
        for mlp in [self.film_mlp, self.local_mlp]:
            last = mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def build_occupancy_grid(
        self,
        gaussian_xyz,
        gaussian_opacity,
        gaussian_scaling,
        resolution=128,
        padding=2.0,
    ):
        """Build the 3D occupancy grid from frozen Gaussians. Call once at start of Phase 2."""
        self.occupancy_grid = OccupancyGrid(
            gaussian_xyz, gaussian_opacity, gaussian_scaling,
            resolution=resolution, padding=padding
        )
        print(f"  Built occupancy grid: {resolution}^3, "
              f"bounds=[{self.occupancy_grid.bounds_min.detach().cpu().numpy()}, "
              f"{self.occupancy_grid.bounds_max.detach().cpu().numpy()}]")

    def forward(
        self,
        fle_coeffs,
        rx_pos,
        gaussian_pos,
    ):
        K = self.n_fle_components
        C = self.num_channels

        # ---- Global branch: (1, K, C) ----
        rx_encoded = self.rx_pos_enc(rx_pos.unsqueeze(0))
        rx_expanded = rx_encoded.expand(K, -1)

        film_input = torch.cat([
            rx_expanded,
            self.struct_features,
            self.component_embed,
        ], dim=-1)

        film_mod = self.film_mlp(film_input)
        film_scale = film_mod[:, :C]
        film_shift = film_mod[:, C:]

        out = (1.0 + film_scale.unsqueeze(0)) * fle_coeffs + film_shift.unsqueeze(0)

        # ---- Local branch (visibility-aware): (N, 1, C) ----
        rel_vec = rx_pos.unsqueeze(0) - gaussian_pos
        dist = rel_vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
        direction = rel_vec / dist

        with torch.no_grad():
            transmittance, mean_density = self.occupancy_grid.ray_transmittance(
                gaussian_pos, rx_pos, n_samples=self.n_probe_samples
            )
        transmittance = transmittance.unsqueeze(1)
        mean_density = mean_density.unsqueeze(1)

        local_input = torch.cat([direction, dist, transmittance, mean_density], dim=-1)
        local_mod = self.local_mlp(local_input)
        local_scale = local_mod[:, :C]
        local_shift = local_mod[:, C:]

        out = (1.0 + local_scale.unsqueeze(1)) * out + local_shift.unsqueeze(1)

        return out
