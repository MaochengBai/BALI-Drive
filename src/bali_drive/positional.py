import torch


def position_3d(grid, dim):
    """Factorized sine/cosine encoding, serialized in (time, row, column) order."""
    dim_t = max(2, (dim // 4) // 2 * 2)
    dim_y = max(2, ((dim - dim_t) // 2) // 2 * 2)
    dims = (dim_t, dim_y, dim - dim_t - dim_y)
    if min(dims) < 2 or any(d % 2 for d in dims):
        raise ValueError("Cannot split dimension into three even position widths")
    coords = torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in grid), indexing="ij")
    values = []
    for coord, d in zip(coords, dims):
        omega = 10000.0 ** (-2 * torch.arange(d // 2, dtype=torch.float32) / d)
        phase = coord.reshape(-1, 1) * omega
        values.append(torch.cat((phase.sin(), phase.cos()), dim=-1))
    return torch.cat(values, dim=-1)
