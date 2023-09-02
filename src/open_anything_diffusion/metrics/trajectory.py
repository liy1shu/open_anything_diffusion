import torch


def normalize_trajectory(pred):  # pred: bs * 1200, traj_len, 3
    pred = pred.reshape(-1, 1200, pred.shape[1], pred.shape[2])
    norm = pred.norm(p=2, dim=-1)
    norm = torch.max(norm, dim=1).values + 1e-6
    pred = pred / norm[:, None, :, None]
    return torch.flatten(pred, start_dim=0, end_dim=1)  # bs * 1200, traj_len, 3


def flow_metrics(pred_flow, gt_flow):
    with torch.no_grad():
        # pred_flow = normalize_trajectory(pred_flow)

        # RMSE
        rmse = (pred_flow - gt_flow).norm(p=2, dim=-1).mean()

        # Cosine similarity, normalized.
        nonzero_gt_flowixs = torch.where(gt_flow.norm(dim=-1) != 0.0)
        gt_flow_nz = gt_flow[nonzero_gt_flowixs]
        pred_flow_nz = pred_flow[nonzero_gt_flowixs]
        cos_dist = torch.cosine_similarity(pred_flow_nz, gt_flow_nz, dim=-1).mean()

        # Magnitude
        mag_error = (
            (pred_flow.norm(p=2, dim=-1) - gt_flow.norm(p=2, dim=-1)).abs().mean()
        )
    print(rmse, cos_dist, mag_error)
    return rmse, cos_dist, mag_error


def artflownet_loss(
    f_pred: torch.Tensor,
    f_target: torch.Tensor,
    n_nodes: torch.Tensor,
) -> torch.Tensor:
    # f_pred = normalize_trajectory(f_pred)

    # Flow loss, per-point.
    raw_se = ((f_pred - f_target) ** 2).sum(dim=-1)

    weights = (1 / n_nodes).repeat_interleave(n_nodes)
    l_se = (raw_se * weights[:, None]).sum() / f_pred.shape[1]  # Trajectory length

    # Full loss, aberaged across the batch.
    loss: torch.Tensor = l_se / len(n_nodes)

    return loss
