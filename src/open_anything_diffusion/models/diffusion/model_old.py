# The model naming as in pipeline training. Not trained under /diffusion directory
from dataclasses import dataclass

import rpad.pyg.nets.pointnet2 as pnp
import torch
import torch_geometric.data as tgd
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

# import open_anything_diffusion.models.diffusion.module as pnp

# from .util import quat2mat


@dataclass
class PointNetOutput(BaseOutput):
    """
    The output of [`DGCNN`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


class PNDiffuser(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels, time_embed_dim, traj_len, cond_emb_dims=64):
        super(PNDiffuser, self).__init__()
        self.in_channels = in_channels
        self.sample_size = 1200
        self.time_embed_dim = time_embed_dim
        self.traj_len = traj_len

        # positional time embeddings
        flip_sin_to_cos = True
        freq_shift = 0
        self.time_proj = Timesteps(64, flip_sin_to_cos, freq_shift)
        timestep_input_dim = 64
        self.time_embed = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # self.condition_encoder = pnp.PN2Dense(
        #     in_channels=1,
        #     out_channels=cond_emb_dims,
        #     p=pnp.PN2DenseParams(),
        # )

        self.backbone = pnp.PN2Dense(
            # in_channels = self.in_channels + timestep_input_dim,
            in_channels=self.in_channels + time_embed_dim,
            out_channels=3 * self.traj_len,
            p=pnp.PN2DenseParams(),
        )

    def forward(
        self,
        noisy_input,
        timestep,
        context,
        return_dict: bool = True,
    ):
        """
        Args:
            x:  Flow at some timestep t, (B, 3, 1, N).
            timestep:     Time. (B, ).
            context: Batched data with .pos (context pointcloud observation), .x (should be replaced to be the noisy input)
        """

        # time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=noisy_input.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(noisy_input.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(
            noisy_input.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        t_emb = self.time_embed(t_emb)

        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, self.sample_size)

        # #  condition embedding
        # cond_emb = self.condition_encoder(condition)

        # concatenate embeddings
        # breakpoint()
        context = context.to(noisy_input.device)
        context.x = torch.cat(
            (torch.flatten(noisy_input, start_dim=1, end_dim=2), t_emb), dim=1
        )  # (B, 3 + 64 , N)
        # breakpoint()
        context.x = torch.flatten(context.x.permute(0, 2, 1), start_dim=0, end_dim=1)

        x = self.backbone(context)
        x = x.reshape(-1, self.sample_size, 3, self.traj_len).permute(0, 2, 3, 1)
        # breakpoint()
        if not return_dict:
            return (x,)

        return PointNetOutput(sample=x)
        # return x


if __name__ == "__main__":
    model = PNDiffuser(3, 64, 1, 64)

    from typing import Protocol, cast

    class Flowbot3DTGData(Protocol):
        id: str  # Object ID.

        pos: torch.Tensor  # Points in the point cloud.
        flow: torch.Tensor  # instantaneous positive 3D flow.
        mask: torch.Tensor  # Mask of the part of interest.

    condition = cast(
        Flowbot3DTGData,
        tgd.Data(
            id="1",
            pos=torch.randn((1200, 3)),
            flow=torch.randn((1200, 3)),
            mask=torch.randn(1200),
            x=torch.randn(1200, 1),
        ),
    )
    input = torch.randn((10, 3, 1200))
    print(model(input, 1, condition))
