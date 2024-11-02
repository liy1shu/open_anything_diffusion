# Taken from https://github.com/WangYueFt/dcp/blob/master/model.py
# Provides the baseline architectures for the DCP model.
# Only changes:
# - Change `from util import quat2mat` to `from .util import quat2mat`.
# - Add this comment.
#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

# from .util import quat2mat


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device("cuda")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    # feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


@dataclass
class DGCNNOutput(BaseOutput):
    """
    The output of [`DGCNN`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


# class DGCNN(nn.Module):
class DGCNN(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels, sample_size, time_embed_dim, emb_dims=512):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.sample_size = sample_size
        self.time_embed_dim = time_embed_dim

        # positional time embeddings
        flip_sin_to_cos = True
        freq_shift = 0
        self.time_proj = Timesteps(64, flip_sin_to_cos, freq_shift)
        timestep_input_dim = 64
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # out_channels = [64, 256, 512, 1024]
        out_channels = [256, 512, 512, 1024]

        # goal embedding
        goal_embed_dim = 64
        goal_channels = [64, 128, 128, goal_embed_dim]
        self.goal_sample_size = sample_size
        self.goal_conv1 = nn.Conv2d(3 * 2, goal_channels[0], kernel_size=1, bias=False)
        self.goal_conv2 = nn.Conv2d(
            goal_channels[0], goal_channels[1], kernel_size=1, bias=False
        )
        self.goal_conv3 = nn.Conv2d(
            goal_channels[1], goal_channels[2], kernel_size=1, bias=False
        )
        self.goal_conv4 = nn.Conv2d(
            sum(goal_channels[:-1]), goal_channels[3], kernel_size=1, bias=False
        )

        # network components
        # self.conv1 = nn.Conv2d((in_channels+time_embed_dim)*2, out_channels[0], kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(
            (in_channels + goal_embed_dim + time_embed_dim) * 2,
            out_channels[0],
            kernel_size=1,
            bias=False,
        )
        # self.conv1 = nn.Conv2d(in_channels*2+time_embed_dim, out_channels[0], kernel_size=1, bias=False)
        # self.conv1 = nn.Conv2d(in_channels*2, 64, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels[0], out_channels[1], kernel_size=1, bias=False
        )
        # self.conv2 = nn.Conv2d(out_channels[0]+time_embed_dim, out_channels[1], kernel_size=1, bias=False)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(
            out_channels[1], out_channels[2], kernel_size=1, bias=False
        )
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        # self.conv4 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(
            out_channels[2], out_channels[3], kernel_size=1, bias=False
        )
        # self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        # self.conv5 = nn.Conv2d(960, emb_dims, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(sum(out_channels), emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.Identity()  # nn.BatchNorm2d(64)
        self.bn2 = nn.Identity()  # nn.BatchNorm2d(64)
        self.bn3 = nn.Identity()  # nn.BatchNorm2d(128)
        self.bn4 = nn.Identity()  # nn.BatchNorm2d(256)
        self.bn5 = nn.Identity()  # nn.BatchNorm2d(emb_dims)

    def forward(
        self,
        x,
        timestep,
        context,
        return_dict: bool = True,
    ):
        """
        Args:
            x:  Point clouds and flows at some timestep t, (B, 3+3, N).
            timestep:     Time. (B, ).
            context: None.
        """

        # time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(
            x.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # goal embedding
        batch_size, num_dims, num_points = x.size()
        goal_pcd = context
        goal_x = get_graph_feature(goal_pcd)
        goal_x = F.relu(self.goal_conv1(goal_x))
        goal_x1 = goal_x.max(dim=-1, keepdim=True)[0]

        goal_x = F.relu(self.goal_conv2(goal_x))
        goal_x2 = goal_x.max(dim=-1, keepdim=True)[0]

        goal_x = F.relu(self.goal_conv3(goal_x))
        goal_x3 = goal_x.max(dim=-1, keepdim=True)[0]

        goal_x = torch.cat((goal_x1, goal_x2, goal_x3), dim=1)
        goal_emb = (self.goal_conv4(goal_x)).view(batch_size, -1, self.goal_sample_size)

        # concatenate embeddings
        emb = emb.view(batch_size, -1, 1).repeat(1, 1, num_points)
        x = torch.cat((x, goal_emb, emb), dim=1)  # (B, d+64+64, N)

        x = get_graph_feature(x)  # (B, (d+64+64)*2, N, k)
        # emb = emb.view(batch_size, -1, 1, 1).repeat(1, 1, num_points, 20)
        # x = torch.cat((x, emb), dim=1) # B, d*2 + 64, N, k)
        # x = get_graph_feature(x) # (B, d*2+64, N, k)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        # emb = emb.view(batch_size, -1, num_points, 1).repeat(1, 1, 1, 20)
        # x = torch.cat([x, emb], dim=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        # x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        x = self.bn5(self.conv5(x)).view(batch_size, -1, num_points)

        if not return_dict:
            return (x,)

        return DGCNNOutput(sample=x)


if __name__ == "__main__":
    model = DGCNN(in_channels=6, sample_size=1200, time_embed_dim=64, emb_dims=3).cuda()

    noise = torch.randn(1, 3, 1200).cuda()
    pcd = torch.randn(1, 3, 1200).cuda()
    x = torch.concat((noise, pcd), dim=1).cuda()
    t = torch.randint(100, size=(1,)).cuda()
    context = pcd
    output = model(x, t, context)
    breakpoint()
