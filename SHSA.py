import torch


class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1
    ):
        super().__init__()

        self.add_module(
            "c",
            torch.nn.Conv2d(
                a,
                b,
                ks,
                stride,
                pad,
                dilation,
                groups,
                bias=False
            )
        )

        self.add_module("bn", torch.nn.BatchNorm2d(b))

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        conv, bn = self._modules.values()

        weight = bn.weight / (bn.running_var + bn.eps) ** 0.5
        weight = conv.weight * weight[:, None, None, None]

        bias = (
            bn.bias
            - bn.running_mean * bn.weight
            / (bn.running_var + bn.eps) ** 0.5
        )

        fused_conv = torch.nn.Conv2d(
            weight.size(1) * self.c.groups,
            weight.size(0),
            weight.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=conv.weight.device
        )

        fused_conv.weight.data.copy_(weight)
        fused_conv.bias.data.copy_(bias)

        return fused_conv


class SHSA(torch.nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()

        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm(pdim)

        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)

        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(),
            Conv2d_BN(dim, dim, bn_weight_init=0)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x1, x2 = torch.split(
            x,
            [self.pdim, self.dim - self.pdim],
            dim=1
        )

        x1 = self.pre_norm(x1)

        qkv = self.qkv(x1)

        q, k, v = qkv.split(
            [self.qk_dim, self.qk_dim, self.pdim],
            dim=1
        )

        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x1 = (v @ attn.transpose(-2, -1)).reshape(
            batch_size,
            self.pdim,
            height,
            width
        )

        x = self.proj(torch.cat([x1, x2], dim=1))

        return x