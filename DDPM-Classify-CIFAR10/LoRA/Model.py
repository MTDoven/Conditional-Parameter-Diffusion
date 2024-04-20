RANK = 2
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch, rank=RANK):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.lora_main_A = nn.Conv2d(in_ch, rank, 1, stride=1, padding=0, bias=False)
        self.lora_main_B = nn.Conv2d(rank, in_ch, 3, stride=2, padding=1, bias=False)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)
        init.kaiming_normal_(self.lora_main_A.weight)
        init.zeros_(self.lora_main_B.weight)

    def forward(self, x, temb):
        x = self.main(x) + self.lora_main_B(self.lora_main_A(x))
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch, rank=RANK):
        super().__init__()
        self.lora_main_A = nn.Conv2d(in_ch, rank, 1, stride=1, padding=0, bias=False)
        self.lora_main_B = nn.Conv2d(rank, in_ch, 1, stride=1, padding=0, bias=False)
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)
        init.kaiming_normal_(self.lora_main_A.weight)
        init.zeros_(self.lora_main_B.weight)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x) + self.lora_main_B(self.lora_main_A(x))
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch, rank=RANK):
        super().__init__()
        self.lora_proj_q_A = nn.Conv2d(in_ch, rank, 1, stride=1, padding=0, bias=False)
        self.lora_proj_q_B = nn.Conv2d(rank, in_ch, 1, stride=1, padding=0, bias=False)
        self.lora_proj_k_A = nn.Conv2d(in_ch, rank, 1, stride=1, padding=0, bias=False)
        self.lora_proj_k_B = nn.Conv2d(rank, in_ch, 1, stride=1, padding=0, bias=False)
        self.lora_proj_v_A = nn.Conv2d(in_ch, rank, 1, stride=1, padding=0, bias=False)
        self.lora_proj_v_B = nn.Conv2d(rank, in_ch, 1, stride=1, padding=0, bias=False)
        self.lora_proj_A = nn.Conv2d(in_ch, rank, 1, stride=1, padding=0, bias=False)
        self.lora_proj_B = nn.Conv2d(rank, in_ch, 1, stride=1, padding=0, bias=False)

        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)
        for module in [self.lora_proj_q_A, self.lora_proj_k_A, self.lora_proj_v_A, self.lora_proj_A]:
            init.kaiming_normal_(module.weight)
        for module in [self.lora_proj_q_B, self.lora_proj_k_B, self.lora_proj_v_B, self.lora_proj_B]:
            init.zeros_(module.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h) + self.lora_proj_q_B(self.lora_proj_q_A(h))
        k = self.proj_k(h) + self.lora_proj_k_B(self.lora_proj_k_A(h))
        v = self.proj_v(h) + self.lora_proj_v_B(self.lora_proj_v_A(h))

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h) + self.lora_proj_B(self.lora_proj_A(h))

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, rank=RANK):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.lora_conv_block1_A = nn.Conv2d(in_ch, rank, 1, stride=1, padding=0, bias=False)
        self.lora_conv_block1_B = nn.Conv2d(rank, out_ch, 1, stride=1, padding=0, bias=False)

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        self.lora_conv_block2_A = nn.Conv2d(out_ch, rank, 1, stride=1, padding=0, bias=False)
        self.lora_conv_block2_B = nn.Conv2d(rank, out_ch, 1, stride=1, padding=0, bias=False)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
        for module in [self.lora_conv_block1_A, self.lora_conv_block2_A]:
            init.kaiming_normal_(module.weight)
        for module in [self.lora_conv_block1_B, self.lora_conv_block2_B]:
            init.zeros_(module.weight)

    def forward(self, x, temb):
        h = self.block1[1](self.block1[0](x))
        h = self.block1[2](x) + self.lora_conv_block1_B(self.lora_conv_block1_A(h))

        h += self.temb_proj(temb)[:, :, None, None]

        h = self.block2[2](self.block2[1](self.block2[0](h)))
        h = self.block2[3](h) + self.lora_conv_block2_B(self.lora_conv_block2_A(h))

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, rank=RANK):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.lora_head_A = nn.Conv2d(3, rank, 1, 1, 0)
        self.lora_head_B = nn.Conv2d(rank, ch, 1, 1, 0)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.lora_tail_A = nn.Conv2d(now_ch, rank, 1, 1, 0)
        self.lora_tail_B = nn.Conv2d(rank, 3, 1, 1, 0)

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)
        for module in [self.lora_head_A, self.lora_tail_A]:
            init.kaiming_normal_(module.weight)
        for module in [self.lora_head_B, self.lora_tail_B]:
            init.zeros_(module.weight)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x) + self.lora_head_B(self.lora_head_A(x))

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)

        h = self.tail[1](self.tail[0](h))
        h = self.tail[2](h) + self.lora_tail_B(self.lora_tail_A(h))

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
