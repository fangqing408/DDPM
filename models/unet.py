import math
from matplotlib.pylab import block
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(
        self, 
        time_channels: int,
    ):
        super().__init__()
        # time_channels = model_channels * 4
        self.time_channels = time_channels

        self.linear_1 = nn.Linear(time_channels // 4, time_channels)
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(time_channels, time_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # d = model_channels // 2 == time_channels // 8
        d = self.time_channels // 8
        freq = torch.exp(-math.log(10000) / (d - 1) * torch.arange(d, device=t.device))
        # t.shape == (batch_size, ), freq.shape == (d, )
        # emb.shape == (batch_size, d)
        emb = t[:, None] * freq[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.linear_2(self.act_1(self.linear_1(emb)))

class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_channels: int,
        num_groups: int,
        drop_out: float = 0.1,

    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.num_groups = num_groups
        self.drop_out = drop_out

        self.norm_1 = nn.GroupNorm(num_groups, in_channels)
        self.act_1 = nn.SiLU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_act = nn.SiLU()
        self.proj_time = nn.Linear(time_channels, out_channels)

        self.norm_2 = nn.GroupNorm(num_groups, out_channels)
        self.act_2 = nn.SiLU()
        self.dropout = nn.Dropout(drop_out)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else: 
            self.shortcut = nn.Identity()
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self. conv_1(self.act_1(self.norm_1(x)))
        # t.shape == (batch_size, time_channels)
        # x.shape == (batch_size, out_channels, height, width)
        h += self.proj_time(self.time_act(t))[:, :, None, None]
        h = self.conv_2(self.dropout(self.act_2(self.norm_2(h))))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(
        self, 
        out_channels: int, 
        num_heads: int = 1,
        d_k: int | None = None,
        num_groups: int = 32
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        if d_k is None:
            assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
            d_k = out_channels // num_heads
        self.d_k = d_k
        self.num_groups = num_groups
        
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.input_proj = nn.Linear(out_channels, num_heads * d_k * 3)
        self.output = nn.Linear(num_heads * d_k, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x

        x = self.norm(x)
        seq_len = h * w
        # (b, c, h, w) -> (b, c, seq_len)
        x = x.reshape(b, c, seq_len)

        # (b, c, seq_len) -> (b, seq_len, c)
        x = x.permute(0, 2, 1)

        # (b, seq_len, c) -> (b, seq_len, num_heads * d_k * 3)
        qkv: torch.Tensor = self.input_proj(x)

        # (b, seq_len, num_heads * d_k * 3) -> (b, seq_len, num_heads, d_k * 3)
        qkv = qkv.reshape(b, seq_len, self.num_heads, self.d_k * 3)

        # (b, seq_len, num_heads, d_k * 3) -> (b, seq_len, num_heads, d_k) * 3
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # (b, seq_len, num_heads, d_k), (b, seq_len, num_heads, d_k) -> (b, seq_len, seq_len, num_heads)
        self.scale = self.d_k ** -0.5
        attention_scores = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale

        # (b, seq_len, seq_len, num_heads) -> (b, seq_len, seq_len, num_heads)
        attention_weights = torch.softmax(attention_scores, dim=2)

        # (b, seq_len, seq_len, num_heads) * (b, seq_len, num_heads, d_k) -> (b, seq_len, num_heads, d_k)
        attention_aggregation = torch.einsum("bijh,bjhd->bihd", attention_weights, v)

        # (b, seq_len, num_heads, d_k) -> (b, seq_len, num_heads * d_k)
        attention_aggregation = attention_aggregation.reshape(b, seq_len, -1)

        # (b, seq_len, num_heads * d_k) -> (b, seq_len, out_channels)
        res = self.output(attention_aggregation)

        # (b, seq_len, out_channels) -> (b, c, h, w)
        res= res.permute(0, 2, 1).reshape(b, c, h, w)

        return res + x_in
    
class DownBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        time_channels: int,
        is_attention: bool = False,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.is_attention = is_attention

        self.res_block = ResidualBlock(in_channels, out_channels, time_channels, num_groups)
        if is_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, t)
        x = self.attention(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        time_channels: int,
        is_attention: bool = False,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.is_attention = is_attention

        self.res_block = ResidualBlock(in_channels, out_channels, time_channels, num_groups)
        if is_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, t)
        x = self.attention(x)
        return x
    
class DownSample(nn.Module):
    def __init__(
        self, 
        out_channels: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        # H_new = (H_old + 2 * padding - kernel_size) // stride + 1
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)
    
class UpSample(nn.Module):
    def __init__(
        self, 
        out_channels: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        # H_new = (H_old - 1) * stride - 2 * padding + kernel_size + output_padding
        self.transpose_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.transpose_conv(x)
    
class MiddleBlock(nn.Module):
    def __init__(
        self, 
        out_channels: int,
        time_channels: int,
        num_groups: int = 32,
    ):
        super().__init__()
        self.channels = out_channels
        self.time_channels = time_channels

        self.res_block_1 = ResidualBlock(out_channels, out_channels, time_channels, num_groups)
        self.attention = AttentionBlock(out_channels)
        self.res_block_2 = ResidualBlock(out_channels, out_channels, time_channels, num_groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res_block_1(x, t)
        x = self.attention(x)
        x = self.res_block_2(x, t)
        return x
    
class DDPMUNet(nn.Module):
    def __init__(
        self,
        img_channels: int,
        model_channels: int,
        channel_mults: list[int] = [1, 2, 2, 2], 
        is_attention: list[bool] = [False, False, False, False],
        num_res_blocks: int = 2,
        num_groups: int = 32,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.model_channels = model_channels
        self.channel_mults = channel_mults
        self.is_attention = is_attention
        self.num_res_blocks = num_res_blocks
        self.num_groups = num_groups
        self.time_channels = model_channels * 4

        # input projection
        self.img_proj = nn.Conv2d(img_channels, model_channels, kernel_size=3, padding=1)
        # time embedding
        self.time_enbedding = TimeEmbedding(self.time_channels)
        # down blocks
        self.down_blocks_num = len(channel_mults)
        self.down_blocks = nn.ModuleList()
        cur_channels = model_channels
        self.channels_stack = [cur_channels]
        for i in range(self.down_blocks_num):
            nxt_channels = model_channels * channel_mults[i]
            for _ in range(num_res_blocks):
                self.down_blocks.append(DownBlock(cur_channels, nxt_channels, self.time_channels, is_attention[i]))
                cur_channels = nxt_channels
                self.channels_stack.append(cur_channels)
            if i < self.down_blocks_num - 1:
                self.down_blocks.append(DownSample(cur_channels))
        
        # middle block
        self.middle_block = MiddleBlock(cur_channels, self.time_channels)

        # up blocks 
        self.up_blocks = nn.ModuleList()
        for i in range(self.down_blocks_num - 1, -1, -1):
            if i < self.down_blocks_num - 1:
                cur_channels = self.channels_stack[-1]
                self.up_blocks.append(UpSample(cur_channels))
            for _ in range(num_res_blocks):
                cur_channels = self.channels_stack.pop()
                nxt_channels = self.channels_stack[-1]
                self.up_blocks.append(UpBlock(cur_channels * 2, nxt_channels, self.time_channels, is_attention[i]))

        # output projection
        self.norm = nn.GroupNorm(num_groups, model_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(model_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.img_proj(x)
        t = self.time_enbedding(t)

        # down blocks
        h = [x]
        for down_block in self.down_blocks:
            if isinstance(down_block, DownSample):
                x = down_block(x, t)
            else:
                x = down_block(x, t)
                h.append(x)

        # middle block
        x = self.middle_block(x, t)

        # up blocks
        for up_block in self.up_blocks:
            if isinstance(up_block, UpSample):
                x = up_block(x, t)
            else:
                s = h.pop()
                x = torch.cat((s, x), dim=1)
                x = up_block(x, t)

        # output projection
        x = self.conv(self.act(self.norm(x)))
        return x