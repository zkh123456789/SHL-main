import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_
import math
import time
from thop import profile  

def exists(val):
    return val is not None

def is_empty(t):
    return t.nelement() == 0

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def center_iter(x, means, buckets=None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means

class ISAM(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_sizes=[128, 256]):
        super().__init__()
        self.heads = heads
        self.group_sizes = group_sizes
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        

        self.scale_weights = nn.Parameter(torch.ones(len(group_sizes)))
        
    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape
        
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))
        

        multi_scale_outputs = []
        
        for gs in self.group_sizes:
            gs = min(N, gs)  # group size
            ng = (N + gs - 1) // gs
            

            q_groups = rearrange(q, 'b (ng gs) d -> b ng gs d', ng=ng, gs=gs)
            k_groups = rearrange(k, 'b (ng gs) d -> b ng gs d', ng=ng, gs=gs)
            v_groups = rearrange(v, 'b (ng gs) d -> b ng gs d', ng=ng, gs=gs)
            

            q_groups = rearrange(q_groups, 'b ng gs (h d) -> b ng h gs d', h=self.heads)
            k_groups = rearrange(k_groups, 'b ng gs (h d) -> b ng h gs d', h=self.heads)
            v_groups = rearrange(v_groups, 'b ng gs (h d) -> b ng h gs d', h=self.heads)
            

            out1 = F.scaled_dot_product_attention(q_groups, k_groups, v_groups)
            

            k_global_reshaped = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
            v_global_reshaped = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)
            
            out2 = F.scaled_dot_product_attention(q_groups, k_global_reshaped, v_global_reshaped)
            out = out1 + out2
            

            out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")
            out = out[:, :N, :]  
            
            multi_scale_outputs.append(out)
        

        weights = F.softmax(self.scale_weights, dim=0)
        y = sum(w * out for w, out in zip(weights, multi_scale_outputs))
        
        y = y.scatter(dim=-2, index=idx_last.expand(y.shape), src=y)
        y = self.proj(y)
    
        return y

class CGAM(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        

        self.freq_enhance = nn.Sequential(
            nn.Conv1d(dim, dim // 4, 3, padding=1, groups=dim // 4),
            nn.ReLU(),
            nn.Conv1d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
      
    def forward(self, normed_x, x_means):
        x = normed_x
        

        x_transposed = x.transpose(1, 2)  # [B, D, N]
        freq_weights = self.freq_enhance(x_transposed)
        x_enhanced = x_transposed * freq_weights
        x_enhanced = x_enhanced.transpose(1, 2)  # [B, N, D]
        

        x = x + x_enhanced
        
        if self.training:
            x_global = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
        else:
            x_global = x_means

        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k, v, x_global.detach()

class SGAB(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_sizes=[128], ema_decay=0.999,
                 num_semantic_classes=10, semantic_dim=32, spatial_sigma=10.0):
        super().__init__()
        
        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens
        self.spatial_sigma = spatial_sigma
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))
        self.irca_attn = CGAM(dim, qk_dim, heads)
        self.iasa_attn = ISAM(dim, qk_dim, heads, group_sizes)
        

        self.semantic_proj = nn.Linear(num_semantic_classes, semantic_dim)
        self.fusion = nn.Linear(dim + semantic_dim, dim)
        

        self.register_buffer('spatial_positions', None)
        self.register_buffer('center_positions', torch.rand(num_tokens, 2))
        
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)
        
    def init_spatial_positions(self, h, w):

        y_pos = torch.linspace(0, 1, h, device=self.means.device)
        x_pos = torch.linspace(0, 1, w, device=self.means.device)
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
        positions = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
        positions = positions.reshape(-1, 2)  # [H*W, 2]
        self.register_buffer('spatial_positions', positions)
        
    def forward(self, x, semantic_logits=None):
        
        _, _, h, w = x.shape
        

        if self.spatial_positions is None or self.spatial_positions.shape[0] != h * w:
            self.init_spatial_positions(h, w)
            
        x_2d = x
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        

        if semantic_logits is not None:

            if semantic_logits.shape[2:] != (h, w):
                semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode='bilinear', align_corners=False)
            

            semantic_tokens = rearrange(semantic_logits, 'b c h w->b (h w) c')
            semantic_features = self.semantic_proj(semantic_tokens)
            

            x = torch.cat([x, semantic_features], dim=-1)
            x = self.fusion(x)
        
        x = self.norm(x)
        B, N, _ = x.shape
        

        spatial_dists = torch.cdist(self.spatial_positions.unsqueeze(0), 
                                   self.center_positions.unsqueeze(0))  # [1, N, num_tokens]
        spatial_weights = torch.exp(-spatial_dists / self.spatial_sigma)  
        
        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)
        
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:, N-pad_n:N, :], dims=[-2])), dim=-2)
            x_means = torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c', cnt=self.num_tokens), dim=-2).detach()
        else:
            x_means = self.means.detach()
        
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
        
        k_global, v_global, x_means = self.irca_attn(x, x_means)
  

        with torch.no_grad():

            content_similarity = torch.einsum('b i c,j c->b i j', 
                                             F.normalize(x, dim=-1), 
                                             F.normalize(x_means, dim=-1))
            

            combined_similarity = content_similarity * spatial_weights
            
            x_belong_idx = torch.argmax(combined_similarity, dim=-1)
            
            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)
        
        y = self.iasa_attn(x, idx_last, k_global, v_global)

        y = rearrange(y, 'b (h w) c->b c h w', h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x
        


        if self.training:
            with torch.no_grad():

                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    ema_inplace(self.means, new_means, self.ema_decay)
            

                for j in range(self.num_tokens):
                    mask = (x_belong_idx == j).float()
                    if mask.sum() > 0:
                        mean_pos = (mask.unsqueeze(-1) * self.spatial_positions.unsqueeze(0)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
                        mean_pos = mean_pos.mean(dim=0)
                        self.center_positions[j] = ema(self.center_positions[j], mean_pos, self.ema_decay)

        return rearrange(x, 'b (h w) c->b c h w', h=h)


def patch_divide(x, step, ps):
    """Crop image into patches."""
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw

def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image."""
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output

class PreNorm(nn.Module):
    """Normalization layer."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    """Attention module."""
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q,k,v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class LSA(nn.Module):

    def __init__(self, dim, qk_dim, mlp_dim,heads=1):
        super().__init__()
        self.layer = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, qk_dim)),
                PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        return x

@ARCH_REGISTRY.register()
class SHL(nn.Module):
    setting = dict(dim=40, block_num=8, qk_dim=36, mlp_dim=96, heads=4, 
                     patch_size=[16, 20, 24, 28, 16, 20, 24, 28])

    def __init__(self, in_chans=3, n_iters=[5,5,5,5,5,5,5,5],
                 num_tokens=[16,32,64,128,16,32,64,128],
                 group_sizes=[[128, 256]] * 8,  
                 upscale: int = 4, num_semantic_classes=10):
        super().__init__()
        
        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']
        
        self.n_iters = n_iters
        self.num_tokens = num_tokens
        self.group_sizes = group_sizes
        self.num_semantic_classes = num_semantic_classes
    
        #-----------1 shallow--------------
        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)


        self.semantic_net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_semantic_classes, 1)
        )
        
        #----------2 deep--------------
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
   
        for i in range(self.block_num):
            self.blocks.append(nn.ModuleList([
                SGAB(
                    self.dim, self.qk_dim, self.mlp_dim,
                    self.heads, self.n_iters[i], 
                    self.num_tokens[i], self.group_sizes[i],
                    num_semantic_classes=num_semantic_classes
                ), 
                LSA(self.dim, self.qk_dim, self.mlp_dim, self.heads)
            ]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1))
            
        #----------3 reconstruction---------
        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
    
        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, semantic_logits):
        for i in range(self.block_num):
            residual = x
            global_attn, local_attn = self.blocks[i]
        

            if not self.training:
                with torch.no_grad():
                    x = global_attn(x, semantic_logits)
                    x = local_attn(x, self.patch_size[i])
            else:
                x = global_attn(x, semantic_logits)
                x = local_attn(x, self.patch_size[i])
            
            x = residual + self.mid_convs[i](x)
        return x
        
    def forward(self, x):

        semantic_logits = self.semantic_net(x)
        
        if self.upscale != 1: 
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else: 
            base = x
            
        x = self.first_conv(x)
        x = self.forward_features(x, semantic_logits) + x
    
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
            
        out = self.last_conv(out) + base
        return out
    
    def get_flops(self, input_size=(3, 128, 128)):

        device = next(self.parameters()).device
        input_tensor = torch.randn(1, *input_size).to(device)
        

        flops, params = profile(self, inputs=(input_tensor,), verbose=False)
        return flops, params
    
    def get_inference_time(self, input_size=(3, 128, 128), warmup=10, repeats=100):

        device = next(self.parameters()).device
        input_tensor = torch.randn(1, *input_size).to(device)
        

        self.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = self(input_tensor)
        

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(repeats):
                _ = self(input_tensor)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) * 1000 / repeats 
        return avg_time
    
    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        

        try:
            flops, params = self.get_flops()
            inference_time = self.get_inference_time()
            
            info_str = f'#Params: {num_parameters / 10 ** 3:<.4f} [K]\n'
            info_str += f'FLOPs: {flops / 10 ** 9:<.2f} [G]\n'
            info_str += f'Inference Time: {inference_time:<.2f} [ms]'
        except Exception as e:
            info_str = f'#Params: {num_parameters / 10 ** 3:<.4f} [K]\n'
            info_str += f'FLOPs/Time calculation failed: {e}'
        
        return f'{self._get_name()}\n{info_str}'

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SHL(upscale=4).to(device)
    x = torch.randn(2, 3, 64, 64).to(device)

    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    

    print("Testing inference time...")
    inference_time = model.get_inference_time(input_size=(3, 64, 64))
    

    print("Calculating FLOPs...")
    flops, params = model.get_flops(input_size=(3, 64, 64))
    

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"FLOPs: {flops / 1e9:.2f}G")
    print(f"Inference time: {inference_time:.2f}ms")
    

    test_sizes = [(3, 64, 64)]
    print("\nTesting different input sizes:")
    print("Size\t\tFLOPs(G)\tTime(ms)")
    print("-" * 40)
    
    for size in test_sizes:
        try:
            flops, _ = model.get_flops(input_size=size)
            time_ms = model.get_inference_time(input_size=size, repeats=50)
            print(f"{size}\t{flops/1e9:.2f}\t\t{time_ms:.2f}")
        except Exception as e:
            print(f"{size}\tError: {e}")
