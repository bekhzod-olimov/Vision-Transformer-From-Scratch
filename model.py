# Import libraries
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    
    """
    
    This class splits an image into pathces and then embed these patches.
    
    Arguments:
    
        im_size    - input image size, int;
        p_size     - patch size, int;
        in_chs     - number of channels in an input volume, int;
        emb_dim    - embedding dimension, int;
    
    """
    
    def __init__(self, im_size: int, p_size: int, in_chs: int = 3, emb_dim: int = 768):
        super(PatchEmbed, self).__init__()
        
        # Get image size, patch size, and number of patches
        self.im_size, self.p_size, self.n_ps = im_size, p_size, (im_size // p_size) ** 2
        
        # Initialize a convolution operation for projection 
        self.proj = nn.Conv2d(in_channels = in_chs, out_channels = emb_dim, kernel_size = p_size, stride = p_size)
        
    def forward(self, inp: torch.tensor):
        
        """
        
        This function gets an input tensor volume and creates patches with a pre-defined embedding dimension.
        
        Argument:
        
            inp - input volume, tensor;
            
        Output:
        
            out - output volume from PatchEmbed class, tensor
        
        """

        inp = self.proj(inp)                     # (batch, emb_dim, (n_ps ** 0.5) x 2)
        inp = inp.flatten(start_dim = 2)         # (batch, emb_dim, n_ps)
        out = inp.transpose(dim0 = 1, dim1 = 2)  # (batch, n_ps, emb_dim)

        return out
        
class Attention(nn.Module):
    
    """
    
    This class creates an attention layer and passes the input volume through it.
    
    Arguments:
    
        dim         - attention dimension, int;
        n_heads     - number of heads of the attention layer, int;
        qkv_bias    - query, key, and value bias availability, bool;
        attn_p      - attention dropout probability, float;
        proj_p      - projection dropout probability, float.
        
        
    Output:
    
        out         - output volume from the attention layer, tensor.
    
    """
    
    def __init__(self, dim: int, n_heads: int = 12, qkv_bias: bool = True, attn_p: float = 0., proj_p: float = 0.):
        super(Attention, self).__init__()
        
        # Reason to use scale is not to feed extremely big values to SoftMax, which can lead to small gradients
        self.n_heads, self.dim, self.head_dim = n_heads, dim, dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(in_features = dim, out_features = dim * 3, bias = qkv_bias)
        self.proj = nn.Linear(in_features = dim, out_features = dim)
        self.attn_drop = nn.Dropout(p = attn_p)
        self.proj_drop = nn.Dropout(p = proj_p)
        
    def forward(self, inp: torch.tensor):
        
        """
        
        This function gets an input volume, applies attention layer and returns output with attention applied
        
        Argument:
        
            inp     - an input volume, tensor;
            
        Output:
        
            out    - an output volume after attention is applied, tensor.
        
        
        """
        
        batch, n_tokens, dim = inp.shape # inp_shape = (batch, n_ps+1, dim) -> +1 is for the class token as the first token in the sequence
        assert dim == self.dim, "Input and attention dimensions do not match"
        
        # Get query, key, and values
        qkv = self.qkv(inp) # (batch, n_ps + 1, 3 * dim)
        # Reshape the qkv
        qkv = qkv.reshape(batch, n_tokens, 3, self.n_heads, self.head_dim) # (batch, n_ps + 1, 3, n_heads, head_dim) -> 3 is for q,k,v
        # Permute the qkv
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch, n_heads, n_ps + 1, head_dim)
        
        # Get query, key, and values
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Transpose keys
        k_t = k.transpose(-2, -1) # (batch, n_heads, head_dim, n_ps + 1)
        
        # Scaling
        dp = (q @ k_t) * self.scale # (batch, n_heads, n_ps + 1, n_ps + 1)
        
        # Attention
        attn = dp.softmax(dim = -1) # (batch, n_heads, n_ps + 1, n_ps + 1)
        attn = self.attn_drop(attn)
        
        # Get weighted average        
        weighted_avg = attn @ v # (batch, n_heads, n_ps + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (batch, n_ps + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (batch, n_ps + 1, head_dim)
        
        # Apply projection
        inp = self.proj(weighted_avg) # (batch, n_ps + 1, dim)
        
        # Apply dropout
        out = self.proj_drop(inp) # (batch, n_ps + 1, dim) 
        
        return out

class MLP(nn.Module):
    
    """
    
    This class constructs multilayer perceptron and passes input through it.
    
    Arguments:
    
        
    
    """
    
    def __init__(self, in_fs, hid_fs, out_fs, p=0):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(in_fs, hid_fs)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_fs, out_fs)
        self.drop = nn.Dropout(p)
        
    def forward(self, inp):
        
        inp = self.fc1(inp) # (batch, n_ps + 1, hid_fs)
        inp = self.act(inp) # (batch, n_ps + 1, hid_fs)
        inp = self.drop(inp) # (batch, n_ps + 1, hid_fs)
        inp = self.fc2(inp) # (batch, n_ps + 1, out_fs)
        inp = self.drop(inp) # (batch, n_ps + 1, out_fs)
        
        return inp
    
class Block(nn.Module):
    
    def __init__(self, dim, n_heads, mlp_ratio = 4.0, qkv_bias = True, p = 0, attn_p = 0):
        super(Block, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6)
        self.attn = Attention(dim, n_heads = n_heads, qkv_bias = qkv_bias, attn_p = attn_p, proj_p = p)
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        
        hid_fs = int(dim * mlp_ratio)
        self.mlp = MLP(in_fs = dim, hid_fs = hid_fs, out_fs = dim)
        
    def forward(self, inp):
        
        inp = inp + self.attn(self.norm1(inp))
        inp = inp + self.mlp(self.norm2(inp))
        
        return inp
    
    
class VisionTransformer(nn.Module):
    
    def __init__(self, im_size = 384, p_size = 16, in_chs = 3, n_cls = 1000, emb_dim = 768, depth = 12, n_heads = 12, mlp_ratio = 4, qkv_bias = True, p = 0., attn_p = 0.):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbed(im_size = im_size, p_size = p_size, in_chs = in_chs, emb_dim = emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_ps + 1, emb_dim))
        self.pos_drop = nn.Dropout(p = p)
        
        self.blocks = nn.ModuleList(
        
            [Block(dim = emb_dim, n_heads = n_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, p = p, attn_p = attn_p) for _ in range(depth)]
            
        )
        
        self.norm = nn.LayerNorm(emb_dim, eps = 1e-6)
        self.head = nn.Linear(emb_dim, n_cls)
        
    def forward(self, inp):
        
        batch = inp.shape[0]
        inp = self.patch_embed(inp)
        
        cls_token = self.cls_token.expand(batch, -1, -1) # (batch, 1, emb_dim)
        inp = torch.cat((cls_token, inp), dim = 1) # (batch, n_ps + 1, emb_dim)
        inp = inp + self.pos_embed
        inp = self.pos_drop(inp)
        
        for block in self.blocks: inp = block(inp)
        
        inp = self.norm(inp)
        cls_token_out = inp[:, 0] # class token only
        
        return self.head(cls_token_out)
