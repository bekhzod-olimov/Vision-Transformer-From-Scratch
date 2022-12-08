import numpy as np
import timm, torch
from model import VisionTransformer

def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    np.testing.assert_allclose(a1, a2)
    
model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained = True)
model_official.eval()
print(type(model_official))

custom_config = {"im_size": 384, "in_chs": 3, "p_size": 16, "emb_dim": 768, "depth": 12, "n_heads": 12, "qkv_bias": True, "mlp_ratio": 4}
model_custom = VisionTransformer(**custom_config)
model_custom.eval()

for (official_name, official_param), (custom_name, custom_param) in zip(model_official.named_parameters(), model_custom.named_parameters()):
    
    assert official_param.numel() == custom_param.numel()
    print(f"{official_name} | {custom_name}")
    custom_param.data[:] = official_param.data
    assert_tensors_equal(official_param.data, custom_param.data)
    
inp = torch.rand(1, 3, 384, 384)
official_out = model_official(inp)
custom_out = model_custom(inp)

assert get_n_params(model_official) == get_n_params(model_custom)
assert_tensors_equal(official_out, custom_out)
torch.save(model_custom, "test.pth")
