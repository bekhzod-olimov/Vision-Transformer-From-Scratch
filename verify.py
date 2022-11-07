# Import libraries
import timm, torch
from model import VisionTransformer
from utils import get_n_params, assert_tensors_equal

# Set model name
model_name = "vit_base_patch16_384"

# Get official implementation from timm library
model_official = timm.create_model(model_name, pretrained = True)

# Swith to evaluation mode
model_official.eval()

# Initialize config for a custom model
custom_config = {"im_size": 384, "in_chs": 3, "p_size": 16, "emb_dim": 768, "depth": 12, "n_heads": 12, "qkv_bias": True, "mlp_ratio": 4}

# Construct a custom model
model_custom = VisionTransformer(**custom_config)
# Swith to evaluation mode
model_custom.eval()


# Start comparison between a custom model and a timm model
for (official_name, official_param), (custom_name, custom_param) in zip(model_official.named_parameters(), model_custom.named_parameters()):
    
    # Assert that number of parameters are the same 
    assert official_param.numel() == custom_param.numel()
    print(f"{official_name} | {custom_name}")
    
    custom_param.data[:] = official_param.data
    # Assert that values of the tensors are equal
    assert_tensors_equal(official_param.data, custom_param.data)
    

# Set an input tensor to the models     
inp = torch.rand(1, 3, 384, 384)

# Get a timm model output
official_out = model_official(inp)

# Get a custom model output
custom_out = model_custom(inp)

# Assertions
assert get_n_params(model_official) == get_n_params(model_custom)
assert_tensors_equal(official_out, custom_out)

# Save the model
torch.save(model_custom, "test.pth")