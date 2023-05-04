# Import library
import numpy as np

# A function to get number of trainable parameters of a model
def get_n_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# A function to assert that two tensors are equal
def assert_tensors_equal(t1, t2): np.testing.assert_allclose(t1.detach().numpy(), t2.detach().numpy())
