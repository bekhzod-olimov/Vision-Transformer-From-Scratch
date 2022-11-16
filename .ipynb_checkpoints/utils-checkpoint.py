import numpy as np

def get_n_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2): np.testing.assert_allclose(t1.detach().numpy(), t2.detach().numpy())