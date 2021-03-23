# Pytorc and Lightning
import torch
from torch import nn
from pytorch_lightning import seed_everything
from project.basic_gan import generator_block

# Testing libs
import pytest

@pytest.mark.parametrize("in_features,out_features,num_test", [(15,12,1000), (15,18,1000)])
def test_generator_block(in_features, out_features, num_test):
    seed_everything(0) # fixed value. do not change.
    
    block = generator_block(in_features, out_features)
    
    assert len(block) == 3 # We want three layers with the following config
    
    # Check layer configuration
    assert type(block[0]) == nn.Linear
    assert type(block[1]) == nn.BatchNorm1d
    assert type(block[2]) == nn.ReLU
    
    # Check the output shape
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)
    assert tuple(test_output.shape) == (num_test, out_features)
    assert test_output.std() > 0.55
    assert test_output.std() < 0.65
