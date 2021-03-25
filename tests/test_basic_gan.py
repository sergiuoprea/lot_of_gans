# Pytorc and Lightning
import torch
from torch import nn
from pytorch_lightning import seed_everything
from project.basic_gan import generator_block, Generator

# Testing libs
import pytest

# Testing the building blocks used in the Generator
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

# Testing the Generator
@pytest.mark.parametrize("z_dim,im_dim,hidden_dim,num_test", [(5,10,20,10000),(20,8,24,10000)])
def test_generator(z_dim, im_dim, hidden_dim, num_test):
    gen = Generator(z_dim, im_dim, hidden_dim).get_gen()
    
    # Six blocks in the generator
    assert len(gen) == 6
    
    test_input = torch.randn(num_test, z_dim)
    test_output = gen(test_input)
    
    assert tuple(test_output.shape) == (num_test, im_dim) # Output shape is correct
    # Make sure we are using a sigmoid as final layer
    assert test_output.max() < 1
    assert test_output.min() > 0
    # No batchnorm in the output
    assert test_output.std() > 0.05
    assert test_output.std() < 0.15
    