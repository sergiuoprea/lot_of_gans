# Pytorc and Lightning
import torch
from torch import nn
from pytorch_lightning import seed_everything
from project.basic_gan import generator_block, Generator, discriminator_block

# Testing libs
import pytest

seed_everything(0) # fixed value. do not change.

# Testing the building blocks used in the Generator
@pytest.mark.parametrize("in_features,out_features,num_test", [(15,12,1000), (15,18,1000)])
def test_generator_block(in_features, out_features, num_test):
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

@pytest.mark.parametrize("in_features,out_features,num_test"), [(25,12,10000),(15,28,10000)]
def test_discriminator_block(in_features, out_features, num_test):
    block = discriminator_block(in_features, out_features)
    
    assert len(block) == 2 # We want two layers into the block
    assert type(block[0]) == nn.Linear # A linear layer
    assert type(block[1]) == nn.LeakyReLU # A leaky relu act. function
    
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)
    
    assert tuple(test_output.shape) == (num_test, out_features) # Check the output shape is fine
    
    # Check the LeakyReLU slope is about 0.2
    assert -test_output.min() / test_output.max() > 0.1
    assert -test_output.min() / test_output.max() < 0.3
    assert test_output.std() > 0.3
    assert test_output.std() < 0.5