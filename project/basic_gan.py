## Imports
from torch import nn

########################################################
#################### DISCRIMINATOR  ####################
########################################################


########################################################
###################### GENERATOR  ######################
########################################################
def generator_block(input_dim, output_dim):
    """Function for returning a block of the generator's neural network,
    given ints input and output dimensions.

    Args:
        input_dim (int): the dimension of the input vector, a scalar
        output_dim (int): the dimension of the output vector, a scalar

    Returns:
        nn.Sequential: a layer of the GAN generator, featuring a linear transformation,
                       followed by a batch normalization and finally a ReLu activation
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )
    
    
class Generator(nn.Module):
    """Class defining the Generator

    Args:
        z_dim (int): dimension of the noise vector (default: 10).
        im_dim (int): dimension of the images used for training
            (default: 784 as MNIST images are 28x28 pixels).
        hidden_dim (int): the inner dimension (default: 128).
    """
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        
        self.generator = nn.Sequential(
            generator_block(z_dim, hidden_dim),
            generator_block(hidden_dim, hidden_dim * 2),
            generator_block(hidden_dim * 2, hidden_dim * 4),
            generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
        
    def forward(self, noise):
        """Forward pass of the generator given a noise tensor.

        Args:
            noise (tensor): noise tensor with dimensions (n_samples, z_dim).

        Returns:
            tensor: generated image from the input noise (batch_size, im_dim).
        """
        return self.gen(noise)
    
    def get_gen(self):
        """Getter for the generator

        Returns:
            nn.Sequential: a sequential model of the generator.
        """
        return self.gen
