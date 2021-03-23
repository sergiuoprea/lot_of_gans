## Imports
from torch import nn

## Discriminator


## Generator
def generator_block(input_dim, output_dim):
    """Function for returning a block of the generator's neural network,
    given ints input and output dimensions.

    Args:
        input_dim ([int]): [the dimension of the input vector, a scalar]
        output_dim ([int]): [the dimension of the output vector, a scalar]

    Returns:
        [nn.Sequential]: [a layer of the GAN generator, featuring a linear transformation,
                          followed by a batch normalization and finally a ReLu activation]
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

