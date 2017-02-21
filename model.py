from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model_part

class Discriminator():
    def __init__(self, hidden_layer_dim, output_dim):
        self.hidden_layer_dim = hidden_layer_dim
        self.output_dim = output_dim

    def mlp(self, x):
        fc1 = model_part.fc('fc1', x, [x.get_shape()[1], self.hidden_layer_dim], [self.hidden_layer_dim])
        logits = model_part.fc('fc2', fc1, [self.hidden_layer_dim, self.output_dim], [self.output_dim])
        return logits