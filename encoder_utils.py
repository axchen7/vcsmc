from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, width: int, depth: int):
        """
        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.flatten = nn.Flatten()

        mlp = nn.Sequential()

        mlp.add_module("input", nn.Linear(in_features, width))
        mlp.add_module("relu_input", nn.ReLU())

        for i in range(depth - 1):
            mlp.add_module(f"hidden_{i}", nn.Linear(width, width))
            mlp.add_module(f"relu_{i}", nn.ReLU())

        mlp.add_module("output", nn.Linear(width, out_features))
        self.mlp = mlp

    def forward(self, x: Tensor) -> Tensor:
        """
        Flatten the input tensor and pass it through the MLP.
        """
        return self.mlp(self.flatten(x))
