import torch


class L1Regulariser:
    @staticmethod
    def loss(weights):
        """Computes the loss term added by the regulariser (a scalar)."""
        return torch.abs(weights).sum()

    @staticmethod
    def gradients(weights):
        """Computes subgradients of the regularisation term with respect to the weights (a vector)."""
        return (weights > 0).float()


class L2Regulariser:
    @staticmethod
    def loss(weights):
        """Computes the loss term added by the regulariser (a scalar)."""
        return torch.pow(weights, 2).sum()

    @staticmethod
    def gradients(weights):
        """Computes the gradients of the regularisation term with respect to the weights (a vector)."""
        return 2 * weights
