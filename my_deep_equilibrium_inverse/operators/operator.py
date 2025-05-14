import torch

class LinearOperator(torch.nn.Module):
    def __init__(self):
        super(LinearOperator, self).__init__()

    def forward(self, x):
        pass

    def adjoint(self, x):
        pass

    def gramian(self, x):
        return self.adjoint(self.forward(x))

class SelfAdjointLinearOperator(LinearOperator):
    def adjoint(self, x):
        return self.forward(x)

class Identity(SelfAdjointLinearOperator):
    def forward(self, x):
        return x

class OperatorPlusNoise(torch.nn.Module):
    def __init__(self, operator, noise_sigma):
        super(OperatorPlusNoise, self).__init__()
        self.internal_operator = operator
        self.noise_sigma = noise_sigma

    def forward(self, x):
        A_x = self.internal_operator(x)
        return A_x + self.noise_sigma * torch.randn_like(A_x)
    
class OperatorPlusPoissonNoise(torch.nn.Module):
    def __init__(self, operator, scale_factor):
        # Higher the scale factor, lower the effect of Poisson noise
        super(OperatorPlusPoissonNoise, self).__init__()
        self.internal_operator = operator
        self.scale_factor = scale_factor

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        A_x = self.internal_operator(x)
        scaled = A_x * self.scale_factor
        noisy = torch.poisson(scaled)
        noisy = noisy / self.scale_factor

        return noisy