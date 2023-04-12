import numpy as np


class Scaler:
    def __init__(self, x, dx, y, dy):
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0)

    def transform(self, x, dx, y, dy):
        x_scaled = (x - self.x_mean) / self.x_std
        dx_scaled = dx / self.x_std[:, np.newaxis] / self.x_std[np.newaxis, :]
        y_scaled = (y - self.y_mean) / self.y_std
        dy_scaled = dy / self.y_std
        return x_scaled, dx_scaled, y_scaled, dy_scaled

    def transform_coeff(self, alpha, beta, sigma):
        alpha_scaled = (
            (alpha - self.y_mean) + np.dot(beta, self.x_mean)
        ) / self.y_std
        beta_scaled = beta * self.x_std / self.y_std
        sigma_scaled = sigma / self.y_std
        return alpha_scaled, beta_scaled, sigma_scaled

    def inv_transform(self, x_scaled, dx_scaled, y_scaled, dy_scaled):
        x = x_scaled * self.x_std + self.x_mean
        dx = dx_scaled * self.x_std[:, np.newaxis] * self.x_std[np.newaxis, :]
        y = y_scaled * self.y_std + self.y_mean
        dy = dy_scaled * self.y_std
        return x, dx, y, dy

    def inv_transform_coeff(self, alpha_scaled, beta_scaled, sigma_scaled):
        beta = self.y_std * beta_scaled / self.x_std
        alpha = (
            self.y_mean + self.y_std * alpha_scaled - np.dot(self.x_mean, beta)
        )
        sigma = self.y_std * sigma_scaled
        return alpha, beta, sigma
