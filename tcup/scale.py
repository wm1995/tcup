from abc import ABC, abstractmethod

import jax.numpy as jnp


class Scaler(ABC):
    @abstractmethod
    def __init__(self, x, cov_x, y, dy):
        pass

    @abstractmethod
    def transform(self, x, cov_x, y, dy):
        pass

    @abstractmethod
    def transform_coeff(self, alpha, beta, sigma):
        pass

    @abstractmethod
    def inv_transform(self, x_scaled, cov_x_scaled, y_scaled, dy_scaled):
        pass

    @abstractmethod
    def inv_transform_coeff(self, alpha_scaled, beta_scaled, sigma_scaled):
        pass


class NoScaler(Scaler):
    def __init__(self, x, cov_x, y, dy):
        pass

    def transform(self, x, cov_x, y, dy):
        return x, cov_x, y, dy

    def transform_coeff(self, alpha, beta, sigma):
        return alpha, beta, sigma

    def inv_transform(self, x_scaled, cov_x_scaled, y_scaled, dy_scaled):
        return x_scaled, cov_x_scaled, y_scaled, dy_scaled

    def inv_transform_coeff(self, alpha_scaled, beta_scaled, sigma_scaled):
        return alpha_scaled, beta_scaled, sigma_scaled


class StandardScaler(Scaler):
    def __init__(self, x, cov_x, y, dy):
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0)

    def transform(self, x, cov_x, y, dy):
        x_scaled = (x - self.x_mean) / self.x_std
        cov_x_scaled = (
            cov_x / self.x_std[:, jnp.newaxis] / self.x_std[jnp.newaxis, :]
        )
        y_scaled = (y - self.y_mean) / self.y_std
        dy_scaled = dy / self.y_std
        return x_scaled, cov_x_scaled, y_scaled, dy_scaled

    def transform_coeff(self, alpha, beta, sigma):
        alpha_scaled = (
            (alpha - self.y_mean) + jnp.dot(beta, self.x_mean)
        ) / self.y_std
        beta_scaled = beta * self.x_std / self.y_std
        sigma_scaled = sigma / self.y_std
        return alpha_scaled, beta_scaled, sigma_scaled

    def inv_transform(self, x_scaled, cov_x_scaled, y_scaled, dy_scaled):
        x = x_scaled * self.x_std + self.x_mean
        cov_x = (
            cov_x_scaled
            * self.x_std[:, jnp.newaxis]
            * self.x_std[jnp.newaxis, :]
        )
        y = y_scaled * self.y_std + self.y_mean
        dy = dy_scaled * self.y_std
        return x, cov_x, y, dy

    def inv_transform_coeff(self, alpha_scaled, beta_scaled, sigma_scaled):
        beta = self.y_std * beta_scaled / self.x_std[:, jnp.newaxis]
        alpha = (
            self.y_mean
            + self.y_std * alpha_scaled
            - jnp.dot(self.x_mean, beta)
        )
        sigma = self.y_std * sigma_scaled
        return alpha, beta, sigma
