import torch
import torch.nn as nn


class FlowMatching(nn.Module):
    """Flow matching model with a linear interpolation between noise and data (a.k.a. rectified flow)

    Convention: t = 0 is pure noise eps ~ N(0, I), t = 1 is data z.

    Q2 reuses `interpolate`, `get_loss` and `sample` unchanged to train a class-conditional model on images: it
    subclasses this class and overrides `get_velocity` and `sample_times`. So write these three methods for data of
    any shape (N, ...), draw the times with `self.sample_times`, and pass the extra arguments `*cond` through to
    `get_velocity` (in Q1 there are none; in Q2 they are the class labels and the guidance scale).
    """

    def __init__(self, velocity_net):
        """
        :param velocity_net: an `nn.Module` instance that takes the concatenation [x, t] of shape (N, D + 1)
                             and predicts the velocity u_t(x) of shape (N, D)
        """
        super().__init__()
        self.velocity_net = velocity_net

    def get_velocity(self, x, t):
        """
        Predict the vector field u_t(x) at points x and times t (provided)
        :param x: points of shape (N, D)
        :param t: times, either a float or a tensor with N elements
        :return: the predicted velocity, of shape (N, D)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        t = t.to(x).reshape(-1, 1).expand(x.size(0), 1)
        return self.velocity_net(torch.cat([x, t], dim=-1))

    def sample_times(self, n, device=None, dtype=None):
        """
        Draw one training time per data point, t ~ U[0, 1] (provided)
        :return: times of shape (n,)
        """
        return torch.rand(n, device=device, dtype=dtype)

    def interpolate(self, z):
        """
        Sample points on the straight lines between noise and the data z.
        You should draw noise eps ~ N(0, I) of the same shape as z, and draw one time per data point in the batch
        (NOT for each pixel/coordinate!) with `self.sample_times`. Reshape the times to (N, 1, ..., 1), with as many
        dimensions as z, so that they broadcast. Then compute x_t = t * z + (1 - t) * eps.
        :param z: batch of data of shape (N, ...), e.g. (N, D) in Q1 and (N, C, H, W) in Q2
        :return: noise eps of the same shape as z
                 times t of shape (N, 1, ..., 1), e.g. (N, 1) in Q1
                 interpolated points x_t of the same shape as z
        """
        # TODO(student): Complete this function
        raise NotImplementedError

    def get_loss(self, z, *cond):
        """
        Calculate the conditional flow matching loss, with the conditional vector field z - eps as the target.
        Predict the velocity with `self.get_velocity(x_t, t, *cond)`.
        The loss should be averaged over the batch dimension and the data dimensions.
        :param z: batch of data of shape (N, ...)
        :param cond: extra arguments for `get_velocity` (none in Q1)
        :return: conditional flow matching loss, a scalar tensor
        """
        # TODO(student): Complete this function
        raise NotImplementedError

    def sample(self, x, *cond, n_steps=100):
        """
        Generate samples by simulating the ODE dX_t/dt = u_t(X_t) from t = 0 (noise) to t = 1 (data) with the
        Euler method. Predict the velocity with `self.get_velocity(x, t, *cond)`.
        :param x: initial noise X_0 ~ N(0, I) of shape (N, ...)
        :param cond: extra arguments for `get_velocity` (none in Q1)
        :param n_steps: number of Euler steps n
        :return: trajectories of shape (n_steps + 1, N, ...), where index 0 is the initial noise X_0
                 and index -1 is the final sample X_1
        """
        self.eval()
        ts = torch.linspace(0, 1, n_steps + 1).tolist()

        traj = [x]
        with torch.no_grad():  # not a @torch.no_grad() decorator: Jupyter's autoreload cannot update decorated methods
            # TODO(student): Complete this function
            pass
        traj = torch.stack(traj, dim=0)
        return traj

    def forward(self, x, t):
        """
        Predict the vector field u_t(x) at points x and times t
        :param x: points of shape (N, D)
        :param t: times, either a float or a tensor with N elements
        :return: the predicted velocity, of shape (N, D)
        """
        return self.get_velocity(x, t)
