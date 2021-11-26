"""Poincare ball manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import artanh, tanh
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-8
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.max_norm = 1 - 1e-5

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

    ########################################################################
    ########################################################################
    ########################################################################
    @classmethod
    def distance(self, x, y, keepdim=True):
        """Hyperbolic distance on the Poincare ball with curvature c.
        Args:
            x: torch.Tensor of size B x d with hyperbolic points
            y: torch.Tensor of size B x d with hyperbolic points
        Returns: torch,Tensor with hyperbolic distances, size B x 1
        """
        pairwise_norm = self.mobius_add(-x, y).norm(dim=-1, p=2, keepdim=True)
        dist = 2.0 * torch.atanh(pairwise_norm.clamp(-1 + MIN_NORM, 1 - MIN_NORM))
        if not keepdim:
            dist = dist.squeeze(-1)
        return dist
    
    
    @classmethod
    def pairwise_distance(self, x, keepdim=False):
        """All pairs of hyperbolic distances (NxN matrix)."""
        return self.distance(x.unsqueeze(-2), x.unsqueeze(-3), keepdim=keepdim)
    
    
    @classmethod
    def distance0(self, x, keepdim=True):
        """Computes hyperbolic distance between x and the origin."""
        x_norm = x.norm(dim=-1, p=2, keepdim=True)
        d = 2 * torch.atanh(x_norm.clamp(-1 + 1e-15, 1 - 1e-15))
        if not keepdim:
            d = d.squeeze(-1)
        return d
    
    @classmethod
    def mobius_add(self, x, y):
        """Mobius addition."""
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        # num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        # denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        denom = 1 + 2 * xy + x2 * y2
        return num / denom.clamp_min(MIN_NORM)

    @classmethod
    def mobius_midpoint(self, x, y):
        """Computes hyperbolic midpoint beween x and y."""
        t1 = self.mobius_add(-x, y)
        t2 = self.mobius_mul(t1, 0.5)
        return self.mobius_add(x, t2)
    
    @classmethod
    def mobius_mul(self, x, t):
        """Mobius multiplication. 
        
        t*x = tanh(t*arctanh(|x|)) * x/|x|
        
        Note: arctanh(x) is only defined for x < 1
        """
        normx = x.norm(dim=-1, p=2, keepdim=True).clamp(min=MIN_NORM, max=1. - 1e-5)
        return torch.tanh(t * torch.atanh(normx)) * x / normx
        
    @classmethod
    def _sq_gamma(self, v, r):
        """Calculates Gamma factor for poincare ball.
        
        according to eq. 2.45 of "Gyrovector space" by Ungar.
        L  = 1/sqrt(1 - (distance0(v)/ball_radius)^2)
        
        args:
            v: vector of dime (N) in poincare ball.
            r: radius of poincare ball.
        returns:
            a scalar that corresponds to the gamma factor for a given vector
            in r-ball.
        """
        return 1/(1 - torch.pow(self.distance0(v)/r, 2)).clamp_min(min=self.min, max=self.max_norm)
        
        
    @classmethod
    def _mobius_midpoint(self, a, v):
        """Calculates the Einstein Midpoint for a weighted list of vectors.
        
        Note taht the eq. is the generalization of eq. 3.134 of  
        "Gyrovector space" by Ungar.

        args:
            a: The list of co-efficients (weights) in which each co-efficient 
                belongs to the vector with same index.
            v: The list of N dimensional time-like vectors.
        returns:
            An N dimensional time-like vector.
        """
        gamma_ws = torch.FloatTensor([self._sq_gamma(v_j) for _, v_j in enumerate(v)])
        total_gamma_ws = torch.sum(gamma_ws) - len(a) / 2
        n_gamma_ws = gamma_ws / total_gamma_ws
        # Calculates the weighted vectors
        # assert len(a) == v.size(dim=0)
        normalized_a = torch.nn.functional.normalize(a)
        # Note: attention weights are scalars, we probably can have non-mobius mul here.
        # weighted_v = self.mobius_mul(v[idx], normalized_a[idx])
        weights = (n_gamma_ws * normalized_a).reshape(len(a), 1)
        return self.mobius_mul(0.5, torch.sum(weights * v, dim=0))
          
    @classmethod
    def poincare_attention_weight(self, q, k, beta, c):
        """Calculate an attention wight for a given query, and keys.
        
        a(q_i,k_j) = expmap(-Beta*distnace(q,k) - Constant)
        more details: "Hyperbolic attention network" eq. 2
        
        Note: Beta can be either set manually or learned from query vector.
        Note: Both vectors must already be in hyperbolic space. (no poincare, no klein)
    
        args:
            q: an N-dimensional query vector for a location i.
            k: keys for the memory locations (N-dimensional)
        returns:
            a tesnor of dim (N) where contains the attention weight based on 
            corresponding query and key vectors.
        """
        # assert q.size(dim=0) == k.size(dim=0)
        # TODO(): in tensorflow, the for loop is optimized in compile time.
        # either impletemtn it in tensorflow, or find a way to unroll the loop
        # for pytorch for GPU performance.
        return torch.stack([self.expmap(beta*self.distance(q[idx], k[idx]) - c) for idx in enumerate(q)])
    
    @classmethod
    def hyperbolic_poincare_aggregation(self, q, k, v):
        """Calculates the poincare attention for the given query, key, and values.
    
        args:
            q: N*M dimensional matrix of queries, where M is the number of 
                locations to attend to.
            k: N*M dimensional matrix of keys, where M is the number of 
                locations to attend to.
            v: N*M dimensional matrix of values, where M is the number of 
                locations to attend to. Note that v is a matrix where each row is
                a vector in poincate model.
        returns:
            z: The self-attention calculation in N*M dimensional matrix form .
                i_th row corresponds to the attention embeddings for a location i.
        """
        # assert q.size(dim=x) == k.size(dim=x) == v.size(dim=x) for all dims
        a = self.poincare_attention_weight(q, k)
        return self._mobius_midpoint(a, v)
