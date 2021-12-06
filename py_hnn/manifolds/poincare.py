"""Poincare ball manifold."""

import torch
#from scipy.special import beta

from manifolds.base import Manifold
from manifolds.hyperboloid import Hyperboloid
from utils.math_utils import artanh, tanh
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """
    name = 'PoincareBall'
    min_norm = 1e-8
    eps = {torch.float32: 4e-3, torch.float64: 1e-5}
    max_norm = 1 - 1e-5

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    @classmethod
    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * PoincareBall.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    @classmethod
    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(PoincareBall.min_norm)

    @classmethod
    def egrad2rgrad(self, p, dp, c):
        lambda_p = PoincareBall._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    @classmethod
    def proj(self, x, c):
        """Project points to Poincare ball with curvature c.
        Args:
            x: torch.Tensor of dim(M,N) with hyperbolic points
            c: manifold curvature.
        Returns:
            torch.Tensor with projected hyperbolic points.
        """
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), PoincareBall.min_norm)
        maxnorm = (1 - PoincareBall.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    @classmethod
    def proj_tan(self, u, p, c):
        return u

    @classmethod
    def proj_tan0(self, u, c):
        return u

    @classmethod
    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(PoincareBall.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * PoincareBall._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = PoincareBall.mobius_add(p, second_term, c)
        return gamma_1

    @classmethod
    def logmap(self, p1, p2, c):
        sub = PoincareBall.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(PoincareBall.min_norm)
        lam = PoincareBall._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    @classmethod
    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), PoincareBall.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    @classmethod
    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(PoincareBall.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    @classmethod
    def mobius_matvec(self, m, x, c):
        """Calculates the mobius matrix multiplication.
        Args:
            m: matrix of dim (M,N).
            a: matix of dim (N,P).
            c: curvature of the poincare ball.
        Returns:
             the matrix product of two arrays of dim (M,P)
        """
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(PoincareBall.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(PoincareBall.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    @classmethod
    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    @classmethod
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
        return w + 2 * (a * u + b * v) / d.clamp_min(PoincareBall.min_norm)

    @classmethod
    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = PoincareBall._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    @classmethod
    def ptransp(self, x, y, u, c):
        lambda_x = PoincareBall._lambda_x(x, c)
        lambda_y = PoincareBall._lambda_x(y, c)
        return PoincareBall._gyration(y, -x, u, c) * lambda_x / lambda_y

    @classmethod
    def ptransp_(self, x, y, u, c):
        lambda_x = PoincareBall._lambda_x(x, c)
        lambda_y = PoincareBall._lambda_x(y, c)
        return PoincareBall._gyration(y, -x, u, c) * lambda_x / lambda_y

    @classmethod
    def ptransp0(self, x, u, c):
        lambda_x = PoincareBall._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(PoincareBall.min_norm)

    @classmethod
    def to_hyperboloid(self, x, c=1, ideal=False):
        return Hyperboloid.from_poincare(x, c, ideal)
    
    @classmethod
    def from_hyperboloid(self, x, c=1, ideal=False):
        return Hyperboloid.to_poincare(x, c, ideal)

    @classmethod
    def distance(self, x, y, keepdim=True):
        """Hyperbolic distance on the Poincare ball with curvature c.
        Args:
            x: torch.Tensor of size B x d with hyperbolic points
            y: torch.Tensor of size B x d with hyperbolic points
        Returns: torch,Tensor with hyperbolic distances, size B x 1
        """
        pairwise_norm = PoincareBall.mobius_add(-x, y).norm(dim=-1, p=2, keepdim=True)
        dist = 2.0 * torch.atanh(pairwise_norm.clamp(-1 + PoincareBall.min_norm, 1 - PoincareBall.max_norm))
        if not keepdim:
            dist = dist.squeeze(-1)
        return dist
    
    
    @classmethod
    def pairwise_distance(self, x, keepdim=False):
        """All pairs of hyperbolic distances (NxN matrix)."""
        return PoincareBall.distance(x.unsqueeze(-2), x.unsqueeze(-3), keepdim=keepdim)
    
    
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
        return num / denom.clamp_min(PoincareBall.min_norm)

    @classmethod
    def mobius_midpoint(self, x, y):
        """Computes hyperbolic midpoint beween x and y."""
        t1 = PoincareBall.mobius_add(-x, y)
        t2 = PoincareBall.mobius_mul(t1, 0.5)
        return PoincareBall.mobius_add(x, t2)
    
    @classmethod
    def mobius_mul(self, x, t):
        """Mobius multiplication. 
        
        t*x = tanh(t*arctanh(|x|)) * x/|x|
        
        Note: arctanh(x) is only defined for x < 1
        """
        normx = x.norm(dim=-1, p=2, keepdim=True).clamp(min=PoincareBall.min_norm, max=PoincareBall.max_norm)
        return torch.tanh(t * torch.atanh(normx)) * x / normx
        
    @classmethod
    def _sq_gamma(self, v, r = 1):
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
        return 1/(1 - torch.pow(PoincareBall.distance0(v)/r, 2)).clamp(min=PoincareBall.min_norm, max=PoincareBall.max_norm)
        
        
    @classmethod
    def _mobius_midpoint(self, a, v):
        """Calculates the Einstein Midpoint for a weighted list of vectors.
        
        Note taht the eq. is the generalization of eq. 3.134 of  
        "Gyrovector space" by Ungar.

        args:
            a: The list of co-efficients (weights) in which each co-efficient 
                belongs to the vector with same index.
            v: The M*N dimensional time-like vectors.
        returns:
            An N dimensional time-like vector.
        """
        # Calculates the weighted vectors
        # Note: attention weights are scalars, we probably can have non-mobius mul here.
        # assert len(a) == v.size(dim=0)
        n_a = torch.nn.functional.normalize(a, dim=-1)
        w_v = torch.stack([PoincareBall.mobius_mul(v[i], n_a[i]) for i in range(0, len(n_a))])

        # Calculates the gamma factors for all vectors        
        gamma_ws = torch.FloatTensor([PoincareBall._sq_gamma(w_v_j) for _, w_v_j in enumerate(w_v)])
        weights = (gamma_ws / (torch.sum(gamma_ws) - len(a) / 2)).reshape(len(a), 1)

        # Generalized mobius midpoint
        return PoincareBall.mobius_mul(x=torch.sum(weights * v, dim=0), t=0.5)
          
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
            c: 
        returns:
            a tesnor of dim (M, 1) where contains the attention weight based on 
            corresponding query and key vectors.
        """
        # assert q.size(dim=0) == k.size(dim=0)
        # TODO(): in tensorflow, the for loop is optimized in compile time.
        # either impletemtn it in tensorflow, or find a way to unroll the loop
        # for pytorch for GPU performance.
        return torch.stack([PoincareBall.expmap(beta*PoincareBall.distance(q[idx], k[idx]) - c) for idx in enumerate(q)])
    
    @classmethod
    def poincare_aggregation(self, q, k, v):
        """Calculates the poincare attention for the given query, key, and values.
    
        Note: if used in Graph attention mechansim, all the masked vectors need 
            to be filtered out from q,k,v before invoking this function.

        args:
            q: M*M*N dimensional matrix of queries, where M is the number of 
                locations to attend to.
            k: M*M*N dimensional matrix of keys, where M is the number of 
                locations to attend to.
            v: M*M*N dimensional matrix of values, where M is the number of 
                locations to attend to. Note that v is a matrix where each row 
                is a vector in poincate model.
        returns:
            The self-attention calculation in M*N dimensional matrix form .
            i_th row corresponds to the attention embeddings for a location i.
        """
        # assert q.size(dim=x) == k.size(dim=x) == v.size(dim=x) for all dims
        h = []
        for i, v_i in enumerate(v):
            h.append([PoincareBall._mobius_midpoint(PoincareBall.poincare_attention_weight(q[i], k[i]), v_i)])
        return torch.stack(h)
        