"""Particle Beam Microscope Time-Resolved Measurement Simulation and Plug-and-Play Reconstruction Methods

Author: Ruangrawee (Kao) Kitichotkul
Credit: 
    Thanks Minxu Peng for a code reference which inspires this code. The function trmlEstimate is mostly Minxu's code.
"""

import numpy as np
import torch
import sigpy as sp
import bm3d
from skimage.restoration import denoise_tv_chambolle
from scipy.signal import convolve2d
from scipy.stats import poisson
from scipy.special import lambertw
from tqdm import tqdm

import util

"""Forward Model"""

def poiPoiSampling(etaImage, lamb, retM = False):
    M = np.random.poisson(lamb, size=etaImage.shape)
    y = np.zeros_like(etaImage)
    for i in range(1, M.max() + 1):
        indices = np.where(M >= i)
        y[indices] += np.random.poisson(etaImage[indices])
    if retM:
        return y, M
    else:
        return y

def gaussianSampling(etaImage, lamb):
    return lamb * etaImage + np.random.randn(*etaImage.shape) * \
            np.sqrt((lamb * etaImage * (etaImage + 1)))

def poiPoiTrSampling(etaImage, lamb, n, retM = False):
    y = [None] * n
    M = [None] * n
    for it in range(n):
        thisY, thisM = poiPoiSampling(etaImage, lamb / n, retM = True)
        y[it] = thisY
        M[it] = thisM
    y = np.stack(y, axis=0)
    M = np.stack(M, axis=0)
    if retM:
        return y, M
    else:
        return y

def minxuPoiPoiSampling(etaImage, lamb, n):
    """Should be the same as poiPoiTrSampling"""
    dose_per_acqu = lamb / n
    ions = np.random.poisson(lam=dose_per_acqu, size=(*etaImage.shape, n))
    y = np.random.poisson(lam=ions * np.expand_dims(etaImage, axis=-1))
    return y.transpose((2, 0, 1)), ions.transpose((2, 0, 1))

"""Data-fidelity Proximal Operators and Gradients"""

class GaussNaiveFidProx(sp.prox.Prox):
    def __init__(self, y, lamb):
        super().__init__(y[0].shape)
        self.etaNaive = np.sum(y, axis=0) / lamb
        self.reqPrevEta = False

    def _prox(self, rho, input):
        return (self.etaNaive + rho * input) / (1 + rho)

class GaussApproxFidProx:
    def __init__(self, y, lamb):
        self.yConv = np.sum(y, axis=0)
        self.lamb = lamb
        self.reqPrevEta = True
        self.kernel = np.ones((3, 3)) / 9

    def __call__(self, rho, input, etaPrev):
        etaPrev = convolve2d(etaPrev, self.kernel, mode='same', boundary='symm') # average etaPrev by neighboring pixels
        enumTerm = self.yConv / (etaPrev * (etaPrev + 1)) + rho * input
        denomTerm = self.lamb / (etaPrev * (etaPrev + 1)) + rho
        return enumTerm / denomTerm

class GaussFidGrad:
    def __init__(self, y, lamb):
        self.yConv = np.sum(y, axis=0)
        self.lamb = lamb

    def __call__(self, eta):
        grad = 0.5 / (eta + 1)
        grad += 0.5 / eta
        termA = -self.lamb * eta + self.yConv
        grad -= termA / (eta * (eta + 1))
        grad -= (termA ** 2) / (2 * self.lamb * eta * ((eta + 1 ** 2)))
        grad -= (termA ** 2) / (2 * self.lamb * (eta + 1) * (eta ** 2))
        return grad

class OracleFidProx(sp.prox.Prox):
    def __init__(self, y, M):
        super().__init__(y[0].shape)
        self.yConv = np.sum(y, axis=0)
        self.M = M
        self.reqPrevEta = False
        assert self.M.shape == self.yConv.shape

    def _prox(self, rho, input):
        diff = self.M / rho - input
        return 0.5 * (-diff + np.sqrt(diff ** 2 + 4 * self.yConv / rho))

class OracleFidGrad:
    def __init__(self, y, M):
        self.yConv = np.sum(y, axis=0)
        self.M = M
        assert self.M.shape == self.yConv.shape

    def __call__(self, eta):
        return self.M - self.yConv / eta

class QMFidProx(sp.prox.Prox):
    def __init__(self, y):
        super().__init__(y[0].shape)
        self.yConv = np.sum(y, axis=0)
        self.L = np.sum(y > 0, axis=0).astype(float)
        self.reqPrevEta = False

    def _prox(self, rho, input):
        diff = self.L / rho - input
        return 0.5 * (-diff + np.sqrt(diff ** 2 + 4 * self.yConv / rho))

class QMFidGrad:
    def __init__(self, y):
        self.yConv = np.sum(y, axis=0)
        self.L = np.sum(y > 0, axis=0).astype(float)

    def __call__(self, eta):
        return self.L - self.yConv / eta

class LQMApproxFidProx:
    def __init__(self, y):
        self.yConv = np.sum(y, axis=0)
        self.L = np.sum(y > 0, axis=0).astype(float)
        self.reqPrevEta = True

    def __call__(self, rho, input, etaPrev):
        diff = self.L / (rho * (1 - np.exp(-etaPrev))) - input
        return 0.5 * (-diff + np.sqrt(diff ** 2 + 4 * self.yConv / rho))

class LQMFidGrad:
    def __init__(self, y):
        self.yConv = np.sum(y, axis=0)
        self.L = np.sum(y > 0, axis=0).astype(float)

    def __call__(self, eta):
        firstTerm = - self.L * eta * np.exp(-eta)
        firstTerm /= (1 - np.exp(-eta)) ** 2
        secondTerm = self.L + self.yConv * np.exp(-eta)
        secondTerm /= (1 - np.exp(-eta))
        thirdTerm = - self.yConv / eta
        return firstTerm + secondTerm + thirdTerm

class TRApproxFidProx:
    def __init__(self, y, lamb, stepSize, numSteps):
        self.yConv = np.sum(y, axis=0)
        self.L = np.sum(y > 0, axis=0).astype(float)
        self.n = y.shape[0]
        self.lamb = lamb
        self.y = y

        self.mask = (y > 0).astype(float)
        self.sumEnumTerm = (2 ** y - 1) * lamb
        self.sumDenomTerm = (2 ** (y - 1) - 1) * lamb

        self.stepSize = stepSize
        self.numSteps = numSteps
        self.reqPrevEta = True

    def __call__(self, rho, input, etaPrev):
        etaEstimate = np.copy(etaPrev)
        for it in range(self.numSteps):
            etaEstimate -= self.stepSize * self._computeGrad(etaEstimate, input, rho)
        return etaEstimate

    def _computeGrad(self, eta, input, rho):
        firstTerm = (self.n - self.L) * (self.lamb / self.n) * np.exp(-eta)
        secondTerm = - self.yConv / eta
        thirdTerm = (self.n + self.sumEnumTerm * np.exp(-eta)) / (self.n + self.sumDenomTerm * np.exp(-eta))
        thirdTerm *= self.mask
        thirdTerm = np.sum(thirdTerm, axis=0)
        fgrad = firstTerm + secondTerm + thirdTerm
        squarePenaltyGrad = rho * (eta - input)
        return fgrad + squarePenaltyGrad

class TRApproxFidGrad:
    def __init__(self, y, lamb):
        self.yConv = np.sum(y, axis=0)
        self.L = np.sum(y > 0, axis=0).astype(float)
        self.n = y.shape[0]
        self.lamb = lamb
        self.y = y

        self.mask = (y > 0).astype(float)
        self.sumEnumTerm = (2 ** y - 1) * lamb
        self.sumDenomTerm = (2 ** (y - 1) - 1) * lamb


    def __call__(self, eta):
        firstTerm = (self.n - self.L) * (self.lamb / self.n) * np.exp(-eta)
        secondTerm = - self.yConv / eta
        thirdTerm = (self.n + self.sumEnumTerm * np.exp(-eta)) / (self.n + self.sumDenomTerm * np.exp(-eta))
        thirdTerm *= self.mask
        thirdTerm = np.sum(thirdTerm, axis=0)
        return firstTerm + secondTerm + thirdTerm

"""Regularization Proximal Operators"""

class TVPriorProx:
    """
    Note:
        The TV prior prox is
            prox(x) = argmin_v beta * ||Dv||_2 + rho / 2 * ||v - x||_2^2
        which is the solution of the TV-regularized denoising problem;
        we can solve the problem using an iterative optimization algorithm,
        e.g. skimage's Chambolle-Pock implementation of TV denoising.
        
        Empirically, the optimal weight seems to be 0.5 * std for
        an image corrupted by Gaussian noise with strength std.

        We use the isotropic version of the TV regularizer, and thus the l2 norm at ||Dv||_2, in this work;
        this is what scikit-image's denoise_tv_chambolle function provides.
    """
    def __init__(self, std):
        self.std = std

    def __call__(self, input):
        return denoise_tv_chambolle(input, weight = 0.5 * self.std)

class Bm3dPriorProx:
    def __init__(self, std):
        self.std = std

    def __call__(self, input):
        return bm3d.bm3d(input, self.std)

class DnCNNPriorProx:
    def __init__(self, model, device=torch.device('cpu'), mu = 1.0,
                imRange = [2, 8], denRange = [0, 1]):
        model.eval()
        self.model = model.to(device=device)
        self.device = device
        self.preTransform = lambda x: (x - imRange[0]) / (imRange[1] - imRange[0]) * (denRange[1] - denRange[0]) + denRange[0]
        self.postTransform = lambda x: (x - denRange[0]) / (denRange[1] - denRange[0]) * (imRange[1] - imRange[0]) + imRange[0]

        # Denoiser scaling (Xu et al., 2020)
        # Boosting the Performance of Plug-and-Play Priors via Denoiser Scaling
        self.mu = mu

    @torch.no_grad()
    def __call__(self, input):
        shape = input.shape
        input = self.preTransform(input) * self.mu
        input = torch.Tensor(input).view(1, 1, *shape).to(device=self.device)
        out = self.model(input).to(torch.device('cpu'))
        out = np.array(out[0, 0]) / self.mu
        out = self.postTransform(out)
        return out

"""Solver"""

class ADMM(sp.alg.Alg):
    """Alternating Direction Method of Multipliers"""
    def __init__(self, proxf, proxg, x, v, u, rho, maxIter, tol):
        super().__init__(maxIter)
        self.proxf = proxf
        self.proxg = proxg
        self.x = x
        self.v = v
        self.u = u
        self.rho = rho

        # For stopping criterion
        self.tol = tol
        self.numPixels = np.prod(x.shape)
        self.residue = np.inf
        
    def _update(self):
        # Back up variables for stopping criterion
        xPrev = np.copy(self.x)
        vPrev = np.copy(self.v)
        uPrev = np.copy(self.u)

        # ADMM updates
        if self.proxf.reqPrevEta:
            self.x = self.proxf(self.rho, self.v - self.u, xPrev)
        else:
            self.x = self.proxf(self.rho, self.v - self.u)
        self.v = self.proxg(self.x + self.u)
        self.u += self.x - self.v

        # Compute residue for stopping criterion
        self.residue = np.linalg.norm(self.x - xPrev) \
                    + np.linalg.norm(self.v - vPrev) \
                    + np.linalg.norm(self.u - uPrev)
        self.residue /= np.sqrt(self.numPixels)

    def _done(self):
        return (self.iter >= self.max_iter) or (self.residue <= self.tol)

class ADMMApp(sp.app.App):
    def __init__(self, fidProx, priorProx, rho, xInit, maxIter,
                tol = 5e-4, xGt = None, fullLogging = False, retV = True):

        # Initialize variables
        # Note that initial self.x doesn't matter to soln/convergence, 
        # because x is updated first.
        self.x = np.copy(xInit)
        self.v = np.copy(xInit)
        self.u = np.zeros(xInit.shape)

        # Whether to return x or v
        self.retV = retV
        
        # Define algorithm
        alg = ADMM(fidProx, priorProx, self.x, self.v, self.u, rho, maxIter, tol)
        super().__init__(alg)

        # Logging
        self.it = 0
        self.fullLogging = fullLogging
        self.xGt = xGt
        self.hist = {
            'mse': [],
            'snr': [],
            'x': [],
            'v': [],
            'u': []
        }
        self._log()  # Initial logging

    def _output(self):
        if self.retV:
            return self.v
        else:
            return self.x

    def _post_update(self):
        self.it += 1
        self.x = self.alg.x
        self.v = self.alg.v
        self.u = self.alg.u
        self._log()

    def _log(self):
        if self.fullLogging:
            self.hist['x'] += [self.x]
            self.hist['v'] += [self.v]
            self.hist['u'] += [self.u]
        if self.xGt is not None:
            self.hist['mse'] += [util.computeMSE(self.v, self.xGt)]
            self.hist['snr'] += [util.computeSNR(self.v, self.xGt)]

    def getNumIter(self):
        return self.it

    def getResidue(self):
        return self.alg.residue

    def isConverged(self):
        return self.alg.residue <= self.alg.tol

class PGM(sp.alg.Alg):
    """Proximal Gradient Method
    
    Note: a.k.a. Iterative Shrinkage-Thresholding Algorithm (ISTA) or Forward-Backward Splitting (FBS).
    """
    def __init__(self, gradf, proxg, x, stepSize, maxIter, tol, fista):
        super().__init__(maxIter)
        self.gradf = gradf
        self.proxg = proxg
        self.x = x
        self.stepSize = stepSize

        # For Nesterov acceleration (FISTA)
        self.fista = fista
        self.s = np.copy(x)
        self.q = 1

        # For stopping criterion
        self.tol = tol
        self.numPixels = np.prod(x.shape)
        self.residue = np.inf
        
    def _update(self):
        # Back up variables for stopping criterion
        xPrev = np.copy(self.x)

        # PGM updates
        if self.fista:
            self.x = self.s - self.stepSize * self.gradf(self.s)
            self.x = self.proxg(self.x)
            qNext = 0.5 * (1 + np.sqrt(1 + 4 * self.q ** 2))
            self.s = self.x + ((self.q - 1) / qNext) * (self.x - xPrev)
            self.q = qNext
        else:
            self.x = self.x - self.stepSize * self.gradf(self.x)
            self.x = self.proxg(self.x)

        # Compute residue for stopping criterion
        self.residue = np.linalg.norm(self.x - xPrev) / np.sqrt(self.numPixels)

    def _done(self):
        return (self.iter >= self.max_iter) or (self.residue <= self.tol)

class PGMApp(sp.app.App):
    def __init__(self, fidGrad, priorProx, stepSize, xInit, maxIter,
                fista = True, tol = 5e-4, xGt = None, fullLogging = False):

        # Initialize variables
        self.x = np.copy(xInit)
        
        # Define algorithm
        alg = PGM(fidGrad, priorProx, self.x, stepSize, maxIter, tol, fista = fista)
        super().__init__(alg)

        # Logging
        self.it = 0
        self.fullLogging = fullLogging
        self.xGt = xGt
        self.hist = {
            'mse': [],
            'snr': [],
            'x': []
        }
        self._log()  # Initial logging

    def _output(self):
        return self.x

    def _post_update(self):
        self.it += 1
        self.x = self.alg.x
        self._log()

    def _log(self):
        if self.fullLogging:
            self.hist['x'] += [self.x]
        if self.xGt is not None:
            self.hist['mse'] += [util.computeMSE(self.x, self.xGt)]
            self.hist['snr'] += [util.computeSNR(self.x, self.xGt)]

    def getNumIter(self):
        return self.it

    def getResidue(self):
        return self.alg.residue

    def isConverged(self):
        return self.alg.residue <= self.alg.tol

"""Solvers without regularization"""

class NoRegApp:
    """Placeholder just so (non-iterative) estimates without regularization
        have the same formats as other methods (ADMM and PGM).
    """
    def __init__(self, method, dataDict):
        self.method = method
        self.data = dataDict
        self.hist = {}

    def run(self):
        return self.method(**self.data)

    def getNumIter(self):
        return 0

    def getResidue(self):
        return 0

    def isConverged(self):
        return True

def convEstimate(yConv, lamb):
    return yConv / lamb

def oracleEstimate(yConv, MConv):
    return yConv / MConv

def qmEstimate(yTr):
    L = np.sum(yTr > 0, axis=0).astype(float)
    yConv = yTr.sum(axis=0)
    etaEst = yConv / L
    etaEst[np.isnan(etaEst)] = 0.0
    return etaEst

def lqmEstimate(yTr):
    etaQm = qmEstimate(yTr)
    etaEst = lambertw(-etaQm * np.exp(-etaQm)) + etaQm
    etaEst[np.isnan(etaEst)] = 0.0
    return np.real(etaEst)

def trmlEstimate(yTr, lamb):
    """From Minxu's code
    Computes the estimated eta value from using Expectation Maximization (EM) algorithm.
    It will terminate after 5000 steps or the absolute difference of two consecutive estimated etas
    is less than tolerance.

    Args:
        y (ndarray): total SE counts.
        lam_single (float): dose for each sub-acquisition.

    Returns:
        eta (ndarray): estimated eta.

    Shapes:
        Inputs
            y: (nn, dd), where nn is the number of pixels. dd is the number of sub-acquisitions.
            lam_single: scalar.
        Output:
            eta: (nn,)
    """
    n = yTr.shape[0]
    shape = (yTr.shape[1], yTr.shape[2])
    lam_single = lamb / n
    y = yTr.reshape(n, -1).T
    m = np.arange(15).reshape((1, 1, -1))
    pmf_m = poisson.pmf(m, lam_single)
    y = y[:, :, np.newaxis]
    nn, dd, _ = y.shape

    # Initialize the prior probability w_m^{(i)} = P(M=m|Y_i=y_i).
    # Firstly initiate eta which is the conventional estimator.
    eta = y.sum(axis=1) / lam_single / dd
    eta = eta.reshape((-1, 1, 1))

    steps = 5000
    tol = 1e-5
    # t_start = timer()
    eta_hat = []
    for i in tqdm(range(y.shape[0])):
        y_vec = y[i, :, :]
        eta_vec = eta[i, :, :]

        eta_store = []
        for step in range(steps):
            # E-step.
            # Calculate the prior probability w_m^{(i)} = P(M=m|Y_i=y_i).
            pmf_y_given_m = poisson.pmf(y_vec, eta_vec * m)
            pmf_y_and_m = pmf_y_given_m * pmf_m
            pmf_y_and_m_sum = np.sum(pmf_y_and_m, axis=2, keepdims=True)

            w_m = pmf_y_and_m / pmf_y_and_m_sum

            # M-step.
            # Update eta.
            y_wm_sum = np.sum(y_vec * w_m)
            m_wm_sum = np.sum(m * w_m)

            eta_new = y_wm_sum / m_wm_sum
            eta_vec.fill(eta_new)

            if step > 0 and np.abs(eta_new - eta_store[-1]) <= tol:
                eta_hat.append(eta_new)
                # print("Pixel {}, elapsed time = {:.2f} seconds".format(i, timer() - t_start))
                break
            eta_store.append(eta_new)

    # print("TRML method takes {:.2f} seconds".format(timer() - t_start))

    return np.array(eta_hat).reshape(shape)