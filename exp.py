"""Particle-Beam Micrograph Denoising Experiments

Author: Ruangrawee (Kao) Kitichotkul
Note:
    The argument parsing at in main() should explain how to use this code via command line arguments.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse

import numpy as np
import torch
import pandas as pd

import micrograph as mg
import dncnn
import util
from timeit import default_timer as timer

def solverSelector(solverInfo, data, arg):

    if solverInfo.method == 'noreg':
        if solverInfo.f == 'naive':
            dataDict = dict(yConv=data['yTr'].sum(axis=0), lamb=arg.lamb)
            method = mg.convEstimate
        elif solverInfo.f == 'oracle':
            dataDict = dict(yConv=data['yTr'].sum(axis=0), MConv=data['MConv'])
            method = mg.oracleEstimate
        elif solverInfo.f == 'qm':
            dataDict = dict(yTr=data['yTr'])
            method = mg.qmEstimate
        elif solverInfo.f == 'lqm':
            dataDict = dict(yTr=data['yTr'])
            method = mg.lqmEstimate
        elif solverInfo.f == 'tr':
            dataDict = dict(yTr=data['yTr'], lamb=arg.lamb)
            method = mg.trmlEstimate
        else:
            raise ValueError('invalid method = {}, f = {}'.format(solverInfo.method, solverInfo.f))
        return mg.NoRegApp(method, dataDict)

    # Choose the data-fidelity update
    if solverInfo.method == 'admm':
        if solverInfo.f == 'gaussian':
            dataFidUpdate = mg.GaussApproxFidProx(data['yTr'], arg.lamb)
        elif solverInfo.f == 'oracle':
            dataFidUpdate = mg.OracleFidProx(data['yTr'], data['MConv'])
        elif solverInfo.f == 'qm':
            dataFidUpdate = mg.QMFidProx(data['yTr'])
        elif solverInfo.f == 'lqm':
            dataFidUpdate = mg.LQMApproxFidProx(data['yTr'])
        elif solverInfo.f == 'tr':
            dataFidUpdate = mg.TRApproxFidProx(data['yTr'], arg.lamb, 1e-4, 100)
        elif solverInfo.f == 'naive':
            dataFidUpdate = mg.GaussNaiveFidProx(data['yTr'], arg.lamb)
        else:
            raise ValueError('invalid method = {}, f = {}'.format(solverInfo.method, solverInfo.f))
    elif solverInfo.method in ['pgm', 'fista']:
        if solverInfo.f == 'gaussian':
            dataFidUpdate = mg.GaussFidGrad(data['yTr'], arg.lamb)
        elif solverInfo.f == 'oracle':
            dataFidUpdate = mg.OracleFidGrad(data['yTr'], data['MConv'])
        elif solverInfo.f == 'qm':
            dataFidUpdate = mg.QMFidGrad(data['yTr'])
        elif solverInfo.f == 'lqm':
            dataFidUpdate = mg.LQMFidGrad(data['yTr'])
        elif solverInfo.f == 'tr':
            dataFidUpdate = mg.TRApproxFidGrad(data['yTr'], arg.lamb)
        else:
            raise ValueError('invalid method = {}, f = {}'.format(solverInfo.method, solverInfo.f))
    else:
        raise ValueError('invalid method = {}'.format(solverInfo.method))

    # Choose the prior update
    if solverInfo.g == 'tv':
        priorUpdate = mg.TVPriorProx(float(solverInfo.sigma))
    elif solverInfo.g == 'bm3d':
        priorUpdate = mg.Bm3dPriorProx(float(solverInfo.sigma))
    elif solverInfo.g == 'dncnn':
        model = dncnn.DnCNN(arg.numDncnnLayers)
        dncnn.loadCheckpoint(arg.modeldir, model)
        priorUpdate = mg.DnCNNPriorProx(model, arg.device, mu=float(solverInfo.sigma), 
                                        imRange=arg.etaRange, denRange=arg.denRange)

    # Set up the solver
    if solverInfo.method == 'admm':
        solver = mg.ADMMApp(dataFidUpdate, priorUpdate, solverInfo.rho, 
                            data['etaInit'], arg.maxIter,
                            tol=arg.tol, xGt=data['etaGt'], fullLogging=arg.fullLogging)
    elif solverInfo.method == 'pgm':
        solver = mg.PGMApp(dataFidUpdate, priorUpdate, solverInfo.rho,
                            data['etaInit'], arg.maxIter, fista=False,
                            tol=arg.tol, xGt=data['etaGt'], fullLogging=arg.fullLogging)
    elif solverInfo.method == 'fista':
        solver = mg.PGMApp(dataFidUpdate, priorUpdate, solverInfo.rho,
                            data['etaInit'], arg.maxIter, fista=True,
                            tol=arg.tol, xGt=data['etaGt'], fullLogging=arg.fullLogging)
    else:
        raise ValueError('invalid method = {}. How did it get here?'.format(solverInfo.method))

    return solver

def main(arg):

    np.random.seed(0)
    torch.manual_seed(0)
    
    util.mkdir(arg.savedir)
    util.logArguments(arg, os.path.join(arg.savedir, 'arg.txt'))
    if arg.dataFromSaved:
        loadlist = util.generateLoadlist(arg.datadir, num_files=arg.numImages, suffix='data.npz')
        namelist = [path[path.rfind('/') + 1:] for path in loadlist]
        namelist = [name[:name.find('-data')] for name in namelist]
        numImages = len(namelist)
    else:
        loadlist, namelist, numImages = util.prepareImagePath(arg.datadir, arg.numImages)
    solverList = pd.read_csv(arg.solverdir)
    dynamicRange = arg.etaRange[1] - arg.etaRange[0]
    results = []

    for loadname, name in zip(loadlist, namelist):

        if arg.dataFromSaved:
            # Load measurement
            data = dict(np.load(loadname))
            etaGt = data['etaGt']
        else:
            # Simulate measurement
            etaGt = util.readImageToEta(loadname, arg.etaRange[0], arg.etaRange[1])
            yTr, MTr = mg.poiPoiTrSampling(etaGt, arg.lamb, arg.n, retM = True)
            yConv = np.sum(yTr, axis=0)
            MConv = np.sum(MTr, axis=0)
            etaInit = yConv / arg.lamb  # initialize eta with conventional estimate

            data = {
                'yTr': yTr,
                'MConv': MConv,
                'etaInit': etaInit,
                'etaGt': etaGt
            }
            np.savez(os.path.join(arg.savedir, '{}-data.npz'.format(name)), **data)

        # Solving the inverse problem using each method
        for idx, solverInfo in solverList.iterrows():
            try:
                solver = solverSelector(solverInfo, data, arg)
            except ValueError as err:
                print('ValueError: {}'.format(err))
                continue
            print(name, '\n', solverInfo)
            timeStart = timer()
            etaRecon = solver.run()
            time = timer() - timeStart

            # Log result
            mse, rmse, psnr, snr, ssim = util.computeErrorMetrics(etaRecon, etaGt, dynamicRange)
            results += [{
                'image': name,
                'solverRef': idx,
                **solverInfo,
                'mse': mse,
                'rmse': rmse,
                'psnr': psnr,
                'snr': snr,
                'ssim': ssim,
                'time': time,
                'numIter': solver.getNumIter(),
                'residue': solver.getResidue(),
                'converged': solver.isConverged()
            }]
            solver.hist['etaRecon'] = etaRecon
            np.savez(os.path.join(arg.savedir, '{}-{}.npz'.format(name, idx)), **solver.hist)
            util.saveImage(etaRecon, arg.etaRange[0], arg.etaRange[1],
                            os.path.join(arg.savedir, '{}-{}.png'.format(name, idx)))

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(arg.savedir, 'results.csv'), sep=',', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Particle-Beam Micrograph Denoising Experiments')

    # Common arguments
    parser.add_argument('--datadir', type=str, default='data/bsd/val', 
                        help='Path of image directory.')
    parser.add_argument('--dataFromSaved', action='store_true',
                        help='Whether to use measurements from saved .npz files in the datadir directory.')
    parser.add_argument('--savedir', type=str, default='result/exp00', 
                        help='Path to save the results.')
    parser.add_argument('--solverdir', type=str, default='experiment/solvers.csv', 
                        help='Path to a csv file listing the solvers to use.')
    parser.add_argument('--numImages', type=int,
                        default=None, help='Number of images in datadir to use.')
    parser.add_argument('--fullLogging', action='store_true',
                        help='Whether to save all intermediate variables.')

    # Forward model arguments
    parser.add_argument('--etaRange', type=float, default=[2., 8.], nargs='*',
                        help='Secondary electron (SE) yield range, i.e. range to scale pixel values to.')
    parser.add_argument('--lamb', type=float,
                        default=20., help='Ion beam dose (# ions).')
    parser.add_argument('--n', type=int,
                        default=200, help='Number of subacquisitions (as in TR measurement).')                    

    # General solver arguments              
    parser.add_argument('--maxIter', type=int,
                        default=50, help='Maximum number of iterations.')
    parser.add_argument('--tol', type=float, default=5e-4, 
                        help='Tolerance for stopping criteria.')
    
    # DnCNN arguments
    parser.add_argument('--modeldir', type=str, default='model/std25.ckpt', 
                        help='Path to model weights.')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device for using NN denoisers.')
    parser.add_argument('--numDncnnLayers', type=int,
                        default=17, help='Number of layers of DnCNN.')
    parser.add_argument('--denRange', type=float, default=[0., 1.], nargs='*',
                        help='Range of pixel values the denoiser is trained for.')

    arg = parser.parse_args()
    main(arg)
