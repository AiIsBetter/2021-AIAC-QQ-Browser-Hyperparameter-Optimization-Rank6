# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from ..base_model import BaseModel
from ..layers import EmbTransform, OneHotTransform
from ..scalers import TorchMinMaxScaler, TorchStandardScaler

import GPy
import torch
import torch.nn as nn
import numpy as np
import warnings
from torch import Tensor, FloatTensor, LongTensor

class GPyGP(BaseModel):
    """
    Input warped GP model implemented using GPy instead of GPyTorch

    Why doing so:
    - Input warped GP
    """
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        if num_enum > 0:
            self.one_hot = OneHotTransform(self.conf['num_uniqs'])
        self.xscaler    = TorchMinMaxScaler((-1, 1))
        self.yscaler    = TorchStandardScaler()
        self.verbose    = self.conf.get('verbose', False)
        self.num_epochs = self.conf.get('num_epochs', 200)
        self.warp       = self.conf.get('warp', True)
        self.space      = self.conf.get('space') # DesignSpace
        if self.space is None and self.warp:
            warnings.warn('Space not provided, set warp to False')
            self.warp = False

    def fit_scaler(self, Xc : FloatTensor, y : FloatTensor):
        if Xc is not None and Xc.shape[1] > 0:
            if self.space is not None:
                cont_lb = self.space.opt_lb[:self.space.num_numeric].view(1, -1).float()
                cont_ub = self.space.opt_ub[:self.space.num_numeric].view(1, -1).float()
                self.xscaler.fit(torch.cat([Xc, cont_lb, cont_ub], dim = 0))
            else:
                self.xscaler.fit(Xc)
        self.yscaler.fit(y)

    def trans(self, Xc : Tensor, Xe : Tensor, y : Tensor = None):
        if Xc is not None and Xc.shape[1] > 0:
            Xc_t = self.xscaler.transform(Xc)
        else:
            Xc_t = torch.zeros(Xe.shape[0], 0)

        if Xe is None or Xe.shape[1] == 0:
            Xe_t = torch.zeros(Xc.shape[0], 0)
        else:
            Xe_t = self.one_hot(Xe.long())
        Xall = torch.cat([Xc_t, Xe_t], dim = 1)

        if y is not None:
            y_t = self.yscaler.transform(y)
            return Xall.numpy(), y_t.numpy()
        return Xall.numpy()

    def fit(self, Xc : FloatTensor, Xe : LongTensor, y : LongTensor): 
        self.fit_scaler(Xc, y)
        X, y = self.trans(Xc, Xe, y)

        # k1  = GPy.kern.Linear(X.shape[1],   ARD = False)
        k2  = GPy.kern.Matern52(X.shape[1], ARD = True)
        k2.lengthscale = np.std(X, axis = 0)
        k2.variance    = 0.5
        k2.variance.set_prior(GPy.priors.Gamma(1.5, 1))
        kern = k2
        if not self.warp:
            self.gp = GPy.models.GPRegression(X, y, kern)
        else:
            xmin    = np.zeros(X.shape[1])
            xmax    = np.ones(X.shape[1])
            xmin[:Xc.shape[1]] = -1
            warp_f  = GPy.util.input_warping_functions.KumarWarping(X, Xmin = xmin, Xmax = xmax)
            self.gp = GPy.models.InputWarpedGP(X, y, kern, warping_function = warp_f)
        self.gp.likelihood.variance.set_prior(GPy.priors.LogGaussian(-4.63, 2.5))

        self.gp.optimize_restarts(max_iters = self.num_epochs/3, verbose = self.verbose, num_restarts = 1)
        # print(self.gp.likelihood.variance, flush = True)
        # print(self.gp.likelihood.variance[0], flush = True)
        return self

    def predict(self, Xc : FloatTensor, Xe : LongTensor) -> (FloatTensor, FloatTensor):
        Xall    = self.trans(Xc, Xe)
        py, ps2 = self.gp.predict(Xall)
        mu      = self.yscaler.inverse_transform(FloatTensor(py).view(-1, 1))
        var     = (self.yscaler.std**2 * FloatTensor(ps2).view(-1, 1)).clamp(min = 1e-6)
        return mu, var

    def sample_f(self):
        raise RuntimeError('Thompson sampling is not supported for GP, use `sample_y` instead')

    @property
    def noise(self):
        var_normalized = self.gp.likelihood.variance[0]
        return (var_normalized * self.yscaler.std**2).view(self.num_out)

    @property
    def support_ts(self):
        return False

    @property
    def support_grad(self):
        return False

    @property
    def support_multi_output(self):
        return False

    @property
    def support_warm_start(self) ->bool:
        return False
