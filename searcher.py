# coding=utf-8
"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2021.08.01
import copy
import random

from .bo.design_space.design_space import DesignSpace
# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher

from sklearn.preprocessing import power_transform

import numpy  as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from .bo.models.model_factory import get_model
from .bo.acquisitions.acq import LCB, Mean, Sigma, MOMeanSigmaLCB, MACE
from .bo.optimizers.evolution_optimizer import EvolutionOpt
import time
torch.set_num_threads(min(1, torch.get_num_threads()))

class Searcher(AbstractSearcher):

    def __init__(self, parameters_config, n_iter, n_suggestion):
        """ Init searcher

        Args:
            parameters_config: parameters configuration, consistent with the definition of parameters_config of EvaluateFunction. dict type:
                    dict key: parameters name, string type
                    dict value: parameters configuration, dict type:
                        "parameter_name": parameter name
                        "parameter_type": parameter type, 1 for double type, and only double type is valid
                        "double_max_value": max value of this parameter
                        "double_min_value": min value of this parameter
                        "double_step": step size
                        "coords": list type, all valid values of this parameter.
                            If the parameter value is not in coords,
                            the closest valid value will be used by the judge program.

                    parameter configuration example, eg:
                    {
                        "p1": {
                            "parameter_name": "p1",
                            "parameter_type": 1
                            "double_max_value": 2.5,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0, 2.5]
                        },
                        "p2": {
                            "parameter_name": "p2",
                            "parameter_type": 1,
                            "double_max_value": 2.0,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0]
                        }
                    }
                    In this example, "2.5" is the upper bound of parameter "p1", and it's also a valid value.

        n_iteration: number of iterations
        n_suggestion: number of suggestions to return
        """
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)

        self._bounds = self.get_bounds()
        self._bounds = self._bounds.T
        self.lb, self.ub = self._bounds[0], self._bounds[1]
        self.dim = len(self.lb)
        self.batch_size = None

        self.space = self.parse_space(parameters_config)
        self.X = pd.DataFrame(columns=self.space.para_names)
        self.y = np.zeros((0, 1))
        self.X_drop = pd.DataFrame(columns=self.space.para_names)
        self.y_drop = np.zeros((0, 1))
        self.model_name = 'gpy'

        self.sobol = SobolEngine(self.space.num_paras, scramble=False)

    def init_param_group(self, n_suggestions):
        """ Suggest n_suggestions parameters in random form

        Args:
            n_suggestions: number of parameters to suggest in every iteration

        Return:
            next_suggestions: n_suggestions Parameters in random form
        """
        next_suggestions = [{p_name: p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)]
                            for p_name, p_conf in self.parameters_config.items()} for _ in range(n_suggestions)]

        return next_suggestions

    def parse_suggestions_history(self, suggestions_history):
        """ Parse historical suggestions to the form of (x_datas, y_datas), to obtain GP training data

        Args:
            suggestions_history: suggestions history

        Return:
            x_datas: Parameters
            y_datas: Reward of Parameters
        """
        suggestions_history_use = copy.deepcopy(suggestions_history)
        x_datas = [[item[1] for item in sorted(suggestion[0].items(), key=lambda x: x[0])]
                   for suggestion in suggestions_history_use]
        y_datas = [suggestion[1] for suggestion in suggestions_history_use]
        x_datas = np.array(x_datas)
        y_datas = np.array(y_datas)
        return x_datas, y_datas

    def init_param_group(self, n_suggestions):
        """ Suggest n_suggestions parameters in random form

        Args:
            n_suggestions: number of parameters to suggest in every iteration

        Return:
            next_suggestions: n_suggestions Parameters in random form
        """
        next_suggestions = [{p_name: p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)]
                             for p_name, p_conf in self.parameters_config.items()} for _ in range(n_suggestions)]

        return next_suggestions

    def parse_suggestions_history(self, suggestions_history):
        """ Parse historical suggestions to the form of (x_datas, y_datas), to obtain GP training data

        Args:
            suggestions_history: suggestions history

        Return:
            x_datas: Parameters
            y_datas: Reward of Parameters
        """
        suggestions_history_use = copy.deepcopy(suggestions_history)
        x_datas = [[item[1] for item in sorted(suggestion[0].items(), key=lambda x: x[0])]
                   for suggestion in suggestions_history_use]
        y_datas = [suggestion[1] for suggestion in suggestions_history_use]
        x_datas = np.array(x_datas)
        y_datas = np.array(y_datas)
        return x_datas, y_datas

    def random_sample(self):
        """ Generate a random sample in the form of [value_0, value_1,... ]

        Return:
            sample: a random sample in the form of [value_0, value_1,... ]
        """
        sample = [p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)] for p_name, p_conf
                  in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        return sample

    def get_bounds(self):
        """ Get sorted parameter space

        Return:
            _bounds: The sorted parameter space
        """

        def _get_param_value(param):
            value = [param['double_min_value'], param['double_max_value']]
            return value

        _bounds = np.array(
            [_get_param_value(item[1]) for item in sorted(self.parameters_config.items(), key=lambda x: x[0])],
            dtype=np.float
        )
        return _bounds
    def parse_suggestions_dict(self, suggestions):
        """ Parse the parameters result

        Args:
            suggestions: Parameters

        Return:
            suggestions: The parsed parameters
        """

        def get_param_value(p_name, value):
            p_coords = self.parameters_config[p_name]['coords']
            if value in p_coords:
                return value
            else:
                subtract = np.abs([p_coord - value for p_coord in p_coords])
                min_index = np.argmin(subtract, axis=0)
                return p_coords[min_index]

        suggestions = [{p_name: get_param_value(p_name, value) for p_name, value in suggestion.items()}
                       for suggestion in suggestions]
        return suggestions
    def parse_suggestions(self, suggestions):
        """ Parse the parameters result

        Args:
            suggestions: Parameters

        Return:
            suggestions: The parsed parameters
        """

        def get_param_value(p_name, value):
            p_coords = self.parameters_config[p_name]['coords']
            if value in p_coords:
                return value
            else:
                subtract = np.abs([p_coord - value for p_coord in p_coords])
                min_index = np.argmin(subtract, axis=0)
                return p_coords[min_index]

        p_names = [p_name for p_name, p_conf in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        suggestions = [{p_names[index]: suggestion[index] for index in range(len(suggestion))}
                       for suggestion in suggestions]
        suggestions = [{p_name: get_param_value(p_name, value) for p_name, value in suggestion.items()}
                       for suggestion in suggestions]
        return suggestions


    def filter(self, y: torch.Tensor) -> [bool]:
        if not (np.all(y.numpy() > 0) and (y.max() / y.min() > 20)):
            return [True for _ in range(y.shape[0])], np.inf
        else:
            data = y.numpy().reshape(-1)
            quant = min(data.min() * 20, np.quantile(data, 0.95, interpolation='lower'))
            return (data <= quant).tolist(), quant

    def quasi_sample(self, n):
        samp = self.sobol.draw(n)
        # samp    = torch.FloatTensor(lhs(self.space.num_paras, n))
        samp = samp * (self.space.opt_ub - self.space.opt_lb) + self.space.opt_lb
        x = samp[:, :self.space.num_numeric]
        xe = samp[:, self.space.num_numeric:]
        df_samp = self.space.inverse_transform(x, xe)
        return df_samp

    def parse_space(self, api_config):
        space = DesignSpace()
        params = []
        for param_name in api_config:
            param_conf = api_config[param_name]
            bo_param_conf = {'name': param_name}
            if isinstance(param_conf['coords'][0], int):
                bo_param_conf['type'] = 'int'
                bo_param_conf['lb'] = param_conf['double_min_value']
                bo_param_conf['ub'] = param_conf['double_max_value']
            else:
                bo_param_conf['type'] = 'num'
                bo_param_conf['lb'] = param_conf['double_min_value']
                bo_param_conf['ub'] = param_conf['double_max_value']
            params.append(bo_param_conf)
        # print(params)
        space.parse(params)
        return space

    @property
    def model_config(self):
        if self.model_name == 'gp':
            cfg = {
                'lr': 0.01,
                'num_epochs': 100,
                'verbose': False,
                'noise_lb': 8e-4,
                'pred_likeli': False
            }
        elif self.model_name == 'gpy':
            cfg = {
                'verbose': False,
                'warp': True,
                'space': self.space
            }
        elif self.model_name == 'gpy_mlp':
            cfg = {
                'verbose': False
            }
        elif self.model_name == 'rf':
            cfg = {
                'n_estimators': 20
            }
        else:
            cfg = {}
        if self.space.num_categorical > 0:
            cfg['num_uniqs'] = [len(self.space.paras[name].categories) for name in self.space.enum_names]
        return cfg


    def random_sample(self):
        """ Generate a random sample in the form of [value_0, value_1,... ]

        Return:
            sample: a random sample in the form of [value_0, value_1,... ]
        """
        sample = [p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)] for p_name, p_conf
                  in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        return sample


    def suggest_old(self, suggestions_history, n_suggestions=1):
        if (suggestions_history is None) or (len(suggestions_history) <= 1):
            next_suggestions = self.init_param_group(n_suggestions)
        else:
            start_all = time.time()
            # x_datas, y_datas = self.parse_suggestions_history(suggestions_history)
            x_datas = [i[:-1] for i in suggestions_history]
            y_datas = [-i[-1] for i in suggestions_history]
            # y_datas = -y_datas
            # y_datas = y_datas.reshape((-1, 1))
            self.observe( x_datas[-n_suggestions:], y_datas[-n_suggestions:])
            if self.X.shape[0] < 4  * n_suggestions:
                df_suggest = self.quasi_sample(n_suggestions)
                x_guess    = []
                for i, row in df_suggest.iterrows():
                    x_guess.append(row.to_dict())
                # next_suggestions = x_guess
            else:

                X, Xe   = self.space.transform(self.X)
                try:
                    start = time.time()
                    if self.y.min() <= 0:
                        y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                    else:
                        y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'box-cox'))
                        if y.std() < 0.5:
                            y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                    if y.std() < 0.5:
                        raise RuntimeError('Power transformation failed')
                    model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                    model.fit(X, Xe, y)
                    print('model time:', time.time() - start)
                except:
                    # print('Error fitting GP')
                    y       = torch.FloatTensor(self.y).clone()
                    filt, q = self.filter(y)
                    # print('Q = %g, kept = %d/%d' % (q, y.shape[0], self.y.shape[0]))
                    X       = X[filt]
                    Xe      = Xe[filt]
                    y       = y[filt]
                    model   = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                    model.fit(X, Xe, y)
                # print('Noise level: %g' % model.noise, flush = True)
                start = time.time()
                best_id = np.argmin(self.y.squeeze())
                best_x  = self.X.iloc[[best_id]]
                best_y  = y.min()
                py_best, ps2_best = model.predict(*self.space.transform(best_x))
                py_best = py_best.detach().numpy().squeeze()
                ps_best = ps2_best.sqrt().detach().numpy().squeeze()

                # XXX: minimize (mu, -1 * sigma)
                #      s.t.     LCB < best_y
                iter  = max(1, self.X.shape[0] // n_suggestions)
                upsi  = 0.5
                delta = 0.025
                kappa = np.sqrt(upsi * 2 * np.log(iter **  (2.0 + self.X.shape[1] / 2.0) * 3 * np.pi**2 / (3 * delta)))

                acq = MOMeanSigmaLCB(model, kappa=kappa, best_y=py_best)  # LCB < py_best
                # acq = MACE(model, py_best, kappa = kappa) # LCB < py_best
                mu  = Mean(model)
                sig = Sigma(model, linear_a = -1.)
                opt = EvolutionOpt(self.space, acq, pop = 100, iters = 25, verbose = False)
                rec = opt.optimize(initial_suggest = best_x).drop_duplicates()
                print('optimize time:', time.time() - start)

                rec = self.check_unique(rec)
                if self.X_drop.shape[0] > 0:
                    rec = self.check_unique_drop(rec)
                cnt = 0
                while rec.shape[0] < n_suggestions:
                    rand_rec = self.quasi_sample(n_suggestions - rec.shape[0])
                    rec = rec.append(rand_rec, ignore_index=True)
                    rec = self.check_unique(rec)
                    if self.X_drop.shape[0] > 0:
                        rec = self.check_unique_drop(rec)
                    cnt += 1
                    if cnt > 5:
                        break
                if rec.shape[0] < n_suggestions:
                    rand_rec = self.quasi_sample(n_suggestions - rec.shape[0])
                    rec = rec.append(rand_rec, ignore_index=True)

                rec_torch = torch.from_numpy(rec.values.astype(np.float32))
                best_r_torch = torch.from_numpy(best_x.values.astype(np.float32))
                distance = torch.cdist(rec_torch.unsqueeze(0), best_r_torch.unsqueeze(0), p=2).squeeze(0)
                distance = distance.numpy()

                select_id = np.random.choice(rec.shape[0], n_suggestions, replace = False).tolist()
                x_guess = []
                with torch.no_grad():
                    py_all       = mu(*self.space.transform(rec)).squeeze().numpy()
                    ps_all       = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
                    best_pred_id = np.argmin(py_all)
                    best_unce_id = np.argmax(ps_all)
                    if best_unce_id not in select_id and n_suggestions > 2:
                        select_id[0]= best_unce_id
                    if best_pred_id not in select_id and n_suggestions > 2:
                        select_id[1]= best_pred_id
                    if distance.shape[0]>0:
                        long_distance = np.argmax(distance)
                        short_distance = np.argmin(distance)
                        if long_distance not in select_id and n_suggestions > 3:
                            select_id[2]= long_distance
                        if short_distance not in select_id and n_suggestions > 3:
                            select_id[3]= short_distance
                    print(select_id)
                    rec_selected = rec.iloc[select_id].copy()
                    py,ps2 = model.predict(*self.space.transform(rec_selected))
                    rec_selected['py'] = py.squeeze().numpy()
                    rec_selected['ps'] = ps2.sqrt().squeeze().numpy()
                    # print(rec_selected)
                # print('Best y is %g %g %g %g' % (self.y.min(), best_y, py_best, ps_best), flush = True)
                for idx in select_id:
                    x_guess.append(rec.iloc[idx].to_dict())

            # for rec in x_guess:
            #     for name in rec:
            #         if self.api_config[name]['type'] == 'int':
            #             rec[name] = int(rec[name])

            # suggestions = [ suggestion for suggestion in x_guess]
            # suggestions = suggestions.values()
            # next_suggestions  = x_guess
            next_suggestions = self.parse_suggestions_dict(x_guess)

            print('total time:' ,time.time()-start_all)

        return next_suggestions

    def get_my_score(self, reward):
        """ Get the most trusted reward of all iterations.

        Returns:
            most_trusted_reward: float
        """
        return reward[-1]['value']

    def suggest(self, iteration_number, running_suggestions, suggestion_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters. new implementation of final competition

        Args:
            iteration_number: int ,the iteration number of experiment, range in [1, 140]

            running_suggestions: a list of historical suggestion parameters and rewards, in the form of
                    [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
                Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                    {'p1': 0, 'p2': 0, 'p3': 0}
                Reward: a list of dict, each dict of the list corresponds to an iteration,
                    the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound}
                    Reward example:
                        [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
                         {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
                        ]

            suggestion_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions

            n_suggestion: int, number of suggestions to return

        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        MIN_TRUSTED_ITERATION = 7
        new_suggestions_history = []
        for suggestion in suggestion_history:
            iterations_of_suggestion = len(suggestion['reward'])
            if iterations_of_suggestion >= MIN_TRUSTED_ITERATION:
                cur_score = self.get_my_score(suggestion['reward'])
                new_suggestions_history.append([suggestion["parameter"], cur_score])
        return self.suggest_old(new_suggestions_history, n_suggestions)

    def is_early_stop(self, iteration_number, running_suggestions, suggestion_history):
        """ Decide whether to stop the running suggested parameter experiment.

        Args:
            iteration_number: int, the iteration number of experiment, range in [1, 140]

            running_suggestions: a list of historical suggestion parameters and rewards, in the form of
                    [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
                Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                    {'p1': 0, 'p2': 0, 'p3': 0}
                Reward: a list of dict, each dict of the list corresponds to an iteration,
                    the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound}
                    Reward example:
                        [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
                         {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
                        ]

            suggestions_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions

        Returns:
            stop_list: list of bool, indicate whether to stop the running suggestions.
                    len(stop_list) must be the same as len(running_suggestions), for example:
                        len(running_suggestions) = 3, stop_list could be :
                            [True, True, True] , which means to stop all the three running suggestions
        """

        # Early Stop algorithm demo 2:
        #
        #   If there are 3 or more suggestions which had more than 7 iterations,
        #   the worst running suggestions will be stopped
        #
        # MIN_ITERS_TO_STOP = 7
        # MIN_SUGGUEST_COUNT_TO_STOP = 3
        # MAX_ITERS_OF_DATASET = self.n_iteration
        # ITERS_TO_GET_STABLE_RESULT = 14
        # INITIAL_INDEX = -1
        #
        # res = [False] * len(running_suggestions)
        # if iteration_number + ITERS_TO_GET_STABLE_RESULT <= MAX_ITERS_OF_DATASET:
        #     score_min_idx = INITIAL_INDEX
        #     score_min = float("inf")
        #     count = 0
        #     # Get the worst suggestion of current running suggestions
        #     for idx, suggestion in enumerate(running_suggestions):
        #         if len(suggestion['reward']) >= MIN_ITERS_TO_STOP:
        #             count = count + 1
        #             cur_score = self.get_my_score(suggestion['reward'])
        #             if score_min_idx == INITIAL_INDEX or cur_score < score_min:
        #                 score_min_idx = idx
        #                 score_min = cur_score
        #     # Stop the worst suggestion
        #     if count >= MIN_SUGGUEST_COUNT_TO_STOP and score_min_idx != INITIAL_INDEX:
        #         res[score_min_idx] = True
        res = [False] * len(running_suggestions)
        return res

    # def check_unique(self, rec: pd.DataFrame) -> [bool]:
    #     return (~pd.concat([self.X, rec], axis=0).duplicated().tail(rec.shape[0]).values).tolist()
    def check_unique(self, rec: pd.DataFrame) -> pd.DataFrame:
        rec_parse = [row.values.tolist() for index, row in rec.iterrows()]
        rec_parse = pd.DataFrame(self.parse_suggestions(rec_parse))
        index = rec_parse.drop_duplicates().index
        rec = rec.loc[index].reset_index(drop=True)
        rec_parse = rec_parse.loc[index].reset_index(drop=True)
        rec_drop = rec[~pd.concat([self.X, rec_parse], axis=0).duplicated(keep=False)[self.X.shape[0]:]].reset_index(
            drop=True)
        return rec_drop

    def check_unique_drop(self, rec: pd.DataFrame) -> pd.DataFrame:
        rec_parse = [row.values.tolist() for index, row in rec.iterrows()]
        rec_parse = pd.DataFrame(self.parse_suggestions(rec_parse))
        index = rec_parse.drop_duplicates().index
        rec = rec.loc[index].reset_index(drop=True)
        rec_parse = rec_parse.loc[index].reset_index(drop=True)
        rec_drop = rec[~pd.concat([self.X_drop, rec_parse], axis=0).duplicated(keep=False)[self.X_drop.shape[0]:]].reset_index(
            drop=True)
        return rec_drop
    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Random search so don't do anything
        y = np.array(y).reshape(-1)
        valid_id = np.where(np.isfinite(y))[0].tolist()
        XX = [X[idx][0] for idx in valid_id]
        yy = y[valid_id].reshape(-1, 1)
        self.X = self.X.append(XX, ignore_index=True)
        self.y = np.vstack([self.y, yy])
        # print(yy)