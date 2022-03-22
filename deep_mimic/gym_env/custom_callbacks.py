from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
# from stable_baselines.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

# class PlottingCallback(BaseCallback):
#     """
#     Callback for plotting the performance in realtime.
#
#     :param verbose: (int)
#     """
#
#     def __init__(self, log_dir, verbose=1):
#         super(PlottingCallback, self).__init__(verbose)
#         self._plot = None
#         self.log_dir = log_dir
#
#     def _on_step(self) -> bool:
#         # get the monitor's data
#         x, y = ts2xy(load_results(self.log_dir), 'timesteps')
#         if self._plot is None:  # make the plot
#             plt.ion()
#             fig = plt.figure(figsize=(6, 3))
#             ax = fig.add_subplot(111)
#             line, = ax.plot(x, y)
#             self._plot = (line, ax, fig)
#             plt.show()
#         else:  # update and rescale the plot
#             self._plot[0].set_data(x, y)
#             self._plot[-2].relim()
#             self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02,
#                                      self.locals["total_timesteps"] * 1.02])
#             self._plot[-2].autoscale_view(True, True, True)
#             self._plot[-1].canvas.draw()

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _dsum(self, *dicts):
        ret = defaultdict(int)
        for d in dicts:
            for k, v in d.items():
                ret[k] += v
        return dict(ret)

    def _on_step(self) -> bool:
        # info_rew = dict(
        #       pose_reward=pose_reward,
        #       vel_reward=vel_reward,
        #       end_eff_reward=end_eff_reward,
        #       root_reward=root_reward,
        #       imitation_reward=reward
        #     )
        #     info_errs = dict(
        #       pose_err=pose_err,
        #       vel_err=vel_err,
        #       end_eff_err=end_eff_err,
        #       root_err=root_err,
        #     )

        rews = [env._humanoid._info_rew for env in self.training_env.venv.get_attr("internal_env")]#[env.env.env.internal_env._humanoid._info_rew for env in self.training_env.venv.envs]
        n_envs = len(rews)
        errs = [env._humanoid._info_err for env in self.training_env.venv.get_attr("internal_env")]
        rews = self._dsum(*rews)
        errs = self._dsum(*errs)

        for name, value in rews.items():
            self.logger.record("reward/"+name, value/n_envs)
        for name, value in errs.items():
            self.logger.record("error/"+name, value/n_envs)
        return True
