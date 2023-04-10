from loguru import logger
from .fit import loglikelihood, gradient
from .type import ParetoMixtureParameters, DeltaParetoMixtureParameters, Sample
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class Solver:
    def __init__(
        self,
        initial_step_size: float = 0.5,
        step_multiplier: float = 0.8,
        max_step_iterations: int = 100,
        max_steps: int = 100,
        make_steps_for_the_sake_of_it: bool = True,
    ):
        self._path_history = None
        self.initial_step_size = initial_step_size
        self.step_multiplier = step_multiplier
        self.max_step_iterations = max_step_iterations
        self.max_steps = max_steps
        self.make_steps_for_the_sake_of_it = make_steps_for_the_sake_of_it
        pass

    @property
    def path_history(self):
        if self._path_history is None:
            raise Exception("No path history available, did you run a search?")
        return self._path_history

    def backtracking_line_search(
        self,
        sample: Sample,
        pmp: ParetoMixtureParameters,
        direction: DeltaParetoMixtureParameters,
        p_greater_than_1: bool,
        pre_normalize: bool = False,
    ):
        """
        Backtracking line search
        https://en.wikipedia.org/wiki/Backtracking_line_search

        sample: Sample
            The sample we are fitting to
        pmp: ParetoMixtureParameters
            The current parameters we are at
        direction: DeltaParetoMixtureParameters
            The initial direction we are moving in
        p_greater_than_1: bool
            Whether p should be greater than 1 or not
        pre_normalize: bool
            Whether we should normalize the initial direction before we start the search (set norm to 1)
        """
        old_ll = loglikelihood(pmp, sample)
        logger.debug(
            f"Doing backtracking line search in direction {direction}. Old ll: {old_ll}"
        )

        if pre_normalize:
            direction = direction * (
                1
                / np.linalg.norm(
                    np.array([direction.dalpha, direction.dbeta, direction.dp])
                )
            )
            logger.debug(f"We normalize our direction: {direction}")

        multiplier = self.initial_step_size
        new_ll = None
        for step in range(1, self.max_step_iterations + 1):
        # for step in range(1, 10):
            new_pmp = pmp + direction * multiplier

            logger.debug(f"New pmp in backtracking line search: {new_pmp}")
            # we ensure our new pmp do not surpass the thresholds we are given
            if not p_greater_than_1 and new_pmp.p >= 1:
                logger.debug(f"Encountered p >= 1 in {pmp} where p should be < 1")
                new_ll = -np.inf
            elif p_greater_than_1 and new_pmp.p <= 1:
                logger.debug(f"Encountered p <= 1 in {pmp} where p should be > 1")
                new_ll = -np.inf
            else:
                new_ll = loglikelihood(new_pmp, sample)

            if new_ll == np.nan or new_ll == np.inf:
                logger.debug(f"Encountered np.nan ll with params {pmp}")
                new_ll = -np.inf

            logger.debug(f"Its likelihood is {new_ll}")

            if new_ll > old_ll:
                logger.debug(
                    f"Found direction: {direction * multiplier}; new ll: {new_ll}"
                )
                return direction * multiplier
            elif (
                self.make_steps_for_the_sake_of_it
                and step > (self.max_step_iterations - 20)
                and new_ll > old_ll - 1e-5
            ):
                logger.debug("Making a step for the sake of it.")
                return direction * multiplier

            multiplier *= self.step_multiplier
        raise Exception(
            f"Backtracking line search did not converge in {self.max_step_iterations}, old_ll={old_ll}, new_ll={new_ll}, multiplier={multiplier}, direction={direction}"
        )

    def search(self, sample: Sample, p_greater_than_1: bool):
        self._path_history = defaultdict(dict[str, object])
        if not p_greater_than_1:
            # p smaller than 1
            # alpha < hill estimate
            # D > 0
            # bias hill > 0
            # need a beta > alpha
            # alpha = hill(sample)
            alpha = 1.5
            beta = alpha
            p = 1

            # theta = np.array([alpha, beta, p])
            # direction = np.array([1, 0, -1])
            pmp = ParetoMixtureParameters(alpha, beta, p)
            direction = DeltaParetoMixtureParameters(-1, 0, -1)

            direction = self.backtracking_line_search(
                sample, pmp, direction, p_greater_than_1
            )
            self._path_history[0]["pmp"] = pmp
            self._path_history[0]["direction"] = direction
            self._path_history[0]["ll"] = loglikelihood(pmp, sample)
            self._path_history[0]["gradient"] = gradient(pmp, sample)
            pmp = pmp + direction
            old_ll = loglikelihood(pmp, sample)

            # for step in range(1, self.max_steps + 1):
            for step in range(1, 5):
                logger.debug("-" * 80)
                logger.debug(f"Got new pmp: {pmp}")

                direction = gradient(pmp, sample)
                direction = DeltaParetoMixtureParameters(
                    direction.dll_dalpha, direction.dll_dbeta, direction.dll_dp
                )
                direction = self.backtracking_line_search(
                    sample, pmp, direction, p_greater_than_1, pre_normalize=False
                )
                self._path_history[step]["pmp"] = pmp
                self._path_history[step]["ll"] = loglikelihood(pmp, sample)
                self._path_history[step]["direction"] = direction
                self._path_history[step]["gradient"] = gradient(pmp, sample)
                pmp = pmp + direction
                new_ll = loglikelihood(pmp, sample)
                if (new_ll - old_ll) < 1e-6:
                    logger.debug(f"Got final pmp: {pmp}")
                    self._path_history[step + 1]["pmp"] = pmp
                    self._path_history[step + 1]["ll"] = loglikelihood(pmp, sample)
                    self._path_history[step + 1]["gradient"] = gradient(pmp, sample)
                    return pmp
                else:
                    logger.debug(f"old_ll={old_ll}, new_ll={new_ll}")
                old_ll = new_ll
            raise Exception(f"Did not finish in {self.max_steps} iterations")
        else:
            raise NotImplementedError("p > 1 not implemented yet")

    def visualize_path_history(self, sample, true_alpha: float, true_beta: float, true_p: float, figsize = (16/1.4, 9 / 1.4)):
        min_alpha, max_alpha = 0, -np.inf
        min_beta, max_beta = 0, -np.inf
        min_p, max_p = 0, 1
        for step, values in self.path_history.items():
            min_alpha = min(min_alpha, values["pmp"].alpha)  # type: ignore
            max_alpha = max(max_alpha, values["pmp"].alpha)  # type: ignore
            min_beta = min(min_beta, values["pmp"].beta)  # type: ignore
            max_beta = max(max_beta, values["pmp"].beta)  # type: ignore
            min_p = min(min_p, values["pmp"].p)  # type: ignore
            max_p = max(max_p, values["pmp"].p)  # type: ignore
        
        min_alpha = min(min_alpha, true_alpha)
        max_alpha = max(max_alpha, true_alpha)
        min_beta = min(min_beta, true_beta)
        max_beta = max(max_beta, true_beta)
        min_p = min(min_p, true_p)
        max_p = max(max_p, true_p)


        logger.debug(f"min_alpha={min_alpha},\tmax_alpha={max_alpha}\nmin_beta={min_beta},\tmax_beta={max_beta}\nmin_p={min_p},\tmax_p={max_p}")

        # we broaden our alphas and betas such that there's a 10% margin on either side
        margin_size = 0.1 # 10%
        margin_alpha = (max_alpha - min_alpha) * margin_size + 0.1
        margin_beta = (max_beta - min_beta) * margin_size + 0.1
        margin_p = (max_p - min_p) * margin_size + 0.1
        alpha_range = np.linspace(
            min_alpha - margin_alpha, max_alpha + margin_alpha, 50
        )
        beta_range = np.linspace(min_beta - margin_beta, max_beta + margin_beta, 50)
        p_range = np.linspace(min_p - margin_p, max_p + margin_p, 50)
        alpha_grid, beta_grid, p_grid = np.meshgrid(alpha_range, beta_range, p_range, indexing = "ij")
        ll_grid = np.zeros_like(alpha_grid)
        for i, alpha in enumerate(alpha_range):
            for j, beta in enumerate(beta_range):
                for k, p in enumerate(p_range):
                    ll_grid[i, j, k] = loglikelihood(ParetoMixtureParameters(alpha, beta, p), sample)
                    try:
                        assert alpha_grid[i, j, k] - alpha < 0.001
                        assert beta_grid[i, j, k] - beta < 0.001
                        assert p_grid[i, j, k] - p < 0.001
                    except AssertionError:
                        print(f"alpha_grid[{i}, {j}, {k}]={alpha_grid[i, j, k]}, alpha={alpha}")
                        print(f"beta_grid[{i}, {j}, {k}]={beta_grid[i, j, k]}, beta={beta}")
                        print(f"p_grid[{i}, {j}, {k}]={p_grid[i, j, k]}, p={p}")
                        raise

        import plotly.graph_objects as go

        fig = go.Figure(data=go.Volume(
            x=alpha_grid.flatten(),
            y=beta_grid.flatten(),
            z=p_grid.flatten(),
            value=ll_grid.flatten(),
            # isomin=-0.1,
            # isomax=0.8,
            opacity=0.2, # needs to be small to see through all surfaces
            surface_count=21, # needs to be a large number for good volume rendering
        ))
        # we set axis labels
        fig.update_layout(scene = dict(
            xaxis_title='alpha',
            yaxis_title='beta',
            zaxis_title='p',
            ),
            margin=dict(r=20, l=10, b=10, t=10)
        )

        # we draw a yellow plane through the 3d plot at p = 1
        alpha_plane, beta_plane = np.meshgrid(alpha_range, beta_range)
        p_plane = np.ones_like(alpha_plane)

        # Then, create the go.Surface object for the plane
        plane = go.Surface(
            x=alpha_plane,
            y=beta_plane,
            z=p_plane,
            surfacecolor=np.ones_like(alpha_plane) * 1,  # Color based on p value
            colorscale=[[0, 'yellow'], [1, 'yellow']],  # Set the color for the plane
            showscale=False,  # Disable the color scale
            opacity=0.3  # Set the opacity for the plane
        )

        fig.add_trace(plane)


         # and through p = 0
        alpha_plane, beta_plane = np.meshgrid(alpha_range, beta_range)
        p_plane = np.zeros_like(alpha_plane)

        # Then, create the go.Surface object for the plane
        plane = go.Surface(
            x=alpha_plane,
            y=beta_plane,
            z=p_plane,
            surfacecolor=np.ones_like(alpha_plane) * 1,  # Color based on p value
            colorscale=[[0, 'yellow'], [1, 'yellow']],  # Set the color for the plane
            showscale=False,  # Disable the color scale
            opacity=0.3  # Set the opacity for the plane
        )

        # Finally, add the plane to the figure
        fig.add_trace(plane)
        
        point = go.Scatter3d(
            x=[2],  # X-coordinate (alpha)
            y=[3],  # Y-coordinate (beta)
            z=[0.95],  # Z-coordinate (p)
            mode='markers',
            marker=dict(
                size=6,  # Set the size of the marker
                color='red',  # Set the color of the marker
                symbol='circle',  # Set the marker symbol
            ),
            name=f'Point ({true_alpha}, {true_beta}, {true_p})',  # Set the trace name (optional)
        )

        # Add the point to the figure
        fig.add_trace(point)

        # Add historical path
        sorted_path_history = sorted(self.path_history.items(), key=lambda x: x[0])

        # Extract alpha, beta, and p values from the sorted path history
        alpha_steps = [values["pmp"].alpha for step, values in sorted_path_history]
        beta_steps = [values["pmp"].beta for step, values in sorted_path_history]
        p_steps = [values["pmp"].p for step, values in sorted_path_history]

        # Create the go.Scatter3d object for the steps
        steps_trace = go.Scatter3d(
            x=alpha_steps,
            y=beta_steps,
            z=p_steps,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue', symbol='circle'),
            name='Line Search Steps',
        )

        # Add the steps trace to the figure
        fig.add_trace(steps_trace)

        return fig
    
