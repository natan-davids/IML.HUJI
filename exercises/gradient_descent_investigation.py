import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

from IMLearn.model_selection import cross_validate
from utils import custom
from IMLearn.metrics import mean_square_error

import plotly.graph_objects as go
import plotly.express as px


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    Values = []
    Weights = []
    def callback(solver, weights, val, grad, t, eta, delta, **kwargs):
        Values.append(val)
        Weights.append(weights)
        return
    return callback, Values, Weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    module_names = {L1: "L1", L2: "L2"}
    min_loss = float('inf')
    for module in [L1, L2]:
        for base_lr in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            lr = FixedLR(base_lr)
            f = module(init)
            gd = GradientDescent(learning_rate=lr, callback=callback)
            gd.fit(f, np.array([]), np.array([]))
            descent_path = np.array(weights)

            descent_plot_title = f"Fixed LR Descent Path _ module={module_names[module]} eta={base_lr}"
            descent_plot = plot_descent_path(module, descent_path, descent_plot_title)
            descent_plot.show()
            descent_plot.write_image(f'ex6_plots/{descent_plot_title}.png')

            convergence_plot_title = f"Fixed LR Convergence Rate _ module={module_names[module]} eta={base_lr}"
            convergence_plot = px.line(y=np.array(values), title=convergence_plot_title)
            convergence_plot.show()
            convergence_plot.write_image(f'ex6_plots/{convergence_plot_title}.png')

            lowest_loss = min(values)
            min_loss = min(min_loss, lowest_loss)

            print(f"lowest achieved loss for module {module_names[module]} with eta={base_lr}:\n{lowest_loss}")

    print(f'\nMinimum loss for all modules: {min_loss}ֿ\n')



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    convergence_df = pd.DataFrame()
    for g in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        f = L1(init)
        lr = ExponentialLR(eta, g)
        gd = GradientDescent(learning_rate=lr, callback=callback)
        gd.fit(f, np.array([]), np.array([]))

        descent_path = np.array(weights)
        descent_plot_title = f"Exponential LR Descent Path for L1 _ gamma={g} base_lr={eta}"
        descent_plot = plot_descent_path(L1, descent_path, descent_plot_title)
        descent_plot.show()
        descent_plot.write_image(f'ex6_plots/{descent_plot_title}.png')

        title = f"gamma={g}"
        convergence_df[title] = pd.Series(values)
        print(f"lowest achieved loss with gamma={g}:\n{min(values)}")

    # Plot algorithm's convergence for the different values of gamma
    convergence_plot_title = f"Exponential LR Convergence Rate _ base_lr={eta} gammas={gammas}"
    convergence_plot = px.line(convergence_df, title=convergence_plot_title)
    convergence_plot.show()
    convergence_plot.write_image(f'ex6_plots/{convergence_plot_title}.png')

    # Plot descent path for gamma=0.95



def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    train_X, train_y, test_X, test_y = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    callback, values, weights = get_gd_state_recorder_callback()
    solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=2000, callback=callback)
    model = LogisticRegression(solver=solver)
    model.fit(train_X.to_numpy(), train_y.to_numpy())
    y_prob = model.predict_proba(test_X.to_numpy())

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(test_y, y_prob)
    c = [custom[0], custom[-1]]

    roc_plot = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    roc_plot.show()
    roc_plot.write_image(f"ex6_plots/roc_curve_logistic.png")

    a_star = thresholds[np.argmax(tpr-fpr)]
    model.alpha_ = a_star

    print(f"optimal roc with alpha={a_star}")
    print(f"test error: {model.loss(test_X.to_numpy(), test_y.to_numpy())}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    vals = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    l1_training_loss = np.empty(len(vals))
    l1_validation_loss = np.empty(len(vals))

    l2_training_loss = np.empty(len(vals))
    l2_validation_loss = np.empty(len(vals))

    for i, lambda_val in enumerate(vals):
        l1_model = LogisticRegression(lam=lambda_val,
                                      penalty='l1',
                                      solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=2000))
        l1_mean_loss, _ = cross_validate(l1_model, train_X.to_numpy(), train_y.to_numpy(), mean_square_error)
        l1_training_loss[i] = l1_mean_loss
        l1_validation_loss[i] = l1_model.loss(test_X.to_numpy(), test_y.to_numpy())

        l2_model = LogisticRegression(lam=lambda_val,
                                      penalty='l2',
                                      solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=2000))
        l2_mean_loss, _ = cross_validate(l2_model, train_X.to_numpy(), train_y.to_numpy(), mean_square_error)
        l2_training_loss[i] = l2_mean_loss
        l2_validation_loss[i] = l2_model.loss(test_X.to_numpy(), test_y.to_numpy())

    l1_optimal = vals[np.argmin(l1_validation_loss)]
    l2_optimal = vals[np.argmin(l2_validation_loss)]

    params = [(l1_optimal, 'l1'), (l2_optimal, 'l2')]
    optimal_models = [LogisticRegression(lam=opt_lam,
                                      penalty=penalty,
                                      solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=2000))
                      for opt_lam, penalty in params]
    for i, mod in enumerate(optimal_models):
        mod.fit(train_X.to_numpy(), train_y.to_numpy())
        print(f"fitting data with optimal {params[i][1]} regularization, lambda={params[i][0]}")
        print(f"model loss: {mod.loss(test_X.to_numpy(), test_y.to_numpy())}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
