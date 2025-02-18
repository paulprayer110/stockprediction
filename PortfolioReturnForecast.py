import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from statsmodels.nonparametric.kde import KDEUnivariate  # For statsmodels KDE
from arch import arch_model
from scipy.stats import kstest
from fitter import Fitter
from scipy.stats import norm, t, skewnorm, uniform, lognorm, weibull_min, cauchy
from scipy.stats import rankdata, norm, t, kendalltau
from copulas.bivariate import Clayton, Gumbel
from sklearn.neighbors import KernelDensity  # For sklearn KDE
from scipy.stats import gaussian_kde