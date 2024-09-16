# Experimental setup (ICASSP 2024 submission)

This document contains a detailed description of the experimental setup used  in the submission 'Covariance Change Point Detection for Graph Signals' at ICASSP 2024. More precisely, it focuses on the default parametrization of the *glasso* and *covcp* methods and is intended to increase our work reproducibility.

## Default signal generation setting

:construction: In progress :construction:


## The *covcp* method

Like many test statistics, the one proposed in [[Avanesov2018]](#Avanesov2018) relies on several hyper-parameters: the window size $\eta$, the bootstrap set length $|\mathcal{I}|$ and the test level $\alpha$. The choice of the bootstrap length and the window size require a bit of knowledge from the user as the samples included in $\mathcal{I}$ should not contain change points. Similarly, the larger the window size, the higher the sensitivity of a test but $\eta$ should not be higher than the stationarity segment length $\ell$ either. 

In our experiments default setting, we set these values to $|\mathcal{I}| = 80$ and $\eta = 80$ because they correspond to the largest admissible values, allowing an improved statistic computation. Based on these values, we calibrated the test level to $\alpha = 0.3$, which is much higher than the usual level values. Empirically, this was a good trade-off with respect to the target task. As we know the number of groundtruth change points, this allows us to prevent the whole algorithm (see below) to return too many false positives, while not predicting change-points on short stationarity segments. Hence the chosen value of $\alpha$.

In order to detect multiple change points, Avanesov and Buzun suggest to apply a BS procedure and to repeat the threshold calibration. In practice, we used the R library provided by the authors at [https://github.com/akopich/covcp](https://github.com/akopich/covcp), and we wrapped it into a suitable BS procedure in python thanks to the [rpy2](https://centre-borelli.github.io/ruptures-docs/) library. For each new segment $\{a+1, \ldots, b\}$ fed to the algorithm, we adapt the bootstrap set length $|\mathcal{I}|$ and window size $\eta$ by respectively truncating them to $(b-a) -1$ and $(b-a)/2 -1$. Also, we stop the Binary Segmentation procedure as soon as the predicted number of change points reaches the value $|\mathcal{T}^*|$. The authors clearly state that, statistically, the localization of the change point should be given with an incertitude equal to the size of the considered window. However, we decide to flag the change points at the location of the maximum value of the test-statistic. Therefore, we only use one window size (the largest possible, \ie $\eta = 80$) because the precision in time of the method does not depend on this parameter anymore.

## The *glasso* method

We set the sparsity pernalty to $\rho = 4\sqrt{\left( \log N / (b-a) \right)}$ in the minimization problem from [[Friedman2008]](#Friedman2008).

The above comes from the following parametrization of the sparsity penalty $\rho$:

$$
    \rho = m \sqrt{\left( \log N / (b-a) \right)}
$$

where $m$ is a multiplicative coefficient and $b - a$ the length of the segment over which the precision matrix is estimated. This formula is inspired from the bound derived in [[Friedman2008]](Friedman2008), but we essentially use it for convenience as it depends on the dimensions of the estimation problem. 

We calibrated the value of $m$ with a grid search applied over the values $\{0.001, 0.01, 0.1, 1, 10, 100\}$ for signals of length $T=1000$, dimension $N=20$ and with $\ell = 0.4 N(N+1)/2$, to be consistent with the hyper-parameters used in our simulation study. We observed that for high values of $m$, the sparsity assumption led to almost constant precision matrix estimation, so the predicted change points were evenly spaced over $\{1, T\}$. This could induce a performance bias as change points tend to be evenly spaced with our chosen signal generation algorithm. Thus, we also performed our grid search with signals containing onlyv $4$ change points randomly located and respecting $\ell = 0.4N(N+1)/2$.

We then refined the range of values between $1$ and $10$ and we finally chose $m=4$ as this visually led to better performances.


## References

<a id="Avanesov2018">[Avanesov2018]</a>
V. Avanesov  and N. Buzun, Change-point detection in high-dimensional covariance structure, Electronic Journal of Statistics, vol. 12, no. 2, pp. 3254–3294, Jan. 2018, Publisher: Institute of Mathematical

<a id="Friedman2008">[Friedman2008]</a>
J. Friedman, T. Hastie, and R. Tibshirani, Sparse inverse covariance estimation with the
graphical lasso, Biostatistics, vol. 9, no. 3, pp. 432–441, July 2008.