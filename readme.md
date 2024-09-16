# Change Point for Graph Signals

This repository includes all the material used in the experimental study of the work on **Covariance Change Point Detection (CPD) for Graph Signals**. It also provides additional content related to CPD for Graph Signals and Graph Stationarity.

## Structure

The repository is organized as follows:

### Python regular scripts

<details><summary>Detailed list</summary>
    <ul>
        <li>[utils](https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/utils.py): file writing and processing.</li>
        <li>[signal_related](signal_related.py): signal generation and modification.</li>
    </ul>
</details>



- `signal_related`: signal generation and modification. 
- `graph_related`: graph generation and modification.
- `result_related`: metrics computation, processing and storage.
        - `custom_cost_functions`: cost function classes, from the [ruptures](https://centre-borelli.github.io/ruptures-docs/) BaseCost class. 
        - `numba_cost_functions`: [numba](https://numba.pydata.org/)-compatible implementations of the cost functions.
        - `running_cpd`: utils and dynamic programming implementation for CPD solving.
        - `rpy2_related`: utils and wrapping functions for the apllication of the Graph Lasso algorithm from [glasso](https://cran.r-project.org/web/packages/glasso/index.html) [[Friedman2008]](Friedman2008) and the [covcp](https://github.com/akopich/covcp) [[Avanesov2018]](#Avanesov2018) method based on the [rpy2](https://centre-borelli.github.io/ruptures-docs/) package.

### Command Line scripts

- `run_cpd_cmd_line`: produces predictions files for the target signals and methods.
- `compute_metrics_cmd_line`: produces metrics files based on prediction files. 


### Jupyter notebooks

- `covariance_matrix_cpd`: contains the main utilities such as predictions generation (same as but more flexible than `run_cpd_cmd_line`), metrics computation (same as but more flexible than `compute_metrics_cmd_line`), plotting utils and cells as well as experiments presentation and visualization.
- `cpd_playground`: additional content related to CPD (search algorithms comparions) and the application to local mean change for graph signals, through the Graph Fourier Scan Statistic [[Ferrari2019]](#Ferrari2019).
- `graph_stationarity`: additional content related to graph stationarity. Includes the appendix material of the work on Covariance CPD.
- `real_dataset_preprocessing`: utils, visualization and preprocessing for different CPD real datasets.
- `rpy2_covariance_cpd`: experiments on different R libraries with rpy2.




## References

<a id="Avanesov2018">[Avanesov2018]</a>
V. Avanesov  and N. Buzun, Change-point detection in high-dimensional covariance structure, Electronic Journal of Statistics, vol. 12, no. 2, pp. 3254–3294, Jan. 2018, Publisher: Institute of Mathematical

<a id="Ferrari2019">[Ferrari2019]</a>
A. Ferrari, C. Richard, and L. Verduci. Distributed Change Detection in Streaming Graph Signals. In 2019 IEEE 8th International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP), pages 166–170,

<a id="Friedman2008">[Friedman2008]</a>
J. Friedman, T. Hastie, and R. Tibshirani, Sparse inverse covariance estimation with the
graphical lasso, Biostatistics, vol. 9, no. 3, pp. 432–441, July 2008.








