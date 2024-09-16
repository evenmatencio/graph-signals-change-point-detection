# Change Point for Graph Signals

This repository includes all the material used in the experimental study of the work on **Covariance Change Point Detection (CPD) for Graph Signals**. It also provides additional content related to CPD for Graph Signals and Graph Stationarity.

## Structure

The repository is organized as follows:

### Python regular scripts

<details><summary>Detailed list</summary>
    <ul>
        <li><a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/utils.py'>utils.py</a>: file writing and processing.</li>
        <li><a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/signal_related.py'>signal_related.py</a>: signal generation and modification.</li>
        <li><a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/graph_related.py'>graph_related.py</a>: graph generation and modification.</li>
        <li><a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/result_related.py'>result_related.py</a>: metrics computation, processing and storage.</li>
        <li><a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/custom_cost_functions.py'>custom_cost_functions.py</a>: cost function classes, from the <a href='https://centre-borelli.github.io/ruptures-docs/'>ruptures</a>  <i>BaseCost</i> class.</li>
        <li><a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/numba_cost_functions.py'>numba_cost_functions.py</a>: numba-compatible implementations of the cost functions.</li>
        <li><a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/running_cpd.py'>running_cpd.py</a>: utils and dynamic programming implementation for CPD solving.</li>
        <li><a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/rpy2_related.py'>rpy2_related.py</a>: utils and wrapping functions for the apllication of the Graph Lasso algorithm from <a href="#Friedman2008">[Friedman2008]</a> and the <a href='https://github.com/akopich/covcp'>covcp</a> method from <a href="#Avanesov2018">[Avanesov2018]</a>.</li>
    </ul>
</details>


### Command Line scripts

<details><summary>Detailed list</summary>
    <ul>
        <li> <a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/run_cpd_cmd_line.py'>run_cpd_cmd_line.py</a>: produces predictions files for the target signals and methods.
        <li> <a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/compute_metrics_cmd_line.py'>compute_metrics_cmd_line.py</a>: produces metrics files based on prediction files. 
    </ul>
</details>


### Jupyter notebooks

<details><summary>Detailed list</summary>
    <ul>
    <li> <a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/covariance_matrix_cpd.ipynb'>covariance_matrix_cpd.ipynb</a>: contains the main utilities such as predictions generation (same as but more flexible than <a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/run_cpd_cmd_line.py'>run_cpd_cmd_line.py</a>), metrics computation (same as but more flexible than <a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/compute_metrics_cmd_line.py'>compute_metrics_cmd_line.py</a>), plotting utils and cells as well as experiments presentation and visualization.
    <li> <a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/cpd_playground.ipynb'>cpd_playground.ipynb</a>: additional content related to CPD (search algorithms comparions) and the application to local mean change for graph signals, through the Graph Fourier Scan Statistic <a href="#Ferrari2019">[Ferrari2019]</a>
    <li> <a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/graph_stationarity.ipynb'>graph_stationarity.ipynb</a>: additional content related to graph stationarity. Includes the appendix material of the work on Covariance CPD.
    <li> <a href='https://github.com/evenmatencio/graph-signals-change-point-detection/blob/main/real_dataset_preprocessing.ipynb'>real_dataset_preprocessing</a>: utils, visualization and preprocessing for different CPD real datasets.
    </ul>
</details>

## Usage

:construction: In progress :construction:


## References

<a id="Avanesov2018">[Avanesov2018]</a>
V. Avanesov  and N. Buzun, Change-point detection in high-dimensional covariance structure, Electronic Journal of Statistics, vol. 12, no. 2, pp. 3254–3294, Jan. 2018, Publisher: Institute of Mathematical

<a id="Ferrari2019">[Ferrari2019]</a>
A. Ferrari, C. Richard, and L. Verduci. Distributed Change Detection in Streaming Graph Signals. In 2019 IEEE 8th International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP), pages 166–170,

<a id="Friedman2008">[Friedman2008]</a>
J. Friedman, T. Hastie, and R. Tibshirani, Sparse inverse covariance estimation with the
graphical lasso, Biostatistics, vol. 9, no. 3, pp. 432–441, July 2008.








