# Description

Code for the paper: Conservative Online Convex Optimization

Submitted to: ICML 2021  
Id: 4470

### Requirements
 Install the requirements with:

 ```
 pip3 install -r requirements.txt
 ```

It is suggested to use a virtual environment to run these experiments.

Install [parallel](https://www.gnu.org/software/parallel/) to parallalize the experiments.

### Experiments

The provided experimental environments are: 

* Synthetic Online Linear Regression
* IMDB
* SpamBase
* Online Portfolio Optimization

The experiments can be run with the following command:

```bash
./run_experiments.sh n s
```

Where `n` is the number of workers used to parellalize the experiments, and `s` is the number of seeds for the OLR experiment (30 has been used in the paper).

This will save the results of the experiments in the `./experiments` folder.

Run `export PYTHONPATH='.'` to make sure that python add `.` to the `sys.path` directory list.

After this step, one could plot the figures in the paper with the python scripts provided in the `./plots` folder, e.g. `python3 plots/IMDB_figure_4.py`.

---

The experimental pipeline is organized as follows: for each experiment the corresponding main file contains an instance of the experiment class, on which we call the run() method. After that we commented out the call to the save method that saves a yaml file (describing completely the experiment) and the results.
These output files should be used to compute the performance metrics and to plot the results.

---

The data used to run the Online Portfolio Optimization experiment is quite large, and it is not provided in this folder. In order to obtain it run the following command:

```bash
python3 download_fin_data.py
```

This will put the required data into the `./data` folder.

After downloading the data you can run the command `python3 src/FIN_main.py` to generate the experiment results. Then you can plot it with `python3 plots/FIN_figure_7.py`.
