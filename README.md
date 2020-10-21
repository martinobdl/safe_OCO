# Description

Code for the paper: Conservative Online Convex Optimization

Submitted to: AISTATS 2021  
Id: 873 

### Requirements
 Install the requirements with:

 ```
 pip3 install -r requirements.txt
 ```

### Experiments

The provided experimental environments are: 

* Synthetic Online Linear Regression
* IMDB
* SpamBase
* Online Portfolio Optimization

Sample experiments can be run with the following commands:

`
python3 src/OLR_main.py  
python3 src/IMDB_main.py  
python3 src/SPAM_main.py  
python3 src/FIN_main.py  
`

---

The experimental pipeline is organized as follows: the main file contains an instance of the experiment class, on which we call the run() method. After that we commented out the call to the save method that saves a yaml file (describing completely the experiment) and the results.  
These output files should be used to compute the performance metrics and to plot the results.


Moreover, we shortened the time horizon of the scripts of the experiments in order to run them faster.
To reproduce the exact results, they should have to be modified as described in the experimental section of the paper.


The data for the financial experiment is not provided therein this zip archive for space constraint. The complete code code can be found at <https://drive.google.com/drive/folders/1RkE6hiI9ZBhLL5z2cR7s8Zuw1RT2smpP?usp=sharing>  
