## Python scripts for analyzing seasonal infection risk from serology samples

- `age_analysis.py`: An implementation of the survival analysis that incorporates host-age.  Can be used to reproduce Figure 4 in the main text.

- `leftcensored_analysis.py`: An implementation of the survival analysis that incorporates left-censoring. Can be used to reproduce Figure 5 in the main text.

- `rightcensored_analysis.py`: An implementation of the standard right-censored
survival analysis. Can be used to reproduce Figure 3 in the main text.

- `survival.py`: Accompanying functions used to prepare simulations and data for the survival analyses. Documentation of functions provided in the file.

- `estimating_infection_risk_tutorial*`: The .ipynb and .html files give a brief tutorial on how one can use the Stan models provided in `stan_code/` to estimate
time since infection from serology data. Moreover, it shows how some of the functions provided in `survival.py` can be used in an analysis workflow of
serology data.  Additional description is given in the files. The .ipynb file
is an editable Jupyter notebook. 

- `stan_code/`: Folder containing Stan Models. See `stan_code/README.txt` for descriptions