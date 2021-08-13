# wrf_tools

Tools for extracting primary and diagnostic variables from from the Weather Research and Forecasting Model (WRF).

Authors: Michael Smith (michaesm@marine.rutgers.edu) and Lori Garzio (lgarzio@marine.rutgers.edu)

Rutgers Center for Ocean Observing Leadership

## Installation Instructions
Add the channel conda-forge to your .condarc. You can find out more about conda-forge from their website: https://conda-forge.org/

`conda config --add channels conda-forge`

Clone the wrf_tools repository

`git clone https://github.com/rucool/wrf_tools.git`

Change your current working directory to the location that you downloaded wrf_tools. 

`cd /Users/garzio/Documents/repo/wrf_tools/`

Create conda environment from the included environment.yml file:

`conda env create -f environment.yml`

Once the environment is done building, activate the environment:

`conda activate wrf`

Install the toolbox to the conda environment from the root directory of the wrf_tools toolbox:

`pip install .`

The toolbox should now be installed to your conda environment.


## Citations
Ladwig, W. (2017). wrf-python (Version 1.3.2) [Software]. Boulder, Colorado: UCAR/NCAR. https://doi.org/10.5065/D6W094P1
