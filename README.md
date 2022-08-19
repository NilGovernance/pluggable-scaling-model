# Pluggable scaling model for Ethereum cluster

## Requirements
- Python >= 3.10

## Installation
1. Install pip 
> sudo apt install python3-pip
2. Install pandas, plotly, numpy, scipy
> pip install pandas scipy numpy plotly

## Run script
Change assumptions in the **Input assumptions** section (please, read the comments with description first)

Run script 
> python3 main.py

## Results
Script generates **results.html** with charts:
- gas price
- average wait time
- average transaction fee
- % transactions stuck in mempool (stuck = more than hour in mempool)