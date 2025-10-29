# Inhibition Stabilized Network history dependence
Codes for Hilty and Miller, Unpublished: "Inhibitory-stabilization is sufficient for history-dependent computation in a randomly connected attractor network"

_bioRxiv_ preprint release: [![DOI](https://zenodo.org/badge/956121706.svg)](https://doi.org/10.5281/zenodo.17470740)

## File Structure
    |-- undefined
        |-- README.md
        |-- ISN.yml
        |-- model
        |   |-- fiducial_network.csv
        |   |-- left_right_task.py
        |   |-- network_model.py
        |   |-- plot_style.py
        |   |-- util.py
        |   |-- Wji_10_reliability_1.00.npy
        |-- figures
        |   |-- figure1
        |   |-- figure2
        |   |-- figure3
        |   |-- figure4
        |   |   |-- run `data_figure4_all_states.py` before `data_figure4_stim_sweep_vs_max_itinerancy.py`
        |   |-- figure5
        |   |-- figure6
        |   |-- figure7
        |   |   |-- `make_Wjis.py` generates `all data_{}.py` files which are necessary for other data `.py` files

- `model`: Contains the core model code, including fiducial network parameters and simulation functions.
- `figures`: Contains the code and data for generating figures. Each folder contains two types of `.py` scripts: those for generating the data and one for plotting the figure. Data used in the paper is also included in these folders. Note 1: If you generate your own data, you will need to change file paths accordingly in the plotting scripts. Note 2: some data scripts have dependencies on each other -- this is specified above when relevant.
- `__dev` folders: Contain miscelaneous .ipynb files, old/bugged data used during development.

## Usage
1) Clone the code locally.
2) Create the conda environment
```
conda evn create -f ISN.yml
```
3) Activate the environment
```
conda activate ISN
```
4) Navigate to the directory containing both the `model` and `figures` folders
5) Compute and plot figure1
```
python figures/figure1/figure1.py
```

### Troubleshooting
Many scripts take a long time and it is desirable to run them in the background, which can present challenges. Here is some guidance for running these scripts in the background on different OS.

#### Windows:
To run a script:
```
$cwd = Get-Location
$outputFile = Join-Path $cwd "output.txt"
$job = Start-Job -scriptblock {Set-Location -Path $using:cwd; Start-Process -FilePath "\absolute\path\to\miniconda\env\python.exe" -ArgumentList "figures/figureN/script.py" -RedirectStandardOutput $using:outputFile -NoNewWindow -Wait}
```
Helpful commands to manipulate/check on the Job while it runs:
```
Receive-Job $job
```
```
Get-Job
```
```
Stop-Job -State Running
```

#### Linux/MacOS:
Typically, `nohup` can be used to run scripts in the background.
```
nohup python figures/figureN/script.py
```
Certain scripts involving multiprocessing require a more complicated use of nohup:
```
nohup bash -lc 'PYTHONUNBUFFERED=1 exec python /absolute/path/to/heterogeneity_vs_reliability.py </dev/null >>/absolute/path/to/run.out 2>>/absolute/path/to/run.err' &
disown
```
