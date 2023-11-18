# Reproducing the arXiv paper results:

The instructions below applies to the tag `arxiv-version` which you can get via cloning the tag:

```terminal 
git clone --branch arxiv-version https://github.com/owensgroup/BGHT.git
```

## Summary
```bash
# Building the benchmarks
mkdir build && cd build
cmake ..
make
# Running the benchmarks
source ../scripts/benchmark.sh
# Plotting
cd ../figs
# Note the results will be stored under ../build/results/gpu_name
# e.g., TITAN V GPU results will be under ../build/results/NVIDIA-TITAN-V/
python -W ignore ./plot.py -d  ../build/results/NVIDIA-TITAN-V/
# Plots will be located at the current directory `figs`
```

## Running the benchmarks
After building you should have the following binaries
```bash
user:~/github/bght/build/bin$ ls
probes_per_technique  rates_per_technique  rates_per_technique_fixed_lf
```
1. `probes_per_technique`: An experiment to evaluate the average number of probes per key for a constant number of keys and all different probing schemes and load factors<br />
2. `rates_per_technique`: An experiment to evaluate the average insertion and find throughputs for a constant number of keys and all different probing schemes and load factors<br />
3. `rates_per_technique_fixed_lf`: An experiment to evaluate the average insertion and find throughputs for two constant load factors, a range of keys, and all different probing schemes<br />

All executables have CPU validation that can be turned on by using the flag `--validate=true`. Additional options are shown below:
```
./probes_per_technique
    --num-keys      Number of keys for all experiments e.g, --num-keys=512
./rates_per_technique
    --num-keys      Number of keys for all experiments e.g, --num-keys=512
./rates_per_technique_fixed_lf
    --min-keys      Minimum number of keys for all experiments e.g, --min-keys=512
    --max-keys      Maximum number of keys for all experiments e.g, --max-keys=1024
    --step          Step between the minimum and maximum for all experiments e.g, --step=32
    --load-factor1  First constant load factor for all experiments e.g, --load-factor1=0.8
    --load-factor2  First constant load factor for all experiments e.g, --load-factor1=0.9
```

## Plotting the results
A Python script is under `bght/figs` which will parse all outputs of the benchmarking step and reproduce the figures (in SVG format) in the paper. The script requires the packages: pandas, Matplotlib, NumPy, and SciPy. The script takes the following arguments:
```
python ./plot.py
    -d              Root directory that contains the results
    -mf             Minimum find rate for the charts axes (optional)
    -xf             Maximim find rate for the charts axes (optional)
    -mi             Minimum insertion rate for the charts axes (optional)
    -xi             Maximim insertion rate for the charts axes (optional)
    -p              Specify the probing scheme to plot results for (optiona - default plots all probing schemes)
```
