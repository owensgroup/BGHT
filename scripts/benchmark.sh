#!/bin/bash

# Pick a suitable device id
device=0

# Probing counts per scheme
./bin/probes_per_technique --validate=true --num-keys=20000000 --device=$device 2>&1 | tee probes_per_technique.log

# Rates for fixed number of keys
./bin/rates_per_technique --validate=true --num-keys=50000000 --device=$device 2>&1 | tee rates_per_technique.log

# Rates for fixed load factor
./bin/rates_per_technique_fixed_lf --validate=true --min-keys=1000000 --max-keys=96000000 --step=5000000 --device=$device 2>&1 | tee rates_per_technique_fixed_lf.log