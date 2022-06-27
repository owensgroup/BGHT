#!/bin/bash

# probing counts per scheme
./bin/probes_per_technique --validate=false --num-keys=20000000
# rates for fixed number of keys
./bin/rates_per_technique --validate=false --num-keys=50000000
# rates for fixed load factor
./bin/rates_per_technique_fixed_lf --validate=false --min-keys=1000000 --max-keys=96000000 --step=5000000