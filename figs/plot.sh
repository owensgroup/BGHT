gpu="NVIDIA-GeForce-RTX-2080"
python -W ignore ./plot.py -d  ../results/$gpu/ -od ./$gpu &> $gpu/$gpu.txt

# python -W ignore ./plot.py -d  ../results/NVIDIA-TITAN-V/ -od ./NVIDIA-TITAN-V
# python -W ignore ./plot.py -d  ../build/results/NVIDIA-A100-PCIE-40GB/ -od ./NVIDIA-A100-PCIE-40GB &> NVIDIA-A100-PCIE-40GB.txt
