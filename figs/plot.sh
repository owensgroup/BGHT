gpu="Radeon-RX-7900-XT"
mkdir -p $gpu
python -W ignore ./plot.py -d  ../results/$gpu/ -od ./$gpu 2>&1 | tee "$gpu/$gpu.txt"

# python -W ignore ./plot.py -d  ../results/NVIDIA-TITAN-V/ -od ./NVIDIA-TITAN-V
# python -W ignore ./plot.py -d  ../build/results/NVIDIA-A100-PCIE-40GB/ -od ./NVIDIA-A100-PCIE-40GB &> NVIDIA-A100-PCIE-40GB.txt
