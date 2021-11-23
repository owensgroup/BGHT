mkdir -p ./arxiv/Quadro-GV100
mkdir -p ./arxiv/NVIDIA-TITAN-V
mkdir -p ./arxiv/NVIDIA-TITAN-Xp
mkdir -p ./arxiv/Tesla-V100-DGXS-32GB
min_max='-mf 0 -xf 5000 -mi 0 -xi 1800'
python -W ignore ./plot.py $min_max -d ../results/arxiv/Quadro-GV100/ -od ./arxiv/Quadro-GV100/ &> arxiv/Quadro-GV100/summary.txt
python -W ignore ./plot.py $min_max -d ../results/arxiv/NVIDIA-TITAN-V/ -od ./arxiv/NVIDIA-TITAN-V/ &> arxiv/NVIDIA-TITAN-V/summary.txt
python -W ignore ./plot.py $min_max -d ../results/arxiv/NVIDIA-TITAN-Xp/ -od ./arxiv/NVIDIA-TITAN-Xp/ &> arxiv/NVIDIA-TITAN-Xp/summary.txt
python -W ignore ./plot.py $min_max -d ../results/arxiv/Tesla-V100-DGXS-32GB/ -od ./arxiv/Tesla-V100-DGXS-32GB/ &> arxiv/Tesla-V100-DGXS-32GB/summary.txt