# llmoptim
2024 Fall CMPT-981: LLMs for Optimization

## Installation
```
conda create -n llmoptim python=3.10
conda activate llmoptim
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
pip install flash-attn==2.5.5 --no-build-isolation
```