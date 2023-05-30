## Set up the python environment

```shell
conda create -n audiounet python=3.7
conda activate audiounet
# make sure that the pytorch cuda is consistent with the system cuda
# pytorch version must >= 1.9.0
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```