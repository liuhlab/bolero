mamba env create -n bolero python=3.11
mamba install -n bolero pytorch=2.4 torchvision torchaudio pytorch-cuda=12.1 cudnn -c pytorch -c nvidia
pip install ray=2.34
# install prebuilt flash_attn using correct version and platform
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install -e path_to_bolero
pip install -e path_to_bolerodata
