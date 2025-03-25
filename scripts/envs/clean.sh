
conda create -n clean python==3.8 -y # 3.10 does not work
conda activate clean
pip install -r CLEAN/app/requirements.txt

#conda install pytorch==1.11.0 cpuonly -c pytorch
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch

python build.py install

git clone https://github.com/facebookresearch/esm.git CLEAN/app/esm

pip install gdwon

gdown gdown https://drive.google.com/uc?id=1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ -O CLEAN/app/pretrained.zip

unzip -j  CLEAN/app/pretrained.zip  -d CLEAN/app/data/pretrained

