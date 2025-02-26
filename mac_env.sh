cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../..
pip install -e .

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

pip install hydra-core pyrsistent
pip install pandas
pip install scipy
pip install opt_einsum

git clone --recurse-submodules https://github.com/dmlc/dgl.git
export DGL_LIBRARY_PATH=/Users/wojtek/Documents/coding/RFdiffusion/dgl
cd dgl

mkdir build
cd build
cmake -DUSE_OPENMP=off -DUSE_LIBXSMM=OFF ..
make -j4
cd ../python
python setup.py install
# Build Cython extension
python setup.py build_ext --inplace

cd ../..


# https://github.com/RosettaCommons/RFdiffusion/issues/306
pip install torchdata==0.9.0

#https://discuss.pytorch.org/t/modulenotfounderror-no-module-named-torch-utils-import-utils/208935
# suggest pip install torch==2.1.0 torchdata==0.7.0
# but oldest torch na mac m1/m4 is 2.2.0 :( -> dead end