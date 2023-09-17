# Set up ProteinMPNN
mkdir -p packages
cd packages
git clone https://github.com/dauparas/ProteinMPNN.git

# Set up ESMFold
pip install "fair-esm[esmfold]"
pip install "dllogger @ git+https://github.com/NVIDIA/dllogger.git"
pip install "openfold @ git+https://github.com/aqlaboratory/openfold.git"
pip install modelcif

# Set up TMscore/TMalign
mkdir -p TMscore
cd TMscore
wget https://zhanggroup.org/TM-score/TMscore.cpp
g++ -O3 -ffast-math -lm -o TMscore TMscore.cpp
chmod +x TMscore
wget https://zhanggroup.org/TM-align/TMalign.cpp
g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp
chmod +x TMalign
