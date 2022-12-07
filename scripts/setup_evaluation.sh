pip install --upgrade pip
pip install tqdm
pip install numpy
pip install torch
pip install pandas

git clone https://github.com/dauparas/ProteinMPNN.git

git clone --branch beta https://github.com/sokrypton/OmegaFold.git
cd OmegaFold
python setup.py install
cd ..

mkdir TMscore
cd TMscore
wget https://zhanggroup.org/TM-score/TMscore.cpp
g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp
chmod +x TMscore
wget https://zhanggroup.org/TM-align/TMalign.cpp
g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp
chmod +x TMalign
cd ..
