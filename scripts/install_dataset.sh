mkdir data
cd data
echo 'Downloading SCOPe dataset ...'
wget https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
wget https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz
tar -xf pdbstyle-sel-gs-bib-40-2.08.tgz
rm pdbstyle-sel-gs-bib-40-2.08.tgz
cd ..

echo 'Preprocessing SCOPe dataset ...'
python scripts/generate_scope_coords.py
