mkdir -p data
cd data
mkdir cath
cd cath
mkdir raw
cd raw
wget http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt
wget http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.atom.fa
wget http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz
tar zxvf cath-dataset-nonredundant-S40.pdb.tgz
rm cath-dataset-nonredundant-S40.pdb.tgz
cd ../../..

python scripts/generate_cath_coords.py