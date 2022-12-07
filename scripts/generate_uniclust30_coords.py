import os
import glob
import numpy as np
from tqdm import tqdm


RES_MAP_3_TO_1 = {
    'ALA': 'a',
    'VAL': 'v',
    'PHE': 'f',
    'PRO': 'p',
    'MET': 'm',
    'ILE': 'i',
    'LEU': 'l',
    'ASP': 'd',
    'GLU': 'e',
    'LYS': 'k',
    'ARG': 'r',
    'SER': 's',
    'THR': 't',
    'TYR': 'y',
    'HIS': 'h',
    'CYS': 'c',
    'ASN': 'n',
    'GLN': 'q',
    'TRP': 'w',
    'GLY': 'g',
    'SEC': 'u',
    'GLX': 'z',
    'UNK': 'x'
}


def parse_pdb(filepath):
	with open(filepath) as file:
		models = []
		for line in file:
			if line.startswith('MODEL'):
				seq, chains, coords, plddts = '', set(), [], []
			elif line.startswith('ATOM') and line[13:15] == 'CA':
				seq += RES_MAP_3_TO_1[line[17:20]]
				chains.add(line[21:22])
				coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
				plddts.append(float(line[60:66]))
			elif line.startswith('ENDMDL'):
				models.append({
					'model_idx': len(models),
					'seq': seq,
					'chains': sorted(list(chains)),
					'coords': coords,
					'plddts': plddts
				})
				seq, chains, coords, plddts = '', set(), [], []
	return models

indir = os.path.join('data', 'uniclust30')
outdir = os.path.join('data', 'uniclust30_coords')
if os.path.exists(outdir):
	print('Coords existed!')
	exit(0)
print('INFO: Approximate runtime is 30 minutes')
os.mkdir(outdir)

count = sum([os.path.isdir(os.path.join(indir, name)) for name in os.listdir(indir)])

num_structures = 0
num_multiple_models = 0
num_multiple_chains = 0
num_low_confidence = 0

for filepath in tqdm(glob.iglob(os.path.join(indir, '*', 'pdb', '*.pdb')), total=count):
	num_structures += 1
	name = filepath.split('/')[-1].split('.')[0]
	models = parse_pdb(filepath)

	# filter out multi-model structures
	if len(models) > 1:
		num_multiple_models += 1
		continue

	# filter out multi-chain structures
	model = models[0]
	if len(model['chains']) > 1:
		num_multiple_chains += 1
		continue

	# filter out low-confidence structures
	pLDDT = np.mean(model['plddts'])
	if pLDDT < 70:
		num_low_confidence += 1
		continue

	# save
	np.savetxt(
		os.path.join(outdir, f'{name}.npy'),
		model['coords'], delimiter=',', fmt='%.3f'
	)


print('Summary')
print(f'\tnumber of structures: {num_structures}')
print(f'\t\tmulti models: {num_multiple_models}')
print(f'\t\tmulti chains: {num_multiple_chains}')
print(f'\t\tlow confidence: {num_low_confidence}')
num_remainings = num_structures - num_multiple_models - num_multiple_chains - num_low_confidence
print(f'\t\tremaining: {num_remainings}')