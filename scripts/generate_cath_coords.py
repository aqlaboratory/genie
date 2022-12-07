import os
import numpy as np
from tqdm import tqdm


ROOT_DIR = 'data/cath'
RAW_DIR = os.path.join(ROOT_DIR, 'raw')
CATHDOM_ATOM_FILEPATH = os.path.join(RAW_DIR, 'cath-dataset-nonredundant-S40.atom.fa')
CATHDOM_LIST_FILEPATH = os.path.join(RAW_DIR, 'cath-domain-list.txt')
CATHDOM_DIR = os.path.join(RAW_DIR, 'dompdb')


def remove(domains, idxs):
    for idx in sorted(idxs, reverse=True):
        del domains[idx]

def load_domains():
	domains = []
	with open(CATHDOM_ATOM_FILEPATH) as file:
		for line in file:
			if line.startswith('>'):
				domains.append({
					'name': line.strip().split('|')[-1].split('/')[0],
					'residue_indices': line.strip().split('|')[-1].split('/')[1]
				})
			else:
				domains[-1]['sequence'] = line.strip()
	print(f'Number of domains: {len(domains)}')
	return domains

def filter_domains_with_multi_chains(domains):
	remove_idxs = []
	for idx, domain in enumerate(domains):
		if '_' in domain['residue_indices']:
			remove_idxs.append(idx)
	remove(domains, remove_idxs)
	print(f'After filtering domains with multi chains: {len(domains)}')

def filter_domains_with_missing_atoms(domains):
	remove_idxs = []
	for idx, domain in enumerate(tqdm(domains)):
		pdb_filepath = os.path.join(CATHDOM_DIR, domain['name'])
		coords = []
		with open(pdb_filepath) as file:
			for line in file:
				if line.startswith('ATOM') and line[13:15].strip() in ['N', 'CA', 'C']:
					coords.append([
						float(line[30:38].strip()),
						float(line[38:46].strip()),
						float(line[46:54].strip())
					])
		if (len(coords) / 3) != len(domain['sequence']):
			remove_idxs.append(idx)
		else:
			domain['coords'] = coords
	remove(domains, remove_idxs)
	print(f'After filtering domains with missing residues: {len(domains)}')

def filter_domains_with_missing_classes(domains):

	# load
	classes = {}
	with open(CATHDOM_LIST_FILEPATH) as file:
		for line in file:
			if not line.startswith('#'):
				elts = line.split()
				classes[elts[0]] = f'{elts[1]}.{elts[2]}.{elts[3]}.{elts[4]}'

	# filter
	remove_idxs = []
	for idx, domain in enumerate(domains):
		if domain['name'] not in classes:
			remove_idxs.append(idx)
		domain['class'] = classes[domain['name']]
	remove(domains, remove_idxs)
	print(f'After filtering domains with missing classes: {len(domains)}')

def save_domains(domains):

	classes_filepath = os.path.join(ROOT_DIR, 'classes.txt')
	with open(classes_filepath, 'w') as file:
		for domain in domains:
			file.write(f'{domain["name"]},{domain["class"]}\n')

	coords_dir = os.path.join(ROOT_DIR, 'coords')
	os.mkdir(coords_dir)
	for domain in domains:
		coords_filepath = os.path.join(coords_dir, f'{domain["name"]}.npy')
		np.savetxt(coords_filepath, domain['coords'], delimiter=',')


def main():
	domains = load_domains()
	filter_domains_with_multi_chains(domains)
	filter_domains_with_missing_atoms(domains)
	filter_domains_with_missing_classes(domains)
	save_domains(domains)


if __name__ == '__main__':
	main()
