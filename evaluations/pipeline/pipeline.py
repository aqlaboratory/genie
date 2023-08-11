import os
import glob
import torch
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
	hcluster,
	save_as_pdb,
	parse_pdb_file,
	parse_tm_file,
	parse_pae_file,
	assign_secondary_structures,
	assign_left_handed_helices
)


class Pipeline:

	def __init__(
		self,
		inverse_fold_model,
		fold_model,
		tm_score_exec='packages/TMscore/TMscore',
		tm_align_exec='packages/TMscore/TMalign'
	):
		self.inverse_fold_model = inverse_fold_model
		self.fold_model = fold_model
		self.tm_score_exec = tm_score_exec
		self.tm_align_exec = tm_align_exec

	def evaluate(self, input_dir, output_dir, info=None, clean=True, verbose=True):

		##################
		###   Set up   ###
		##################

		assert os.path.exists(input_dir), 'Missing input directory'
		coords_dir = os.path.join(input_dir, 'coords')
		assert os.path.exists(coords_dir), 'Missing coordinate directory'
		assert not os.path.exists(output_dir), 'Output directory existed'
		os.mkdir(output_dir)

		###################
		###   Process   ###
		###################

		# main pipeline
		pdbs_dir = self._preprocess(coords_dir, output_dir, verbose)
		sequences_dir = self._inverse_fold(pdbs_dir, output_dir, verbose)
		structures_dir = self._fold(sequences_dir, output_dir, verbose)
		scores_dir = self._compute_scores(pdbs_dir, structures_dir, output_dir, verbose)
		results_dir, designs_dir = self._aggregate_scores(scores_dir, structures_dir, output_dir, verbose)
		self._compute_secondary_diversity(designs_dir, results_dir, verbose)
		self._compute_tertiary_diversity(designs_dir, results_dir, verbose)

		# branched pipeline
		self._evaluate_motif_scaffolding(input_dir, designs_dir, results_dir, info, verbose)

		####################
		###   Clean up   ###
		####################

		self._process_results(results_dir, output_dir)

		if clean:
			shutil.rmtree(pdbs_dir)
			shutil.rmtree(sequences_dir)
			shutil.rmtree(structures_dir)
			shutil.rmtree(scores_dir)
			shutil.rmtree(results_dir)

	def _preprocess(self, coords_dir, output_dir, verbose):
		"""
		Convert coordinate files to pdb files.
		"""

		# create output directory
		pdbs_dir = os.path.join(output_dir, 'pdbs')
		assert not os.path.exists(pdbs_dir), 'Output pdbs directory existed'
		os.mkdir(pdbs_dir)

		# process
		for filepath in tqdm(
			glob.glob(os.path.join(coords_dir, '*.npy')),
			desc='Preprocessing', disable=not verbose
		):
			try:
				domain_name = filepath.split('/')[-1].split('.')[0]
				pdb_filepath = os.path.join(pdbs_dir, f'{domain_name}.pdb')
				coords = np.loadtxt(filepath, delimiter=',')
				if np.isnan(coords).any():
					print(f'Error: {domain_name}')
				else:
					seq = 'A' * coords.shape[0]
					save_as_pdb(seq, coords, pdb_filepath)
			except:
				os.remove(pdb_filepath)
				print(f'Error: {domain_name}')
				continue

		return pdbs_dir

	def _inverse_fold(self, pdbs_dir, output_dir, verbose):
		"""
		Run inverse folding to obtain sequences.
		"""

		# create output directory
		sequences_dir = os.path.join(output_dir, 'sequences')
		assert not os.path.exists(sequences_dir), 'Output sequences directory existed'
		os.mkdir(sequences_dir)

		# process
		for pdb_filepath in tqdm(
			glob.glob(os.path.join(pdbs_dir, '*.pdb')),
			desc='Inverse folding', disable=not verbose
		):
			domain_name = pdb_filepath.split('/')[-1].split('.')[0]
			sequences_filepath = os.path.join(sequences_dir, f'{domain_name}.txt')
			with open(sequences_filepath, 'w') as file:
				file.write(self.inverse_fold_model.predict(pdb_filepath))

		return sequences_dir

	def _fold(self, sequences_dir, output_dir, verbose):
		"""
		Run folding to obtain structures.
		"""

		# create output directory
		structures_dir = os.path.join(output_dir, 'structures')
		assert not os.path.exists(structures_dir), 'Output structures directory existed'
		os.mkdir(structures_dir)

		# process
		for filepath in tqdm(
			glob.glob(os.path.join(sequences_dir, '*.txt')),
			desc='Folding', disable=not verbose
		):
			domain_name = filepath.split('/')[-1].split('.')[0]
			with open(filepath) as file:
				seqs = [line.strip() for line in file if line[0] != '>']
			for i in range(len(seqs)):

				# define output filepaths
				output_pdb_filepath = os.path.join(structures_dir, f'{domain_name}-resample_{i}.pdb')
				output_pae_filepath = os.path.join(structures_dir, f'{domain_name}-resample_{i}.pae.txt')
				
				# run structure prediction
				pdb_str, pae = self.fold_model.predict(seqs[i])
				
				# save
				np.savetxt(output_pae_filepath, pae, '%.3f')
				with open(output_pdb_filepath, 'w') as f:
					f.write(pdb_str)

		return structures_dir

	def _compute_scores(self, pdbs_dir, structures_dir, output_dir, verbose):
		"""
		Compute self-consistency scores.
		"""

		# create output directory
		scores_dir = os.path.join(output_dir, 'scores')
		assert not os.path.exists(scores_dir), 'Output scores directory existed'
		os.mkdir(scores_dir)

		# process
		for designed_pdb_filepath in tqdm(
			glob.glob(os.path.join(structures_dir, '*.pdb')),
			desc='Computing scores', disable=not verbose
		):

			# parse
			filename = designed_pdb_filepath.split('/')[-1].split('.')[0]
			domain_name = filename.split('-')[0]
			seq_name = filename.split('-')[1]

			# compute score
			generated_pdb_filepath = os.path.join(pdbs_dir, f"{domain_name}.pdb")
			output_filepath = os.path.join(scores_dir, f'{domain_name}-{seq_name}.txt')
			subprocess.call(f'{self.tm_score_exec} {generated_pdb_filepath} {designed_pdb_filepath} > {output_filepath}', shell=True)

		return scores_dir

	def _aggregate_scores(self, scores_dir, structures_dir, output_dir, verbose):
		"""
		Aggregate self-consistency scores and structural confidence scores.
		Save best resampled structures.
		"""

		# create output directory
		results_dir = os.path.join(output_dir, 'results')
		designs_dir = os.path.join(output_dir, 'designs')
		assert not os.path.exists(results_dir), 'Output results directory existed'
		assert not os.path.exists(designs_dir), 'Output designs directory existed'
		os.mkdir(results_dir)
		os.mkdir(designs_dir)

		# create scores filepath
		scores_filepath = os.path.join(results_dir, 'single_scores.csv')
		with open(scores_filepath, 'w') as file:
			columns = ['domain', 'seqlen', 'scTM', 'scRMSD', 'pLDDT', 'pAE']
			file.write(','.join(columns) + '\n')

		# get domains
		domains = set()
		for filepath in glob.glob(os.path.join(scores_dir, '*.txt')):
			domains.add(filepath.split('/')[-1].split('-')[0])
		domains = list(domains)

		# process
		for domain in tqdm(domains, desc='Aggregating scores', disable=not verbose):

			# find best sample based on scRMSD
			resample_idxs, scrmsds = [], []
			for filepath in glob.glob(os.path.join(scores_dir, f'{domain}-resample_*.txt')):
				resample_idx = int(filepath.split('_')[-1].split('.')[0])
				resample_results = parse_tm_file(filepath)
				resample_idxs.append(resample_idx)
				scrmsds.append(resample_results['rmsd'])
			best_resample_idx = resample_idxs[np.argmin(scrmsds)]

			# parse scores
			tm_filepath = os.path.join(
				scores_dir,
				f'{domain}-resample_{best_resample_idx}.txt'
			)
			output = parse_tm_file(tm_filepath)
			sctm, scrmsd, seqlen = output['tm'], output['rmsd'], output['seqlen']

			# parse pLDDT
			pdb_filepath = os.path.join(
				structures_dir,
				f'{domain}-resample_{best_resample_idx}.pdb'
			)
			output = parse_pdb_file(pdb_filepath)
			plddt = output['pLDDT']

			# parse pAE
			pae_filepath = os.path.join(
				structures_dir,
				f'{domain}-resample_{best_resample_idx}.pae.txt'
			)
			pae = parse_pae_file(pae_filepath)['pAE'] if os.path.exists(pae_filepath) else None

			# save results
			with open(scores_filepath, 'a') as file:
				file.write('{},{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
					domain, seqlen, sctm, scrmsd, plddt, pae
				))

			# save best resampled structure
			design_filepath = os.path.join(designs_dir, f'{domain}.pdb')
			shutil.copyfile(pdb_filepath, design_filepath)

		return results_dir, designs_dir

	def _compute_secondary_diversity(self, designs_dir, results_dir, verbose):
		"""
		Compute secondary diversity.
		TODO: add DSSP
		"""

		# create output filepath
		assert os.path.exists(results_dir), 'Missing output results directory'
		secondary_filepath = os.path.join(results_dir, 'single_secondary.csv')
		assert not os.path.exists(secondary_filepath), 'Output secondary filepath existed'
		with open(secondary_filepath, 'w') as file:
			columns = ['domain', 'pct_helix', 'pct_strand', 'pct_ss', 'pct_left_helix']
			file.write(','.join(columns) + '\n')

		# process
		for design_filepath in tqdm(
			glob.glob(os.path.join(designs_dir, '*.pdb')),
			desc='Computing secondary diversity', disable=not verbose
		):

			# parse filepath
			domain = design_filepath.split('/')[-1].split('.')[0]

			# parse pdb file
			output = parse_pdb_file(design_filepath)

			# parse secondary structures
			ca_coords = torch.Tensor(output['ca_coords']).unsqueeze(0)
			pct_ss = torch.sum(assign_secondary_structures(ca_coords, full=False), dim=1).squeeze(0) / ca_coords.shape[1]
			pct_left_helix = torch.sum(assign_left_handed_helices(ca_coords).squeeze(0)) / ca_coords.shape[1]
			
			# save
			with open(secondary_filepath, 'a') as file:
				file.write('{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
					domain, pct_ss[0], pct_ss[1], pct_ss[0] + pct_ss[1], pct_left_helix
				))

	def _compute_tertiary_diversity(self, designs_dir, results_dir, verbose):
		"""
		Compute tertiary diversity.
		"""

		# create output filepath
		assert os.path.exists(results_dir), 'Missing output results directory'
		tertiary_filepath = os.path.join(results_dir, 'pair_tertiary.csv')
		assert not os.path.exists(tertiary_filepath), 'Output tertiary filepath existed'
		with open(tertiary_filepath, 'w') as file:
			columns = ['domain_1', 'domain_2', 'tm']
			file.write(','.join(columns) + '\n')
		clusters_filepath = os.path.join(results_dir, 'single_clusters.csv')
		assert not os.path.exists(clusters_filepath), 'Output clusters filepath existed'
		with open(clusters_filepath, 'w') as file:
			columns = ['domain', 'single_cluster_idx', 'complete_cluster_idx', 'average_cluster_idx']
			file.write(','.join(columns) + '\n')

		# create design filepath pairs
		design_filepath_pairs = []
		domains, domain_idx_map = [], {}
		filepaths = glob.glob(os.path.join(designs_dir, '*.pdb'))
		for idx1, filepath1 in enumerate(filepaths):
			domain = filepath1.split('/')[-1].split('.')[0]
			domains.append(domain)
			domain_idx_map[domain] = len(domain_idx_map)
			for idx2, filepath2 in enumerate(filepaths):
				if idx1 < idx2:
					design_filepath_pairs.append((filepath1, filepath2))

		# compute pairwise tm scores
		dists = np.zeros((len(domains), len(domains)))
		for design_filepath_1, design_filepath_2 in tqdm(
			design_filepath_pairs,
			desc='Computing tertiary diversity',
			disable=not verbose
		):

			# parse filepath
			domain_1 = design_filepath_1.split('/')[-1].split('.')[0]
			domain_2 = design_filepath_2.split('/')[-1].split('.')[0]

			# compare pdb files
			output_filepath = os.path.join(results_dir, 'internal_tm_output.txt')
			subprocess.call(f'{self.tm_align_exec} {design_filepath_1} {design_filepath_2} -fast > {output_filepath}', shell=True)
			
			# parse TMalign output 
			rows = []
			with open(output_filepath) as file:
				for line in file:
					if line.startswith('TM-score') and 'Chain_1' in line:
						tm = float(line.split('(')[0].split('=')[-1].strip())
						rows.append((domain_1, domain_2, tm))
					if line.startswith('TM-score') and 'Chain_2' in line:
						tm = float(line.split('(')[0].split('=')[-1].strip())
						rows.append((domain_2, domain_1, tm))

			# clean up
			os.remove(output_filepath)

			# save
			with open(tertiary_filepath, 'a') as file:
				for domain_1, domain_2, tm in rows:
					domain_idx_1 = domain_idx_map[domain_1]
					domain_idx_2 = domain_idx_map[domain_2]
					dists[domain_idx_1][domain_idx_2] = tm
					file.write('{},{},{:.3f}\n'.format(domain_1, domain_2, tm))

		# compute clusters
		columns = []
		linkages = ['single', 'complete', 'average']
		for linkage in linkages:

			# perform hierarchical clustering
			clusters = hcluster(dists, linkage)

			# map domain to cluster idx
			domain_cluster_idx_map = {}
			for cluster_idx, cluster in enumerate(clusters):
				for domain_idx in cluster:
					domain = domains[domain_idx]
					domain_cluster_idx_map[domain] = cluster_idx

			# create column
			columns.append([domain_cluster_idx_map[domain] for domain in domains])

		# save cluster information
		with open(clusters_filepath, 'a') as file:
			for i, domain in enumerate(domains):
				file.write('{},{},{},{}\n'.format(domain, columns[0][i], columns[1][i], columns[2][i]))

	def _process_results(self, results_dir, output_dir):
		"""
		Combine files in the results directory.
		Save output files in the output directory.
		"""

		# create output filepath
		assert os.path.exists(results_dir), 'Missing output results directory'
		info_filepath = os.path.join(output_dir, 'info.csv')
		pair_info_filepath = os.path.join(output_dir, 'pair_info.csv')
		assert not os.path.exists(info_filepath), 'Output info filepath existed'
		assert not os.path.exists(pair_info_filepath), 'Output pair info filepath existed'

		# process single level information
		for idx, filepath in enumerate(glob.glob(os.path.join(results_dir, 'single_*.csv'))):
			if idx == 0:
				df = pd.read_csv(filepath)
			else:
				df = df.merge(pd.read_csv(filepath), on='domain')

		# save single level information
		df.to_csv(info_filepath, index=False)

		# process pair level information
		for idx, filepath in enumerate(glob.glob(os.path.join(results_dir, 'pair_*.csv'))):
			if idx == 0:
				df = pd.read_csv(filepath)
			else:
				df = df.merge(pd.read_csv(filepath), on=['domain_1', 'domain_2'])

		# save pair level information
		df.to_csv(pair_info_filepath, index=False)

	def _evaluate_motif_scaffolding(self, input_dir, designs_dir, results_dir, info, verbose):
		"""
		Evaluate motif scaffolding (if applicable).
		"""

		# sanity check
		if info is None or 'motif_filepath' not in info:
			return
		motif_filepath = info['motif_filepath']
		motif_name = motif_filepath.split('/')[-1].split('.')[0]
		assert os.path.exists(motif_filepath), 'Missing input motif filepath'
		motif_masks_dir = os.path.join(input_dir, 'motif_masks')
		assert os.path.exists(motif_masks_dir), 'Missing motif masks directory'

		# create output filepath
		motif_scores_filepath = os.path.join(results_dir, 'single_motif_scores.csv')
		assert not os.path.exists(motif_scores_filepath), 'Output motif scores filepath existed'
		with open(motif_scores_filepath, 'w') as file:
			columns = ['domain', 'motif_name', 'rmsd']
			file.write(','.join(columns) + '\n')

		# process
		for design_filepath in tqdm(
			glob.glob(os.path.join(designs_dir, '*.pdb')),
			desc='Evaluating motif scaffolding', disable=not verbose
		):

			# parse
			domain = design_filepath.split('/')[-1].split('.')[0]

			# load mask
			mask = np.loadtxt(os.path.join(motif_masks_dir, f'{domain}.npy'))
			mask = torch.Tensor(mask).bool().unsqueeze(1).repeat(1, 4).view(-1,)

			# create pdb file for designed motif
			motif_design_filepath = os.path.join(results_dir, 'motif_design.pdb')
			design_coords = parse_pdb_file(design_filepath)['bb_coords'][mask]
			design_sequence = 'A' * (design_coords.shape[0] // 4)
			save_as_pdb(design_sequence, design_coords, motif_design_filepath, ca_only=False)

			# create pdb file for target motif
			motif_target_filepath = os.path.join(results_dir, 'motif_target.pdb')
			target_coords = parse_pdb_file(motif_filepath)['bb_coords']
			target_sequence = 'A' * (target_coords.shape[0] // 4)
			save_as_pdb(target_sequence, target_coords, motif_target_filepath, ca_only=False)

			# compute motif rmsd
			assert design_coords.shape[0] == target_coords.shape[0], 'Motif length mismatch'
			output_filepath = os.path.join(results_dir, 'motif_eval_output.txt')
			subprocess.call(f'{self.tm_score_exec} {motif_design_filepath} {motif_target_filepath} > {output_filepath}', shell=True)
			rmsd = parse_tm_file(output_filepath)['rmsd']

			# write
			with open(motif_scores_filepath, 'a') as file:
				file.write('{},{},{:.3f}\n'.format(domain, motif_name, rmsd))

			# clean up
			os.remove(motif_design_filepath)
			os.remove(motif_target_filepath)
			os.remove(output_filepath)