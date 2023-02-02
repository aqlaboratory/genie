import os
import glob
import argparse
import numpy as np
from tqdm import tqdm


DEFAULT_SCOPEDOM_FILEPATH = 'data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa'
DEFAULT_SCOPE_PDB_DIR = 'data/pdbstyle-2.08'
DEFAULT_OUTPUT_DIR = 'data/scope'

SCOPE_CLASSES = ['a', 'b', 'c', 'd']
IGNORE_DOMAIN_IDS = [
    'd6m9za_',
    'd6a1ia1' # missing coordinate information
]

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

def load_domains(scopedom_filepath, scope_pdb_dir):

    def remove(domains, idxs):
        for idx in sorted(idxs, reverse=True):
            del domains[idx]

    # load domains from scope file
    domains = load_scope_file(scopedom_filepath)
    print('Number of Domains: {}'.format(len(domains)))

    # remove ignored domains
    remove_idxs = [idx for idx, domain in enumerate(domains) if domain['domain_id'] in IGNORE_DOMAIN_IDS]
    remove(domains, remove_idxs)
    print('After ignoring problematic domains: {}'.format(len(domains)))

    # update domains from pdb files
    remove_idxs = []
    for idx, domain in enumerate(tqdm(domains)):
        scope_pdb_filepath = os.path.join(scope_pdb_dir, domain['domain_id'][2:4], '{}.ent'.format(domain['domain_id']))
        if os.path.exists(scope_pdb_filepath):
            update_domain_with_pdb_file(domain, scope_pdb_filepath)
        else:
            remove_idxs.append(idx)
    remove(domains, remove_idxs)
    print('After removing domains with missing pdb information: {}'.format(len(domains)))

    # remove domains with unknown residues
    remove_idxs = []
    for idx, domain in enumerate(domains):
        if any([model['missing'] for model in domain['models']]):
            remove_idxs.append(idx)
    remove(domains, remove_idxs)
    print('After removing domains with unknown residues: {}'.format(len(domains)))

    # remove domains with mismatched sequences
    remove_idxs = []
    for idx, domain in enumerate(domains):
        if any([model['mismatched'] for model in domain['models']]):
            remove_idxs.append(idx)
    remove(domains, remove_idxs)
    print('After removing domains with mismatched sequences: {}'.format(len(domains)))
    
    # remove domains with multiple segments
    remove_idxs = []
    for idx, domain in enumerate(domains):
        if any([model['multisegmented'] for model in domain['models']]):
            remove_idxs.append(idx)
    remove(domains, remove_idxs)
    print('After removing domains with multiple segments: {}'.format(len(domains)))
    
    # remove domains with missing backbone atoms
    remove_idxs = []
    for idx, domain in enumerate(domains):
        if any([model['backbone_coords'] is None for model in domain['models']]):
            remove_idxs.append(idx)
    remove(domains, remove_idxs)
    print('After removing domains with missing atoms: {}'.format(len(domains)))
    
    return domains

def load_scope_file(filepath):
    domains = []
    
    with open(filepath, 'r') as file:
        for line in file:
            if line[0] == '>':
                domain_id = 'd' + line[2:8]
                
                segments = []
                region_str = line.split('(', 1)[1].split(')', 1)[0].strip()
                for segment in parse_region(region_str):
                    segments.append({
                        'chain_id': segment[0],
                        'start_res_seq': segment[1],
                        'start_ins_code': segment[2],
                        'end_res_seq': segment[3],
                        'end_ins_code': segment[4]
                    })

                domains.append({
                    'domain_id': domain_id,
                    'segments': segments,
                    'full_sequence': ''
                })
            
            else:
                domains[-1]['full_sequence'] += line.strip()
    
    for domain in domains:
        seqs = domain['full_sequence'].split('X')        
        assert len(seqs) == len(domain['segments'])
        for segment_idx, segment in enumerate(domain['segments']):
            segment['sequence'] = seqs[segment_idx]
    
    return domains


def update_domain_with_pdb_file(domain, filepath):
    
    remarks = {}
    atoms, models = [], []
    with open(filepath, 'r') as file:
        for line in file:
            row_type = line[:6].strip()

            if row_type == 'HEADER':
                continue
            
            if row_type == 'REMARK':

                remark = line[18:].strip()
                if len(remark) > 0:
                    key, val = remark.split(':', 1)
                    remarks[key] = val.strip()
                    if key == 'SCOPe-sccs':
                        sccs_elts = remarks[key].split('.')
                        remarks['SCOPe-class'] = sccs_elts[0]
                        remarks['SCOPe-fold'] = int(sccs_elts[1])
                        remarks['SCOPe-superfamily'] = int(sccs_elts[2])
                        remarks['SCOPe-family'] = int(sccs_elts[3])
                    if key == 'Region':
                        remarks['Segments'] = parse_region(val.strip())

            elif row_type == 'ATOM' or row_type == 'HETATM':

                atom_name, res_name = line[12:16].strip(), line[17:20].strip()
                alt_loc = None if line[16] == ' ' else line[16]
                chain_id, res_seq_num = line[21], int(line[22:26].strip())
                ins_code = None if line[26] == ' ' else line[26]
                x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())                
                
                atoms.append({
                    'atom_name': atom_name,
                    'res_name': res_name,
                    'alt_loc': alt_loc,
                    'chain_id': chain_id,
                    'res_seq_num': res_seq_num,
                    'ins_code': ins_code,
                    'coord': [x, y, z]
                })
            
            elif row_type == 'MODEL':
                assert len(atoms) == 0
            
            elif row_type == 'ENDMDL':
                assert len(atoms) == 0
            
            elif row_type == 'TER':
                
                # compile atom information into residues
                residues = []
                for atom in atoms:
                    if len(residues) == 0 or \
                        atom['chain_id'] != residues[-1]['chain_id'] or \
                        atom['res_seq_num'] != residues[-1]['res_seq_num'] or \
                        atom['ins_code'] != residues[-1]['ins_code']:
                        residues.append({
                            'res_name': atom['res_name'],
                            'chain_id': atom['chain_id'],
                            'res_seq_num': atom['res_seq_num'],
                            'ins_code': atom['ins_code'],
                            'atoms': {}
                        })                    
                    assert atom['res_name'] == residues[-1]['res_name']
                    atom_name = atom['atom_name']
                    if atom['alt_loc'] is None:
                        assert atom_name not in residues[-1]['atoms']
                        residues[-1]['atoms'][atom_name] = {
                            'coord': atom['coord'],
                            'alt_locs': None,
                            'alt_loc_used': None
                        }
                    elif atom_name not in residues[-1]['atoms']:
                        residues[-1]['atoms'][atom_name] = {
                            'coord': atom['coord'],
                            'alt_locs': {
                                atom['alt_loc']: atom['coord']
                            },
                            'alt_loc_used': atom['alt_loc']
                        }
                    else:
                        assert atom_name in residues[-1]['atoms']
                        assert atom['alt_loc'] > residues[-1]['atoms'][atom_name]['alt_loc_used']
                        residues[-1]['atoms'][atom_name]['alt_locs'][atom['alt_loc']] = atom['coord']
                
                # separate residues into segments
                # step 1: compute starting indices from region information (in remarks)
                start_idxs, remark_segment_idx = [], 0
                for residue_idx, residue in enumerate(residues):
                    if remark_segment_idx >= len(remarks['Segments']):
                        break
                    chain_id, start_res_seq, start_ins_code, _, _ = remarks['Segments'][remark_segment_idx]
                    if chain_id == residue['chain_id']:
                        if start_res_seq is None and start_ins_code is None:
                            start_idxs.append(residue_idx)
                            remark_segment_idx += 1
                        else:
                            assert start_res_seq is not None
                            if start_res_seq == residue['res_seq_num'] and start_ins_code == residue['ins_code']:
                                start_idxs.append(residue_idx)
                                remark_segment_idx += 1
                assert len(start_idxs) == len(remarks['Segments'])
                end_idxs = start_idxs[1:] + [len(residues)]
                
                # step 2: create segments and verify positions
                segments = []
                assert len(remarks['Segments']) == len(domain['segments'])
                for remark_segment_idx in range(len(remarks['Segments'])):
                    chain_id, start_res_seq, start_ins_code, end_res_seq, end_ins_code = remarks['Segments'][remark_segment_idx]
                    
                    # check against ASTRAL file
                    assert domain['segments'][remark_segment_idx]['chain_id'] == chain_id
                    assert domain['segments'][remark_segment_idx]['start_res_seq'] == start_res_seq
                    assert domain['segments'][remark_segment_idx]['start_ins_code'] == start_ins_code
                    assert domain['segments'][remark_segment_idx]['end_res_seq'] == end_res_seq
                    assert domain['segments'][remark_segment_idx]['end_ins_code'] == end_ins_code
                    
                    # check against residue information
                    segment_residues = residues[start_idxs[remark_segment_idx]:end_idxs[remark_segment_idx]]
                    assert segment_residues[0]['chain_id'] == chain_id
                    if start_res_seq is not None:
                        assert segment_residues[0]['res_seq_num'] == start_res_seq
                        assert segment_residues[0]['ins_code'] == start_ins_code
                    else:
                        assert start_ins_code is None
                    if end_res_seq is not None:
                        assert segment_residues[-1]['res_seq_num'] == end_res_seq
                        assert segment_residues[-1]['ins_code'] == end_ins_code
                    else:
                        assert end_ins_code is None
                    segments.append({
                        'chain_id': chain_id,
                        'start_res_seq': start_res_seq,
                        'start_ins_code': start_ins_code,
                        'end_res_seq': end_res_seq,
                        'end_ins_code': end_ins_code,
                        'residues': segment_residues,
                        'scope_sequence': domain['segments'][remark_segment_idx]['sequence']
                    })
                assert len(segments) == len(remarks['Segments'])
                
                # step 3: verify sequence                
                for segment in segments:
                    sequence = ''
                    for residue in segment['residues']:
                        if residue['res_name'] not in RES_MAP_3_TO_1:
                            sequence = None
                            break
                        else:
                            sequence += RES_MAP_3_TO_1[residue['res_name']]
                    assert sequence is None or len(sequence) > 0
                    segment['sequence'] = sequence
                
                # extract backbone coordinates
                backbone_coords = None
                if len(segments) == 1:
                    try:
                        backbone_coords = []
                        for residue in segments[0]['residues']:
                            for atom_name in ['N', 'CA', 'C']:
                                backbone_coords.append(residue['atoms'][atom_name]['coord'])
                    except:
                        backbone_coords = None
                
                # store model information
                model_id = len(models)
                models.append({
                    'model_id': model_id,
                    'segments': segments,
                    'missing': any([segment['sequence'] is None for segment in segments]),
                    'mismatched': any([segment['sequence'] != segment['scope_sequence'] for segment in segments]),
                    'multisegmented': len(segments) > 1,
                    'backbone_coords': backbone_coords
                })

                atoms = []
            
            elif row_type == 'END':
                
                assert 'SCOPe-sccs' in remarks
                assert 'Region' in remarks
                assert len(atoms) == 0
                
            else:
                print(f'Unhabled Row Type: {row_type}')
                exit(0)
    
    # update domain
    del domain['segments']
    del domain['full_sequence']
    domain['remarks'] = remarks
    domain['models'] = models


def parse_region(region_str):
    segments = []

    for segment in region_str.split(','):
        segment_elts = segment.split(':')
        chain_id = segment_elts[0]
        start_res_seq, start_ins_code = None, None
        end_res_seq, end_ins_code = None, None
        if len(segment_elts[1]) > 0:
            segment_start, segment_end = segment_elts[1].rsplit('-', 1)
            segment_start, segment_end = segment_start.strip(), segment_end.strip()
            if segment_start[-1].isalpha():
                assert segment_start[-1].isupper()
                start_res_seq = int(segment_start[:-1])
                start_ins_code = segment_start[-1]
            else:
                start_res_seq = int(segment_start)
            if segment_end[-1].isalpha():
                assert segment_end[-1].isupper()
                assert start_res_seq is not None
                end_res_seq = int(segment_end[:-1])
                end_ins_code = segment_end[-1]
            else:
                assert start_res_seq is not None
                end_res_seq = int(segment_end)
        segments.append((chain_id, start_res_seq, start_ins_code, end_res_seq, end_ins_code))
    
    return segments


def generate_coords(domains, output_dir, model_id=0):

    # set up directories
    coords_dir = os.path.join(output_dir, 'coords')
    os.mkdir(output_dir)
    os.mkdir(coords_dir)

    # set up file to save domain class information
    class_filepath = os.path.join(output_dir, 'classes.txt')
    open(class_filepath, 'w').close()

    # extract coordinates
    for domain in tqdm(domains):
        scope_class = domain['remarks']['SCOPe-class']
        if scope_class not in SCOPE_CLASSES:
            continue

        # add domain class
        with open(class_filepath, 'a+') as file:
            file.write(','.join([
                domain['domain_id'],
                '.'.join([
                    str(domain['remarks']['SCOPe-class']),
                    str(domain['remarks']['SCOPe-fold']),
                    str(domain['remarks']['SCOPe-superfamily']),
                    str(domain['remarks']['SCOPe-family'])
                ])
            ]))
            file.write('\n')

        # extract backbone coordinates
        model = domain['models'][model_id]
        assert len(model['segments']) == 1
        residues = model['segments'][0]['residues']
        coords = []
        for residue in residues:
            for atom_name in ['N', 'CA', 'C']:
                coords.append(residue['atoms'][atom_name]['coord'])

        # save coordinates
        coords_filepath = os.path.join(coords_dir, '{}.npy'.format(domain['domain_id']))
        np.savetxt(coords_filepath, coords, delimiter=',')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--scopedom_filepath', type=str,
        help='Filepath for SCOPe domain file',
        default=DEFAULT_SCOPEDOM_FILEPATH
    )
    parser.add_argument(
        '-d', '--scope_pdb_dir', type=str,
        help='Directory for SCOPe PDB files',
        default=DEFAULT_SCOPE_PDB_DIR
    )
    parser.add_argument(
        '-o', '--output_dir', type=str,
        help='Output directory',
        default=DEFAULT_OUTPUT_DIR
    )
    args = parser.parse_args()

    scopedom_filepath = args.scopedom_filepath
    scope_pdb_dir = args.scope_pdb_dir
    output_dir = args.output_dir

    # io check
    if not os.path.exists(scopedom_filepath):
        print('Missing SCOPe Domain File: {}'.format(scopedom_filepath))
        exit(0)
    elif not os.path.exists(scope_pdb_dir):
        print('Missing SCOPe PDB Directory: {}'.format(scope_pdb_dir))
        exit(0)
    elif os.path.exists(output_dir):
        print('Output Directory Existed: {}'.format(output_dir))
        exit(0)

    generate_coords(
        load_domains(scopedom_filepath, scope_pdb_dir),
        output_dir
    )