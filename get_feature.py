import numpy as np
import torch
import torch.nn.functional as F
import joblib
import pandas as pd
import os
import subprocess

# Set the absolute paths of blast+, HHBlits and their databases in here.
PSIBLAST = '/fxh/pred_ppis/tool/ncbi-blast-2.10.1+/bin/psiblast'
PSIBLAST_DB = '/fxh/pred_ppis/database/nr_19_01_06/nr_v5'
HHblits = '/fxh/pred_ppis/tool/hhsuite-3.0.3/bin/hhblits'
HHblits_DB = '/fxh/pred_ppis/database/uniclust30_2018_08/uniclust30_2018_08'

DSSP = '../AGF-PPIS/dssp'

def SaveChainPDB(chain_id,query_path,filename,query_id):

    pdb_file = "{}/{}".format(query_path,filename)
    chain_file = '{}/{}.pdb'.format(query_path,query_id)

    with open(pdb_file,'r') as f:
        pdb_text = f.readlines()
    text = []

    if chain_id == ' ':
        chainid_list = set()
        for line in pdb_text:
            if line.startswith('ATOM'):
                chainid_list.add(line[21])
        chainid_list = list(chainid_list)
        if len(chainid_list) == 1:
            chain_id = chainid_list[0]
        else:
            print('ERROR: Your query structure has multiple chains, please input the chain ID!')
            raise ValueError



    for line in pdb_text:
        if line.startswith('ATOM') and line[21] == chain_id:
            text.append(line)
        if line.startswith('TER') and line[21] == chain_id:
            break
    text.append('\nTER\n')
    text.append('END\n')

    with open(chain_file, 'w') as f:
        f.writelines(text)
    return

def def_atom_features():
    A = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,3,0]}
    V = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,1,0], 'CG1':[0,3,0], 'CG2':[0,3,0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,1,1] }
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,1], 'CG':[0,2,1], 'CD':[0,2,1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,1,0], 'CD1':[0,3,0], 'CD2':[0,3,0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'CG1':[0,2,0], 'CG2':[0,3,0], 'CD1':[0,3,0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,2,0], 'CD':[0,2,0], 'NE':[0,1,0], 'CZ':[1,0,0], 'NH1':[0,2,0], 'NH2':[0,2,0] }
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[-1,0,0], 'OD1':[-1,0,0], 'OD2':[-1,0,0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[-1,0,0], 'OE1':[-1,0,0], 'OE2':[-1,0,0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'OG':[0,1,0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'OG1':[0,1,0], 'CG2':[0,3,0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'SG':[-1,1,0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,0,0], 'OD1':[0,0,0], 'ND2':[0,2,0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,0,0], 'OE1':[0,0,0], 'NE2':[0,2,0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'ND1':[-1,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'NE2':[-1,1,1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,2,0], 'CE':[0,2,0], 'NZ':[0,3,1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,0,1], 'OH':[-1,1,0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'SD':[0,0,0], 'CE':[0,3,0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,0,1], 'NE1':[0,1,1], 'CE2':[0,0,1], 'CE3':[0,1,1], 'CZ2':[0,1,1], 'CZ3':[0,1,1], 'CH2':[0,1,1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                   'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0]/2+0.5,i_fea[1]/3,i_fea[2]]

    return atom_features

def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
               'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R'}
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path,'r')
    pdb_res = pd.DataFrame(columns=['ID','atom','res','res_id','xyz','B_factor'])
    res_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {'H':1,'C':12,'O':16,'N':14,'S':32,'FE':56,'P':31,'BR':80,'F':19,'CO':59,'V':51,
                            'I':127,'CL':35.5,'CA':40,'B':10.8,'ZN':65.5,'MG':24.3,'NA':23,'HG':200.6,'MN':55,
                            'K':39.1,'AP':31,'AC':227,'AL':27,'W':183.9,'SE':79,'NI':58.7}


    while True:
        line = pdb_file.readline()
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count+=1
            res_pdb_id = int(line[22:26])
            if res_pdb_id != before_res_pdb_id:
                res_count +=1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N','CA','C','O','H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5,0.5,0.5]

            try:
                bfactor = float(line[60:66])
            except ValueError:
                bfactor = 0.5

            tmps = pd.Series(
                {'ID': atom_count, 'atom':line[12:16].strip(),'atom_type':atom_type, 'res': res, 'res_id': int(line[22:26]),
                 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                 'B_factor': bfactor,'mass':Relative_atomic_mass[atom_type],'is_sidechain':is_sidechain,
                 'charge':atom_fea[0],'num_H':atom_fea[1],'ring':atom_fea[2]})
            if len(res_id_list) == 0:
                res_id_list.append(int(line[22:26]))
            elif res_id_list[-1] != int(line[22:26]):
                res_id_list.append(int(line[22:26]))
            pdb_res = pdb_res.append(tmps,ignore_index=True)
        if line.startswith('TER'):
            break

    return pdb_res,res_id_list

def PDBFeature(query_id,PDB_chain_dir,results_dir):

    print('PDB_chain -> PDB_DF')
    pdb_path = PDB_chain_dir+'/{}.pdb'.format(query_id)
    pdb_DF, res_id_list = get_pdb_DF(pdb_path)

    with open(results_dir+'/{}.df'.format(query_id),'wb') as f:
        joblib.dump({'pdb_DF':pdb_DF,'res_id_list':res_id_list},f)

    print('Extract PDB_feature')
    print(query_id)

    res_sidechain_centroid = []
    res_types = []
    for res_id in res_id_list:
        res_type = pdb_DF[pdb_DF['res_id'] == res_id]['res'].values[0]
        res_types.append(res_type)

        res_atom_df = pdb_DF[pdb_DF['res_id'] == res_id]
        xyz = np.array(res_atom_df['xyz'].tolist())
        masses = np.array(res_atom_df['mass'].tolist()).reshape(-1,1)
        centroid = np.sum(masses*xyz,axis=0)/np.sum(masses)
        res_sidechain_atom_df = pdb_DF[(pdb_DF['res_id'] == res_id) & (pdb_DF['is_sidechain'] == 1)]
        if len(res_sidechain_atom_df) == 0:
            res_sidechain_centroid.append(centroid)
        else:
            xyz = np.array(res_sidechain_atom_df['xyz'].tolist())
            masses = np.array(res_sidechain_atom_df['mass'].tolist()).reshape(-1, 1)
            sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
            res_sidechain_centroid.append(sidechain_centroid)

    res_sidechain_centroid = np.array(res_sidechain_centroid)
    with open(results_dir + '/'+query_id+'_psepos_SC.pkl', 'wb') as f:
        joblib.dump(res_sidechain_centroid, f)
    sequence = ''.join(res_types)
    with open(results_dir+'/'+query_id+'.seq','w') as f:
        f.write('>{}\n'.format(query_id))
        f.write(sequence)

    return

def norm_pssm(query_path,query_id):

    with open('{}/{}.pssm'.format(query_path,query_id),'r') as f:
        text = f.readlines()

    pssm = []
    for line in text[3:]:
        if line=='\n':
            break
        else:
            res_pssm = np.array(list(map(int,line.split()[2:22]))).reshape(1,-1)
            pssm.append(res_pssm)
    pssm = np.concatenate(pssm,axis=0)
    pssm = 1/(1+np.exp(-pssm))

    return pssm

def norm_hhm(query_path,query_id):

    with open('{}/{}.hhm'.format(query_path,query_id),'r') as f:
        text = f.readlines()
    hhm_begin_line = 0
    hhm_end_line = 0
    for i in range(len(text)):
        if '#' in text[i]:
            hhm_begin_line = i + 5
        elif '//' in text[i]:
            hhm_end_line = i
    hhm = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])

    axis_x = 0
    for i in range(hhm_begin_line, hhm_end_line, 3):
        line1 = text[i].split()[2:-1]
        line2 = text[i + 1].split()
        axis_y = 0
        for j in line1:
            if j == '*':
                hhm[axis_x][axis_y] = 9999 / 10000.0
            else:
                hhm[axis_x][axis_y] = float(j) / 10000.0
            axis_y += 1
        for j in line2:
            if j == '*':
                hhm[axis_x][axis_y] = 9999 / 10000.0
            else:
                hhm[axis_x][axis_y] = float(j) / 10000.0
            axis_y += 1
        axis_x += 1
    hhm = (hhm - np.min(hhm)) / (np.max(hhm) - np.min(hhm))

    return hhm

def norm_DSSP(query_path,query_id):
    maxASA = {'G': 188, 'A': 198, 'V': 220, 'I': 233, 'L': 304, 'F': 272, 'P': 203, 'M': 262, 'W': 317, 'C': 201,
              'S': 234, 'T': 215, 'N': 254, 'Q': 259, 'Y': 304, 'H': 258, 'D': 236, 'E': 262, 'K': 317, 'R': 319}
    map_ss_8 = {' ': [1, 0, 0, 0, 0, 0, 0, 0], 'S': [0, 1, 0, 0, 0, 0, 0, 0], 'T': [0, 0, 1, 0, 0, 0, 0, 0],
                'H': [0, 0, 0, 1, 0, 0, 0, 0],
                'G': [0, 0, 0, 0, 1, 0, 0, 0], 'I': [0, 0, 0, 0, 0, 1, 0, 0], 'E': [0, 0, 0, 0, 0, 0, 1, 0],
                'B': [0, 0, 0, 0, 0, 0, 0, 1]}

    with open('{}/{}.dssp'.format(query_path,query_id),'r') as f:
        text = f.readlines()

    start_line = 0
    for i in range(0, len(text)):
        if text[i].startswith('  #  RESIDUE AA STRUCTURE'):
            start_line = i + 1
            break

    dssp = {}
    for i in range(start_line, len(text)):
        line = text[i]
        if line[13] not in maxASA.keys() or line[9] == ' ':
            continue
        res_id = float(line[5:10])
        res_dssp = np.zeros([14])
        res_dssp[:8] = map_ss_8[line[16]]  # SS
        res_dssp[8] = min(float(line[35:38]) / maxASA[line[13]], 1)
        res_dssp[9] = (float(line[85:91]) + 1) / 2
        res_dssp[10] = min(1, float(line[91:97]) / 180)
        res_dssp[11] = min(1, (float(line[97:103]) + 180) / 360)
        res_dssp[12] = min(1, (float(line[103:109]) + 180) / 360)
        res_dssp[13] = min(1, (float(line[109:115]) + 180) / 360)
        dssp[res_id] = res_dssp.reshape((1, -1))

    return dssp

def PDBResidueFeature(query_path,query_id,pssm,hhm,dssp):

    atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85,'H':1.2,'D':1.2,'SE':1.9,'P':1.8,'FE':2.23,'BR':1.95,
                        'F':1.47,'CO':2.23,'V':2.29,'I':1.98,'CL':1.75,'CA':2.81,'B':2.13,'ZN':2.29,'MG':1.73,'NA':2.27,
                        'HG':1.7,'MN':2.24,'K':2.75,'AC':3.08,'AL':2.51,'W':2.39,'NI':2.22}
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

    with open('{}/{}.df'.format(query_path,query_id), 'rb') as f:
        tmp = joblib.load(f)
    pdb_DF, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
    pdb_DF = pdb_DF[pdb_DF['atom_type']!='H']

    # atom features
    mass = np.array(pdb_DF['mass'].tolist()).reshape(-1, 1)
    mass = mass / 32
    B_factor = np.array(pdb_DF['B_factor'].tolist()).reshape(-1, 1)
    if (max(B_factor) - min(B_factor)) == 0:
        B_factor = np.zeros(B_factor.shape) + 0.5
    else:
        B_factor = (B_factor - min(B_factor)) / (max(B_factor) - min(B_factor))
    is_sidechain = np.array(pdb_DF['is_sidechain'].tolist()).reshape(-1, 1)
    charge = np.array(pdb_DF['charge'].tolist()).reshape(-1, 1)
    num_H = np.array(pdb_DF['num_H'].tolist()).reshape(-1, 1)
    ring = np.array(pdb_DF['ring'].tolist()).reshape(-1, 1)
    atom_type = pdb_DF['atom_type'].tolist()
    atom_vander = np.zeros((len(atom_type), 1))
    for i, type in enumerate(atom_type):
        try:
            atom_vander[i] = atom_vander_dict[type]
        except:
            atom_vander[i] = atom_vander_dict['C']

    atom_feas = [mass, B_factor, is_sidechain, charge, num_H, ring, atom_vander]
    atom_feas = np.concatenate(atom_feas,axis=1)

    res_atom_feas = []
    atom_begin = 0
    for i, res_id in enumerate(res_id_list):
        res_atom_df = pdb_DF[pdb_DF['res_id'] == res_id]
        atom_num = len(res_atom_df)
        res_atom_feas_i = atom_feas[atom_begin:atom_begin + atom_num]
        res_atom_feas_i = np.average(res_atom_feas_i, axis=0).reshape(1, -1)
        res_atom_feas.append(res_atom_feas_i)
        atom_begin += atom_num
    res_atom_feas = np.concatenate(res_atom_feas, axis=0)

    dssp_ = []
    for res_id_i in res_id_list:
        if res_id_i in dssp.keys():
            dssp_.append(dssp[res_id_i])
        else:
            dssp_.append(np.zeros(list(dssp.values())[0].shape))
    dssp_ = np.concatenate(dssp_, axis=0)

    residue_feas = [res_atom_feas,pssm,hhm,dssp_]
    residue_feas = np.concatenate(residue_feas, axis=1)
    with open('{}/{}.resfea'.format(query_path,query_id),'wb') as f:
        joblib.dump(residue_feas,f)
    return

def cal_dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

def get_distance_map(query_path,query_id):
    with open('{}/{}_psepos_SC.pkl'.format(query_path, query_id), 'rb') as f:
        pos = joblib.load(f)
    dis_matrix = []
    for radius in pos:
        dis_radius = []
        for i in range(len(pos)):
            dis_radius.append(cal_dist(radius, pos[i]))
        dis_matrix.append(dis_radius)
    dis_matrix = np.row_stack(dis_matrix)
    np.save(query_path+'/'+query_id+'_dia_matrix',dis_matrix)


def Feature(query_path,filename,chain_id,fea_num_threads):

    if chain_id == '':
        chain_id = ' '
    query_id = 'chain'
    # feature extract
    print('1.extracting query chain...')
    SaveChainPDB(chain_id, query_path,filename, query_id)
    with open('{}/{}.pdb'.format(query_path, query_id),'r') as f:
        text = f.readlines()

    residue_num = 0
    for line in text:
        if line.startswith('ATOM'):
            residue_type = line[17:20]
            if residue_type not in ["GLY","ALA","VAL","ILE","LEU","PHE","PRO","MET","TRP","CYS",
                                    "SER","THR","ASN","GLN","TYR","HIS","ASP","GLU","LYS","ARG"]:
                print("ERROR: There are mutant residues in your structure!")
                raise ValueError

            residue_num += 1
    if residue_num == 0:
        print('ERROR: Your query chain id "{}" is not in the uploaded structure, please check the chain ID!'.format(chain_id))
        raise ValueError

    DSSP_code = subprocess.call([DSSP, '-i', '{}/{}.pdb'.format(query_path, query_id),
                                 '-o', '{}/{}.dssp'.format(query_path, query_id)])
    if not os.path.exists('{}/{}.dssp'.format(query_path, query_id)):
        print("ERROR: The upload protein structure is not in correct PDB format, please check the structure!")
        raise ValueError

    print('2.extracting features...')
    PDBFeature(query_id, query_path, query_path)
    PSIBLAST_code = subprocess.call([PSIBLAST, '-db', PSIBLAST_DB, '-evalue', '0.001', '-num_iterations', '3',
                                     '-num_threads', str(fea_num_threads),
                                     '-query', '{}/{}.seq'.format(query_path, query_id),
                                     '-out_ascii_pssm', '{}/{}.pssm'.format(query_path, query_id)])
    HHblits_code = subprocess.call([HHblits, '-d', HHblits_DB, '-cpu', str(fea_num_threads),
                                    '-i', '{}/{}.seq'.format(query_path, query_id),
                                    '-ohhm', '{}/{}.hhm'.format(query_path, query_id)])
    query_pssm = norm_pssm(query_path, query_id)
    query_hhm = norm_hhm(query_path, query_id)
    query_dssp = norm_DSSP(query_path, query_id)

    PDBResidueFeature(query_path, query_id,query_pssm, query_hhm, query_dssp)

    get_distance_map(query_path,query_id)
    return

def cal_adj_matrix(query_path,query_id,radius):
    dist_matrix = np.load(query_path+'/'+query_id+'_dia_matrix' + '.npy')
    mask = ((dist_matrix >= 0)*(dist_matrix <= radius))
    adjency_matrix = mask.astype(np.int32)
    return adjency_matrix

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

def get_onehot(seq):
    aLetters_ = np.array(
        ['H', 'D', 'R', 'F', 'A', 'C', 'G', 'Q', 'E', 'K', 'L', 'M', 'N', 'S', 'Y', 'T', 'I', 'W', 'P', 'V', 'U', 'O',
         'B', 'J', 'Z', 'X'])
    index_ = []
    for i in seq:
        for j in range(len(aLetters_)):
            if aLetters_[j] == i:
                index_.append(j)
    index_ = torch.tensor(index_)
    one_hot = torch.squeeze(F.one_hot(index_, 26).float())
    return one_hot










































