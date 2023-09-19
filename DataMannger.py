from torch.utils.data import Dataset
from get_feature import *
class ProDataset(Dataset):
    def __init__(self,query_path,query_id,MAP_CUT = 14,DIST_NORM=15):
        self.query_path = query_path
        self.query_id  = query_id

        with open('{}/{}.seq'.format(query_path, query_id), 'r') as f:
            self.sequence = list(f.readlines()[1].strip())
        with open('{}/{}_psepos_SC.pkl'.format(query_path, query_id), 'rb') as f:
            self.residue_psepos = joblib.load(f)
        self.map_cut = MAP_CUT
        self.dist = DIST_NORM

    def __getitem__(self, index):

        pos = self.residue_psepos
        reference_res = pos[0]
        pos = pos - reference_res
        pos = torch.from_numpy(pos)

        adj_matrix = cal_adj_matrix(self.query_path,self.query_id,self.map_cut)
        adj = normalize(adj_matrix)

        one_hot_feature = get_onehot(self.sequence)
        with open('{}/{}.resfea'.format(self.query_path,self.query_id), 'rb') as f:
            res_feature = joblib.load(f)
        node_features = np.concatenate([res_feature,one_hot_feature], axis=1)
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist],dim=-1)
        adj_matrix = torch.from_numpy(adj_matrix).type(torch.FloatTensor)
        mask_adj = (adj_matrix == 0)
        adj = torch.from_numpy(adj).type(torch.FloatTensor)

        return node_features,mask_adj,adj_matrix,adj

    def __len__(self):
        return 1





