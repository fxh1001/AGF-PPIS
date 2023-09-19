import warnings
from torch.utils.data import DataLoader
from DataMannger import *
import argparse
from Bio.PDB.PDBParser import PDBParser
from torch.autograd import Variable
warnings.filterwarnings("ignore")
from Model import *

# Seed
SEED = 2023
Layers = 12
Input_Size = 88
Hidden_Size = 256
Fliter_Size = 512
Output_Size = 2
Dropout = 0.1
BATCH_SIZE = 1
NUM_EPOCHS = 80
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.empty_cache()

device = torch.device('cuda')



def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--querypath", dest="query_path", help="The path of query structure")
    parser.add_argument("--filename", dest="filename", help="The file name of the query structure which should be in PDB format.")
    parser.add_argument("--chainid", dest="chain_id", default = '', help="The query chain id(case sensitive).")
    parser.add_argument("--cpu", dest="fea_num_threads", default='1',help="The number of CPUs used for calculating PSSM and HMM profile.")
    return parser.parse_args()

def predict(model,dataloader,query_path,query_id,filename):
    with open('{}/{}.seq'.format(query_path, query_id), 'r') as f:
        sequence = list(f.readlines()[1].strip())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(f"../AGF-PPIS/model/best_model.pkl"))
    model.to(device)
    model.eval()
    pred = []
    for data in dataloader:
        with torch.no_grad():
            node_feature, mask, adj_matrix, adj = data
            node_feature = Variable(node_feature.cuda().float())
            mask = Variable(mask.cuda())
            adj_matrix = Variable(adj_matrix.cuda().float())
            adj = Variable(adj.cuda().float())
            output = model(node_feature, mask, adj_matrix, adj)
            output = output.squeeze()
            soft = nn.Softmax(dim=1)
            y_pred = soft(output)
            y_pred = y_pred.cpu().detach().numpy()
            pred += [pred[1] for pred in y_pred]


    binary_preds = [1 if score >= 0.43 else 0 for score in pred]

    with open( filename[0:4]+"Pred_results.txt", "w") as f:

        f.write("AA\tProb\tPred\n")
        for i in range(len(sequence)):
            f.write(sequence[i] + "\t" + str(pred[i]) + "\t" + str(binary_preds[i]) + "\n")
def main(query_path,filename,chain_id,fea_num_threads):
    query_id = 'chain'

    Feature(query_path,filename,chain_id,fea_num_threads)

    dataloader = DataLoader(ProDataset(query_path,query_id), batch_size=1, shuffle=False)

    model = FinalModel(Input_Size,Hidden_Size,Fliter_Size,Output_Size,Dropout,Layers)

    predict(model,dataloader,query_path, query_id,filename)

if __name__ == '__main__':
    print('Start...')

    args = parse_args()
    if args.query_path is None:
        print('ERROR: please --querypath!')
        raise ValueError
    if args.filename is None:
        print('ERROR: please --filename!')
        raise ValueError
    fea_num_threads = args.fea_num_threads
    query_path = args.query_path.rstrip('/')
    filename = args.filename
    chain_id = args.chain_id


    if not os.path.exists('{}/{}'.format(query_path,filename)):
        print('ERROR: Your query structure "{}/{}" is not found!'.format(query_path,filename))
        raise ValueError


    p1 = PDBParser(PERMISSIVE=1)
    try:
        structure = p1.get_structure('chain', '{}/{}'.format(query_path,filename))
        a=0
    except:
        print('ERROR: The query structure "{}/{}" is not in correct PDB format, please check the structure!'.format(query_path,filename))
        raise ValueError


    main(query_path,filename,chain_id,fea_num_threads)
    print('End...')



