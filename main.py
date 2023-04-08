import torch
import networkx as nx
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from converter import assignEdgesToFaces, getFacesTypes
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, InMemoryDataset

########## DETERMINE A SPECIFIC PATH FOR TEXT REPRESENTATION OF .STP FILE ###########
stp_file_path = '00350186_fb1b60ebf52b00f7b8cbbefd_step_023_step.txt'

with open(stp_file_path, 'r') as f2:
    data = f2.read()

########## FACES & EDGES EXTRACTION PROCESS ##########
faces = getFacesTypes(data)
edges = assignEdgesToFaces(data)

########## FACES TYPE CHECKER ##########
def checkCylindrical(input):
  if input == 'Cylindrical':
    return 1
  else:
    return 0
  
def checkConical(input):
  if input == 'Conical':
    return 1
  else:
    return 0
  
def checkSwept(input):
  if input == 'Swept':
    return 1
  else:
    return 0

def checkParametric(input):
  if input == 'Parametric':
    return 1
  else:
    return 0

def checkPlanar(input):
  if input == 'Planar':
    return 1
  else:
    return 0
  
def checkSurfaceOfRevolution(input):
  if input == 'SurfaceOfRevolution':
    return 1
  else:
    return 0
  
def checkSpherical(input):
  if input == 'Spherical':
    return 1
  else:
    return 0

def checkOffset(input):
  if input == 'Offset':
    return 1
  else:
    return 0

########## GRAPH BUILDER ##########
def buildGraph(faces, edges):
  X = nx.Graph()
  graph_nodes_X = []
  for i in faces:
    row = (i[0], {"isCylindrical" : checkCylindrical(i[1]), "isConical" : checkConical(i[1]), "isSwept" : checkSwept(i[1]),
                  "isParametric" : checkParametric(i[1]), "isPlanar" : checkPlanar(i[1]), "isSurfaceOfRevolution" : checkSurfaceOfRevolution(i[1]),
                  "isSpherical" : checkSpherical(i[1]), "isOffset" : checkOffset(i[1]), "is_feature" : 0})
    graph_nodes_X.append(row)

  graph_edges_X = []
  for i in edges:
    row = (i[1], i[2], {"shape" : 'i[1]', "type" : 'i[2]'})
    graph_edges_X.append(row)
  
  X.add_nodes_from(graph_nodes_X)
  X.add_edges_from(graph_edges_X)
  return X

########### NODE EMBEDDING ###########
node_feats = []
my_graph = buildGraph(faces, edges)
adj = nx.to_scipy_sparse_array(my_graph).tocoo()
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)
degrees = np.array(list(dict(my_graph.degree()).values()))
counter = 0
scale = StandardScaler()
degrees = scale.fit_transform(degrees.reshape(-1,1))

for g in my_graph:
  deg = degrees[counter][0]
  isCylindrical = my_graph.nodes[g]["isCylindrical"]
  isConical = my_graph.nodes[g]["isConical"]
  isSwept = my_graph.nodes[g]["isSwept"]
  isParametric = my_graph.nodes[g]["isParametric"]
  isPlanar = my_graph.nodes[g]["isPlanar"]
  isSurfaceOfRevolution = my_graph.nodes[g]["isSurfaceOfRevolution"]
  isSpherical = my_graph.nodes[g]["isSpherical"]
  isOffset = my_graph.nodes[g]["isOffset"]
  list_feat = [deg, isCylindrical, isConical, isSwept, isParametric, isPlanar, isSurfaceOfRevolution, isSpherical, isOffset]
  node_feats.append(list_feat)
  counter += 1
    
embeddings = np.array(node_feats)
labels = np.asarray([my_graph.nodes[i]['is_feature'] == 1 for i in my_graph.nodes]).astype(np.int64)

########### DATASET BUILDER ###########
class ModelVerificationData(InMemoryDataset):
  def __init__(self, transform=None):
      super(ModelVerificationData, self).__init__('.', transform, None, None)
      data = Data(edge_index=edge_index)
      data.num_nodes = my_graph.number_of_nodes()
      data.x = torch.from_numpy(embeddings).type(torch.float32)
      y = torch.from_numpy(labels).type(torch.long)
      data.y = y.clone().detach()
      data.num_classes = 2
      X_test = pd.Series(labels)
      n_nodes = my_graph.number_of_nodes()
      test_mask = torch.zeros(n_nodes, dtype=torch.bool)
      test_mask[X_test.index] = True
      data['test_mask'] = test_mask
      self.data, self.slices = self.collate([data])

  def _download(self):
    return

  def _process(self):
    return

  def __repr__(self):
    return '{}()'.format(self.__class__.__name__)
    
dataset = ModelVerificationData()
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

########### GNN MODEL ###########
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 512)
        self.conv2 = GCNConv(512, 128)
        self.conv3 = GCNConv(128, int(data.num_classes))

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

trained_model = Net()
trained_model = torch.load('modelGNN.pth', map_location ='cpu')

########### PREDICTION ###########
@torch.no_grad()
def showMeWhatYouGot():
  trained_model.eval()
  logits = trained_model()
  mask = data['test_mask']
  pred = logits[mask].max(1)[1]
  facesNo = []
  counter = 0
  for i in pred.numpy():
    if i == 1:
      facesNo.append('FACE' + str(counter))
    counter += 1

  return facesNo

print('CLOSED POCKETs DETECTED IN:', showMeWhatYouGot())
