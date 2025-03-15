# %%
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
# import faiss 
import numpy as np
import matplotlib.pyplot as plt



# %%
# Load CIFAR10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
 #                                         shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False)
#print("hi")
#classes = ('plane', 'car', 'bird', 'cat',
 #          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
from resnet_cifar import resnet18

class SimCLRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the CIFAR ResNet from your code snippet
        backbone_dict = resnet18(in_channel=3)
        self.backbone = backbone_dict['backbone']  # a ResNet instance
        backbone_dim  = backbone_dict['dim']       # should be 512
        print("backbone done")

        self.contrastive_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)

        )

    def forward(self, x, return_features=False):
        feats = self.backbone(x)          # shape [bs, 512]
        proj  = self.contrastive_head(feats) # shape [bs, 128]
        if return_features:
            return proj, feats
        else:
            return proj


# Instantiate model
simclr_model = SimCLRNet ()  #features_dim=128

# Load the checkpoint
checkpoint_path = 'simclr_cifar-10.pth/simclr_cifar-10.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print("checkpoint done")
simclr_model.load_state_dict(checkpoint, strict=False)  # or checkpoint['state_dict']
simclr_model.eval()
print("model lpoaded")
# Extract embeddings from the loaded model
def extract_embeddings(model, dataset, device='cpu', batch_size=256):
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_feats = []
    model.to(device)
    model.eval()
    print("loop embed")
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            # return_features=True  gets the encoder output
            _, feats = model(imgs, return_features=True)
            all_feats.append(feats.cpu().numpy())
        print(np.concatenate(all_feats, axis=0))
    return np.concatenate(all_feats, axis=0)
print("embeddding done")

# Now actually call the extraction function
#embeddings = extract_embeddings(simclr_model, trainset, device='cpu', batch_size=256)
#print("Embeddings shape:", embeddings.shape)


# %%
# Clustering
def cluster_embeddings(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print("Cluster distribution:", dict(zip(unique_clusters, counts)))

    return cluster_labels
#cluster_labels = cluster_embeddings(embeddings, 10)

# %%


# %%
from sklearn.neighbors import NearestNeighbors

def compute_typicality_sklearn(embeddings, k=20):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
    dist, _ = nbrs.kneighbors(embeddings)
    avg_dist = dist[:, 1:].mean(axis=1)
    return 1.0 / (avg_dist + 1e-8)
#typicality_vals = compute_typicality_sklearn(embeddings, k=20)

# %%
def select_typical_samples(embeddings, cluster_labels, B, typicality_vals):
    """
    Pick the most typical (highest density) sample in each of the B clusters.
    Paper: "We pick arg max_{x in cluster} typicality(x) for each cluster."
    """
    chosen_indices = []
    for cluster_id in range(B):
        cluster_idx = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_idx) == 0:
            continue
        cluster_tvals = typicality_vals[cluster_idx]
        best_local_idx = cluster_idx[np.argmax(cluster_tvals)]
        chosen_indices.append(best_local_idx)
    return chosen_indices


# %%
# Initial pool selection
def initial_pool_selection(model, unlabelled_indices, dataset, B=10, device='cpu'):
    """
    Implements Algorithm 1 from the paper for initial pool selection.
    unlabelled_indices: list of indices in dataset that are unlabelled
    B: number of points to query
    Returns the selected indices for labeling.
    """
    # Extract embeddings for the unlabelled set
    unlabelled_subset = Subset(dataset, unlabelled_indices)
    embeddings = extract_embeddings(model, unlabelled_subset, device=device)

    # Clustering (K-means with B clusters)
    cluster_labels = cluster_embeddings(embeddings, B)

    # For each cluster, picks the point with highest typicality
    typicality_vals = compute_typicality_sklearn(embeddings, k=20)
    local_chosen = select_typical_samples(embeddings, cluster_labels, B, typicality_vals)
    # map from local indices to global dataset indices
    chosen_global = [unlabelled_indices[loc] for loc in local_chosen]

    return chosen_global

# %%
class SimpleClassifier(nn.Module):
    """
    A standard CNN or ResNet. For demonstration, a small net:
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(64*4*4, num_classes)
        )
    def forward(self, x):
        return self.net(x)
    


# %%
def train_supervised(model, dataset, labeled_indices, device='cpu', epochs=5, lr=1e-3, batch_size=64):
    """
    Trains 'model' on the subset of 'dataset' specified by labeled_indices.
    Paper: "We re-initialize the classifier each iteration" (Sec. 4.2).
    """
    subset = Subset(dataset, labeled_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()
    for ep in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# %%
def evaluate(model, test_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total

# %%


# Active learning loop
def active_learning_loop(simclr_model, trainset, test_loader, initial_B=10, cycle_B=10, cycles=5, device='cpu'):
    """
    Integrates the initial selection + iterative queries.
   
    """
    # Start with no labeled data
    all_indices = np.arange(len(trainset))
    unlabelled_indices = list(range(len(trainset))) #list(all_indices)  # initially, everything is unlabelled
    labeled_indices   = []

    # Initial pool selection
    init_selection = initial_pool_selection(simclr_model, unlabelled_indices, trainset, B=initial_B, device=device)
    for idx in init_selection:
        unlabelled_indices.remove(idx)
        labeled_indices.append(idx)

    print(f"Initial labelled set size = {len(labeled_indices)}")

    # AL cycles
    accuracies = []
    for cycle in range(cycles):
        # Train a supervised model from scratch using the labeled_indices
        classifier = SimpleClassifier(num_classes=10)  
        train_supervised(classifier, trainset, labeled_indices, device=device)
        
        # Evaluate on test set
        acc = evaluate(classifier, test_loader, device=device)
        accuracies.append(acc)
        print(f"Cycle {cycle}: Accuracy {acc:.2f}% with {len(labeled_indices)} labels")

        # B. Query new points (reuse embeddings or recalc with your SimCLR model, etc.)
        # For demonstration, let's do random here:
        # but you'd do something like TPC-RP again (clustering the unlabelled, picking typical).
        new_selection = initial_pool_selection(simclr_model, unlabelled_indices, trainset, B=cycle_B, device=device)
        
        # Add them to labeled set
        for idx in new_selection:
            unlabelled_indices.remove(idx)
            labeled_indices.append(idx)

    plt.plot(range(len(accuracies)), accuracies, marker='o')
    plt.xlabel("Active Learning Cycle")
    plt.ylabel("Test Accuracy (%)")
    plt.title("TPC(RP) Accuracy over AL Cycles")
    plt.show()

    print("Active learning finished.")
    return accuracies

# %%
final_accuracies = active_learning_loop(
    simclr_model,
    trainset,
    testloader,
    initial_B=10,   # how many labels to pick initially
    cycle_B=10,     # how many labels to pick each cycle
    cycles=5,       # total number of AL cycles
    device='cpu'
)

# %%


# %%
def random_selection(unlabelled_indices, B=10):
    return list(np.random.choice(unlabelled_indices, size=B, replace=False))


