import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm as tqdm

def sim_graph(features):
    simGraph = []
    for i in tqdm(range(len(features))):
        line = []
        for j in range(len(features)):
            if i == j:
                line.append(0)
            else:
                line.append(np.dot(features[i], features[j])/(np.linalg.norm(features[i]) * np.linalg.norm(features[j])))
        simGraph.append(line)
    return simGraph

def overlap(a, b):
    ret = 0
    for item in a:
        if item in b:
            ret += 1
    return ret

def kernel_alignment(simGraph_1, simGraph_2, k = 64):
    # kernel_alignment: calculate the alignment of two similarity graphs
    # simGraph_1: The first similarity graph
    # simGraph_2: The second similarity graph
    # k: The number of k-nearest neighbors
    # Return type: The mean, std, and a list of the individual alignment of each node
    aligns = []
    for i in range(len(simGraph_1)):
        aligns.append(
            overlap(np.argsort(simGraph_1[i])[::-1][:k], np.argsort(simGraph_2[i])[::-1][:k]) / k
        )
    return np.mean(aligns), np.std(aligns), aligns

def kernel_alignment_on_datasets(feature_set1_layered, feature_set2):
    ICL_sim_map = []
    for layer_hidden_state in feature_set1_layered:
        ICL_sim_map.append(sim_graph(layer_hidden_state))
    encoder_sim_map = sim_graph(feature_set2)

    ## Calculate the kernel alignment.
    ### The organization of the results: res_kernel_alignment[layer_index]: (mean, std, individual_values)
    res_kernel_alignment = []
    for layer_sim_graph in ICL_sim_map:
        res_kernel_alignment.append(kernel_alignment(layer_sim_graph, encoder_sim_map))
    return res_kernel_alignment

def scree_plot_from_pca(feature_set):
    pca = PCA()
    feature_set = np.array(feature_set)
    pca.fit(feature_set)
    explained_variance = pca.explained_variance_ratio_
    return explained_variance, pca

def linear_regression(
        X, 
        Y, 
        torch_model,
        epoch
    ):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    torch_model.train()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    pbar = tqdm(range(epoch))
    for _ in pbar:
        optimizer.zero_grad()
        output = torch_model(X_tensor)
        loss = loss_fn(output, Y_tensor)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")

    torch_model.eval()
    with torch.no_grad():
        output = torch_model(X_tensor)
        train_loss = loss_fn(output, Y_tensor).item()
    return train_loss