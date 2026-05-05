import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(353)

def generate_mpc_data(
    num_clusters=10,
    mpcs_per_cluster=20,
    delay_spread=50e-9,      
    angle_spread_deg=5,
    power_decay=0.5
):

    M = num_clusters * mpcs_per_cluster
    X = []
    true_labels = []

    angle_spread = np.deg2rad(angle_spread_deg)

    for c in range(num_clusters):
        # cluster center
        tau_c = np.random.uniform(0, 3e-6)  
        aoa_c = np.random.uniform(-np.pi, np.pi)
        aod_c = np.random.uniform(-np.pi, np.pi)
        eoa_c = np.random.uniform(-np.pi/2, np.pi/2)
        eod_c = np.random.uniform(-np.pi/2, np.pi/2)

        for m in range(mpcs_per_cluster):

            alpha = np.random.rand()

            tau = tau_c + np.random.normal(0, delay_spread)

            aoa = aoa_c + np.random.laplace(0, angle_spread)
            aod = aod_c + np.random.laplace(0, angle_spread)
            eoa = eoa_c + np.random.laplace(0, angle_spread)
            eod = eod_c + np.random.laplace(0, angle_spread)

            cluster_power = np.random.uniform(0.5, 2.0) 
            P = cluster_power * np.exp(-power_decay * alpha)

            X.append([P, tau, aod, aoa, eod, eoa])
            true_labels.append(c)

    return np.array(X), np.array(true_labels)

def angular_diff_norm(a, b):
    diff = abs(a - b) % (2*np.pi)
    if diff > np.pi:
        diff = 2*np.pi - diff
    return diff


def mcd_distance(x1, x2):
    d_tau = (x1[1] - x2[1])**2

    d_ang = (
    angular_diff_norm(x1[2], x2[2])**2 +
    angular_diff_norm(x1[3], x2[3])**2 +
    angular_diff_norm(x1[4], x2[4])**2 +
    angular_diff_norm(x1[5], x2[5])**2
    )

    return np.sqrt(d_tau + d_ang)

def get_knn_indices(X, K):
    M = len(X)
    knn = []

    for i in range(M):
        dists = [mcd_distance(X[i], X[j]) for j in range(M)]
        idx = np.argsort(dists)[1:K+1] 
        knn.append(idx)

    return knn

def compute_density(X, knn):

    M = len(X)
    rho = np.zeros(M)

    sigma_tau = np.std(X[:,1]) + 1e-12
    sigma_ang = np.std(X[:,2:]) + 1e-12

    for i in range(M):
        for j in knn[i]:

            Pj = X[j][0]

            tau_diff = abs(X[i][1] - X[j][1])

            ang_diff = (
                angular_diff_norm(X[i][2], X[j][2]) +
                angular_diff_norm(X[i][3], X[j][3]) +
                angular_diff_norm(X[i][4], X[j][4]) +
                angular_diff_norm(X[i][5], X[j][5])
            )

            rho[i] += (
                np.exp(Pj / (np.mean(X[:,0]) + 1e-12)) *
                np.exp(-(tau_diff**2) / (2 * sigma_tau**2)) *
                np.exp(-ang_diff / sigma_ang)
            )

    return rho


def compute_rho_star(rho, knn):
    rho_star = np.zeros_like(rho)

    for i in range(len(rho)):
        neighbors = list(knn[i]) + [i]
        rho_star[i] = rho[i] / np.max(rho[neighbors])

    return rho_star


def compute_delta(X, rho_star, knn):

    M = len(X)
    delta = np.zeros(M)

    for i in range(M):

        higher = [j for j in knn[i] if rho_star[j] > rho_star[i]]

        if len(higher) > 0:
            delta[i] = min(mcd_distance(X[i], X[j]) for j in higher)
        else:
           
            delta[i] = max(mcd_distance(X[i], X[j]) for j in range(M) if j != i)

    return delta


def compute_delta_star(delta, knn):
    delta_star = np.zeros_like(delta)

    for i in range(len(delta)):
        neighbors = list(knn[i]) + [i]
        delta_star[i] = delta[i] / np.max(delta[neighbors])

    return delta_star

def assign_clusters(X, rho_star, key_indices, knn):
    M = len(X)
    parent = np.full(M, -1, dtype=int)

    # Restrict search to K-nearest neighbors
    for i in range(M):
        higher = [j for j in knn[i] if rho_star[j] > rho_star[i]]
        
        if len(higher) == 0:
            parent[i] = i
        else:
            dists = [mcd_distance(X[i], X[j]) for j in higher]
            parent[i] = higher[np.argmin(dists)]

    def find_root(i):
        visited = set()
        while i not in key_indices and parent[i] != i:
            if i in visited: break
            visited.add(i)
            i = parent[i]
            
        if i not in key_indices:
            dists = [mcd_distance(X[i], X[k]) for k in key_indices]
            if len(dists) > 0:
                i = key_indices[np.argmin(dists)]
        return i

    labels = np.zeros(M, dtype=int)
    for i in range(M):
        root = find_root(i)
        labels[i] = np.where(key_indices == root)[0][0]

    return labels

def compute_clusters(X, labels, key_indices):
    clusters = []

    for k in range(len(key_indices)):
        cluster_points = np.where(labels == k)[0]

        total_power = np.sum(X[cluster_points, 0])

        centroid_idx = key_indices[k]
        centroid_params = X[centroid_idx]

        Phi_n = {
            "centroid_index": centroid_idx,
            "power": total_power,
            "tau": centroid_params[1],
            "AoD": centroid_params[2],
            "AoA": centroid_params[3],
            "EoD": centroid_params[4],
            "EoA": centroid_params[5],
            "members": cluster_points
        }

        clusters.append(Phi_n)

    return clusters


def select_key_mpcs(gamma):

    sorted_idx = np.argsort(gamma)[::-1]
    sorted_gamma = gamma[sorted_idx]

    diffs = sorted_gamma[:-1] - sorted_gamma[1:]

    smooth_diffs = np.convolve(diffs, np.ones(3)/3, mode='same')

    cut = np.argmax(smooth_diffs) + 1

    cut = max(2, min(cut, len(gamma)//5))

    return sorted_idx[:cut]

def clusters_to_matrix(clusters):
    X_c = []
    for c in clusters:
        X_c.append([
            c["power"],
            c["tau"],
            c["AoD"],
            c["AoA"],
            c["EoD"],
            c["EoA"]
        ])

    return np.array(X_c)


def compute_density_stable(X, knn, sigma_tau, sigma_ang):
    M = len(X)
    log_rho = np.full(M, -np.inf)

    for i in range(M):
        vals = []

        for j in knn[i]:
            Pj = X[j][0]

            tau_diff = abs(X[i][1] - X[j][1])
            ang_diff = np.linalg.norm(X[i][2:] - X[j][2:])

            val = (
                Pj
                - (tau_diff**2) / (2 * sigma_tau**2)
                - ang_diff / sigma_ang
            )

            vals.append(val)

        max_val = np.max(vals)
        log_rho[i] = max_val + np.log(np.sum(np.exp(np.array(vals) - max_val)))

    return np.exp(log_rho)

def normalize_data(X):
    X = X.copy()

    tau = X[:,1]
    X[:,1] = tau / (np.max(tau) + 1e-12)

    for i in range(2,6):
        col = X[:,i]
        X[:,i] = (col - np.min(col)) / (np.max(col) - np.min(col) + 1e-12)

    return X


def compute_f_measure(true_labels, pred_labels):

    clusters_pred = np.unique(pred_labels)
    clusters_true = np.unique(true_labels)

    N = len(true_labels)
    F_total = 0

    for k in clusters_pred:

        idx_k = np.where(pred_labels == k)[0]
        best_F = 0

        for j in clusters_true:

            idx_j = np.where(true_labels == j)[0]

            Nij = len(np.intersect1d(idx_k, idx_j))
            if Nij == 0:
                continue

            precision = Nij / len(idx_k)
            recall    = Nij / len(idx_j)

            F = 2 * precision * recall / (precision + recall)
            best_F = max(best_F, F)

        F_total += (len(idx_k) / N) * best_F

    return F_total


def compute_s_measure(X, labels):

    if len(np.unique(labels)) < 2:
        return 0

    from sklearn.metrics import silhouette_score, pairwise_distances

    D = pairwise_distances(X, metric=mcd_distance)

    return silhouette_score(D, labels, metric='precomputed')


def plot_decision_graph(rho_star, delta_star, gamma):

    plt.figure(figsize=(6,5))
    plt.scatter(rho_star, delta_star, c=gamma, cmap='jet')
    plt.xlabel("Relative Density (ρ*)")
    plt.ylabel("Relative Distance (δ*)")
    plt.title("Decision Graph (Improved KPD)")
    plt.colorbar(label="γ")
    plt.grid()
    plt.show()



def plot_centroids(rho_star, delta_star, gamma, key_indices):

    plt.figure(figsize=(6,5))

    plt.scatter(rho_star, delta_star, c='lightgray', s=20, label="All MPCs")

    plt.scatter(
        rho_star[   key_indices],
        delta_star[key_indices],
        c='red',
        s=80,
        edgecolors='black',
        label="Cluster Centroids"
    )

    plt.xlabel("Relative Density (ρ*)")
    plt.ylabel("Relative Distance (δ*)")
    plt.title("Centroid Selection (Improved KPD)")
    plt.legend()
    plt.grid()
    plt.show()

def conventional_kpd(X, K):

    knn = get_knn_indices(X, K)

    rho = compute_density(X, knn)
    rho_star = compute_rho_star(rho, knn)

    key_indices = np.where(rho_star == 1)[0]

    labels = assign_clusters(X, rho_star, key_indices,knn)

    clusters = compute_clusters(X, labels, key_indices)
    X_c = normalize_data(clusters_to_matrix(clusters))

    N = len(X_c)
    knn_c = [np.array([j for j in range(N) if j != i]) for i in range(N)]

    rho_c = compute_density(X_c, knn_c)
    rho_star_c = rho_c / np.max(rho_c)

    chi = 0.8

    merge_indices = np.where(rho_star_c > chi)[0]

    parent_c = np.full(N, -1)

    for i in range(N):

        if i not in merge_indices:
            parent_c[i] = i
            continue

        candidates = [
            j for j in range(N)
            if rho_star_c[j] > rho_star_c[i] and j not in merge_indices
        ]

        if len(candidates) == 0:
            parent_c[i] = i
        else:
            dists = [mcd_distance(X_c[i], X_c[j]) for j in candidates]
            parent_c[i] = candidates[np.argmin(dists)]

    def find_root_c(i):
        while i in merge_indices and parent_c[i] != i:
            i = parent_c[i]
        return i

    final_labels_c = np.array([find_root_c(i) for i in range(N)])

    pred_labels = final_labels_c[labels]

    return pred_labels

def improved_kpd(X, K):

    knn = get_knn_indices(X, K)

    rho = compute_density(X, knn)
    rho_star = compute_rho_star(rho, knn)

    M = len(X)
    delta = np.zeros(M)

    for i in range(M):
        higher = [j for j in knn[i] if rho_star[j] > rho_star[i]]

        if len(higher) > 0:
            delta[i] = min(mcd_distance(X[i], X[j]) for j in higher)
        else:
            delta[i] = max(mcd_distance(X[i], X[j]) for j in knn[i])

    delta_star = compute_delta_star(delta, knn)

    gamma = rho_star * delta_star

    sorted_idx = np.argsort(gamma)[::-1]
    sorted_gamma = gamma[sorted_idx]

    diffs = sorted_gamma[:-1] - sorted_gamma[1:]
    cut = np.argmax(diffs) + 1

    key_indices = sorted_idx[:cut]

    labels = assign_clusters(X, rho_star, key_indices,knn)
    clusters = compute_clusters(X, labels, key_indices)

    X_c = normalize_data(clusters_to_matrix(clusters))

    N = len(X_c)
    knn_c = [np.array([j for j in range(N) if j != i]) for i in range(N)]

    rho_c = compute_density(X_c, knn_c)
    rho_star_c = rho_c / (np.max(rho_c) + 1e-12)

    delta_c = np.zeros(N)
    for i in range(N):
        higher = [j for j in range(N) if rho_star_c[j] > rho_star_c[i]]

        if len(higher) > 0:
            delta_c[i] = min(mcd_distance(X_c[i], X_c[j]) for j in higher)
        else:
            delta_c[i] = max(mcd_distance(X_c[i], X_c[j]) for j in range(N) if j != i)

    delta_star_c = delta_c / (np.max(delta_c) + 1e-12)
    delta_star_c = np.maximum(delta_star_c, 1e-6)

    psi_c = rho_star_c / (delta_star_c + 1e-8)

    psi_norm = (psi_c - np.min(psi_c)) / (np.max(psi_c) - np.min(psi_c) + 1e-12)
    centroid_indices = np.where(psi_norm > 0.6)[0]
    merge_indices = np.where((psi_norm > 0.3) & (psi_norm <= 0.6))[0]

    parent_c = np.full(N, -1)

    for i in range(N):
        if i not in merge_indices:
            parent_c[i] = i
            continue

        candidates = [j for j in range(N) if rho_star_c[j] > rho_star_c[i]]

        if len(candidates) == 0:
            parent_c[i] = i
        else:
            dists = [mcd_distance(X_c[i], X_c[j]) for j in candidates]
            parent_c[i] = candidates[np.argmin(dists)]

    def find_root_c(i):
        visited = set()
        while i in merge_indices and parent_c[i] != i:
            if i in visited:
                break
            visited.add(i)
            i = parent_c[i]
        return i

    final_labels_c = np.array([find_root_c(i) for i in range(N)])

    # map back to MPCs
    pred_labels = final_labels_c[labels]

    return pred_labels
