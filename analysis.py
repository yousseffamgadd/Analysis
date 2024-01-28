import numpy as np
import time
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture




def gmm_clustering(data, k):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(data)
    return gmm


class GMM:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    
    def initialize_parameters(self, X):
        n_samples, _ = X.shape
    
        self.weights = np.ones(self.k) / self.k
        #print("weights= " ,self.weights)
        self.means = X[np.random.choice(n_samples, self.k, replace=False)]
       # print("means= ", self.means)
        self.covariances = [np.cov(X.T) for _ in range(self.k)]
      #  print("covariances= ", self.covariances)

    def multivariate_normal(self, X, mean, covariance):
        n_features = X.shape[1]
        diff = X - mean
        inv = np.linalg.inv(covariance + 1e-6 * np.eye(covariance.shape[0]))

        exponent = -0.5 * np.sum(np.dot(diff, inv) * diff, axis=1)
        norm_const = 1.0 / ((2 * np.pi) ** (n_features / 2) * np.linalg.det(covariance) ** 0.5)
        
        return norm_const * np.exp(exponent)


    def expectation_step(self, X):
        likelihoods = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            likelihoods[:, i] = self.multivariate_normal(X, self.means[i], self.covariances[i])
      
        weighted_likelihoods = likelihoods * self.weights
        total_weighted_likelihoods = np.sum(weighted_likelihoods, axis=1, keepdims=True)
        total_weighted_likelihoods[total_weighted_likelihoods == 0] = np.finfo(float).eps  # Prevent division by zero
        self.responsibilities = weighted_likelihoods / total_weighted_likelihoods
    
    def maximization_step(self, X):
        total_weight = np.sum(self.responsibilities, axis=0)
        self.weights = total_weight / np.sum(total_weight)
        self.means = np.dot(self.responsibilities.T, X) / total_weight[:, np.newaxis]
        for i in range(self.k):
            diff = X - self.means[i]
            self.covariances[i] = np.dot(self.responsibilities[:, i] * diff.T, diff) / total_weight[i]

    def fit(self, X):
        
        prev_log_likelihood = float('-inf')
        for iteration in range(self.max_iters):
            self.expectation_step(X)
            self.maximization_step(X)
            
            log_likelihood = 0
            for i in range(self.k):
                log_likelihood += np.sum(np.log(self.multivariate_normal(X, self.means[i], self.covariances[i])))
            
            if iteration > 0:
                change_in_likelihood = abs(log_likelihood - prev_log_likelihood)
                if change_in_likelihood < self.tol:
                    break
            prev_log_likelihood = log_likelihood

   
        # Complexity is O(n * 3k * number of k (3 & 5) = 2 * iterations)
        
    
    
    
    
def read_csv_part(file_path, num_samples):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]
    return np.array(data[:num_samples])
  







N_values = list(range(1000, 100001, 10000))  # You can add more values for N
K_values = [3,5]
runtimes = {k: [] for k in K_values}
runtimes2 = {k: [] for k in K_values}


file_path = 'C:/Users/hp/Desktop/Analysis/Analysisproject/DataSet.csv'



for k in K_values:
    
    
    gmm = GMM(k)
    full_data = read_csv_part(file_path, 100000)  # Reading the full data

# Normalize and perform PCA on full data
    normalized_data2 = (full_data - np.mean(full_data, axis=0)) / np.std(full_data, axis=0)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_data2)
    components = pca.components_

# Fit GMM once on the full dataset
    
    gmm.initialize_parameters(normalized_data2)
    gmm.fit(normalized_data2)
    
    cluster_labels = gmm.responsibilities.argmax(axis=1)

    # Plotting the data before PCA with cluster colors
    plt.scatter(normalized_data2[:, 0], normalized_data2[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Data Before PCA with Clusters for K={k}')
    plt.colorbar(label='Cluster')
    plt.show()
    
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=gmm.responsibilities.argmax(axis=1))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'GMM Clustering with K={k} using PCA-reduced dimensions')
    plt.colorbar(label='Cluster')
    plt.show()

        
    unique_clusters, counts = np.unique(gmm.responsibilities.argmax(axis=1), return_counts=True)
    percentages = counts / len(gmm.responsibilities) * 100
    print(f"In {k} clusters")
    for i, percentage in enumerate(percentages):
        print(f"Cluster {i+1} took {percentage:.2f}%")
    
   
    
   
    
    for N in N_values:
        data = read_csv_part(file_path, N)  # Replace this with your actual data
        normalized_data=(data - np.mean(data, axis=0)) / np.std(data, axis=0)
        
        # print(N)
        
        

        
        start_time = time.time()
        gmm_model = gmm_clustering(normalized_data, k)
        runtime2 = time.time() - start_time
        runtimes2[k].append(runtime2)
        
        

        start_time = time.time()
        gmm.fit(normalized_data)
        runtime = time.time() - start_time
        runtimes[k].append(runtime)
        




for k, runtime_list in runtimes.items():
    plt.plot(N_values, runtime_list, label=f'GMM, K={k}')
   
# for k, runtime_list2 in runtimes2.items():
#     plt.plot(N_values, runtime_list2, label=f'Sklearn GMM, K={k}')
    
    
plt.xlabel('Number of Samples (N)')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison for GMM Implementations')
plt.legend() 
plt.show()



