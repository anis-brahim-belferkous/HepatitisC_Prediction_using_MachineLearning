import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")

# Load the data from the CSV file
data = pd.read_csv('HepatitisCdata.csv')

var = data.shape
print("Shape of our data is : ",var)

data.head(5)
data.info()

# Drop the 'Unnamed: 0' column
data.drop('Unnamed: 0', axis=1, inplace=True)
# Drop rows with NaN values
data.dropna(inplace=True)

# Encode categorical variables
data['Sex'] = data['Sex'].map({'f': 0, 'm': 1})
data['Category'] = data['Category'].apply(lambda x: 0 if x not in ['1=Hepatitis', '2=Fibrosis', '3=Cirrhosis'] else 1)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data[['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']] = imputer.fit_transform(
    data[['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']])



#Boxplot graph of all columns
data.plot(kind='box')
plt.show()

# Identify and remove outliers using z-score
z_scores = np.abs(zscore(data[['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]))
data_no_outliers = data[(z_scores < 1.5).all(axis=1)]



"""
# Calculate the IQR (Interquartile Range)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify and remove outliers
X =data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]
"""

# Separate features and target labels
X = data_no_outliers.drop('Category', axis=1)
y = data_no_outliers['Category']


# Descriptive statistics for numerical features
print(X.describe())

# Distribution of categorical features
print(X['Sex'].value_counts())
print(y.value_counts())




X.hist(figsize=(10, 10))
plt.title("Histogram Before Improving",horizontalalignment='center', fontsize=16, pad=20)
plt.show()


# Replace the large value in 'CREA'&'AST' with the mean of 'CREA'&'AST'
X.loc[X['CREA'] > 120, 'CREA'] = X['CREA'].mean()
X.loc[X['CREA'] < 40, 'CREA'] = X['CREA'].mean()
X.loc[X['AST'] > 50, 'AST'] = X['AST'].mean()


#New Histo Results After replacing values by mean of each column
X.hist(figsize=(10, 10))
plt.title("Histogram After Improving",horizontalalignment='center', fontsize=16, pad=20)
plt.show()

# Calculate the correlation matrix
correlation_matrix = X.corr()

# Set the figure size
plt.figure(figsize=(12, 8))

# Plot the heatmap with larger annotations
sns.heatmap(correlation_matrix, annot=True, linewidths=0.2)

# Show the heatmap
plt.show()


# Calculate silhouette scores for each number of clusters
range_n_clusters = range(2, 11)

silhouette_scores = []
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)

    labels = kmeans.labels_
    silhouette_score_ = silhouette_score(data, labels)
    silhouette_scores.append(silhouette_score_)

# Plot the silhouette scores
plt.plot(range_n_clusters, silhouette_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()

# Identify the optimal number of clusters based on the highest silhouette score
optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
print("Optimal number of clusters:", optimal_n_clusters)

kmeans = KMeans(n_clusters=2)
cluster_labels = kmeans.fit_predict(X)

# Calculate the silhouette score
silhouette_score_value = silhouette_score(X, cluster_labels)
print("Silhouette score:", silhouette_score_value)


# Visualizing the clusters
# Getting unique labels
u_labels = np.unique(cluster_labels)

# Plotting the clusters
for i in u_labels:
    plt.scatter(X.iloc[cluster_labels == i, 2], X.iloc[cluster_labels == i, 11], label=f'Cluster {i}')

# Plotting the centroids of the clusters
plt.legend()
plt.title('Scatter Plot with Cluster Labels')
plt.xlabel(X.columns[2])
plt.ylabel(X.columns[11])

# Add annotations for each cluster
for i in u_labels:
    centroid_x = np.mean(X.iloc[cluster_labels == i, 3])
    centroid_y = np.mean(X.iloc[cluster_labels == i, 9])
    plt.annotate(f'Cluster {i}', (centroid_x, centroid_y), textcoords="offset points", xytext=(0, 5), ha='center')

plt.show()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit a KNeighborsClassifier model
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
accuracy = knn.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)
