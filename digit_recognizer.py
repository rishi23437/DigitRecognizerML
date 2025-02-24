import numpy as np
import idx2numpy
import matplotlib.pyplot as plt


# 1. DATA PREPROCESSING

def filter_classes(images, labels):
    ''' Filters images and labels for classes 0, 1 and 2'''
    classes = [0, 1, 2]
    mask = np.isin(labels, classes)
    return images[mask], labels[mask]

def extracting_samples(images, labels, num_of_samples = 100):
    '''Extracting 100 samples from train and test set'''
    sampled_images, sampled_labels = [], []
    classes = np.unique(labels)

    for classs in classes:
        indices = np.where(labels == classs)[0]
        chosen_indices = np.random.choice(indices, num_of_samples, replace = False)
        sampled_images.append(images[:, chosen_indices])     # selecting all pixels of an image, for the chosen images
        sampled_labels.append(labels[chosen_indices])

    return np.hstack(sampled_images), np.hstack(sampled_labels)

# Loading the dataset
train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
test_images = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")

# Filtering classes 0, 1, 2
train_images, train_labels = filter_classes(train_images, train_labels)
test_images, test_labels = filter_classes(test_images, test_labels)

# Converting 28 x 28 images to 784 x 1 feature vectors, normalizing them
train_images = (train_images.reshape(train_images.shape[0], -1) / 255.0).T
test_images =  (test_images.reshape(test_images.shape[0], -1) / 255.0).T
train_labels = train_labels.T
test_labels = test_labels.T

# Selecting 100 samples from each class
train_images, train_labels = extracting_samples(train_images, train_labels, 100)
test_images, test_labels = extracting_samples(test_images, test_labels, 100)

print("\nDataset after preprocessing:")
print("Train images: ", train_images.shape, "Train labels: ", train_labels.shape)
print("Test images: ", test_images.shape, "Test labels: ", test_labels.shape, "\n")


# 2. MLE Estimates of Mean and Variance

def find_mean_MLE(images):
    '''Adding image vectors, then dividing by n'''
    num_samples = images.shape[1]
    mean = np.zeros((images.shape[0], 1))

    for i in range(num_samples):
        mean += images[:, i].reshape(-1, 1)                                         

    mean /= num_samples
    return mean

def find_cov_MLE(images, mean):
    centered_images = images - mean
    covariance = (centered_images @ centered_images.T) / images.shape[1]  # cov = (x-mean)(x-mean)^T / N
    return covariance
    
def MLE(images, labels):
    """Compute mean and covariance for each digit class using MLE."""
    means, covariances = [], []

    # computing the mean and covariance for the 3 classes iteratively, then stacking them in the same array
    for classs in range(3):
        indices = np.where(labels == classs)[0]
        class_images = images[:, indices]

        mean = find_mean_MLE(class_images)
        covariance = find_cov_MLE(class_images, mean)

        means.append(mean)
        covariances.append(covariance)

        # print(f"Class {classs}: Mean shape: {mean.shape}, Covariance shape: {covariance.shape}")

    return np.squeeze(np.array(means)), np.array(covariances)


MLE_means, MLE_covariances = MLE(train_images, train_labels)
print("MLE means and covariances for the 3 classes: ", MLE_means.shape, MLE_covariances.shape, "\n")


# 3. Dimensionality Reduction using PCA

def pca_center_data(images):
    mean_pca = np.mean(images, axis = 1, keepdims = True)
    centered_data_matrix = images - mean_pca
    return centered_data_matrix, mean_pca

def eigen(cov):
    '''
    Computing eigen values and vectors for a covariance matrix, and sorting according to decreasing eigen value
    Since S is PSD, it is symmetric, so using eigh.
    '''
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]                       # sorting in descending order
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors

def reduce_dimensions(data_matrix, eigenvalues, eigenvectors, variance = 0.95):
    ''' Choosing principle components for given variance, obtaining Y = (Up^T).Xc'''
    total_var = np.sum(eigenvalues)
    curr_var, p = 0, 0                                  # p: number of principle components to be chosen
    for i in range(len(eigenvalues)):
        curr_var += eigenvalues[i]
        if curr_var/total_var >= variance:
            p = i + 1
            break
    Up = eigenvectors[:, :p]
    Y = Up.T @ data_matrix
    return Y, Up, p

def transform_test_data(test_images, pca_mean, Up):
    ''' Projecting test data onto the PCA space(the p vectors generated above)'''
    Xc_test = test_images - pca_mean
    Y_test = Up.T @ Xc_test
    return Y_test

def test_components_var(eigenvalues, num_components):
    ''' Finding out the variance retained for different number of components'''
    total_var, curr_var = np.sum(eigenvalues), 0
    for i in range(num_components):
        curr_var += eigenvalues[i]
    return curr_var/total_var

Xc, pca_mean = pca_center_data(train_images)
S = (Xc @ Xc.T)/(300 - 1)                                                       # S -> Covariance matrix
eigenvalues, eigenvectors = eigen(S)                                            # obtained eigen values
Y_train, Up, p = reduce_dimensions(Xc, eigenvalues, eigenvectors, 0.95)
Y_test = transform_test_data(test_images, pca_mean, Up)

test_var = test_components_var(eigenvalues, 50)                          

print("PCA transformed train set shape: ", Y_train.shape)
print("PCA transformed test set shape: ", Y_test.shape)


# 4. Class Projection using FDA

def compute_S_B(images, labels):
    ''' Computing the between class scatter matrix S_B'''
    num_features = images.shape[0]
    classes = [0, 1, 2]
    mean_data = np.mean(images, axis=1, keepdims=True)

    S_B = np.zeros((num_features, num_features))
    for c in classes:
        indices = np.where(labels == c)[0]
        samples_c = images[:, indices]
        mean_c = np.mean(samples_c, axis=1, keepdims=True)          # mean for class c
        N_c = samples_c.shape[1]

        S_B += N_c * (mean_c - mean_data) @ (mean_c - mean_data).T          

    return S_B

def compute_S_W(images, labels):
    ''' Computing within class scatter matrix S_W'''
    num_features = images.shape[0]
    classes = [0, 1, 2]

    S_W = np.zeros((num_features, num_features))
    for c in classes:
        indices = np.where(labels == c)[0]
        samples_c = images[:, indices]
        mean_c = np.mean(samples_c, axis=1, keepdims=True)

        centered_data_c = samples_c - mean_c
        S_W += centered_data_c @ centered_data_c.T                   # adding the covariance within each class

    return S_W

def compute_fda_projection(images, labels, num_components):
    ''' Computing W(fda projection matrix) by maximizing the trace ratio.'''
    num_features = images.shape[0]
    S_B = compute_S_B(images, labels)
    S_W = compute_S_W(images, labels)

    # Solve the generalized eigenvalue problem - finding eigen values and vectors for (S_W inv)*(S_B)
    S_W_inv = np.linalg.inv(S_W + 0.001*np.identity(num_features))  # added bI where b = 0.001 cuz I got a singular matrix error
    eigenvalues, eigenvectors = np.linalg.eigh(S_W_inv @ S_B)

    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    W = eigenvectors[:, :num_components]

    return W

W_fda = compute_fda_projection(train_images, train_labels, 2)    # number of components = number of classes - 1 = 2

train_images_fda = W_fda.T @ train_images
test_images_fda = W_fda.T @ test_images

print("\nProjection Matrix W shape: ", W_fda.shape)
print("FDA transformed train set Shape: ", train_images_fda.shape)
print("FDA transformed test set Shape: ", test_images_fda.shape)


# 5. b) Classification using LDA on PCA transformed data

def compute_lda_params(Y_train, train_labels):
    ''' Computing means, prios and covariance matrix. '''
    classes = np.unique(train_labels)
    num_features = Y_train.shape[0]                             # gives reduced dimensions after pca
    num_samples = Y_train.shape[1]

    means, priors = {}, {}
    for c in classes:                                            # Finding means, priors
        indices = np.where(train_labels == c)[0]
        samples_c = Y_train[:, indices]
        
        mean_c = np.mean(samples_c, axis=1, keepdims=True)
        means[c] = mean_c
        priors[c] = len(indices) / num_samples

    S = np.zeros((num_features, num_features))
    for c in classes:                                            # Computing covariance matrix
        indices = np.where(train_labels == c)[0]
        samples_c = Y_train[:, indices] - means[c]
        S += samples_c @ samples_c.T
    S /= (num_samples - len(classes))                               # dividing by n - c

    return means, S, priors

def lda_classify_sample(x, means, S_inv, priors):
    ''' Classifying by computing discriminants, choosing the class with the max g.'''
    predicted_class = None
    max_discriminant = -np.inf

    for c in means.keys():
        W_c = S_inv @ means[c]
        b_c = -0.5*(means[c].T @ S_inv @ means[c]) + np.log(priors[c])

        g_c = (W_c.T @ x) + b_c                                                 # discriminant for class c

        if g_c > max_discriminant:
            max_discriminant = g_c
            predicted_class = c

    return predicted_class

def lda_classify_dataset(Y_test, means, S_inv, priors):
    predictions = np.zeros(Y_test.shape[1])

    for i in range(Y_test.shape[1]):
        predictions[i] = lda_classify_sample(Y_test[:, i].reshape(-1, 1), means, S_inv, priors)

    return predictions

# Training LDA and classifying the test set
means_pca, S_pca, priors_pca = compute_lda_params(Y_train, train_labels)
S_inv_pca = np.linalg.inv(S_pca)                                                # inverse of S(Covariance matrix)
y_predicted_pca = lda_classify_dataset(Y_test, means_pca, S_inv_pca, priors_pca)

# Analysing the accuracy of LDA
accuracy_lda_pca = np.mean(y_predicted_pca == test_labels) * 100
print(f"\nLDA Classification Accuracy on PCA transformed dataset: {accuracy_lda_pca:.2f}%")


# 5. a1) LDA on FDA transformed data

means_fda, S_fda, priors_fda = compute_lda_params(train_images_fda, train_labels)
S_inv_fda = np.linalg.inv(S_fda)
y_predicted_fda = lda_classify_dataset(test_images_fda, means_fda, S_inv_fda, priors_fda)

accuracy_lda_fda = np.mean(y_predicted_fda == test_labels) * 100
print(f"\nLDA Classification Accuracy on FDA transformed dataset: {accuracy_lda_fda:.2f}%")


# 5. a2) QDA on FDA transformed data

def compute_qda_params(images, train_labels):
    ''' Computing params for qda'''
    classes = np.unique(train_labels)
    num_samples = images.shape[1]

    means, covariances, priors = {}, {}, {}
    for c in classes:
        indices = np.where(train_labels == c)[0]
        class_samples = images[:, indices]

        mean_c = np.mean(class_samples, axis=1, keepdims=True)
        covariance_c = np.cov(class_samples)

        means[c] = mean_c
        covariances[c] = covariance_c
        priors[c] = len(indices)/num_samples

    return means, covariances, priors

def qda_classify_sample(x, means, covariances, priors):
    ''' classifies a specific sample using QDA'''
    predicted_class = None
    max_discriminant = -np.inf

    for c in means.keys():
        mean_c = means[c]
        cov_c = covariances[c]
        prior_c = priors[c]

        cov_c_inv = np.linalg.inv(cov_c)
        cov_c_det = np.linalg.det(cov_c)

        # calculating the discriminant
        g_c = (-0.5 * np.log(cov_c_det)) + (-0.5 * (x - mean_c).T @ cov_c_inv @ (x - mean_c)) + (np.log(prior_c))

        if g_c > max_discriminant:
            max_discriminant = g_c
            predicted_class = c

    return predicted_class

def qda_classify_dataset(test_dataset, means, covariances, priors):
    ''' Classifies the dataset '''
    predictions = np.zeros(Y_test.shape[1])

    for i in range(Y_test.shape[1]):
        predictions[i] = qda_classify_sample(test_dataset[:, i].reshape(-1, 1), means, covariances, priors)

    return predictions

means_qda_fda, covariances_qda_fda, priors_qda_fda = compute_qda_params(train_images_fda, train_labels)
y_predicted_qda_fda = qda_classify_dataset(test_images_fda, means_qda_fda, covariances_qda_fda, priors_qda_fda)

accuracy_qda_fda = np.mean(y_predicted_qda_fda == test_labels) * 100
print(f"\nQDA Classification Accuracy on FDA transformed dataset: {accuracy_qda_fda:.2f}%")


# 6. Plotting the Transformed Feature Spaces for PCA and FDA

def plot_projection(data, labels, title, flag):
    ''' Plotting the 2D transformed data '''
    classes = np.unique(labels)
    colours = ['r', 'g', 'b']                                    # red, green, blue
    markers = ['o', 's', '^']

    plt.figure(figsize=(7, 5))
    
    for cla, col, m in zip(classes, colours, markers):
        indices = np.where(labels == cla)[0]
        plt.scatter(data[0, indices], data[1, indices], c=col, marker=m, label=f'Class {cla}', alpha=0.7)

    if not flag:                                                            # flag == 0 for PCA
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    else:                                                                   # flag == 1 for FDA
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_projection(Y_train[:2, :], train_labels, "PCA: 2D Projected Feature Space", 0)    # plotting pca transformed data for p = 2
plot_projection(train_images_fda, train_labels, "FDA: 2D Projected Feature Space", 1)  # plotting fda transformed data
