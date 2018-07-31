import numpy as np
class HistogramBasedNaiveBayes():
    """
    Histogram based Naive Bayes for images where the sample space for each feature (pixel) is the same: Typically
    256 for gray images
    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : boolean, optional (default=True)
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like, size (n_classes,), optional (default=None)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
        
    Attributes
    ----------
    class_prior_ : array, shape (n_classes,)
        probability of each class.
    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.
    sample_space_ : posible values for features. Must all be the same
    n_features_: number of features. For a 28x28 image is 784
    histograms_matrix_ : Shape (Number of clases, n_features, sample_space len)
    """
    
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
    
    def _pixel_probability(self, clase, feature, valor, alpha=1, n_features=256):
        p_x_y_clase = ((clase[:, feature] == valor).sum()+alpha)/(len(clase) + n_features*alpha)
        return p_x_y_clase
    
    def fit(self, X, y):
        """Fit Naive Bayes classifier according to X, y
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        """
        self.classes_ = np.array(list(set(y))).astype(int)
        
        if self.fit_prior:
            self.class_count_ = np.array([(y==cl).sum() for cl in self.classes_])
            self.class_prior_ = self.class_count_/len(y)
        else:
            self.class_count_ = np.array([(y==cl).sum() for cl in self.classes_])
            self.class_prior_ = class_prior
            print('To do manage error')
        
        self.sample_space_ = np.array(list(set(X.reshape(-1))))
        self.n_features_ = X.shape[1]
        
        self.histograms_matrix_ = np.zeros([len(self.classes_), self.n_features_, len(self.sample_space_)])
        
        for cl in self.classes_:
            class_data = X[y==cl]
            for val in self.sample_space_:
                for feature in range(self.n_features_):
                    self.histograms_matrix_[cl, feature, val] = self._pixel_probability(class_data, feature, val)
        
    def _get_predicted_class(self, X):
        class_probs = []
        for i, cl in enumerate(self.histograms_matrix_):
            probs = cl[range(self.n_features_),X]
            log_probs = np.log(probs)
            class_probs.append(log_probs.sum() + np.log(self.class_prior_[i]))
        return np.argmax(np.array(class_probs))

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        predictions = []
        for i, x in enumerate(X):
            predictions.append(self._get_predicted_class(x))
        return predictions