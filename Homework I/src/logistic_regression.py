from typing import Tuple
import numpy as np
from src.logger import Logger
from sklearn.metrics import confusion_matrix

 

class DataLoader():
    """ Batch Data Loader that shuffles the data at every epoch

    Args:
        data (np.ndarray): 2D Data array
        labels (np.ndarray): 1D class/label array
        batch_size (int): Batch size
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int):
        self.batch_size = batch_size
        self.data = self.preprocess(data)
        self.labels = labels
        self.index = None

    def __iter__(self) -> "DataLoader":
        """ Shuffle the data and reset the index

        Returns:
            DataLoader: self object
        """
        self.shuffle()
        self.index = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Return a batch of data and label starting at <self.index>. Also increment <self.index>.

        Raises:
            StopIteration: If builtin "next" function is called when the data is fully passed 

        Returns:
            Tuple[np.ndarray, np.ndarray]: Batch of data (B, D) and label (B) arrays
        """
        max_batch_idx = len(self.data) / self.batch_size
        if self.index < max_batch_idx:
          self.index += 1
          return self.data[self.batch_size*(self.index-1) : self.batch_size*self.index], self.labels[self.batch_size*(self.index-1) : self.batch_size*self.index]
        else:
          raise StopIteration
          
    def shuffle(self) -> None:
        """ Shuffle the data
        """
        perm = np.random.permutation(len(self.data))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    @staticmethod
    def preprocess(data: np.ndarray) -> np.array:
        """ Preprocess the data

        Args:
            data (np.ndarray): data array

        Returns:
            np.array: Float data array with values ranging from 0 to 1 
        """
        return data.astype(np.float32) / 255


class LogisticRegresssionClassifier():
    """ Logistic Regression Classifier

    Args:
        n_features (int): Number of features
        n_classes (int): Number of unique classes/labels
    """

    def __init__(self, n_features: int, n_classes: int):
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights, self.bias = self._initialize()

    def _initialize(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Initialize weights and biases

        Returns:
            Tuple[np.ndarray, np.ndarray]: weight (D, C) and bias (C) arrays
        """
        np.random.seed(1)
        self.weights = np.random.randn(self.n_features, self.n_classes)*0.01
        self.bias    = np.zeros((self.n_classes))
        print(self.weights)
        print(self.bias)
        return self.weights, self.bias

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """ Return class probabilities of the given input array 

        Args:
            inputs (np.ndarray): input array of shape (B, D)

        Returns:
            np.ndarray: Class probabilities of shape (B, C)
        """
        outputs = np.matmul(inputs, self.weights) + self.bias
        preds = self.softmax(outputs)
        if np.sum((preds == 0)):
          print(outputs)
        return preds

    def fit(self,
            train_data_loader: DataLoader,
            eval_data_loader: DataLoader,
            learning_rate: float,
            l2_regularization_coeff: float,
            epochs: int,
            logger: Logger) -> None:
        """ Main training function

        Args:
            train_data_loader (DataLoader): Data loader with training data
            eval_data_loader (DataLoader): Data loader with evaluation data
            learning_rate (float): Learning rate
            l2_regularization_coeff (float): L2 regularization coefficient
            epochs (int): Number of epochs
            logger (Logger): Logger object for logging accuracies and losses
        """

        for epoch_index in range(epochs):
            train_epoch_accuracy_list = []
            train_epoch_loss_list = []
            eval_epoch_accuracy_list = []

            for iter_index, (train_data, train_label) in enumerate(train_data_loader):

                probs = self.predict(train_data)
                predictions = probs.argmax(axis=-1)
            

                train_loss = self.nll_loss(probs, train_label)
                print("LOSS = ", train_loss)
                train_accuracy = self.accuracy(predictions, train_label)
                print("ACC = ", train_accuracy)

                gradient = self.nll_gradients(probs, train_data, train_label)
                self.update(gradient, learning_rate, l2_regularization_coeff)

                train_epoch_accuracy_list.append(train_accuracy)
                train_epoch_loss_list.append(train_loss)
                logger.iter_train_accuracy.append(train_accuracy)
                logger.iter_train_loss.append(train_loss)
                logger.log_iteration(epoch_index, iter_index)

            for eval_data, eval_lavel in eval_data_loader:
                probs = self.predict(eval_data)
                predictions = probs.argmax(axis=-1)
                eval_accuracy = self.accuracy(predictions, eval_lavel)
                eval_epoch_accuracy_list.append(eval_accuracy)

            logger.epoch_train_accuracy.append(np.mean(train_epoch_accuracy_list))
            logger.epoch_eval_accuracy.append(np.mean(eval_epoch_accuracy_list))
            logger.log_epoch(epoch_index)

    def update(self, gradients: Tuple[np.ndarray, np.ndarray], learning_rate: float, l2_regularization_coeff: float) -> None:
        """ Update weight and biases with the given gradients and regularization 

        Args:
            gradients (Tuple[np.ndarray, np.ndarray]): gradient of weights and biases [(D, C), (C)]
            learning_rate (float): Learning rate
            l2_regularization_coeff (float): L2 regularization coefficient
        """
        l2_w_grads = gradients[0] - 2*l2_regularization_coeff*self.weights
        l2_b_grads = gradients[1] - 2*l2_regularization_coeff*self.bias

        self.weights = self.weights - learning_rate*l2_w_grads
        self.bias = self.bias - learning_rate*l2_b_grads
        
        return self.weights, self.bias

    @staticmethod
    def nll_gradients(probs: np.ndarray, inputs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculate the gradients of negative log likelihood loss with respect to weights and biases

        Args:
            probs (np.ndarray): Softmax output of shape: (B, C)
            inputs (np.ndarray): Input array of shape: (B, D)
            labels (np.ndarray): True class labels: (B,)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Gradients with respect to weights and biases. Shape: [(B, C), (C)]
        """
        if not len(labels.shape):
          labels     = np.array([labels])
        batch_size = probs.shape[0]
        classes    = probs.shape[1]
        gt         = np.zeros((batch_size, classes, 1))
        weight_grads = np.zeros(((batch_size, classes, inputs.shape[1])))
        bias_grads = np.zeros((batch_size, classes))
        for batch_idx in range(batch_size):
          gt[batch_idx, labels[batch_idx]] = 1
          prob =  probs[batch_idx]
          error = prob.reshape(prob.shape[0], 1) - gt[batch_idx]
          weight_grads[batch_idx] = np.dot(error, np.expand_dims(inputs[batch_idx], axis=0))
          bias_grads[batch_idx]   = np.squeeze(error)
        return np.sum(weight_grads, axis=0).T, np.sum(bias_grads, axis=0).T


    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """ Softmax function

        Args:
            logits (np.ndarray): input array of shape (B, C)

        Returns:
            np.ndarray: output array of shape (B, C)
        """
        prob_arr = np.zeros_like(logits)
        for batch_idx in range(logits.shape[0]):
          sample = logits[batch_idx]
          prob_arr[batch_idx] = np.exp(sample)/sum(np.exp(sample))
          #print(np.exp(sample)/sum(np.exp(sample)))
        if np.sum((np.sum(prob_arr, axis=1) <= 0.99)):
          raise Exception("Sorry, sum of class probabilities are not equal to 1 !")
        return prob_arr

    @staticmethod
    def accuracy(prediction: np.ndarray, label: np.ndarray) -> np.float32:
        """ Calculate mean accuracy

        Args:
            prediction (np.ndarray): Prediction array of shape (B)
            label (np.ndarray): Ground truth array of shape (B)

        Returns:
            np.float32: Average accuracy
        """
        return np.sum(prediction == label) / len(prediction)

    @staticmethod
    def nll_loss(prediction_probs: np.ndarray, label: np.ndarray) -> np.float32:
        """ Calculate mean negative log likelihood

        Args:
            prediction_probs (np.ndarray): Prediction probabilitites of shape (B, C)
            label (np.ndarray): Ground truth array of shape (B)

        Returns:
            np.float32: _description_
        """
        loss_arr = np.zeros((len(prediction_probs)))
        for batch_idx in range(len(prediction_probs)):
          gt_idx = label[batch_idx]
          gt     = 1
          prob   = prediction_probs[batch_idx, gt_idx]
          #loss_arr[batch_idx]  = gt * -np.log(prob) + (1-gt) * np.log(1-prob)
          loss_arr[batch_idx]  = -np.log(prob)
          #print("loss array ============ ", loss_arr)
        return np.mean(loss_arr)


    @staticmethod
    def confusion_matrix(predictions: np.ndarray, label: np.ndarray) -> np.ndarray:
        """ Calculate confusion matrix

        Args:
            predictions (np.ndarray): Prediction array of shape (B)
            label (np.ndarray): Ground truth array of shape (B)

        Returns:
            np.ndarray: Confusion matrix of shape (C, C)
        """
        #confusion_matrix(label, predictions).ravel()
        cm = confusion_matrix(label, predictions).T
        return cm











