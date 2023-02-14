from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy, one_hot
from torch.nn.modules.loss import _WeightedLoss, MSELoss, _Loss
from sklearn.metrics import recall_score

from ..distributions import (get_beta_probabilities,
                             get_binomial_probabilities,
                             get_exponential_probabilities,
                             get_poisson_probabilities)

# Params [a,b] for beta distribution
_beta_params_sets = {
    'standard' : {
        3 : [[1,4,1], [4,4,1], [4,1,1]],
        4 : [[1,6,1], [6,10,1], [10,6,1], [6,1,1]],
        5 : [[1,8,1], [6,14,1], [12,12,1], [14,6,1], [8,1,1]],
        6 : [[1,10,1], [7,20,1], [15,20,1], [20,15,1], [20,7,1], [10,1,1]],
        7 : [[1,12,1], [7,26,1], [16,28,1], [24,24,1], [28,16,1], [26,7,1], [12,1,1]],
        8 : [[1,14,1],[7,31,1], [17,37,1], [27,35,1], [35,27,1], [37,17,1], [31,7,1], [14,1,1]],
        9 : [[1,16,1], [8,40,1], [18,47,1], [30,47,1], [40,40,1], [47,30,1], [47,18,1], [40,8,1], [16,1,1]], 
        10 : [[1,18,1], [8,45,1], [19,57,1], [32,59,1], [45,55,1], [55,45,1], [59,32,1], [57,19,1], [45,8,1], [18,1,1]],
        11 : [[1,21,1], [8,51,1], [20,68,1], [34,73,1], [48,69,1], [60,60,1], [69,48,1], [73,34,1], [68,20,1], [51,8,1], [21,1,1]],
        12 : [[1,23,1], [8,56,1], [20,76,1], [35,85,1], [51,85,1], [65,77,1], [77,65,1], [85,51,1], [85,35,1], [76,20,1], [56,8,1], [23,1,1]],
        13 : [[1,25,1], [8,61,1], [20,84,1], [36,98,1], [53,100,1], [70,95,1], [84,84,1], [95,70,1], [100,53,1], [98,36,1], [84,20,1], [61,8,1], [25,1,1]],
        14 : [[1,27,1], [2,17,1], [5,23,1], [9,27,1], [13,28,1], [18,28,1], [23,27,1], [27,23,1], [28,18,1], [28,13,1], [27,9,1], [23,5,1], [17,2,1], [27,1,1]]
    }
}

class UnimodalCrossEntropyLoss(CrossEntropyLoss):
    """Base class to implement a unimodal regularised cross entropy loss.
    Vargas, V. M., Gutiérrez, P. A., & Hervás-Martínez, C. (2022). 
    Unimodal regularisation based on beta distribution for deep ordinal regression.
    Pattern Recognition, 122, 108310.

    Implementations must redefine the cls_probs attribute.
    """

    def __init__(self, num_classes: int = 5, eta: float = 0.85, **kwargs):
        """
        Parameters
        ----------
        num_classes : int, default=5
            Number of classes.
        eta : float, default=0.85
            Parameter that controls the influence of the regularisation.
        """

        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.eta = eta

        # Default class probs initialized to ones
        self.register_buffer('cls_probs', torch.ones(num_classes, num_classes).float())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Method that is called to compute the loss.

        Parameters
        ----------
        input : Tensor
            The input tensor.
        target : Tensor
            The target tensor.

        Returns
        -------
        loss: Tensor
            The computed loss.
        """

        y_prob = self.get_buffer('cls_probs')[target]
        target_oh = one_hot(target, self.num_classes)

        y_true = (1.0 - self.eta) * target_oh + self.eta * y_prob

        return super().forward(input, y_true)

class BetaCrossEntropyLoss(UnimodalCrossEntropyLoss):
    """Beta unimodal regularised cross entropy loss.
    It takes the parameters for the beta distribution from the _beta_params_set dictionary.
    """

    def __init__(self, num_classes: int = 5, params_set: str = 'standard', eta: float = 1.0, **kwargs):
        """
        Parameters
        ----------
        num_classes : int, default=5
            Number of classes.
        params_set : str, default='standard'
            The set of parameters to use for the beta distribution (chosen from the _beta_params_set dictionary).
        eta : float, default=1.0
            Parameter that controls the influence of the regularisation.
        """

        super().__init__(num_classes, eta, **kwargs)
        self.params = _beta_params_sets[params_set]

        # Precompute class probabilities for each label
        self.cls_probs = torch.tensor([get_beta_probabilities(
                                     num_classes,
                                     self.params[num_classes][i][0],
                                     self.params[num_classes][i][1],
                                     self.params[num_classes][i][2])
                                for i in range(num_classes)]).float()
                            

class ExponentialCrossEntropyLoss(UnimodalCrossEntropyLoss):
    """Exponential unimodal regularised cross entropy loss.
    """
    def __init__(self, num_classes: int = 5, eta: float = 0.85, **kwargs):
        """
        Parameters
        ----------
        num_classes : int, default=5
            Number of classes.
        eta : float, default=0.85
            Parameter that controls the influence of the regularisation.
        """

        super().__init__(num_classes, eta, **kwargs)

        # Precompute class probabilities for each label
        self.cls_probs = torch.tensor(get_exponential_probabilities(num_classes)).float()

class BinomialCrossEntropyLoss(UnimodalCrossEntropyLoss):
    """Binomial unimodal regularised cross entropy loss.
    """

    def __init__(self, num_classes: int = 5, eta: float = 0.85, **kwargs):
        """
        Parameters
        ----------
        num_classes : int, default=5
            Number of classes.
        eta : float, default=0.85
            Parameter that controls the influence of the regularisation.
        """

        super().__init__(num_classes, eta, **kwargs)

        # Precompute class probabilities for each label
        self.cls_probs = torch.tensor(get_binomial_probabilities(num_classes)).float()

class PoissonCrossEntropyLoss(UnimodalCrossEntropyLoss):
    def __init__(self, num_classes: int = 5, eta: float = 0.85, **kwargs):
        """
        Parameters
        ----------
        num_classes : int, default=5
            Number of classes.
        eta : float, default=0.85
            Parameter that controls the influence of the regularisation.
        """

        super().__init__(num_classes, eta, **kwargs)

        # Precompute class probabilities for each label
        self.cls_probs = torch.tensor(get_poisson_probabilities(num_classes)).float()


class WKLoss(_WeightedLoss):
    """Weighted Kappa loss implementation.
    de La Torre, J., Puig, D., & Valls, A. (2018).
    Weighted kappa loss function for multi-class classification of ordinal data in deep learning.
    Pattern Recognition Letters, 105, 144-154.
    """

    def __init__(self, num_classes: int, penalization_type: str ='quadratic', weight: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Number of classes.
        penalization_type : str, default='quadratic'
            The penalization type of WK loss to use (quadratic or linear).
        weight : np.ndarray, default=None
            The weight matrix that is applied to the cost matrix.
        """

        super().__init__(**kwargs)

        # Create cost matrix and register as buffer
        cost_matrix = np.reshape(np.tile(range(num_classes), num_classes), (num_classes, num_classes))
        
        if penalization_type == 'quadratic':
            cost_matrix = np.power(cost_matrix - np.transpose(cost_matrix), 2) / (num_classes - 1) ** 2.0
        else:
            cost_matrix = np.abs(cost_matrix - np.transpose(cost_matrix)) / (num_classes - 1) ** 2.0
        
        if weight is not None:
            cost_matrix = cost_matrix * (1. / weight)
        
        self.register_buffer("cost_matrix", torch.tensor(cost_matrix, dtype=torch.float))

        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            The input tensor.
        target : torch.Tensor
            The target tensor.

        Returns
        -------
        loss: Tensor
            The WK loss.
        """

        costs = self.cost_matrix[target] #type: ignore

        numerator = costs * input
        numerator = torch.sum(numerator)

        sum_prob = torch.sum(input, dim=0)
        target_prob = one_hot(target, self.num_classes)
        n = torch.sum(target_prob, dim=0)

        a = torch.reshape(torch.matmul(self.cost_matrix, torch.reshape(sum_prob, shape=[-1,1])), shape=[-1]) #type: ignore
        b = torch.reshape(n / torch.sum(n), shape=[-1])

        epsilon = 1e-9

        denominator = a * b
        denominator = torch.sum(denominator) + epsilon

        result = numerator / denominator

        return result

class MSLoss(_Loss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def compute_sensitivities(y_true = torch.Tensor, y_pred = torch.Tensor):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            Grount truth labels
        y_pred : torch.Tensor
            Predicted labels

        Returns:
        sensitivities : torch.Tensor
            Sensitivities tensor
        """
        
        diff = (1.0 - torch.pow(y_true - y_pred, 2)) / 2.0 # [0,1]        
        diff_class = torch.sum(diff, axis=1) # TP
        sum = torch.sum(diff_class) # total sum of that vector
        sensitivities = diff_class / sum # vector of size N with    
        
        return sensitivities
        
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Parameters
        ----------
        y_true : torch.Tensor 
            Grount truth labels
        y_pred : torch.Tensor
            Predicted labels

        Returns
        -------
        mean_sensitivities : torch.Tensor
            Mean sensitivities tensor
        """
        
        sensitivities = self.compute_sensitivities(y_true, y_pred)
        
        return torch.mean(sensitivities)


class MSAndQWKLoss(_WeightedLoss):
    def __init__(self, num_classes: int, alpha: 0.5) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Number of classes
        alpha 0.5:
            Is the weight for qwk in comparaison with MS. It must be between 1 and 0
        """
        
        self.alpha = alpha
        self.num_classes = num_classes
            
    def forward(self, y_true = torch.Tensor, y_pred = torch.Tensor):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth labels
        y_pred : torch.Tensor
            Predicted labels

        Returns
        -------
        loss : torch.Tensor
            The weighted sum of MS and QWK loss
        """
        
        qwk = WKLoss(self.num_classes)
        qwk_result = qwk(y_true, y_pred)
        
        ms = MSLoss()
        ms_result = ms(y_true, y_pred)
        
        return self.alpa * qwk_result + (1 - self.alpha) * ms_result


class OrdinalEcocDistanceLoss(_WeightedLoss):
    def __init__(self, n_classes: int, device, class_weights: Optional[torch.Tensor] = None) -> None:
        """
        Parameters
        ----------
        n_classes : int
            Number of classes
        device : torch.device
            Contains the device on which the model is running
        class_weights : Optional[torch.Tensor]
            Contains the weights for each class
        """
        
        self.target_class = np.ones((n_classes, n_classes-1), dtype=np.float32)
        self.target_class[np.triu_indices(n_classes, 0, n_classes-1)] = 0.0
        self.target_class = torch.tensor(self.target_class, dtype=torch.float32, device=device, requires_grad=False)
        self.mse = 0
        
        if class_weights is not None:
            assert class_weights.shape == (n_classes,)
            class_weights = class_weights.float().to(device)
            self.mse = MSELoss(reduction='none')
        else:
            self.mse = MSELoss(reduction='sum')
            
    def forward(self, target: torch.Tensor, net_output: torch.Tensor, class_weights: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        target : torch.Tensor
            Contains the target labels
        net_output : torch.Tensor
            Contains the output of the network
        class_weights : Optional[torch.Tensor]
            Contains the weights for each class

        Returns
        -------
        loss : torch.Tensor
            If class_weights is not None, the weighted sum of the MSE loss
            Else the sum of the MSE loss
        """
        
        if class_weights is not None:
            target_vector = self.target_class[target]
            weights = class_weights[target]  # type: ignore
            return (self.mse(net_output, target_vector).sum(dim=1) * weights).sum()
        else:
            target_vector = self.target_class[target]
            return self.mse(net_output, target_vector)