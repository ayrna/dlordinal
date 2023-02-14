import json
from pathlib import Path
from typing import Union, Callable, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_confusion_matrices(path: Union[str, Path], filter: Callable = lambda key: True) -> Dict[str, np.ndarray]:
    """Load confusion matrices from a json file that contains a dictionary
    where each key identifies a confusion matrix and the value is a list
    of lists representing the confusion matrix.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the json file containing the confusion matrices.
    filter : Callable, optional
        Filter function that takes the key of the confusion matrix and
        returns True if the confusion matrix should be loaded. By default,
        all confusion matrices are loaded.

    Returns
    -------
    confusion_matrices: Dict[str, array-like]
        Dictionary containing the confusion matrices.

    Examples
    --------
    >>> from dlmisc.results import load_confusion_matrices
    >>> load_confusion_matrices('confusion_matrices.json')
    {'conf_mat_1': array([[1, 2], [3, 4]]), 'conf_mat_2': array([[5, 6], [7, 8]])}
    >>> load_confusion_matrices('confusion_matrices.json', filter=lambda key: key == 'conf_mat_1')
    {'conf_mat_1': array([[1, 2], [3, 4]])}
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f'{path.absolute()} is not a file.')

    with open(path, 'r') as f:
        conf_mats = json.load(f)

    for k, v in conf_mats.items():
        if not filter(k):
            del conf_mats[k]
        else:
            conf_mats[k] = np.array(v)
    
    return conf_mats
    
def reduce_confusion_matrices(conf_mats: Dict[str, np.ndarray], reduction_method: str = 'sum') -> np.ndarray:
    """Reduce a dictionary of confusion matrices to a single confusion matrix.
    Different reduction methods are available.

    Parameters
    ----------
    conf_mats : Dict[str, array-like]
        Dictionary containing the confusion matrices. Keys represent the
        name of the confusion matrix and the values are the confusion
        matrices.
    reduction_method : str, default = 'sum'
        Reduction method to use. Available methods are 'sum' and 'average'.

    Returns
    -------
    conf_mat : array-like
        Reduced confusion matrix.

    Examples
    --------
    >>> from dlmisc.results import reduce_confusion_matrices
    >>> conf_mats = {'conf_mat_1': [[1, 2], [3, 4]], 'conf_mat_2': [[5, 6], [7, 8]]}
    >>> reduce_confusion_matrices(conf_mats)
    array([[ 6,  8],
            [10, 12]])
    >>> reduce_confusion_matrices(conf_mats, reduction_method='average')
    array([[3, 4],
            [5, 6]])
    """

    if len(conf_mats) == 0:
        return np.ndarray([])

    shape = None

    for k, v in conf_mats.items():
        if shape is None:
            shape = v.shape
        elif shape != v.shape:
            raise ValueError(f'Confusion matrices must have the same shape. {shape} != {v.shape}')
    
    reduced_matrix = np.zeros(shape) #type: ignore

    for k, v in conf_mats.items():
        reduced_matrix += v

    if reduction_method == 'average':
        reduced_matrix /= len(conf_mats)

    return reduced_matrix

def plot_confusion_matrix(conf_mat: np.ndarray, output_path: Union[str, Path] = 'confusion_matrix.pdf', labels: List[str] = [], cmap: str = 'Blues'):
    """Plot a confusion matrix.

    Parameters
    ----------
    conf_mat : array-like
        Confusion matrix to plot.
    output_path : Union[str, Path], default = 'confusion_matrix.pdf'
        Path to save the confusion matrix plot to.
    labels : List[str], default = []
        List of labels to use for the confusion matrix. If no labels are
        provided, the labels are set to the indices of the confusion matrix.
    cmap : str, default = 'Blues'
        Seaborn colormap to use for the confusion matrix plot.

    Examples
    --------
    >>> from dlmisc.results import plot_confusion_matrix
    >>> plot_confusion_matrix([[1, 2], [3, 4]])
    """

    if len(conf_mat.shape) != 2:
        raise ValueError(f'Confusion matrix must have 2 dimensions. {conf_mat.shape} != (n, n)')

    if len(labels) == 0:
        labels = [str(i) for i in range(conf_mat.shape[0])]
    if len(labels) != conf_mat.shape[0]:
        raise ValueError(f'Number of labels must match number of classes. {len(labels)} != {conf_mat.shape[0]}')

    output_path = Path(output_path)
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise NotADirectoryError(f'{output_path.parent.absolute()} is not a directory.')
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    conf_mat = conf_mat.astype('int32')

    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap=cmap, ax=ax, xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={'size': 16}) #type: ignore
    fig.savefig(str(output_path))

    return fig, ax
