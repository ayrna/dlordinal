import os
from datetime import datetime
import pandas as pd
from pathlib import Path
from shutil import make_archive
from typing import Union, List, Dict, Optional, cast

def merge_results_dataframes(results_path: Union[str, Path]) -> pd.DataFrame:
    """Merge all results csv files in a folder into a single dataframe.
    The last column, that is usually an empty column, is dropped.
    The dataframe should have a column named 'name' that identifies the method.
    If the name starts with 'results_', it is removed.

    Parameters
    ----------
    results_path : Union[str, Path]
        Path to the folder containing the results dataframes.

    Returns
    -------
    merged_dataframe: pd.DataFrame
        Merged dataframe.

    Raises
    ------
    FileNotFoundError
        If the results_path does not exist or it does not contain any csv files.

    Example
    -------
    >>> from dlmisc.results import merge_results_dataframes
    >>> from pathlib import Path
    >>> results_path = Path('./results')
    >>> merged_df = merge_results_dataframes(results_path)
    """

    all_df = None
    results_path = Path(results_path)

    for path in sorted(os.listdir(results_path)):
        if path.endswith('.csv'):
            name = path.replace('.csv', '')
            df = pd.read_csv(results_path / path)
            df = df.iloc[:, :-1] # drop empty column
            df.insert(0, 'name', name.replace('results_', ''))
            all_df = df if all_df is None else pd.concat((all_df, df))

    if all_df is None:
        raise FileNotFoundError(f'No result files were found in "{results_path.absolute()}".')

    return all_df

def create_means_dataframe(all_df: pd.DataFrame, mean_columns: List[str] = ['name']) -> pd.DataFrame:
    """Create a dataframe with the mean values obtained when grouping by the mean_columns.

    Parameters
    ----------
    all_df : pd.DataFrame
        Dataframe with all the results.
    mean_columns : List[str], default = ['name']
        Columns to group by.

    Returns
    -------
    mean_df: pd.DataFrame
        Dataframe with the mean values.

    Raises
    ------
    TypeError
        If all_df is not a pandas dataframe.

    Example
    -------
    >>> from dlmisc.results import create_means_dataframe, merge_results_dataframes
    >>> from pathlib import Path
    >>> results_path = Path('./results')
    >>> all_df = merge_results_dataframes(results_path)
    >>> mean_df = create_means_dataframe(all_df)
    """

    if not isinstance(all_df, pd.DataFrame):
        raise TypeError(f'all_df must be a pandas dataframe')

    mean_df = all_df.drop(['seed', 'fold', 'k'], axis=1, errors='ignore').groupby(mean_columns).mean().reset_index(drop=False)
    return mean_df

def create_stds_dataframe(all_df: pd.DataFrame, std_columns: List[str] = ['name']) -> pd.DataFrame:
    """Create a dataframe with the standard deviation values obtained when grouping by the std_columns.

    Parameters
    ----------
    all_df : pd.DataFrame
        Dataframe with all the results.
    std_columns : List[str], default = ['name']
        Columns to group by.

    Returns
    -------
    std_df: pd.DataFrame
        Dataframe with the standard deviation values.

    Raises
    ------
    TypeError
        If all_df is not a pandas dataframe.

    Example
    -------
    >>> from dlmisc.results import create_stds_dataframe, merge_results_dataframes
    >>> from pathlib import Path
    >>> results_path = Path('./results')
    >>> all_df = merge_results_dataframes(results_path)
    >>> std_df = create_stds_dataframe(all_df)
    """

    if not isinstance(all_df, pd.DataFrame):
        raise TypeError(f'all_df must be a pandas dataframe')

    std_df = all_df.drop(['seed', 'fold', 'k'], axis=1, errors='ignore').groupby(std_columns).std().reset_index(drop=False)
    return std_df

def create_columns_dataframes(all_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create a dataframe for each metric in the dataframe.
    On each dataframe, the values of that metric are sorted in one column.

    Parameters
    ----------
    all_df : pd.DataFrame
        Dataframe with all the results.

    Returns
    -------
    metrics_dfs: Dict[str, pd.DataFrame]
        Dictionary with the dataframes for each metric.

    Raises
    ------
    TypeError
        If all_df is not a pandas dataframe.

    Example
    -------
    >>> from dlmisc.results import create_columns_dataframes, merge_results_dataframes
    >>> from pathlib import Path
    >>> results_path = Path('./results')
    >>> all_df = merge_results_dataframes(results_path)
    >>> metrics_dfs = create_columns_dataframes(all_df)
    """

    if not isinstance(all_df, pd.DataFrame):
        raise TypeError(f'all_df must be a pandas dataframe')

    # Get individual results in columns by metrics
    columns = all_df.columns
    method_columns = ['name']
    metrics_columns = [c for c in columns if c not in method_columns]

    methods = all_df[method_columns].drop_duplicates().reset_index(drop=True)
    metrics_dics = {metric : {} for metric in metrics_columns}

    for i, method in methods.iterrows():
        method_name = method['name']

        for metric in metrics_columns:
            metrics_dics[metric][method_name] = all_df[all_df['name'] == method_name][metric].to_list()

    metrics_dfs = {metric : pd.DataFrame(metrics_dics[metric]) for metric in metrics_columns}
    return metrics_dfs

def create_results_excel_summary(results_path: Union[str, Path] = Path('./results'), destination_path: Optional[Union[str, Path]] = None,
    mean_columns: List[str] = ['name'], destination_appendix: str = '') -> Path:
    """Create an excel file with the results of all the csv files in a folder.
    The excel file contains a sheet with the individual values, a sheet with the mean values and a sheet for each metric.
    The mean values are obtained by grouping by the mean_columns.
    Each sheet for each metric contains the values of that metric for each method in one column.

    Parameters
    ----------
    results_path : Union[str, Path], default = Path('./results')
        Path to the folder containing the results dataframes.
    destination_path : Optional[Union[str, Path]], default = None
        Path to the folder where the excel file will be saved.
        If None, the excel will be saved in a directory called 'joined_results'
        with the name 'all_results_{timestamp}.xlsx'.
    mean_columns : List[str], default = ['name']
        Columns to group by when obtaining the mean values.
    destination_appendix : str, default = ''
        String to append to the destination path.

    Returns
    -------
    excel_path: Path
        Path to the created excel file.

    Example
    -------
    >>> from dlmisc.results import create_results_excel_summary
    >>> from pathlib import Path
    >>> results_path = Path('./results')
    >>> excel_path = create_results_excel_summary(results_path)
    """

    if destination_path is None:
        destination_path = f"./joined_results/{datetime.today().strftime('%Y%m%d%H%M')}{destination_appendix}.xlsx"

    results_path = Path(results_path)
    destination_path = Path(destination_path)

    all_df = merge_results_dataframes(results_path)
    mean_df = create_means_dataframe(all_df, mean_columns=mean_columns)
    std_df = create_stds_dataframe(all_df, std_columns=mean_columns)
    metrics_dfs = create_columns_dataframes(all_df)

    os.makedirs(destination_path.parent, exist_ok=True)

    with pd.ExcelWriter(destination_path) as writer:
        all_df.to_excel(writer, sheet_name="Individual", index=False)
        mean_df.to_excel(writer, sheet_name="Average", index=False)
        std_df.to_excel(writer, sheet_name="Std", index=False)
        for metric, metric_df in metrics_dfs.items():
            metric_df.to_excel(writer, sheet_name=f"Columns-{metric}", index=False)

    return cast(Path, destination_path)

def create_results_zip(results_path: Union[str, Path] = Path('./results'), destination_path: Optional[Union[str, Path]] = None,
    destination_appendix: str = '') -> Path:
    """Create a zip file with the results of all the csv files in a folder.

    Parameters
    ----------
    results_path : Union[str, Path], default = Path('./results')
        Path to the folder containing the results dataframes.
    destination_path : Union[str, Path], default = None
        Path to the folder where the zip file will be saved.
        If None, the zip will be saved in a directory called 'joined_results'
        with the name 'all_results_{timestamp}.zip'.
    destination_appendix : str, default = ''
        String to append to the destination path.

    Returns
    -------
    zip_path: Path
        Path to the created zip file.

    Example
    -------
    >>> from dlmisc.results import create_results_zip
    >>> from pathlib import Path
    >>> results_path = Path('./results')
    >>> zip_path = create_results_zip(results_path)
    """

    if destination_path is None:
        destination_path = f"./joined_results/{datetime.today().strftime('%Y%m%d%H%M')}{destination_appendix}"

    destination_path = Path(destination_path)

    os.makedirs(destination_path.parent, exist_ok=True)
    make_archive(str(destination_path), 'zip', results_path)

    return Path(f'{destination_path}.zip')

def extract_data_as_column(df: pd.DataFrame, origin_column: str, new_column_name: str, new_column_regex: str) -> pd.DataFrame:
    """Extract data from a column and add it as a new column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to extract the data from.
    origin_column : str
        Name of the column to extract the data from.
    new_column_name : str
        Name of the new column.
    new_column_regex : str
        Regex to extract the data from the origin column.

    Raises
    ------
    TypeError
        If df is not a pandas dataframe.

    Example
    -------
    >>> from dlmisc.results import extract_data_as_column
    >>> import pandas as pd
    >>> df = pd.DataFrame({'name': ['model1', 'model2'], 'path': ['model1/1', 'model2/2']})
    >>> extract_data_as_column(df, 'path', 'id', r'(\\d+)')
    >>> df
        name   path   id
    0   model1  model1/1   1
    1   model2  model2/2   2
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'df must be a pandas dataframe')

    df[new_column_name] = df[origin_column].str.extract(new_column_regex)

    return df

def compute_ranks(df: pd.DataFrame, columns: Dict[str, bool]) -> pd.DataFrame:
    """Compute the ranks of the values in the columns of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to compute the ranks.
    columns : dict[str, bool]
        Columns to compute the ranks. The key is the name of the column
        and the value is a boolean indicating if the values are ascending or not.

    Raises
    ------
    TypeError
        If df is not a pandas dataframe.

    Example
    -------
    >>> from dlmisc.results import compute_ranks
    >>> import pandas as pd
    >>> df = pd.DataFrame({'name': ['model1', 'model2'], 'metric1': [0.5, 0.3], 'metric2': [0.2, 0.4]})
    >>> compute_ranks(df, {'metric1': True, 'metric2': False})
    >>> df
        name   metric1   metric2   rank_metric1   rank_metric2
    0   model1  0.5     0.2     1               1
    1   model2  0.3     0.4     2               2
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'df must be a pandas dataframe')

    for column, ascending in columns.items():
        df[f'{column}_rank'] = df[column].rank(ascending=ascending)

    return df

def create_results_excel_with_methods_in_columns(results_path: Union[str, Path], sheet_name: str, destination_path: Union[str, Path], method_column_regex: str, metrics: List[str] = ['QWK', 'CCR', 'MAE', 'MS', '1-off']):
    """Create an excel file with the results in a sheet of other excel file but sorting the methods in columns.
    The excel file contains one sheet for each metric considered and each column is associated with one method.
    Each column contains the results of that method for each dataset and seed (always in the same order).

    Parameters
    ----------
    results_path : Union[str, Path]
        Path to the excel file containing the results.
    sheet_name : str
        Name of the sheet in the excel file containing the results.
    destination_path : Union[str, Path]
        Path to the excel file where the results will be saved.
    metrics : List[str], default = ['QWK', 'CCR', 'MAE', 'MS', '1-off']
        List of metrics to consider.

    Raises
    ------
    TypeError
        If results_path is not a string or a Path.
        If destination_path is not a string or a Path.
        If metrics is not a list.

    Example
    -------
    >>> from dlmisc.results import create_results_excel_with_methods_in_columns
    >>> create_results_excel_with_methods_in_columns('results.xlsx', 'Results', 'results_with_methods_in_columns.xlsx')
    """

    if not isinstance(results_path, (str, Path)):
        raise TypeError(f'results_path must be a string or a Path')
    if not isinstance(destination_path, (str, Path)):
        raise TypeError(f'destination_path must be a string or a Path')
    if not isinstance(metrics, list):
        raise TypeError(f'metrics must be a list')

    results_path = Path(results_path)
    destination_path = Path(destination_path)

    results = pd.read_excel(results_path, sheet_name=sheet_name)
    results = extract_data_as_column(results, 'name', 'method', method_column_regex)
    methods = results['method'].unique()

    with pd.ExcelWriter(destination_path) as writer:
        for metric in metrics:
            df_columns = {}
            for method in methods:
                df_columns[method] = results[(results['method'] == method)][metric].values
            df = pd.DataFrame(df_columns)
            df.to_excel(writer, sheet_name=metric, index=False)

