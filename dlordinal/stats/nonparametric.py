from itertools import combinations
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman


def perform_nonparametric_analysis(results_path, metric_name):
    sheet_name = f'Columns-Test-{metric_name}'

    # Get data
    df = pd.read_excel(results_path, sheet_name=sheet_name)
    df_melt = pd.melt(df.reset_index(), value_vars=df.columns)
    df_melt.columns = ['Method', metric_name]

    # Plot methods in boxplot
    ax = sns.boxplot(x='Method', y=metric_name, data=df_melt, color='#99c2a2')
    ax = sns.swarmplot(x="Method", y=metric_name, data=df_melt, color='#7d0013')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{metric_name}.pdf')
    plt.show()
    plt.close()

    # Kruskal-Wallis test
    print('\nKruskal-Wallis test\n=================')
    statistic, pvalue = stats.kruskal(*[group[metric_name].values for name, group in df_melt.groupby('Method')])
    print(f'{statistic=}, {pvalue=}')

    # Friedman test
    print('\nFriedman test\n=================')
    statistic, pvalue = stats.friedmanchisquare(*[group[metric_name].values for name, group in df_melt.groupby('Method')])
    print(f'{statistic=}, {pvalue=}')

    # Mann-Whitney U test
    print('\nMann-Whitney U test\n=================')
    for method1, method2 in combinations(df_melt['Method'].unique(), 2):
        statistic, pvalue = stats.mannwhitneyu(df_melt[df_melt["Method"] == method1][metric_name],
                                               df_melt[df_melt["Method"] == method2][metric_name])
        print(f'{method1} vs {method2}: {statistic=}, {pvalue=}')

    # Wilcoxon signed-rank test
    print('\nWilcoxon signed-rank test\n=================')
    for method1, method2 in combinations(df_melt['Method'].unique(), 2):
        statistic, pvalue = stats.wilcoxon(df_melt[df_melt["Method"] == method1][metric_name],
                                           df_melt[df_melt["Method"] == method2][metric_name])
        print(f'{method1} vs {method2}: {statistic=}, {pvalue=}')

    # Nemenyi posthoc test
    print('\nNemenyi posthoc test\n=================')
    res = posthoc_nemenyi_friedman(df)
    print(res)