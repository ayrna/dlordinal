import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat

def anova_tukey_oneway(results_path, metric_name):
    sheet_name = f'Columns-Test-{metric_name}'

    # Get data
    df = pd.read_excel(results_path, sheet_name=sheet_name)
    df_melt = pd.melt(df.reset_index(), value_vars=df.columns)
    df_melt.columns = ['Method', metric_name]

    # Plot methods in boxplot
    ax = sns.boxplot(x='Method', y=metric_name, data=df_melt, color='#99c2a2')
    ax = sns.swarmplot(x="Method", y=metric_name, data=df_melt, color='#7d0013')
    plt.show()

    # Kolmogorov-Smirnov test
    print('\nKolmogorov-Smirnov test\n=================')
    for method in df_melt['Method'].unique():
        statistic, pvalue = stats.kstest(df_melt[df_melt["Method"] == method][metric_name], stats.norm.cdf)
        print(f'{method}: {statistic=}, {pvalue=}')
        
    # Anderson-Darling test
    print('\nAnderson-Darling test\n=================')
    for method in df_melt['Method'].unique():
        statistic, critical_values, significance_level = stats.anderson(df_melt[df_melt["Method"] == method][metric_name], dist='norm')
        print(f'{method}: {statistic=}, {critical_values=}, {significance_level=}')

    # ANOVA I test
    print('\nANOVA I test\n=================')
    model = ols(f'{metric_name} ~ C(Method)', data=df_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Tukey posthoc test
    print('\nTukey posthoc test\n=================')
    res = stat()
    res.tukey_hsd(df=df_melt, res_var=metric_name, xfac_var='Method', anova_model=f'{metric_name} ~ C(Method)')
    print(res.tukey_summary)