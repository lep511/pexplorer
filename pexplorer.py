import pandas as __pd
import numpy as __np
import matplotlib.pyplot as __plt
import scipy.stats as __stats
import seaborn as __sns
import sys as __sys
import re as __re
import math as __math
import warnings as __warnings

__warnings.filterwarnings("ignore")


def check_df(dataframe):
    """
    Analyze an dataframe to display a summary of statistics.

    Args:
        dataframe: Pandas dataframe
    """

    df = dataframe.copy().reset_index()

    df_drop = df.dropna()
    is_num = df_drop.apply(
        lambda s: __pd.to_numeric(s, errors="coerce").notnull().all()
    )

    col_names, v_type, count_uniq, v_null = [], [], [], []
    val_max, val_min, val_mean, val_std = [], [], [], []
    val_25, val_50, val_75, binary_val = [], [], [], []

    len_df = len(df)

    for feature in df.columns:
        col_names.append(feature)
        v_type.append(str(df[feature].dtype))

        try:
            count_uniq.append(len(df[feature].unique()))
        except:
            count_uniq.append(0)
        val_n = df[feature].isnull().sum() / len_df
        v_null.append(val_n)
        is_bool = __pd.api.types.is_bool_dtype(df[feature])
        is_null = len(df[feature].dropna())

        if is_num[feature] and not is_bool and is_null > 0:

            try:
                max_v = df[feature].max()
                min_v = df[feature].min()
                mea_v = df[feature].mean()
                std_v = df[feature].std()
                p25_v = __np.nanpercentile(df[feature], 25)
                p50_v = __np.nanpercentile(df[feature], 50)
                p75_v = __np.nanpercentile(df[feature], 75)
            except:
                max_v, min_v, mea_v, std_v = "-", "-", "-", "-"
                p25_v, p50_v, p75_v = "-", "-", "-"
            val_max.append(max_v)
            val_min.append(min_v)
            val_mean.append(mea_v)
            val_std.append(std_v)
            val_25.append(p25_v)
            val_50.append(p50_v)
            val_75.append(p75_v)

            val_unic_temp = df[feature].unique().tolist()
            bi_temp = "-"

            if len(val_unic_temp) == 2:

                bin_lst = df[feature].dropna().unique().tolist()

                if len(bin_lst) == 2:
                    bin_1 = str(bin_lst[0])
                    bin_2 = str(bin_lst[1])
                    bi_temp = bin_1.strip() + "/" + bin_2.strip()
            binary_val.append(bi_temp)
        elif is_bool:
            val_max.append("-")
            val_min.append("-")
            val_mean.append("-")
            val_std.append("-")
            val_25.append("-")
            val_50.append("-")
            val_75.append("-")
            binary_val.append("True/False")
        else:
            val_max.append("-")
            val_min.append("-")
            val_mean.append("-")
            val_std.append("-")
            val_25.append("-")
            val_50.append("-")
            val_75.append("-")

            val_unic_temp = df[feature].unique().tolist()
            bi_temp = "-"

            if len(val_unic_temp) < 4:

                bin_lst = df[feature].dropna().unique().tolist()

                if len(bin_lst) == 2:
                    bin_1 = str(bin_lst[0])
                    bin_2 = str(bin_lst[1])
                    bi_temp = bin_1.strip() + "/" + bin_2.strip()
            binary_val.append(bi_temp)
    nr_col = range(len(col_names))
    ran_col = []

    for num in nr_col:
        ran_col.append(str(num + 1) + ")")
    data = {
        "Num.": ran_col,
        "col. name": col_names,
        "type": v_type,
        "unique": count_uniq,
        "NAN(%)": v_null,
        "min": val_min,
        "max": val_max,
        "mean": val_mean,
        "std": val_std,
        "25%": val_25,
        "50%": val_50,
        "75%": val_75,
        "binary values": binary_val,
    }

    df_new = __pd.DataFrame(data=data)
    df_new = df_new.set_index("Num.")

    cm = __sns.light_palette("green", as_cmap=True)
    text_indx = str(__pd.RangeIndex(df.index))

    return df_new.style.background_gradient(cmap=cm).set_caption(text_indx)


def col_rename(dataframe, inplace=False):
    """
    Change all column names to lowercase,
    remove whitespace for _, do the same with . , :

    Args:
        dataframe: Pandas dataframe
        inplace (bool, optional): [description]. Defaults to False.

    Returns:
        col_newname [list]: New column names.
    """
    col_newname = dict()

    for elem in dataframe.columns:

        col_n = elem
        if not col_n.isupper():
            col_n = "_".join(__re.findall("[A-Z][a-z][a-z]*", col_n))
        col_n = col_n.lower()
        col_n = col_n.strip()
        col_n = col_n.replace(",", "_")
        col_n = col_n.replace(".", "_")
        col_n = col_n.replace(":", "_")
        col_n = col_n.replace("-", "_")
        col_n = col_n.replace(" ", "_")
        col_n = col_n.replace("(", "")
        col_n = col_n.replace(")", "")
        col_newname[elem] = col_n
    if inplace:
        return dataframe.rename(columns=col_newname, inplace=True)
    else:
        return col_newname


def make_cat(dataframe, percent=5, exclude=[]):
    """
    Convert columns that meet certain requirements to categorical.

    Args:
        dataframe: Pandas dataframe
        percent (int, optional): % tolerance. Defaults to 5.
        exclude (list, optional): Columns to exclude. Defaults to [].

    Returns:
        dataframe: Pandas dataframe
    """

    n_df = dataframe.copy()
    is_num = n_df.apply(lambda s: __pd.to_numeric(s, errors="coerce").notnull().all())
    cols = []
    percent = percent / 100

    for feature in n_df.columns:

        if not is_num[feature]:

            data_col = n_df[feature].dropna()
            data_num = len(data_col.unique())

            if data_num == 0 or len(data_col) == 0:
                data_porc = 1
            else:
                data_porc = data_num / len(data_col)
            if data_porc < percent and feature not in exclude:

                try:
                    n_df[feature].fillna("UNKNOWN", inplace=True)
                    n_df[feature] = n_df[feature].astype("category")
                    cols.append(feature)
                except:
                    print("{} no se pudo convertir.".format(feature))
                    pass
    n_str = "\n"
    for d in cols:
        val_uniq = len(n_df[d].unique())
        n_str = n_str + d + " (" + str(val_uniq) + ")\n"
    if len(cols) > 0:

        print(
            "Las columnas convertidas: \
            \n-------------------------{cols}".format(
                cols=n_str
            )
        )

        return n_df
    return dataframe


def glimpse(dataframe):
    """
    Similar to R glimpse function. Shows the name of each column,
    with its dtype and unique values.

    Args:
        dataframe: Pandas dataframe
    """
    rows_n = f"{dataframe.shape[0]:,}"
    print("Rows: {}".format(rows_n))
    print("Columns: {}".format(dataframe.shape[1]))
    glimpse_df = dataframe.apply(lambda x: [x.dtype, x.unique()[0:12]]).T
    glimpse_df.columns = ["dtype", "sample values"]
    return glimpse_df


def check_cat(dataframe, percent=5):

    n_df = dataframe.copy()
    is_num = n_df.apply(lambda s: __pd.to_numeric(s, errors="coerce").notnull().all())
    cols = []
    percent = percent / 100

    for feature in n_df.columns:

        if not is_num[feature]:

            data_col = n_df[feature].dropna()
            data_num = len(data_col.unique())

            if data_num == 0 or len(data_col) == 0:
                data_porc = 1
            else:
                data_porc = data_num / len(data_col)
            if data_porc < percent:
                cols.append(feature)
    n_str = "\n"
    for d in cols:
        val_uniq = len(n_df[d].unique())
        n_str = n_str + d + "(" + str(val_uniq) + ")\n"
    if len(cols) > 0:

        return print(
            "Las columnas recomendadas: \
            \n--------------------------{cols}".format(
                cols=n_str
            )
        )
    return print("No hay columnas recomendadas para convertir.")


def clean_nan(dataframe, percent=0.9):

    n_df = dataframe.copy()
    cols = []
    n_str = "\n"

    for feature in n_df.columns:

        c_null = n_df[feature].isnull().sum()
        c_len = len(n_df[feature])

        if c_null != 0 and c_len != 0:

            n_perc = c_null / c_len

            if n_perc >= percent:
                n_df.drop(feature, axis="columns", inplace=True)
                cols.append(feature)
    if len(cols) > 0:

        for d in cols:
            n_str = n_str + d + "\n"
        print(
            "Se eliminaron las sig. columnas: \
            \n-------------------------{cols}".format(
                cols=n_str
            )
        )
    else:

        print("No se eliminó ninguna columna.")
        return dataframe
    return n_df


def any_value(dataframe, search):
    """
    Performs a search for a value in the entire dataframe.

    Args:
        dataframe: Pandas dataframe
        search: item to search

    Returns:
        dataframe: dataframe filtered with search
    """
    return dataframe[dataframe.eq(search).any(1)]


def missing_values(dataframe):
    """
    Check all columns and return the number of null values.

    Args:
        dataframe: Pandas dataframe

    Returns:
        dataframe: table with the number of null values.

    credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
    """
    # Total missing values
    mis_val = dataframe.isnull().sum()

    # Percentage of missing values
    mis_val_percent = mis_val / len(dataframe)

    # Make a table with the results
    mis_val_table = __pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = (
        mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(2)
    )

    # Print some summary information
    print(
        "Your selected dataframe has " + str(dataframe.shape[1]) + " columns.\n"
        "There are "
        + str(mis_val_table_ren_columns.shape[0])
        + " columns that have missing values."
    )

    # Return the dataframe with missing information
    if mis_val_table_ren_columns.shape[0] != 0:
        return mis_val_table_ren_columns.style.background_gradient(
            cmap="YlOrBr", subset=["Missing Values"]
        )
    return


def missing_graph(dataframe):
    """
    Check all columns and return a graph of null values.

    Args:
        dataframe: Pandas dataframe

    Returns:
    """
    lencol = len(dataframe.columns)
    if lencol < 4:
        lencol = 3
    else:
        lencol = lencol / 1.2
    __plt.figure(figsize=(lencol,8))
    __sns.heatmap(dataframe.isna(), cmap='Greys', cbar=False)
    __plt.xticks(rotation=60)
    return __plt.show()


def split_values(dataframe):
    """
    Returns two dataframes, one with only numeric values
    and the other with only categorical values.

    Args:
        dataframe: Pandas dataframe

    Returns:
        tuple: two dataframes.

    Example:

    >>> data_num, data_cat = split_values(df)
    """

    data_num = dataframe.select_dtypes(include=[__np.number])
    data_cat = dataframe.select_dtypes(include=[__np.object, "category"])

    return data_num, data_cat


def __conv_bytes(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(__math.floor(__math.log(size_bytes, 1024)))
    p = __math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "{} {}".format(s, size_name[i])


def memory_size(dataframe):
    """
    Returns the size that the dataframe occupies in memory.

    Args:
        dataframe: Pandas dataframe

    Returns:
        info (str): size in MB.
    """
    size_bytes = dataframe.memory_usage().sum()
    total_mem = __conv_bytes(size_bytes)
    d_mem = dataframe.memory_usage().to_frame()
    
    print("Total memory usage: {}".format(total_mem))
    return d_mem[0].apply(lambda x: __conv_bytes(x)).to_frame().rename(columns={0: "Memory usage"})
    
    
def correlation_all(dataframe):
    """
    Plotting a diagonal correlation matrix

    Args:
        dataframe: Pandas dataframe
    """
    __sns.set_theme(style="white")
    # Compute the correlation matrix
    corr = dataframe.corr()

    # Generate a mask for the upper triangle
    mask = __np.triu(__np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = __plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = __sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    __sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    return __plt.show()


def correlation_col(dataframe, col_name):
    """
    Plot a correlation matrix based on a column. 

    Args:
        dataframe: Pandas dataframe
    """
    if not col_name in dataframe.columns.to_list():
        print(f"Column {col_name} is not found in the dataframe.")
        return
    lencols = len(dataframe.corr()[[col_name]])
    __plt.figure(figsize=(4, lencols * 1.6))
    df_c = dataframe.dropna().corr()
    heatmap = __sns.heatmap(df_c[[col_name]].sort_values(by=col_name, ascending=False)[1:], 
                                        vmin=-1, 
                                        vmax=1, 
                                        annot=True,
                                        fmt='.2%',
                                        cmap='BrBG'
    )
    heatmap.set_title(f'Features Correlating with {col_name}', fontdict={'fontsize':16}, pad=14)
    __plt.yticks(rotation=30)
    return __plt.show()


def grid_plot(dataframe, hue=None, save=False):
    """
    Trazado de cuadrículas estructuradas de varios gráficos

    Parámetros
    ----------

    `dataframe:` (Pandas dataframe)
    `hue:` (row name) para graficar diferentes niveles con diferentes colores
    `save:` (nombre del archivo) si no se especifica no se guarda en el disco
    """
    n_df = dataframe.copy()
    col_names = n_df.columns.tolist()

    for elem in col_names:

        # Check if column is bool and change to numpy.uint8
        if n_df[elem].dtype == __np.bool:
            n_df[elem] = n_df[elem].astype(__np.uint8)
    print("Aguarde un momento...", end="")
    __sys.stdout.flush()
    __sns.set_style("darkgrid")
    g = __sns.PairGrid(n_df, hue=hue, height=4)
    g.map_diag(__sns.histplot, multiple="stack", element="step")
    g.map_offdiag(__sns.scatterplot)
    g.add_legend()

    if save:
        g.savefig(save, dpi=300)
    print("\r ")

    return __plt.show()


def plot_numcat(dataframe, numeric_col, categoric_col):
    """
    Trazado de un gráfico para una variable numérica y otra categórica

    Parámetros
    ----------

    `dataframe` : (Pandas dataframe)
    `numeric_row` : variable numérica del dataframe
    `categoric_row` : variable categórica del dataframe
    """
    __plt.style.use("seaborn-whitegrid")
    fig = __plt.figure(figsize=(18, 4))
    __plt.subplot(1, 2, 1)
    __sns.countplot(y=categoric_col, data=dataframe)
    __plt.subplot(1, 2, 2)
    __sns.histplot(dataframe[numeric_col])
    __plt.xticks(rotation=90)
    return __plt.show()


def plot_distribution_col(dataframe, col=None, rnd=2, hue=None):

    if col == None:
        data_num = dataframe.select_dtypes(include=[__np.number])
    else:
        data_num = dataframe[col].to_frame().select_dtypes(include=[__np.number])
    if len(data_num.columns) == 0:
        print("The dataframe or column does not have numeric values.")
        return
    for c in data_num.columns:

        info_num(dataframe=dataframe, col_sel=c, rnd=rnd)

        df_notnan = dataframe.dropna()
        val_log = __np.log10(df_notnan[c].abs())
        val_log = val_log[__np.isfinite]
        iq_25 = __np.quantile(dataframe[col].dropna(), 0.25)
        iq_75 = __np.quantile(dataframe[col].dropna(), 0.75)

        __plt.style.use("seaborn-whitegrid")
        fig = __plt.figure(figsize=(18, 7))
        __plt.subplot(1, 2, 1)
        __sns.histplot(
            data=df_notnan, x=c, palette="light:m_r", edgecolor=".3", linewidth=0.5
        )
        __plt.gca().axvspan(iq_25, iq_75, alpha=0.1, color='green')
        __plt.xlabel("")
        __plt.subplot(1, 2, 2)
        __sns.histplot(
            val_log,
            hue=hue,
            kde=True,
            palette="light:m_r",
            edgecolor=".3",
            linewidth=0.5,
            multiple="stack",
        )
        __plt.suptitle(c)
        __plt.xlabel("(logarithm)")
        __plt.show()
    return


def info_num(dataframe, col_sel=None, rnd=2):

    data_num = dataframe.select_dtypes(include=[__np.number])

    if len(data_num.columns) == 0:
        print("The dataframe has no numeric values.")
        return
    if col_sel:
        if not __check_num(dataframe, col_sel):
            print(
                "Column {} not found in dataframe or it is not numeric.".format(col_sel)
            )
            return
        count_v = __np.sum(dataframe[col_sel].notna())
        min_v = __np.round(dataframe[col_sel].min(), rnd)
        max_v = __np.round(dataframe[col_sel].max(), rnd)
        mean_v = __np.round(dataframe[col_sel].mean(), rnd)
        med_v = __np.round(dataframe[col_sel].median(), rnd)
        std_n = __np.round(dataframe[col_sel].std(), rnd)
        var_n = __np.round(dataframe[col_sel].var(), rnd)
        skew = __np.round(dataframe[col_sel].skew(), rnd)
        kurto = __np.round(dataframe[col_sel].kurt(), rnd)

        print("====", col_sel, "====")
        print("Count: {}".format(count_v))
        print("Minimum: {}".format(min_v))
        print("Maximum: {}".format(max_v))
        print("Average: {}".format(mean_v))
        print("Median: {}".format(med_v))
        print("Standard Deviation (dispersion): {}".format(std_n))
        print("Variance (spread out): {}".format(var_n))
        print("Skewness (distortion or asymmetry): {}".format(skew))
        print("Kurtosis (peakedness of a distribution): {}".format(kurto))
        print("")
    else:
        data_num = dataframe.select_dtypes(include=[__np.number])
        df_stats = __np.round(
            data_num.apply(
                lambda x: [
                    x.count(),
                    x.min(),
                    x.max(),
                    x.mean(),
                    x.median(),
                    x.std(),
                    x.var(),
                    x.skew(),
                    x.kurt(),
                ]
            ).T,
            2,
        )

        df_stats.columns = [
            "Count",
            "Minimum",
            "Maximum",
            "Average",
            "Median",
            "Std. Dev.",
            "Variance",
            "Skewness",
            "Kurtosis",
        ]
        return df_stats


def normalize_row(dataframe):
    """
    Normalizes the values of a given dataframe
    by the total sum of each line.

    Args:
        dataframe: Pandas dataframe

    Returns:
        dataframe: new dataframe with normalize values
    """
    # credit: https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value
    df_c = dataframe.copy()
    df_n = df_c._get_numeric_data().div(df_c.sum(axis=1), axis=0)
    for col in df_n.columns:
        df_c[col] = df_n[col]
    return df_c


def normalize_column(dataframe, percent=True):
    """
    Normalizes the values of a given dataframe by the total sum
    of each column.

    Args:
        dataframe: Pandas dataframe
        percent (bool): If percent=True (default), multiplies the final
                        value by 100.

    Returns:
        dataframe: new dataframe with normalize values
    """
    # credit: https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value
    df_c = dataframe.copy()
    df_n = df_c._get_numeric_data()

    if percent:
        df_n = df_c._get_numeric_data().div(dataframe.sum(axis=0), axis=1) * 100
    else:
        df_n = df_c._get_numeric_data().div(dataframe.sum(axis=0), axis=1)
    for col in df_n.columns:
        df_c[col] = df_n[col]
    return df_c


def outliers(dataframe, silent=False, n_round=2):
    """
    Grubbs' test, also called the ESD method (extreme studentized deviate),
    to determine if any of the columns contain outliers.

    Args:
        dataframe (Pandas.dataframe)
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        list: List of columns with outliers values
    """
    out_list = []
    title = 0
    col_dat = dataframe._get_numeric_data().columns.to_list()
    col_num = []

    for c in col_dat:
        if len(dataframe[c].dropna().unique()) > 2:
            col_num.append(c)
    for c in col_num:
        col = dataframe[c]
        n = len(col)
        mean_x = __np.mean(col)
        sd_x = __np.std(col)
        numerator = max(abs(col - mean_x))
        g_calculated = numerator / sd_x
        t_value = __stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
        g_ca = (n - 1) * __np.sqrt(__np.square(t_value))
        g_cb = __np.sqrt(n) * __np.sqrt(n - 2 + __np.square(t_value))
        g_critical = g_ca / g_cb

        if g_critical < g_calculated:
            out_list.append(c)
            if not silent:
                if title == 0:
                    print("=== Grubbs' test ===")
                    title = 1
                print(c)
    # Tukey's method
    # credits: https://gist.github.com/alice203

    title = 0
    outliers_prob = []

    for c in col_num:
        q1 = dataframe[c].quantile(0.25)
        q3 = dataframe[c].quantile(0.75)
        log_vals = __np.log(__np.abs(dataframe[c]) + 1)
        q1_l = __np.quantile(log_vals, 0.25)
        q3_l = __np.quantile(log_vals, 0.75)

        iqr = q3 - q1
        outer_fence = 3 * iqr

        iqr_l = q3_l - q1_l
        outer_fence_l = 3 * iqr_l

        # outer fence lower and upper end
        outer_fence_le = q1 - outer_fence
        outer_fence_ue = q3 + outer_fence

        outer_fence_le_l = q1_l - outer_fence_l
        outer_fence_ue_l = q3_l + outer_fence_l

        for index, x in enumerate(dataframe[c]):

            if x <= outer_fence_le or x >= outer_fence_ue:
                outliers_prob.append(__np.round(x, n_round))
        filter_l = __np.where(
            (log_vals <= outer_fence_le_l) | (log_vals >= outer_fence_ue_l), True, False
        )
        log_result = dataframe[c][filter_l].round(n_round)

        if outliers_prob != []:

            if not silent:

                if title == 0:
                    print("\n", "=== Tukey's method  ===")
                    title = 1
                print(c, list(set(outliers_prob)))

                if len(log_result) != 0:
                    print(c, "(log)", list(set(log_result)))
                print(" ")
            out_list.append(c)
            outliers_prob = []
    # Z-Score method
    title = 0
    for c in col_num:
        col = dataframe[c]
        z_score = __stats.zscore(col, nan_policy="omit")
        z_filter = __np.abs(z_score) > 3
        if __np.sum(z_filter) != 0:
            if not silent:
                if title == 0:
                    print("\n", "=== Z-Score method  ===")
                    title = 1
                col_round = __np.round(col[z_filter], n_round)
                print(c, list(set(col_round)))
                print(" ")
            out_list.append(c)
    out_list = list(set(out_list))

    if out_list:
        return out_list
    else:
        if not silent:
            print("Outliers not found in dataframe")
        return


def outliers_graph(dataframe):
    """
    Plots the outliers found in the data frame.

    Args:
        dataframe: Pandas dataframe
    """
    cols = outliers(dataframe, silent=True)
    if not cols:
        print("Outliers not found in dataframe")
        return
    if len(cols) < 4:
        lencol = 3
    else:
        lencol = len(cols) / 1.8
    vals_norm = normalize_column(dataframe[cols])
    __plt.figure(figsize=(14, lencol))
    __sns.set_style("whitegrid")
    __sns.set(font_scale=1.1)
    ax = __sns.boxplot(data=vals_norm, orient="h", palette="Set2")
    ax.set(xlabel="normalized values")
    __plt.title("Outliers Found")
    return __plt.show()


def plot_distribution(dataframe, norm=True, exclude=None):
    df_c = dataframe
    df_n = df_c._get_numeric_data()
    df_n = df_n.drop(columns=exclude) if exclude else df_n
    cols = df_n.columns.to_list()

    if len(cols) == 0:
        print("Dataframe has no numeric values.")
        return
    elif len(cols) < 4:
        lencol = 3
    else:
        lencol = len(cols) / 1.8
    if norm:
        vals_norm = normalize_column(df_n)
        text_n = "normalized values"
    else:
        vals_norm = df_n
        text_n = ""
    if len(cols) < 4:
        lencol = 3
    else:
        lencol = len(cols) / 1.8
    __plt.figure(figsize=(14, lencol))
    __sns.set_style("whitegrid")
    __sns.set(font_scale=1.1)
    ax = __sns.boxplot(data=vals_norm, orient="h", palette="Set2")
    ax.set(xlabel=text_n)
    __plt.title("Data Distribution")
    return __plt.show()


def __check_num(dataframe, col):
    """
    Check if a column is numeric.

    Args:
        dataframe: Pandas dataframe
        col (str): column name

    Returns:
        [bool]: True if it is numeric or false if it is not.
    """
    try:
        if __np.issubdtype(dataframe[col], __np.number):
            return True
        else:
            return False
    except:
        return False
