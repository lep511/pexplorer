import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import sys
import re
import math
import warnings
warnings.filterwarnings("ignore")


def check_df(dataframe):
    
    df = dataframe.copy()
    
    df_drop = df.dropna()
    is_num = df_drop.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
    
    col_names, v_type, count_uniq, v_null = [], [], [], []
    val_max, val_min, val_mean, val_std  = [], [], [], []
    val_25, val_50, val_75, binary_val = [], [], [], []
    
    len_df = len(df)
    
    for feature in df.columns:
        col_names.append(feature)
        v_type.append(str(df[feature].dtype))
        
        count_uniq.append(len(df[feature].unique()))
        val_n = df[feature].isnull().sum() / len_df
        v_null.append(val_n)
        is_bool = pd.api.types.is_bool_dtype(df[feature])
        is_null = len(df[feature].dropna())
    
        if is_num[feature] and not is_bool and is_null > 0:
            
            try:
                max_v = df[feature].max()
                min_v = df[feature].min()
                mea_v = df[feature].mean()
                std_v = df[feature].std()
                p25_v = np.nanpercentile(df[feature], 25)
                p50_v = np.nanpercentile(df[feature], 50)
                p75_v = np.nanpercentile(df[feature], 75)
                      
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
        ran_col.append(str(num+1) + ")")
    
    data = {"Num.": ran_col,
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
        "binary values": binary_val
    }

    df_new = pd.DataFrame(data=data)
    df_new = df_new.set_index('Num.')
    
    cm = sns.light_palette("green", as_cmap=True)
    text_indx = str(pd.RangeIndex(df.index))
    
    return df_new.style.background_gradient(cmap=cm).set_caption(text_indx)


def col_rename(dataframe):
    """
    Cambia todos los nombres de las columnas
    a minúsculas, elimina espacios en blancos
    por _, lo mismo hace con . , :
    """
    col_newname = []
    
    for elem in dataframe.columns:
        
        col_n = elem
        col_n = '_'.join(re.findall('[A-Z][a-z]*', col_n))
        col_n = col_n.lower()
        col_n = col_n.strip()
        col_n = col_n.replace(",", "_")
        col_n = col_n.replace(".", "_")
        col_n = col_n.replace(":", "_")
        col_n = col_n.replace("-", "_")
        col_n = col_n.replace(" ", "_")
        col_n = col_n.replace("(", "")
        col_n = col_n.replace(")", "")     
        col_newname.append(col_n)
     
    return col_newname



def make_cat(dataframe, percent=5, exclude=[]):
    
    n_df = dataframe.copy()
    is_num = n_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
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
                    n_df[feature] = n_df[feature].astype('category')
                    cols.append(feature)
                        
                except:                
                    print("{} no se pudo convertir.").format(feature)
                    pass
                            
    n_str = "\n"            
    for d in cols:
        val_uniq = len(n_df[d].unique())
        n_str = n_str + d + " (" + str(val_uniq) + ")\n"
        
    if len(cols) > 0:
        
        print("Las columnas convertidas: \
            \n-------------------------{cols}".format(cols=n_str))
        
        return n_df
    
    return dataframe


def glimpse(dataframe):
    """
    Similar a la función glimpse de R. Muestra el nombre de cada
    columna, con su dtype y los valores únicos.
    """
    rows_n = f"{dataframe.shape[0]:,}"
    print("Rows: {}".format(rows_n))
    print("Columns: {}".format(dataframe.shape[1]))
    glimpse_df = dataframe.apply(lambda x: [x.dtype, x.unique()[0:7]]).T
    glimpse_df.columns = ["dtype", "sample values"]
    return glimpse_df


def check_cat(dataframe, percent=5):
    
    n_df = dataframe.copy()
    is_num = n_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
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
        
        return print("Las columnas recomendadas: \
            \n--------------------------{cols}".format(cols=n_str))
    
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
                n_df.drop(feature, axis='columns', inplace=True)
                cols.append(feature)
    
    
    if len(cols) > 0:
        
        for d in cols:
            n_str = n_str + d + "\n"
        
        print("Se eliminaron las sig. columnas: \
            \n-------------------------{cols}".format(cols=n_str))
    
    else:
        
        print("No se eliminó ninguna columna.")
        return dataframe
    
    return n_df


def any_value(df, search):
    """
    Realiza una búsqueda de un valor en todo el dataframe

    Args:
        df (datafrane): dataframe donde buscar
        search: elemento a buscar

    Returns:
        dataframe: dataframe con la busqueda
    """
    return df[df.eq(search).any(1)]


def missing_values(df):
    """
    Chquea todas las columnas y devuelve el número de valores nulos.

    Args:
        df (dataframe): dataframe a analizar

    Returns:
        dataframe: tabla con la cantidad de valores nulos.
    
    credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
    """
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = mis_val / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(2)
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    
    # Return the dataframe with missing information
    if mis_val_table_ren_columns.shape[0] != 0:
        return mis_val_table_ren_columns.style.background_gradient(cmap="YlOrBr", subset=["Missing Values"])
    
    return


def split_values(dataframe):
    """
    Devuelve dos dataframes, uno solo con valores númericos y el otro solo
    con valores categóricos.
    
    ### Ejemplo:
    
    data_num, data_cat = split_values(df)
    """
              
    data_num = dataframe.select_dtypes(include=[np.number])
    data_cat = dataframe.select_dtypes(include=[np.object, "category"])
      
    return data_num, data_cat


def memory_size(dataframe):
    """
    Devuelve el tamaño que ocupa en memoria el dataframe.
    """
    size_bytes = dataframe.memory_usage().sum()
    if size_bytes == 0:
        return "0B"
    size_name = ("Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "{} {}".format(s, size_name[i])


def correlation(dataframe):
    
    # Compute the correlation matrix
    corr = dataframe.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

    return plt.show()


def grid_plot(dataframe, hue=None, save=None):
    """Trazado de cuadrículas estructuradas de varios gráficos
    
    Parámetros
    ----------
    
    `dataframe:` (Pandas dataframe)
    `hue:` (row name) ‎para graficar diferentes niveles con diferentes colores‎
    `save:` (nombre del archivo) si no se especifica no se guarda en el disco
    """
    n_df = dataframe.copy()
    col_names = n_df.columns.tolist()
    
    for elem in col_names:
        
        # Check if column is bool and change to numpy.uint8
        if n_df[elem].dtype == np.bool:
            n_df[elem] = n_df[elem].astype(np.uint8)
    
    print("Aguarde un momento...", end="")
    sys.stdout.flush()
    sns.set_style("darkgrid")
    g = sns.PairGrid(n_df, hue=hue, height=4)
    g.map_diag(sns.histplot, multiple="stack", element="step")
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    
    if save:
        g.savefig(save, dpi=300)
    
    print('\r ')
    
    return plt.show()


def plot_numcat(dataframe, numeric_row, categoric_row):
    """
    Trazado de un gráfico para una variable numérica y otra categórica

    Parámetros
    ----------
    
    `dataframe` : (Pandas dataframe)
    `numeric_row` : variable numérica del dataframe
    `categoric_row` : variable categórica del dataframe
    """
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(18,4)) 
    plt.subplot(1, 2, 1)
    sns.countplot(y=categoric_row, data=dataframe);
    plt.subplot(1, 2, 2)
    sns.histplot(dataframe[numeric_row])
    plt.xticks(rotation=90)
    return plt.show()


def plot_distribution(dataframe, numeric_row):
    """
    Trazado de un gráfico para una variable numérica y otra categórica

    Parámetros
    ----------
    
    `dataframe` : (Pandas dataframe)
    `numeric_row` : variable numérica del dataframe
    """
    val_log = np.log(dataframe[numeric_row])
    std_n = dataframe[numeric_row].std()
    var_n = dataframe[numeric_row].var()
    skew = dataframe[numeric_row].skew()
    kurto = dataframe[numeric_row].kurt()
    
    print("Standard Deviation (dispersion): {}".format(std_n))
    print("Variance (spread out): {}".format(var_n))
    print("Skewness (distortion or asymmetry): {}".format(skew))
    print("Kurtosis (peakedness of a distribution): {}".format(kurto))
    
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(18,4)) 
    plt.subplot(1, 2, 1)
    sns.histplot(dataframe[numeric_row]);
    plt.subplot(1, 2, 2)
    sns.histplot(val_log, color="salmon")
    plt.xlabel(numeric_row + " (logarithm)")
    return plt.show()


def normalize_row(dataframe):
    """
    Normalizes the values of a given dataframe 
    by the total sum of each line.
    Algorithm based on:
    https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value
    """
    df_c = dataframe.copy()
    df_n = df_c._get_numeric_data().div(df_c.sum(axis=1), axis=0)
    for col in df_n.columns:
      df_c[col] = df_n[col]
    return df_c


def normalize_column(dataframe, percent=True):
    """
    Normalizes the values of a given dataframe by the total sum 
    of each column. If percent=True (default), multiplies the final 
    value by 100.
    Algorithm based on:
    https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value
    """
    df_c = dataframe.copy()
    df_n = df_c._get_numeric_data()
    
    if percent:
        df_n = df_c._get_numeric_data().div(dataframe.sum(axis=0), axis=1)*100
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
        mean_x = np.mean(col)
        sd_x = np.std(col)
        numerator = max(abs(col-mean_x))
        g_calculated = numerator/sd_x
        t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
        g_ca = (n - 1) * np.sqrt(np.square(t_value))
        g_cb = np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value))
        g_critical = g_ca / g_cb
        
        if g_critical < g_calculated:
            out_list.append(c)
            if not silent:
                if title == 0:
                    print("=== Grubbs' test ===")
                    title = 1
                print(c)
    
    # Tukey's method
    """
    credits: https://gist.github.com/alice203
    """
    title = 0
    outliers_prob = []
    
    for c in col_num:
        q1 = dataframe[c].quantile(0.25)
        q3 = dataframe[c].quantile(0.75)
        log_vals = np.log(np.abs(dataframe[c]) + 1)
        q1_l = np.quantile(log_vals, 0.25)
        q3_l = np.quantile(log_vals, 0.75)
        
        iqr = q3-q1
        outer_fence = 3 * iqr
        
        iqr_l = q3_l - q1_l
        outer_fence_l = 3 * iqr_l
        
        #outer fence lower and upper end
        outer_fence_le = q1 - outer_fence
        outer_fence_ue = q3 + outer_fence
        
        outer_fence_le_l = q1_l - outer_fence_l
        outer_fence_ue_l = q3_l + outer_fence_l
        
        for index, x in enumerate(dataframe[c]):
            
            if x <= outer_fence_le or x >= outer_fence_ue:
                outliers_prob.append(np.round(x, n_round))
        
        filter_l = np.where((log_vals <= outer_fence_le_l) 
                            | (log_vals >= outer_fence_ue_l), 
                            True, 
                            False
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
        z_score = stats.zscore(col, nan_policy="omit")
        z_filter = np.abs(z_score) > 3
        if np.sum(z_filter) != 0:
            if not silent:
                if title == 0:
                    print("\n", "=== Z-Score method  ===")
                    title = 1
                col_round = np.round(col[z_filter], n_round)
                print(c, list(set(col_round)))
                print(" ")
            out_list.append(c)
    
    out_list = list(set(out_list))
       
    if out_list:
        return out_list
    else:
        if not silent: print("Outliers not found in dataframe")
        return
    
    
def outliers_graph(dataframe):
    """
    Plots the outliers found in the data frame.

    Args:
        dataframe: Pandas.dataframe
    """
    cols = outliers(dataframe, silent=True)
    if not cols:
        print("Outliers not found in dataframe")
        return
    
    vals_norm = normalize_column(dataframe[cols])
    plt.figure(figsize=(14, len(cols) / 1.8))
    sns.set_style("whitegrid")
    sns.set(font_scale = 1.1)
    ax = sns.boxplot(data=vals_norm, orient="h", palette="Set2")
    ax.set(xlabel='normalized values')
    plt.title("Outliers Found")
    return plt.show()