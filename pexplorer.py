import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def check_df(dataframe):
    
    df = dataframe.copy()
    
    df_drop = df.dropna()
    is_num = df_drop.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
    col_names = []
    v_type = []
    count_uniq = []
    v_null = []
    v_no_null = []
    val_max = []
    val_min = []
    val_mean = []
    val_std = []
    val_25 = []
    val_50 = []
    val_75 = []
    binary_val = []
    len_df = len(df)
    
    for feature in df.columns:
        col_names.append(feature)
        v_type.append(str(df[feature].dtype))
        count_uniq.append(len(df[feature].unique()))
        val_n = df[feature].isnull().sum() / len_df
        v_null.append(val_n)
        is_bool = pd.api.types.is_bool_dtype(df[feature])
    
        if is_num[feature] and not is_bool:
            val_max.append(df[feature].max())
            val_min.append(df[feature].min())
            val_mean.append(df[feature].mean())
            val_std.append(df[feature].std())
            val_25.append(np.percentile(df_drop[feature], 25))
            val_50.append(np.percentile(df_drop[feature], 50))
            val_75.append(np.percentile(df_drop[feature], 75))
            
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
    
    data = {"N.": ran_col,
        "col. name": col_names,
        "type": v_type,
        "unique": count_uniq,
        "NAN(%)": v_null,
        "min": val_min,
        "25%": val_25,
        "50%": val_50,
        "75%": val_75,
        "max": val_max,
        "mean": val_mean,
        "std": val_std,
        "binary_values": binary_val
    }

    df_new = pd.DataFrame(data=data)
    df_new = df_new.set_index('N.')
    
    cm = sns.light_palette("green", as_cmap=True)

    return df_new.style.background_gradient(cmap=cm)


def col_rename(dataframe):
    
    n_df = dataframe.copy()
    col_names = n_df.columns.tolist()
    col_newnam = []
    
    for elem in col_names:
        
        col_n = elem
        col_n = col_n.lower()
        col_n = col_n.strip()
        col_n = col_n.replace(",", "_")
        col_n = col_n.replace(".", "_")
        col_n = col_n.replace(":", "_")
        col_n = col_n.replace("-", "_")
        col_n = col_n.replace(" ", "_")
                
        col_newnam.append(col_n)
    
    n_df.columns = col_newnam
    
    return n_df


def make_cat(dataframe, percent=5, exclude=[], only=[]):
    
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
                
                if only != []:
                    
                    if feature in only:
                        try:
                            n_df[feature].fillna("UNKNOWN", inplace=True)
                            n_df[feature] = n_df[feature].astype('category')
                            cols.append(feature)
                        
                        except:
                            
                            print("Una columna ya es CATEGORICAL")
                            return dataframe
                            
                
                else:   
                    
                    try:
                        n_df[feature].fillna("UNKNOWN", inplace=True)
                        n_df[feature] = n_df[feature].astype('category')
                        cols.append(feature)
                        
                    except:
                        
                        print("Una columna ya es CATEGORICAL")
                        return dataframe
     
    n_str = "\n"            
    for d in cols:
        val_uniq = len(n_df[d].unique())
        n_str = n_str + d + " (" + str(val_uniq) + ")\n"
        
    if len(cols) > 0:
        
        print("Las columnas convertidas: \
            \n-------------------------{cols}".format(cols=n_str))
        
        return n_df
    
    return dataframe

        
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
            
            if n_perc >= 0.9:
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


def box_plot_all(dataframe):
    
    data_num = dataframe.select_dtypes(include=[np.number])
    for elem in data_num:
        sns.boxplot(data=dataframe, x=elem);
    
    return plt.show()


def grid_plot(datafrmae, hue=None, save=None):
    
    print("Aguarde un momento...", end="")
    sys.stdout.flush()
    sns.set_style("darkgrid")
    g = sns.PairGrid(datafrmae, hue=hue, height=4)
    g.map_diag(sns.histplot, multiple="stack", element="step")
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    
    if save:
        g.savefig(save, dpi=300)
    
    print('\r ')
    
    return plt.show()