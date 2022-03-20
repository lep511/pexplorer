#load_funct.py
from pathlib import Path
import sys

def load_pexplorer():   
    link_path = "pexplorer"
    q = Path(link_path)
    while not q.exists():
        link_path = "../" + link_path
        q = Path(link_path)
        if len(link_path) > 500: raise Exception("Directory pexplorer not found.") 
    sys.path.append(link_path)
    import pexplorer as px
    return px

px = load_pexplorer()
