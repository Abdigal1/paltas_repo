import os
import pandas as pd


def label(xcl_path = "selecciones_automaticas.xlsx",dir_to_search= os.curdir, dir_to_save = os.path.join(os.curdir, "Sent_lab")):

    df = pd.read_excel(xcl_path)
    
