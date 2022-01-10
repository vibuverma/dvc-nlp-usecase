import os
import logging
import joblib
import pandas as pd
import scipy.sparse as sparse
import numpy as np



def save_matrix(df, matrix, out_path):
    id_matrix= sparse.csr_matrix(df["id"].astype(np.int64)).transpose()
    label_matrix= sparse.csr_matrix(df["label"].astype(np.int64)).transpose()

    result= sparse.hstack([id_matrix, label_matrix, matrix], format="csr")

    msg= f"The output matrix {out_path} has shape: {result.shape} and data type: {result.dtype}"
    logging.info(msg)
    
    joblib.dump(result, out_path)
