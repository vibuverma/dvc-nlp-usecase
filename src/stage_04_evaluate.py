import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories, save_json
import random
import joblib
import numpy as np
import json
import sklearn.metrics as metrics
import math



STAGE = "Four" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts= config["artifacts"]
    

    
    featurized_data_dir_path= os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    featurized_test_data_path= os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_OUT_TEST"])


    model_dir_path= os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    model_path= os.path.join(model_dir_path, artifacts["MODEL_NAME"])

    model= joblib.load(model_path)
    matrix= joblib.load(featurized_test_data_path)


    labels = np.squeeze(matrix[:, 1].toarray())
    X= matrix[:,2:]

    predicitions_by_class= model.predict_proba(X)
    predicitions= predicitions_by_class[:1]

    PRC_json_path= config["plots"]["PRC"]
    ROC_json_path= config["plots"]["ROC"]

    scores_json_path= config["metrics"]["SCORES"]


    avg_precision= metrics.average_precision_score(labels, predicitions)

    roc_auc= metrics.roc_auc_score(labels, predicitions)

    scores= {
        "avg_precision": avg_precision,
        "roc_auc": roc_auc
    }
    save_json(scores_json_path, scores)

    
    precision, recall, prc_thresholds= metrics.precision_recall_curve(labels, predicitions)

    nth_point= math.ceil(len(prc_thresholds)/1000)

    prc_points= list(zip(precision, recall, prc_thresholds))[::nth_point]
    prc_data= {
        "prc": [
            {"precision": p, "recall": r, "thresholds": t}
            for p,r,t in prc_points
        ]
    }
    save_json(PRC_json_path, prc_data)

    
    fpr, trp, roc_thresholds = metrics.roc_curve(labels, predicitions)









if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e