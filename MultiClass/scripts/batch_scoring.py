# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import argparse
import json

from azureml.core.model import Model
from azureml.automl.core.shared import logging_utilities
from azureml.contrib.automl.dnn.vision.common.logging_utils import get_logger
from azureml.contrib.automl.dnn.vision.common.model_export_utils import load_model, run_inference_batch
from azureml.contrib.automl.dnn.vision.classification.inference.score import _score_with_model
from azureml.contrib.automl.dnn.vision.common.utils import _set_logging_parameters

TASK_TYPE = 'image-classification'
logger = get_logger('azureml.automl.core.scoring_script_images')


def init():
    global model
    
    # Set up logging
    _set_logging_parameters(TASK_TYPE, {})
    
    parser = argparse.ArgumentParser(description="Start automl-vision model serving")
    parser.add_argument('--model_name', dest="model_name", required=True)
    args, _ = parser.parse_known_args()

    model_path = os.path.join(Model.get_model_path(args.model_name), 'model.pt')
    print(model_path)


    try:
        logger.info("Loading model from path: {}.".format(model_path))
        model_settings = {}
        model = load_model(TASK_TYPE, model_path, **model_settings)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


def run(mini_batch):
    parser = argparse.ArgumentParser(description="Batch size to use for inferencing")
    parser.add_argument('--batch_size', dest="batch_size", type=int, required=False)
    args, _ = parser.parse_known_args()

    logger.info("Running inference.")
    result = run_inference_batch(model, mini_batch, _score_with_model, args.batch_size)
    logger.info("Finished inferencing.")
    return result
