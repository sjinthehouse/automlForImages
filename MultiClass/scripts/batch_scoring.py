# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import argparse
import json

from azureml.core.model import Model
from azureml.automl.core.shared import logging_utilities
try:
    from azureml.automl.dnn.vision.common.logging_utils import get_logger
    from azureml.automl.dnn.vision.common.model_export_utils import load_model, run_inference
    from azureml.automl.dnn.vision.classification.inference.score import _score_with_model
    from azureml.automl.dnn.vision.common.utils import _set_logging_parameters
except ImportError:
    from azureml.contrib.automl.dnn.vision.common.logging_utils import get_logger
    from azureml.contrib.automl.dnn.vision.common.model_export_utils import load_model, run_inference
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
    result_list = []
    for file_path in mini_batch:
        test_image = open(file_path, 'rb').read()
        logger.info("Running inference.")
        result = run_inference(model, test_image, _score_with_model)
        result_str = result.decode()
        result_json = json.loads(result_str)
        result_json["filename"] = file_path
        logger.info("Finished inferencing.")
        result_list.append(json.dumps(result_json))
    return result_list
