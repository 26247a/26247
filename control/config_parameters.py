import argparse
from control.Enums import (
    LearningType,
    ModelType,
    DatasetName,
    PurposeType,
    TrainingMode4LSTMAE,
    ModelLoading,
    FLEnvironment,
    LearningRateDecay,
    ParametersForFLEnvironment,
    FLFramework,
    FLSubType,
)
from types import SimpleNamespace
import json
import os

PARAMETERS = {
    "purpose": {
        "type": str,
        "default": PurposeType.TRAIN.value,
        "choices": [e.value for e in PurposeType.__members__.values()],
        "description": "purpose of the main function; see details in PurposeType definition",
        "parent": "/",
    },
    "fl_framework": {
        "type": str,
        "default": FLFramework.FEDAVG.value,
        "choices": [e.value for e in FLFramework.__members__.values()],
        "description": "federated learning framework",
        "parent": "/",
    },
    "fl_subtype": {
        "type": str,
        "default": "",
        "choices": [e.value for e in FLSubType.__members__.values()],
        "description": "federated learning framework(subtype)",
        "parent": "/fl_framework:{}/".format(FLFramework.FEDAVG.value),
    },
    "model_id": {
        "type": str,
        "default": "",
        "description": "the sole identification of a model",
        "parent": "/",
    },
    "model_dir": {
        "type": str,
        "default": "trained_models",
        "description": "the directory to store trained models",
        "parent": "/",
    },
    "learning_type": {
        "type": str,
        "default": LearningType.UNSUPERVISED.value,
        "choices": [e.value for e in LearningType.__members__.values()],
        "description": "decides the method to train models",
        "parent": "/",
    },
    "model_type": {
        "type": str,
        "default": ModelType.LSTMAE.value,
        "choices": [e.value for e in ModelType.__members__.values()],
        "description": "decides what model is trained",
        "parent": "/",
    },
    "dataset_name": {
        "type": str,
        "default": DatasetName.HAR.value,
        "choices": [e.value for e in DatasetName.__members__.values()],
        "parent": "/",
    },
    "fl_environmet": {
        "type": str,
        "default": FLEnvironment.TYPE1.value,
        "choices": [e.value for e in FLEnvironment.__members__.values()],
        "description": "decides how to split the data into sub-datasets; see details in DataSplitType definition",
        "parent": "/",
    },
    "batch_size": {
        "type": int,
        "default": 16,
        "parent": "/",
    },
    "learning_rate": {
        "type": float,
        "default": 1e-5,
        "parent": "/",
    },
    "learning_rate_decay": {
        "type": str,
        "default": LearningRateDecay.NONE.value,
        "choices": [e.value for e in LearningRateDecay.__members__.values()],
        "parent": "/",
    },
    "global_epoch_number": {
        "type": int,
        "default": 100,
        "parent": "/",
    },
    "local_epoch_number": {
        "type": int,
        "default": 3,
        "parent": "/",
    },
    "log_interval": {
        "type": int,
        "default": 50,
        "description": "how many batch iteration as an interval to log status",
        "parent": "/",
    },
    "load_model": {
        "type": int,
        "default": ModelLoading.LOADNOTHING.value,
        "choices": [e.value for e in ModelLoading.__members__.values()],
        "parent": "/",
    },
    "load_model_name": {
        "type": str,
        "default": "",
        "description": "the name of the model to be loaded",
        "parent": "/load_model:{}/".format(ModelLoading.LOADMODEL.value),
    },
    "load_aux_model": {
        "type": int,
        "default": ModelLoading.LOADNOTHING.value,
        "choices": [e.value for e in ModelLoading.__members__.values()],
        "parent": "/",
    },
    "load_aux_model_name": {
        "type": str,
        "default": "",
        "description": "the name of the aux model to be loaded",
        "parent": "/load_aux_model:{}/".format(ModelLoading.LOADMODEL.value),
    },
    "pretrained_model_id": {
        "type": str,
        "default": "",
        "description": "the id of the pretrained model trained stored in the local project",
        "parent": "/learning_type:{}/".format(
            LearningType.SUPERVISED_WITH_ENCODER.value
        ),
    },
    "with_aux": {
        "type": int,
        "default": 0,
        "description": "",
        "parent": "/fl_framework:{}/".format(FLFramework.GAMAFEDAC.value),
    },
    "AC_local_epoch": {
        "type": int,
        "default": 0,
        "description": "",
        "parent": "/fl_framework:{}/".format(FLFramework.GAMAFEDAC.value),
    },
    "alternative_alignment": {
        "type": int,
        "default": 0,
        "description": "",
        "parent": "/fl_framework:{}/".format(FLFramework.FEDAVG.value),
    },
    "client_num_per_epoch": {
        "type": int,
        "default": -1,
        "description": "the num of selected clients for one global traininig epoch",
        "parent": "/",
    },
    # for MLP
    "hidden_size_MLP": {
        "type": int,
        "default": 64,
        "parent": "/model_type:{}/".format(ModelType.MLP.value),
    },
    "input_size_MLP": {
        "type": int,
        "default": 64,
        "parent": "/model_type:{}/".format(ModelType.MLP.value),
    },
    "output_size_MLP": {
        "type": int,
        "default": 64,
        "parent": "/model_type:{}/".format(ModelType.MLP.value),
    },
    # for LSTM
    "input_size_LSTMAE": {
        "type": int,
        "default": 3,
        "choices": [3],
        "parent": "/model_type:{}/".format(ModelType.LSTMAE.value),
    },
    "hidden_size_LSTMAE": {
        "type": int,
        "default": 256,
        "parent": "/model_type:{}/".format(ModelType.LSTMAE.value),
    },
    "channel_num": {
        "type": int,
        "default": 3,
        "choices": [1, 2, 3],
        "parent": "/model_type:{}/".format(ModelType.LSTMAE.value),
        "description": "the id of the pretrained model trained stored in the local project",
    },
    "dropout": {
        "type": float,
        "default": 0,
        "parent": "/model_type:{}/".format(ModelType.LSTMAE.value),
        "description": "dropout ratio",
    },
    "seq_len": {
        "type": int,
        "default": 128,
        "choices": [128],
        "parent": "/model_type:{}/".format(ModelType.LSTMAE.value),
        "description": "the length of the input sequence of LSTM",
    },
    "training_mode": {
        "type": str,
        "default": TrainingMode4LSTMAE.SPECIAL_UNCONDITIONED.value,
        "choices": [e.value for e in TrainingMode4LSTMAE.__members__.values()],
        "parent": "/model_type:{}/".format(ModelType.LSTMAE.value),
        "description": "mode of training; see details in LSTM",
    },
}


def is_useful_para(k: str, args_dict: dict):
    parent = PARAMETERS[k]["parent"]
    if parent == "/":
        return True
    parent_list = parent.split("/")
    flag = True
    for info in parent_list:
        if info == "":
            continue
        else:
            parent_k, parent_v = info.split(":")
            if str(args_dict[parent_k]) != str(parent_v):
                flag = False
                break
    return flag


def extract_useful_paras(args_dict):
    args = {}
    for k in args_dict:
        if is_useful_para(k, args_dict):
            args[k] = args_dict[k]
        else:
            args[k] = None
    return args


def args_check(args_dict):
    parameters = ParametersForFLEnvironment(args_dict["fl_environmet"])
    args_dict["channel_num"] = len(parameters.global_m_set)
    total_client_num = sum(parameters.client_nums_for_Dk)
    if args_dict["client_num_per_epoch"] <= 0:
        args_dict["client_num_per_epoch"] = total_client_num
    elif args_dict["client_num_per_epoch"] > total_client_num:
        print(
            "warining: client_num_per_epoch is greater than total_client_num and so it is set to total_client_num"
        )
        args_dict["client_num_per_epoch"] = total_client_num

    if args_dict["learning_type"] in (
        LearningType.SUPERVISED_FOR_AUX.value,
        LearningType.UNSUPERVISED_WITH_AUX.value,
    ):
        assert False
    if (
        args_dict["learning_type"] != LearningType.UNSUPERVISED.value
        and args_dict["fl_framework"] == FLFramework.GAMAFEDAC.value
    ):
        assert False


def args_complete(args_dict):
    for k in PARAMETERS:
        if k not in args_dict and is_useful_para(k, args_dict):
            args_dict[k] = PARAMETERS[k]["default"]
            print(f"Warining: para {k} is added by default")


def get_config(args):
    level1_dir = args.model_dir
    if not os.path.exists(level1_dir):
        os.makedirs(level1_dir)
    level2_dir = level1_dir + "/" + args.dataset_name
    if not os.path.exists(level2_dir):
        os.makedirs(level2_dir)

    level3_dir = level2_dir + "/" + args.fl_environmet
    if not os.path.exists(level3_dir):
        os.makedirs(level3_dir)

    leaf_dir = level3_dir + "/" + args.model_id
    if not os.path.exists(leaf_dir):
        os.mkdir(leaf_dir)

    config_path = leaf_dir + "/" + "config.json"
    args_dict = vars(args)

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            print("has loaded the existing config from " + config_path)
            args_dict = json.load(f)
            args_dict["purpose"] = args.purpose
            args_complete(args_dict)
    else:
        args_dict = extract_useful_paras(args_dict)

    args_dict["model_level_dir"] = leaf_dir
    args_check(args_dict)
    with open(config_path, "w", encoding="utf-8") as f:
        print("has stored the config")
        purpose = args_dict["purpose"]
        del args_dict["purpose"]
        json.dump(args_dict, f, indent=4)
        args_dict["purpose"] = purpose
    print(args_dict)
    return SimpleNamespace(**args_dict)


def get_args():
    parser = argparse.ArgumentParser(description="")
    for key in PARAMETERS:
        value = PARAMETERS[key]
        help_tmp = None
        choices_tmp = None
        if help_tmp in value:
            help_tmp = value["description"]
        if choices_tmp in value:
            choices_tmp = choices_tmp["choices"]
        parser.add_argument(
            "--" + key,
            type=value["type"],
            default=value["default"],
            choices=choices_tmp,
            help=help_tmp,
        )

    args = parser.parse_args()

    return args
