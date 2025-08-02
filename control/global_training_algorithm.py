import torch
from control.Enums import (
    ParametersForFLEnvironment,
    FLFramework,
    InTrainingMode,
    LearningRateDecay,
    DataAnalysisIndicators,
    LearningType,
    FLSubType,
)
from copy import deepcopy
from control.MMULFED.Paras import AuxModelConfig
from typing import List
from control.globalside_toolkit import (
    make_client_group,
    sample_clients_normal,
    collect_encoders_decoders,
    aggregate_model_paras_in_list,
    parse_client_id,
    deserialize_model,
    serialize_model,
    aggregate_collected_encoders_decoders,
)
from control.Client import Client
from control.MMULFED.ClientMMULFED import ClientMMULFED

from tools.utils import get_log_info, cosine_decay_lr


class FLAlgorithmInfo:
    def __init__(
        self,
        algorithm_name: str,
        dataset_name: str,
        learning_rate: float,
        learning_rate_decay: str,
        global_epoch_number: int,
        local_epoch_number: int,
        log_interval: int,
        channel_num: int,
        clients: dict,
        global_client: Client,
        client_num: int,
        model: torch.nn.Module,
        model_type: str,
        model_dir: str,
        log_func,
        split_parameters: ParametersForFLEnvironment,
        with_aux: int = None,
        AC_local_epoch: int = None,
        aux_models: List[torch.nn.Module] = None,
        aux_model_config_list: List[AuxModelConfig] = None,
        fl_subtype: str = "",
    ):
        self.dataset_name = dataset_name
        self.model_dir = model_dir
        self.algorithm_name = algorithm_name
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.global_epoch_number = global_epoch_number
        self.local_epoch_number = local_epoch_number
        self.log_interval = log_interval
        self.clients = clients
        self.global_client = global_client
        self.client_num = client_num
        self.model = model
        self.model_type = model_type
        self.log_func = log_func
        self.channel_num = channel_num
        self.split_parameters = split_parameters

        self.with_aux = with_aux
        self.AC_local_epoch = AC_local_epoch
        self.aux_models = aux_models
        self.aux_model_config_list = aux_model_config_list
        self.fl_subtype = fl_subtype


def global_test_func(
    global_client: Client, global_model_paras: dict, log_func, info_dict
):
    val_loss, val_acc = global_client.test(
        global_model_parameters=global_model_paras,
    )

    e = info_dict["epoch"]
    model_name = info_dict["model_name"]
    log_func(
        get_log_info(
            log_type=InTrainingMode.TRAIN.value,
            global_epoch=e,
            client_id=global_client.client_id,
            in_training_metrics=(
                [0],
                [0],
                val_loss,
                val_acc,
            ),
            metrics_name=(
                DataAnalysisIndicators.TRAIN_LOSS.value,
                DataAnalysisIndicators.TRAIN_ACC.value,
                DataAnalysisIndicators.VAL_LOSS.value,
                DataAnalysisIndicators.VAL_ACC.value,
            ),
            model_name=model_name,
        )
    )


def multimodal_supervised_learning_global_default(fl: FLAlgorithmInfo):
    client_keys = list(fl.clients.keys())
    client_groups = make_client_group(client_keys)

    for e in range(fl.global_epoch_number):
        client_models = []
        selected_clients = []
        if fl.algorithm_name == FLFramework.FEDAVG.value:
            selected_clients = sample_clients_normal(client_keys, fl.client_num)
        else:
            assert False, "Unsupported FLFramework"
        print("global epoch", e, "........", selected_clients)
        lr = fl.learning_rate
        if fl.learning_rate_decay == LearningRateDecay.COSINE.value:
            lr = cosine_decay_lr(fl.learning_rate, e, fl.global_epoch_number)
        for client_id in selected_clients:
            client_model = None
            if fl.algorithm_name == FLFramework.FEDAVG.value:
                client: Client = fl.clients[client_id]
                # ALL TRAININGS ARE DONE HERE! And losses is a list recording all training losses of all epochs; so is acc
                (client_model, train_loss, train_acc, val_loss, val_acc, _) = (
                    client.train(
                        global_model_parameters=fl.model.state_dict(),
                        global_epoch=e,
                        local_epoch=fl.local_epoch_number,
                        log_interval=fl.log_interval,
                        learning_rate=lr,
                    )
                )

                fl.log_func(
                    get_log_info(
                        log_type=InTrainingMode.TRAIN.value,
                        global_epoch=e,
                        client_id=client_id,
                        in_training_metrics=(
                            train_loss,
                            train_acc,
                            val_loss,
                            val_acc,
                        ),
                        metrics_name=(
                            DataAnalysisIndicators.TRAIN_LOSS.value,
                            DataAnalysisIndicators.TRAIN_ACC.value,
                            DataAnalysisIndicators.VAL_LOSS.value,
                            DataAnalysisIndicators.VAL_ACC.value,
                        ),
                        model_name=fl.model_type,
                    )
                )
            client_models.append(serialize_model(client_model))
        aggregated_model = aggregate_model_paras_in_list(client_models)
        deserialize_model(fl.model, aggregated_model)
        global_test_func(
            global_client=fl.global_client,
            global_model_paras=fl.model.state_dict(),
            log_func=fl.log_func,
            info_dict={"epoch": e, "model_name": fl.model_type},
        )


def multimodal_unsupervised_learning_global_default(fl: FLAlgorithmInfo):
    global_contral_variate = None

    if fl.fl_subtype == FLSubType.SCAFFOLD.value:
        global_contral_variate = deepcopy(fl.model)
    for e in range(fl.global_epoch_number):
        model_params_cache_encoders = [[] for i in range(fl.channel_num)]
        model_params_cache_decoders = [[] for i in range(fl.channel_num)]
        control_variate_params_cache_encoders = None
        control_variate_params_cache_decoders = None
        if fl.fl_subtype == FLSubType.SCAFFOLD.value:
            control_variate_params_cache_encoders = [[] for i in range(fl.channel_num)]
            control_variate_params_cache_decoders = [[] for i in range(fl.channel_num)]
        selected_clients = []

        if fl.algorithm_name == FLFramework.FEDAVG.value:
            selected_clients = sample_clients_normal(
                list(fl.clients.keys()), fl.client_num
            )
        else:
            assert False, "Unsupported FLFramework"
        print("global epoch", e, "........", selected_clients)
        for client_id in selected_clients:
            client = None
            if fl.algorithm_name == FLFramework.FEDAVG.value:
                client: Client = fl.clients[client_id]
            else:
                assert False, "Unsupported FLFramework"

            # ALL TRAININGS ARE DONE HERE! And losses is a list recording all training losses of all epochs; so is acc
            (
                client_model,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                client_control_variate,
            ) = client.train(
                global_model_parameters=fl.model.state_dict(),
                global_epoch=e,
                local_epoch=fl.local_epoch_number,
                log_interval=fl.log_interval,
                global_control_variate_parameters=(
                    global_contral_variate.state_dict()
                    if global_contral_variate is not None
                    else None
                ),
            )
            fl.log_func(
                get_log_info(
                    log_type=InTrainingMode.TRAIN.value,
                    global_epoch=e,
                    client_id=client_id,
                    in_training_metrics=(
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                    ),
                    metrics_name=(
                        DataAnalysisIndicators.TRAIN_LOSS.value,
                        DataAnalysisIndicators.TRAIN_ACC.value,
                        DataAnalysisIndicators.VAL_LOSS.value,
                        DataAnalysisIndicators.VAL_ACC.value,
                    ),
                    model_name=fl.model_type,
                )
            )

            collect_encoders_decoders(
                encoder_list=client_model.encoder_list,
                model_params_cache_encoders=model_params_cache_encoders,
                decoder_list=client_model.decoder_list,
                model_params_cache_decoders=model_params_cache_decoders,
                client_M_set=fl.split_parameters.M_sets[parse_client_id(client_id)[0]],
                global_M_set=fl.split_parameters.global_m_set,
            )
            if fl.fl_subtype == FLSubType.SCAFFOLD.value:
                collect_encoders_decoders(
                    encoder_list=client_control_variate.encoder_list,
                    model_params_cache_encoders=control_variate_params_cache_encoders,
                    decoder_list=client_control_variate.decoder_list,
                    model_params_cache_decoders=control_variate_params_cache_decoders,
                    client_M_set=fl.split_parameters.M_sets[
                        parse_client_id(client_id)[0]
                    ],
                    global_M_set=fl.split_parameters.global_m_set,
                )
        aggregate_collected_encoders_decoders(
            encoder_list=fl.model.encoder_list,
            model_params_cache_encoders=model_params_cache_encoders,
            decoder_list=fl.model.decoder_list,
            model_params_cache_decoders=model_params_cache_decoders,
            global_M_set=fl.split_parameters.global_m_set,
        )
        if fl.fl_subtype == FLSubType.SCAFFOLD.value:
            aggregate_collected_encoders_decoders(
                encoder_list=client_control_variate.encoder_list,
                model_params_cache_encoders=control_variate_params_cache_encoders,
                decoder_list=client_control_variate.decoder_list,
                model_params_cache_decoders=control_variate_params_cache_decoders,
                global_M_set=fl.split_parameters.global_m_set,
            )
        global_test_func(
            global_client=fl.global_client,
            global_model_paras=fl.model.state_dict(),
            log_func=fl.log_func,
            info_dict={"epoch": e, "model_name": fl.model_type},
        )
