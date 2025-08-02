from control.Client import Client
from control.Manager import HyperInfo, Manager
from copy import deepcopy
from control.Enums import LearningType
from models.AEwithAux import AEwithAux
from typing import List


class ClientMMULFED(Client):
    def __init__(
        self,
        hyper_info: HyperInfo,
        client_id: str,
        aux_model_hyperinfo_list: list,
    ):
        super(ClientMMULFED, self).__init__(hyper_info, client_id)
        self.aux_managers: List[Manager] = []
        for aux_model_hyperinfo in aux_model_hyperinfo_list:
            aux_model_hyperinfo.train_dataloader = self.manager.train_dataloader
            aux_model_hyperinfo.test_dataloader = self.manager.test_dataloader
            aux_model_hyperinfo.val_dataloader = self.manager.val_dataloader
            aux_model_hyperinfo.modaility_set = self.manager.modaility_set
            aux_model_hyperinfo.model.set_name(
                f"{client_id};{aux_model_hyperinfo.model_type}"
            )
            aux_manager = Manager(aux_model_hyperinfo, client_id)
            aux_manager.learning_type = LearningType.SUPERVISED_FOR_AUX.value
            aux_manager.generator = self.manager.model
            self.aux_managers.append(aux_manager)

        self.aux_models = [manager.model for manager in self.aux_managers]
        if (
            len(self.manager.modaility_set) < len(self.manager.global_modaility_set)
            or True
        ):
            self.manager.ae_aux_model = AEwithAux(
                self.manager.model.encoder_list,
                self.manager.model.decoder_list,
                self.aux_models,
                f"{client_id};ae-aux",
            )

    def train_with_aux(
        self,
        global_model_parameters: dict,
        global_aux_model_parameters: dict,
        global_epoch: int,
        local_epoch: int,
        log_interval: int,
        learning_rate: float = 0,
        extra_info: dict = {},
        batch_start_id: int = -1,
        batch_end_id: int = -1,
    ):
        des = extra_info["b"]
        des_global_idx = self.manager.global_modaility_set.index(des)
        aux_model = self.aux_models[des_global_idx]
        aux_model.load_state_dict(global_aux_model_parameters)
        self.manager.model.load_state_dict(global_model_parameters)
        self.manager.reset_environment(LearningType.UNSUPERVISED_WITH_AUX.value)
        # ALL TRAININGS ARE DONE HERE! And losses is a list recording all training losses of all epochs; so is acc
        print("start-train-with-aux", self.client_id, "........")
        train_losses, train_acc = self.manager.train_model(
            num_epoch=local_epoch,
            log_interval=log_interval,
            global_epoch=global_epoch,
            learning_rate=learning_rate,
            extra_info=extra_info,
            batch_start_id=batch_start_id,
            batch_end_id=batch_end_id,
        )
        val_losses, val_acc = self.manager.test_model(extra_info=extra_info)
        self.manager.reset_environment(LearningType.UNSUPERVISED.value)
        return self.manager.model, train_losses, train_acc, val_losses, val_acc

    def aux_train(
        self,
        global_model_parameters: dict,
        global_aux_model_parameters: dict,
        global_epoch: int,
        local_epoch: int,
        log_interval: int,
        learning_rate: float = 0,
        extra_info: dict = {},
        batch_start_id: int = -1,
        batch_end_id: int = -1,
        learning_rate_decay: str = "",
    ):
        des = extra_info["b"]
        des_idx = self.manager.global_modaility_set.index(des)
        aux_manager: Manager = self.aux_managers[des_idx]
        aux_manager.model.load_state_dict(global_aux_model_parameters)
        aux_manager.generator.load_state_dict(global_model_parameters)
        aux_manager.generator.requires_grad_(False)
        # ALL TRAININGS ARE DONE HERE! And losses is a list recording all training losses of all epochs; so is acc
        print("start-aux-train", self.client_id, "........")
        train_losses, train_acc = aux_manager.train_model(
            num_epoch=local_epoch,
            log_interval=log_interval,
            global_epoch=global_epoch,
            learning_rate=learning_rate,
            extra_info=extra_info,
            learning_rate_decay=learning_rate_decay,
            batch_start_id=batch_start_id,
            batch_end_id=batch_end_id,
        )
        val_losses, val_acc = aux_manager.test_model(extra_info)
        return (
            aux_manager.model,
            train_losses,
            train_acc,
            val_losses,
            val_acc,
        )

    def aux_test(
        self,
        global_model_parameters: dict,
        global_aux_model_parameters: dict,
        extra_info: dict = {},
    ):
        print("start-aux-test", self.client_id, "........")
        des = extra_info["b"]
        des_idx = self.manager.global_modaility_set.index(des)
        aux_manager: Manager = self.aux_managers[des_idx]
        aux_manager.model.load_state_dict(global_aux_model_parameters)
        aux_manager.generator.load_state_dict(global_model_parameters)
        aux_manager.generator.requires_grad_(False)
        val_losses, val_acc = aux_manager.test_model(extra_info)
        return (
            val_losses,
            val_acc,
        )

    def rename(self, name: str):
        self.client_id = name
        self.manager.owner = name
        self.aux_manager.owner = name

    # def test(self):
    #     print("start test", self.client_id, "........")
    #     val_losses, val_acc = self.manager.test_model()
    #     return (
    #         val_losses,
    #         val_acc,
    #     )
