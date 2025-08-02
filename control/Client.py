from control.Manager import Manager, HyperInfo
from data_generating.data_preprocess import get_dataloaders
from control.Enums import LearningType, FLSubType
from copy import deepcopy


class Client:
    def __init__(self, hyper_info: HyperInfo, client_id: str):
        self.client_id = client_id
        train_loader, val_loader, test_loader = get_dataloaders(
            hyper_info.dataset_name, hyper_info.data_split_type, client_id
        )
        hyper_info.train_dataloader = train_loader
        hyper_info.test_dataloader = test_loader
        hyper_info.val_dataloader = val_loader
        hyper_info.model.set_name(f"{client_id};{hyper_info.model_type}")
        self.manager = Manager(hyper_info, client_id)

    def train(
        self,
        global_model_parameters: dict,
        global_epoch: int,
        local_epoch: int,
        log_interval: int,
        test_only: bool = False,
        learning_rate: float = 0,
        learning_type: str = "",
        extra_info: dict = {},
        batch_start_id: int = -1,
        batch_end_id: int = -1,
        global_control_variate_parameters: dict = None,
    ):
        self.manager.model.load_state_dict(global_model_parameters)
        if self.manager.fl_subtype in (
            FLSubType.FEDPROX.value,
            FLSubType.SCAFFOLD.value,
        ):
            if self.manager.reserved_global_model is None:
                self.manager.reserved_global_model = deepcopy(self.manager.model)
            else:
                self.manager.reserved_global_model.load_state_dict(
                    global_model_parameters
                )
            if self.manager.fl_subtype == FLSubType.SCAFFOLD.value:
                if self.manager.reserved_global_control_variate is None:
                    self.manager.reserved_global_control_variate = deepcopy(
                        self.manager.model
                    )
                self.manager.reserved_global_control_variate.load_state_dict(
                    global_control_variate_parameters
                )

        if learning_type != "":
            self.manager.reset_environment(learning_type)
        # ALL TRAININGS ARE DONE HERE! And losses is a list recording all training losses of all epochs; so is acc
        train_losses = [0]
        train_acc = [0]
        returned_model = None
        returned_control_variate = None
        if not test_only:
            print("start", self.client_id, "........")
            train_losses, train_acc = self.manager.train_model(
                num_epoch=local_epoch,
                log_interval=log_interval,
                global_epoch=global_epoch,
                learning_rate=learning_rate,
                extra_info=extra_info,
                batch_start_id=batch_start_id,
                batch_end_id=batch_end_id,
            )
            returned_model = self.manager.model
       
            if self.manager.fl_subtype == FLSubType.SCAFFOLD.value:
                returned_control_variate = self.manager.local_control_variate
              

        val_losses, val_acc = self.manager.test_model(extra_info=extra_info)
        if self.manager.learning_type == LearningType.UNSUPERVISED_AB.value:
            self.manager.reset_environment(LearningType.UNSUPERVISED.value)

        return (
            returned_model,
            train_losses,
            train_acc,
            val_losses,
            val_acc,
            returned_control_variate,
        )

    def test(self, global_model_parameters: dict = None):
        if global_model_parameters is not None:
            self.manager.model.load_state_dict(global_model_parameters)
        print("start test", self.client_id, "........", self.manager.learning_type)
        val_losses, val_acc = self.manager.test_model()
        return (
            val_losses,
            val_acc,
        )

    def get_test_loader(self):
        return self.manager.test_dataloader

    def get_train_loader(self):
        return self.manager.train_dataloader

    def rename(self, name: str):
        self.client_id = name
        self.manager.owner = name
