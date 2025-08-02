from control.Manager import HyperInfo
from control.Client import Client
from control.global_training_algorithm import (
    multimodal_unsupervised_learning_global_default,
    multimodal_supervised_learning_global_default,
    FLAlgorithmInfo,
)
from control.globalside_toolkit import (
    get_model,
    get_local_model,
    create_clients,
)
import os
import torch
import pickle
from tools.plot_tools import (
    plot_losses_acc,
    plot_orig_vs_reconstructed,
    plot_tsne,
    save_reps,
)
from tools.utils import (
    get_log_info,
)
from control.Enums import (
    PurposeType,
    LearningType,
    ModelLoading,
    ParametersForFLEnvironment,
    InTrainingMode,
    DatasetName,
    FLFramework,
    SpeiclaClientType,
    DataAnalysisIndicators,
    DatasetInfo,
)
from types import SimpleNamespace
from data_generating.data_preprocess import create_dataloaders, get_dataloaders
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Server:
    """
    Basic FL Server applying fedavg framework
    """

    def __init__(self, args):
        self.args = args
        self.model_id = self.args.model_id
        self.model_level_dir = self.args.model_level_dir
        self.model_info_path = self.model_level_dir + "/" + "model_training_info.pkl"
        self.model_training_info = []

        self.parameters = None
        self.global_modality_set = None
        self.__init_data_paras()

        self.clients = {}
        self.logs_in_traininig = []

        self.model = None
        if self.args.purpose != PurposeType.GENERATE_DATA.value:
            self.__init_model()

        self.global_test_loader = None
        self.client_for_global_test = None

    def __init_data_paras(self):

        self.parameters = ParametersForFLEnvironment(self.args.data_split_type)
        print(self.parameters.M_sets)
        global_modality_set = []
        for s in self.parameters.M_sets:
            for modality in s:
                if modality not in global_modality_set:
                    global_modality_set.append(modality)
        self.global_modality_set = tuple(global_modality_set)

    def __init_model(self):
        model_args = vars(self.args)

        self.model = get_model(model_args)
        if (
            self.args.purpose == PurposeType.TRAIN.value
            and self.args.load_model == ModelLoading.LOADMODEL.value
        ):
            load_model_id = self.args.load_model_name.split("/")[-1]
            load_model_path = f"{self.args.load_model_name}/{load_model_id}.pt"
            self.model.load_state_dict(torch.load(load_model_path))
            print("successfully load model " + load_model_path)
        elif (
            self.args.purpose == PurposeType.TEST.value
            or self.args.purpose == PurposeType.PLOT.value
        ):
            load_model_path = f"{self.model_level_dir}/{self.model_id}.pt"
            self.model.load_state_dict(torch.load(load_model_path))
            print("successfully load model " + load_model_path)

    def run(self):
        if self.args.purpose == PurposeType.TRAIN.value:
            self.train()
            self.__save_model()
            self.__save_traning_info()
            self.args.purpose = PurposeType.TEST.value
            self.train()
            self.args.purpose = PurposeType.PLOT.value
            self.__draw()
        elif self.args.purpose == PurposeType.TEST.value:
            # task_examine(self.args)
            self.train()
        elif self.args.purpose == PurposeType.PLOT.value:
            self.train()
            self.__draw()
        elif self.args.purpose == PurposeType.GENERATE_DATA.value:
            create_dataloaders(self.args)
        else:
            assert False

    def log(
        self,
        log_info: dict,
    ):
        print(log_info)
        self.model_training_info.append(log_info)

    def __draw(self):
        dataset_info = DatasetInfo(dataset_name=self.args.dataset_name)
        # plot_losses_acc(self.model_dir)
        if (
            self.args.learning_type == LearningType.UNSUPERVISED.value
            and self.args.dataset_name in (DatasetName.HAR.value)
        ):
            if self.args.dataset_name == DatasetName.HAR.value:
                plot_orig_vs_reconstructed(
                    local_modailities=self.global_test_loader.dataset.M_set,
                    global_modailities=self.global_modality_set,
                    model=self.model,
                    model_dir=self.model_level_dir,
                    test_iter=self.global_test_loader,
                    dataset_info=dataset_info,
                    # assigned_ori_modality=0,
                    # assigned_des_modality=2,
                )
            train_loader, val_loader, test_loader = get_dataloaders(
                "HAR", "single02", "0_0_"
            )
            save_reps(
                train_loader.dataset.M_set,
                self.model,
                self.model_level_dir,
                train_loader,
                dataset_info,
            )
            # plot_tsne(
            #     self.global_test_loader.dataset.M_set,
            #     self.model,
            #     self.model_level_dir,
            #     self.global_test_loader,
            #     dataset_info,
            # )

    def __save_model(self):
        print("hihihihi..", self.model_level_dir, self.model_id)
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.model_level_dir,
                self.model_id + ".pt",
            ),
        )

    def __save_traning_info(self):
        with open(self.model_level_dir + "/" + "model_info.pkl", "wb") as f:
            pickle.dump(
                self.model_training_info,
                f,
            )

    def train(self):
        model = self.model
        criterion = None
        if self.args.learning_type == LearningType.UNSUPERVISED.value:
            criterion = torch.nn.MSELoss(reduction="sum")
        elif self.args.learning_type in (
            LearningType.SUPERVISED_WITH_ENCODER.value,
            LearningType.SUPERVISED.value,
        ):
            criterion = torch.nn.CrossEntropyLoss()
            if self.args.dataset_name in (DatasetName.CMUMOSEI.value):
                criterion = torch.nn.MSELoss(reduction="mean")

        encoder = None
        encoder_info = None
        if self.args.learning_type == LearningType.SUPERVISED_WITH_ENCODER.value:
            encoder, encoder_info = get_local_model(self.args.pretrained_model_id)

        hyper_info = HyperInfo(
            learning_type=self.args.learning_type,
            modality_choice_idx_in_SL=None,
            model_type=self.args.model_type,
            dataset_name=self.args.dataset_name,
            data_split_type=self.args.data_split_type,
            modaility_set=None,
            global_modaility_set=self.global_modality_set,
            model=model,
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=None,
            criterion=criterion,
            learning_rate=self.args.learning_rate,
            learning_rate_decay=self.args.learning_rate_decay,
            encoder=encoder,
            encoder_info=encoder_info,
            data_preprocess_method=None,
            alternative_alignment=self.args.alternative_alignment,
            fl_subtype=self.args.fl_subtype,
        )
        self.create_global_test_client(hyper_info)
        if self.args.purpose == PurposeType.PLOT.value:
            return
        self.__init_clients(hyper_info)

        if self.args.purpose == PurposeType.TRAIN.value:
            fl_info = FLAlgorithmInfo(
                algorithm_name=self.args.fl_framework,
                dataset_name=self.args.dataset_name,
                learning_rate=self.args.learning_rate,
                learning_rate_decay=self.args.learning_rate_decay,
                global_epoch_number=self.args.global_epoch_number,
                local_epoch_number=self.args.local_epoch_number,
                log_interval=self.args.log_interval,
                channel_num=len(self.global_modality_set),
                clients=self.clients,
                global_client=self.client_for_global_test,
                client_num=self.args.client_num_per_epoch,
                model=model,
                model_type=self.args.model_type,
                model_dir=self.model_level_dir,
                split_parameters=self.parameters,
                log_func=self.log,
                fl_subtype=self.args.fl_subtype,
            )
            if self.args.learning_type in (
                LearningType.SUPERVISED_WITH_ENCODER.value,
                LearningType.SUPERVISED.value,
            ):
                multimodal_supervised_learning_global_default(fl_info)
            elif self.args.learning_type == LearningType.UNSUPERVISED.value:
                multimodal_unsupervised_learning_global_default(fl_info)

        elif self.args.purpose == PurposeType.TEST.value:
            for client_id in self.clients:
                client: Client = self.clients[client_id]
                loss, acc = client.test()
                self.log(
                    get_log_info(
                        log_type=InTrainingMode.TEST.value,
                        global_epoch=None,
                        client_id=client_id,
                        in_training_metrics=(loss, acc),
                        metrics_name=(
                            DataAnalysisIndicators.VAL_LOSS.value,
                            DataAnalysisIndicators.VAL_ACC.value,
                        ),
                        model_name=self.args.model_type,
                    )
                )
            self.global_eval_while_traininig()

    def __init_clients(self, hyper_info: HyperInfo):
        self.clients = create_clients(
            global_M_set=self.global_modality_set,
            M_sets=self.parameters.M_sets,
            client_nums_for_Dk=self.parameters.client_nums_for_Dk,
            client_modality_choice_idx_in_SL=self.parameters.client_modality_choice_idx_in_SL,
            fl_framework=self.args.fl_framework,
            hyper_info=hyper_info,
            aux_model_hyperinfo_list=None,
        )

    def create_global_test_client(
        self, hyper_info: HyperInfo = None, aux_hyper_info_list: list = None
    ):
        if hyper_info is None:
            return
        self.client_for_global_test = create_clients(
            global_M_set=self.global_modality_set,
            M_sets=[self.global_modality_set],
            client_nums_for_Dk=[1],
            client_modality_choice_idx_in_SL=[0],
            hyper_info=hyper_info,
            fl_framework=self.args.fl_framework,
            aux_model_hyperinfo_list=aux_hyper_info_list,
            is_global=True,
        )[SpeiclaClientType.GLOBAL_CLIENT.value]
        self.global_test_loader = self.client_for_global_test.get_test_loader()

    def global_eval_while_traininig(self, global_epoch=0):
        loss, acc = self.client_for_global_test.test()
        self.log(
            get_log_info(
                log_type=InTrainingMode.TEST.value,
                global_epoch=global_epoch,
                client_id=self.client_for_global_test.client_id,
                in_training_metrics=(loss, acc),
                metrics_name=(
                    DataAnalysisIndicators.VAL_LOSS.value,
                    DataAnalysisIndicators.VAL_ACC.value,
                ),
                model_name=self.args.model_type,
            )
        )
