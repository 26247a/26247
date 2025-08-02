import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
from sklearn import manifold
from control.Enums import DatasetInfo
from control.localtraning_toolkit import data_preprocess_transposed
from typing import List
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_info(model_info_path):
    with open(model_info_path, "rb") as f:
        model_info = pickle.load(f)
    return model_info


def plot_losses_acc(model_dir):
    with open(model_dir + "/" + "model_info.pkl", "rb") as f:
        model_info = pickle.load(f)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(model_info["losses"])
    fig.savefig(model_dir + "/" + "loss_fig.png")

    if len(model_info["acc"]) > 0:
        acc = [float(i) for i in model_info["acc"]]
        # print(model_info["acc"])

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(acc)
        fig.savefig(model_dir + "/" + "acc_fig.png")


def save_reps(modalities, model, model_dir, data_iter, dataset_info: DatasetInfo):
    model.eval()
    # Plot original and reconstructed toy data
    dataloader = torch.utils.data.DataLoader(
        data_iter.dataset, batch_size=1, shuffle=False
    )

    reps_mm = {f"m{local_idx}": [] for local_idx in range(len(modalities))}
    y = []
    plot_test_iter = iter(dataloader)
    for pair in plot_test_iter:
        data_x, target = data_preprocess_transposed(
            pair, dataset_info.batch_dimension_switched
        )
        y.append(int(target[0]))
        # if i == 1:
        #     print(pair[i].shape)

        for local_idx, modality in enumerate(modalities):
            data = data_x[local_idx]
            rep = None
            with torch.no_grad():
                model.set_encoder_only()
                encoders = model.encoder_list
                rep = encoders[local_idx](data).to("cpu")
                # print(rep.shape)
                model.setback()
                reps_mm[f"m{local_idx}"].append(rep.squeeze().tolist())

    for local_idx in range(len(modalities)):
        reps_mm[f"m{local_idx}"] = np.array(reps_mm[f"m{local_idx}"])
    reps_mm["y"] = y
    with open(model_dir + "/" + "reps" + ".pkl", "wb") as f:
        pickle.dump(reps_mm, f)
        print("saved reps")


def plot_tsne(modalities, model, model_dir, test_iter, dataset_info: DatasetInfo):
    model.eval()
    # Plot original and reconstructed toy data
    dataloader = torch.utils.data.DataLoader(
        test_iter.dataset, batch_size=1, shuffle=False
    )

    for local_idx, modality in enumerate(modalities):
        reps = []
        y = []
        plot_test_iter = iter(dataloader)
        for pair in plot_test_iter:
            data_x, target = data_preprocess_transposed(
                pair, dataset_info.batch_dimension_switched
            )
            y.append(int(target[0]))
            # if i == 1:
            #     print(pair[i].shape)
            data = data_x[local_idx]
            rep = None
            with torch.no_grad():
                model.set_encoder_only()
                encoders = model.encoder_list
                rep = encoders[local_idx](data).to("cpu")
                # print(rep.shape)
                model.setback()
                reps.append(rep.squeeze().tolist())
        # 2947
        reps = np.array(reps)
        print(reps.shape)
        # print(reps)
        # print(max(y))
        # reps = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tsne = manifold.TSNE(
            n_components=2, init="pca", perplexity=30, random_state=501
        )
        X_tsne = tsne.fit_transform(reps)

        # print("Org data dimension is {}. Embedded data dimension is {}".format(reps.shape[-1], X_tsne.shape[-1]))

        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for j in range(X_norm.shape[0]):
            plt.text(
                X_norm[j, 0],
                X_norm[j, 1],
                str(y[j]),
                color=plt.cm.Set1(y[j]),
                fontdict={"weight": "bold", "size": 9},
            )
        plt.xticks([])
        plt.yticks([])
        plt.savefig(
            model_dir + "/" + "tsne_test_dataset" + "_m" + str(modality) + ".png",
            bbox_inches="tight",
            pad_inches=0,
        )


def plot_orig_vs_reconstructed(
    local_modailities: List[int],
    global_modailities: List[int],
    model: torch.nn.Module,
    model_dir: str,
    test_iter: DataLoader,
    dataset_info: DatasetInfo,
    assigned_ori_modality: int = -1,
    assigned_des_modality: int = -1,
    num_to_plot: int = 2,
    save_path: str = "",
    only_ori: bool = False,
):
    plot_test_iter = iter(DataLoader(test_iter.dataset, batch_size=1, shuffle=False))
    # for i in range(num_to_plot):
    #     fig, axes = plt.subplots(
    #         nrows=1,
    #         ncols=n_axes,
    #         figsize=(14, 7),
    #     )
    # if only_ori:
    #     return

    # Plot original and reconstructed toy data

    for i in range(num_to_plot):
        pair = next(plot_test_iter)
        data_x, target = data_preprocess_transposed(
            pair, dataset_info.batch_dimension_switched
        )

        for ori_local_idx, ori_modality in enumerate(local_modailities):
            if assigned_ori_modality != -1 and ori_modality != assigned_ori_modality:
                continue
            n_axes = len(local_modailities) if assigned_ori_modality == -1 else 1
            fig, axes = plt.subplots(
                nrows=1,
                ncols=n_axes,
                figsize=(14, 7),
            )
            for des_local_idx, des_modality in enumerate(local_modailities):
                if (
                    assigned_des_modality != -1
                    and des_modality != assigned_des_modality
                ):
                    continue
                data: torch.Tensor = data_x[ori_local_idx]
                ori_global_idx = global_modailities.index(ori_modality)
                des_global_idx = global_modailities.index(des_modality)
                constructed = None
                with torch.no_grad():
                    constructed: torch.Tensor = model(data, ori_global_idx)[
                        des_global_idx
                    ]
                target: torch.Tensor = data_x[des_local_idx]
                time_lst = [t for t in range(data.shape[0])]
                ax = None
                if n_axes == 1:
                    ax = axes
                else:
                    ax: plt.Axes = axes[des_local_idx]
                ax.plot(
                    time_lst,
                    target.squeeze().tolist(),
                    color="g",
                    label="Original signal",
                )
                ax.plot(
                    time_lst,
                    constructed.squeeze().tolist(),
                    color="r",
                    label="Reconstructed signal",
                )
                ax.set_xlabel("Time")
                ax.set_ylabel("Signal Value")
                ax.set_title(f"Reconstruction for m{des_modality}")
                ax.legend()
            title = f"Reconstruction for example #{i + 1} from m{ori_modality}"
            # ax.set_title(title + f"of m{j}")
            file_name = model_dir + "/" + title + ".png"
            if save_path != "":
                file_name = save_path

            fig.savefig(fname=file_name, bbox_inches="tight", pad_inches=0)
        # plt.show()


# if __name__=="__main__":
#     t_sne()
