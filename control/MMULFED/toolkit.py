from control.globalside_toolkit import sample_clients_normal
from control.Enums import ParametersForFLEnvironment
import numpy as np


# def make_data_4_discriminator(data:tuple):


def sample_clients_MMULFED(
    client_group_dict: dict,
    global_epoch_number: int,
    client_num: int,
):
    group_num = len(client_group_dict)
    group = client_group_dict[global_epoch_number % group_num]
    return sample_clients_normal(group, client_num)


def sample_clients_by_modalities(
    a: int,
    b: int,
    n_C_ab: int,
    n_C_a_not_b: int,
    n_C_b_not_a: int,
    parameters: ParametersForFLEnvironment,
):
    def sample_a_subset(U: list, n_subset: int):
        n_subset = min(n_subset, len(U))
        idxes = np.random.choice(list(range(len(U))), n_subset, replace=False)
        n_subset = [U[idx] for idx in idxes]
        return n_subset

    U_a = parameters.modality_2_clients[a]

    U_b = parameters.modality_2_clients[b]
    # print("U", U_a, U_b)
    U_ab = list(set(U_a) & set(U_b))
    U_a_not_b = list(set(U_a) - set(U_ab))
    U_b_not_a = list(set(U_b) - set(U_ab))

    C_ab = sample_a_subset(U_ab, n_C_ab)
    C_a_not_b = sample_a_subset(U_a_not_b, n_C_a_not_b)
    C_b_not_a = sample_a_subset(U_b_not_a, n_C_b_not_a)

    C_ab = list(set(C_ab))
    # print(parameters.modality_2_clients)
    return (C_ab, C_a_not_b, C_b_not_a)
