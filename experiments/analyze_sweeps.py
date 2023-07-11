import os
import csv
import pickle
import random
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import seaborn as sns
sns.set()

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
from torch import nn

from source.evaluation.ood import evaluate_ood
from source.evaluation.uncertainty import combine_sample_softmax, calculate_uncertainty_for_class_our
from source.data import get_dataset_as_configured
from source.evaluation.retention import evaluate_retention
from source.run_experiment import obtain_resource
from source.methods import predict_mc_dropout_multiple
from source.networks.efficientnet import get_efficientnet, get_efficientnet_ll
from source.methods import concat_method_outputs
from source.methods import predict_mc_dropout_multiple, predict_ensemble_multiple, predict_sgmcmc_dl


def load_npz(base_path, num_samples, temperature, split):

    preds_path = os.path.join(base_path, f"{split}_preds.npy")
    aux_path = os.path.join(base_path, f"{split}_aux.pkl")

    values = obtain_resource([preds_path, aux_path])
    unc = calculate_uncertainty_for_class_our(**combine_sample_softmax(temperature=temperature, 
                                                                        num_samples=num_samples, 
                                                                        **values))['epistemic']

    return unc, values["target"], values["average_net_pred"]

def process_preds_ood(base_path, results_folder, num_samples, temperature, splits):

    for i, split in enumerate(splits):

        os.makedirs(os.path.join(base_path, results_folder), exist_ok=True)
        uncerts_path = os.path.join(base_path, results_folder, f"{split}_uncerts.pkl")

        if os.path.exists(uncerts_path):
            with open(uncerts_path, 'rb') as f:
                unc = pickle.load(f)
        else:
            unc, _, _ = load_npz(base_path, num_samples, temperature, split)
            with open(uncerts_path, 'wb') as f:
                pickle.dump(unc, f)
    
        uncerts = unc if i == 0 else torch.hstack([uncerts, unc])

    return uncerts
    
def process_preds_miss(base_path, results_folder, num_samples, temperature, splits):

    for i, split in enumerate(splits):

        os.makedirs(os.path.join(base_path, results_folder), exist_ok=True)
        uncerts_path = os.path.join(base_path, results_folder, f"{split}_uncerts.pkl")
        targets_path = os.path.join(base_path, results_folder, f"{split}_targets.pkl")
        preds_path = os.path.join(base_path, results_folder, f"{split}_preds.pkl")

        if os.path.exists(uncerts_path):
            with open(uncerts_path, 'rb') as f:
                unc = pickle.load(f)
            with open(targets_path, 'rb') as f:
                target = pickle.load(f)
            with open(preds_path, 'rb') as f:
                preds = pickle.load(f)
        else:
            unc, target, preds = load_npz(base_path, num_samples, temperature, split)
            with open(uncerts_path, 'wb') as f:
                pickle.dump(unc, f)
            with open(targets_path, 'wb') as f:
                pickle.dump(target, f)
            with open(preds_path, 'wb') as f:
                pickle.dump(preds, f)

        uncerts = unc if i == 0 else torch.hstack([uncerts, unc])
        targets = target if i == 0 else torch.hstack([targets, target])
        average_net_preds = preds if i == 0 else torch.vstack([average_net_preds, preds])

    return uncerts, targets, average_net_preds

def log_results(path, dataset, split, results):

    with open(path, mode='a') as file:
        csv_writer = csv.writer(file)
        try:
            csv_writer.writerow([dataset, split, results['FPR'], results['AUROC'], results['AUPR']])
        except:
            csv_writer.writerow([dataset, split, results])

def plot_histograms(id_uncerts, id_label, ood_uncerts, ood_label, dir):

    plt.tight_layout()
    plt.hist(x=id_uncerts, color="C0", alpha=0.5, bins=40, label=id_label, histtype ='bar', rwidth=1.)
    plt.hist(x=ood_uncerts, color="C1", alpha=0.5, bins=40, label=ood_label, histtype ='bar', rwidth=1.)
    xmax = max(id_uncerts.max().item(), ood_uncerts.max().item())
    xmax = np.ceil(xmax) if xmax >= 1 else np.ceil(xmax*10) / 10
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.legend()
    plt.xlim([-(xmax/50), xmax])
    plt.xticks(np.arange(0, xmax+1, 1) if xmax >= 1 else np.arange(0, xmax+0.1, 0.1))
    plt.savefig(dir)
    plt.close()

def plot_roc(uncerts, classes, plot_dict, plot_auc=False):

    # Define a common set of FPR thresholds
    mean_fpr = np.linspace(0, 1, 100)

    if plot_dict["name"] == "Random":
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label="Random (AUC: 0.50)" if plot_auc else "Random")
    else:

        tprs = []
        for i in range(len(uncerts)):
            fpr, tpr, _ = roc_curve(classes[i], uncerts[i])
            # Interpolate the TPR at the defined FPR thresholds
            tpr_interpolated = np.interp(mean_fpr, fpr, tpr)
            tprs.append(tpr_interpolated)

        tprs = np.array(tprs)

        # Calculate mean, min and max
        tpr_mean = tprs.mean(axis=0)
        tpr_std = tprs.std(axis=0)
        tpr_min = tprs.min(axis=0)
        tpr_max = tprs.max(axis=0)
    
        plt.plot(mean_fpr, tpr_mean, 
                 color=plot_dict["color"], 
                 label=f'{plot_dict["name"]} (AUC: {plot_dict["auc"]:.2f})' if plot_auc else plot_dict["name"])
        plt.fill_between(mean_fpr, tpr_min, tpr_max, color=plot_dict["color"], alpha=0.2)
        
    plt.xlim([-0.001, 1.001])
    plt.ylim([0.0, 1.001])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

def plot_retention(id_uncerts, targets, average_net_preds, plot_dict, plot_auc=False):

    all_accuracies, all_auc_values = list(), list()

    for idu, t, avp in zip(id_uncerts, targets, average_net_preds):

        sort_indices = np.argsort(idu)
        targets_sorted = t[sort_indices]
        average_net_preds_sorted = avp[sort_indices]

        retention, accuracies,  = list(), list()
        for i in range(1, len(idu)):
            acc = (torch.sum(torch.argmax(average_net_preds_sorted[:i], dim=-1) == targets_sorted[:i]) / average_net_preds_sorted[:i].shape[0])
            accuracies.append(acc.detach().item())
            retention.append(i / len(idu))

        all_accuracies.append(accuracies)
        all_auc_values.append(auc(retention, accuracies).item())

    all_accuracies = np.array(all_accuracies)
    all_auc_values = np.array(all_auc_values)

    span = 100
    # Calculate mean, min and max
    acc_mean = all_accuracies.mean(axis=0)
    # Compute Exponential Moving Average
    acc_mean = pd.Series(acc_mean).ewm(span=span, adjust=False).mean().values
    
    acc_min = all_accuracies.min(axis=0)
    # Compute Exponential Moving Average
    acc_min = pd.Series(acc_min).ewm(span=span, adjust=False).mean().values

    acc_max = all_accuracies.max(axis=0)
    # Compute Exponential Moving Average
    acc_max = pd.Series(acc_max).ewm(span=span, adjust=False).mean().values

    plt.plot(retention, acc_mean, 
             color=plot_dict["color"], 
             linestyle='--' if plot_dict["color"] == "grey" else '-', 
             label=f'{plot_dict["name"]} (AUC: {all_auc_values.mean():.2f})' if plot_auc else plot_dict["name"])
    # plt.plot(retention, all_accuracies[0], color=color, alpha=0.5)
    # plt.plot(retention, all_accuracies[1], color=color, alpha=0.5)
    # plt.plot(retention, all_accuracies[2], color=color, alpha=0.5)
    plt.fill_between(retention, acc_min, acc_max, color=plot_dict["color"], alpha=0.2)
    plt.xlabel('Percentage of retained samples')
    plt.ylabel('Accuracy')
    plt.xlim([0.099, 1.001])
    plt.ylim([0.799, 1.0])
    plt.yticks(np.arange(0.8, 1.05, 0.05))
    plt.legend(loc="lower left")

    return all_auc_values

def mc_dropout(dataset, version, start_samples, end_samples):

    # network
    network = get_efficientnet(version=version, use_projections=True, use_bfloat16=True).to('cuda:0')
    network.out[0].inplace = False  # IMPORTANT!

    # data
    config_dict = {
        "dl_properties": {"batch_size": 500},
        "dl_wrappers": [{"wrap_device": {"device": 'cuda:0'}}],
        "ds_wrappers": [{"wrap_shuffle": {"rng_seed": 42}},
                        {"wrap_subset": {"subset_str": f"{start_samples}:{end_samples}"}}],
        "id": dataset,
        "subset": "test",
        }
    _, test_dl = get_dataset_as_configured(**config_dict)
    results = concat_method_outputs(predict_mc_dropout_multiple(test_dl=test_dl, model=network, n_samples=2048))
    uncerts = calculate_uncertainty_for_class_our(**results)

    return uncerts["epistemic"], uncerts["aleatoric"], results["target"], results["average_net_pred"]

def ensembles(dataset, start_samples, end_samples, ll):

    # network
    ensemble = nn.ModuleList()
    if ll:
        efficientnet_versions = [os.path.join("pretrained_models", "efficientnet_stockpile", f"model_{i}.pt") for i in range(10)]
        for version in efficientnet_versions:
            ensemble.append(get_efficientnet_ll(checkpoint=version, use_bfloat=True).to('cuda:0'))
    else:
        efficientnet_versions = ["s", "m", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
        for version in efficientnet_versions:
            network = get_efficientnet(version=version, use_projections=False, use_bfloat16=True).to('cuda:0')
            network.out[0].inplace = False  # IMPORTANT!
            ensemble.append(network)
    
    # data
    config_dict = {
        "dl_properties": {"batch_size": 500},
        "dl_wrappers": [{"wrap_device": {"device": 'cuda:0'}}],
        "ds_wrappers": [{"wrap_shuffle": {"rng_seed": 42}},
                        {"wrap_subset": {"subset_str": f"{start_samples}:{end_samples}"}}],
        "id": dataset,
        "subset": "test",
        }
    
    _, test_dl = get_dataset_as_configured(**config_dict)
    results = concat_method_outputs(predict_ensemble_multiple(test_dl=test_dl, model=ensemble))
    uncerts = calculate_uncertainty_for_class_our(**results)["epistemic"]

    return uncerts, results["target"], results["average_net_pred"]

def sgmcmc(dataset, version, start_samples, end_samples):

    # network
    network = get_efficientnet(version=version, use_projections=True, use_bfloat16=True).to('cuda:0')
    network.out[0].inplace = False  # IMPORTANT!

    # data
    config_dict = {
        "dl_properties": {"batch_size": 500},
        "dl_wrappers": [{"wrap_device": {"device": 'cuda:0'}}],
        "ds_wrappers": [{"wrap_shuffle": {"rng_seed": 42}},
                        {"wrap_subset": {"subset_str": f"{start_samples}:{end_samples}"}}],
        "id": dataset,
        "subset": "test",
        }
    _, test_dl = get_dataset_as_configured(**config_dict)
    config_dict["subset"] = "train"
    config_dict["id"] = "imagenet-projected"
    _, train_dl = get_dataset_as_configured(**config_dict)
    results = concat_method_outputs(predict_sgmcmc_dl(test_dl=test_dl, model=network, train_dl=train_dl,
                                                      n_cycles=1,
                                                      n_iterations=6000,
                                                      n_samples=2000,
                                                      lr=0.01,
                                                      momentum=0,
                                                      sched_beta=0.1,
                                                      temperature=0.00002))

    uncerts = calculate_uncertainty_for_class_our(**results)

    return uncerts["epistemic"], results["target"], results["average_net_pred"]


def process_all(ood_uncerts_dict, results_folder, all_splits, quam_temperature, quam_num_samples, plot_dict):

    ind_uncerts_dict = {"quam": list(),
                        "mcd": list(), 
                        "det": list(), 
                        "ens": list(), 
                        "ens_ll": list(), 
                        "sgmcmc": list(),
                        "rand": list()}
    ind_classes_dict = {"quam": list(),
                        "mcd": list(), 
                        "det": list(), 
                        "ens": list(), 
                        "ens_ll": list(), 
                        "sgmcmc": list(),
                        "rand": list()}
    
    for s, ind_split in enumerate(all_splits):
        start_samples = ind_split[0] * 200
        end_samples = (ind_split[-1] + 1) * 200

        for method in ood_uncerts_dict.keys():

            if  method == "quam":
                id_uncerts = process_preds_ood(base_path=os.path.join(base_path, "imagenet-projected"), 
                                                results_folder=results_folder, 
                                                num_samples=quam_num_samples, 
                                                temperature=quam_temperature, 
                                                splits=ind_split)
            elif method == "mcd":
                id_uncerts, det_id_aleatoric, _, _ = mc_dropout(dataset="imagenet-projected", version="s", start_samples=start_samples, end_samples=end_samples)
            elif method == "det":
                try:
                    id_uncerts = det_id_aleatoric
                except:
                    raise ValueError('"mcd" must be processed prior to "det"!')
            elif method == "ens":
                id_uncerts, _, _ = ensembles(dataset="imagenet", start_samples=start_samples, end_samples=end_samples, ll=False)
            elif method == "ens_ll":
                id_uncerts, _, _ = ensembles(dataset="imagenet-projected", start_samples=start_samples, end_samples=end_samples, ll=True)
            elif method == "sgmcmc":
                id_uncerts, _, _ = sgmcmc(dataset="imagenet-projected", version="s", start_samples=start_samples, end_samples=end_samples)
            else:
                raise ValueError(f"Method '{method}' not defined!")
                
            ind_uncerts_dict[method].append(torch.cat([id_uncerts, ood_uncerts_dict[method]], dim=0).float())
            ind_classes_dict[method].append(torch.cat([torch.zeros(len(id_uncerts)), torch.ones(len(ood_uncerts_dict[method]))], dim=0).float())

            auroc = evaluate_ood(id_uncerts, ood_uncerts_dict[method])
            log_results(path=csv_results_path, dataset=method+"_"+results_folder, split=ind_split, results=auroc)

            plot_histograms(id_uncerts= id_uncerts.float(), 
                            id_label="ImageNet ID",
                            ood_uncerts=ood_uncerts_dict[method].float(),
                            ood_label="ImageNet-A OOD" if results_folder=="results_a" else "ImageNet-O OOD",
                            dir=method + '_auroc_' + results_folder + f'_split{s}.png')

    for method in plot_dict.keys():
        plot_roc(ind_uncerts_dict[method], ind_classes_dict[method], plot_dict[method])
    plt.tight_layout()
    plt.savefig(results_folder + "_roc_final.png")
    plt.close()


if __name__ == "__main__":

    base_path = os.path.join("quam", "analyze_sweeps")
    csv_results_path = "final_results.csv"

    # ---------------------------------------------------------------------------- imagenet-a ----------------------------------------------------------------------------
    print("imagenet-a")
    num_samples = 100
    temperature = 0.005
    results_folder = "results_a"
    all_splits = [list(range(0, 35)), list(range(35, 70)), list(range(70, 105))]

    plot_dict = {"quam": {"color": "C0", "name":"QUAM", "auc": 0.87},
                "ens": {"color": "C3", "name":"DE (all)", "auc": 0.87}, 
                "ens_ll": {"color": "C4", "name":"DE (LL)", "auc": 0.69}, 
                "mcd": {"color": "C1", "name":"MCD", "auc": 0.83},
                "sgmcmc": {"color": "C5", "name":"cSG-HMC", "auc": 0.80},
                "det": {"color": "C2", "name":"Reference", "auc": 0.79}, 
                "rand": {"color": "grey", "name":"Random", "auc": 0.50}}

    # ood data
    quam_ood_uncerts = process_preds_ood(os.path.join(base_path, "imagenet-a-projected"), results_folder, num_samples, temperature, splits=all_splits[0])
    mcd_ood_uncerts, ood_aleatoric, _, _ = mc_dropout(dataset="imagenet-a-projected", version="s", start_samples=0, end_samples=7000)
    ens_ood_uncerts, _, _ = ensembles(dataset="imagenet-a", start_samples=0, end_samples=7000, ll=False)
    ens_ll_ood_uncerts, _, _ = ensembles(dataset="imagenet-a-projected", start_samples=0, end_samples=7000, ll=True)
    sgmcmc_ood_uncerts, _, _ = sgmcmc(dataset="imagenet-a-projected", version="s", start_samples=0, end_samples=7000)

    ood_uncerts_dict = {"quam": quam_ood_uncerts,
                        "mcd": mcd_ood_uncerts,
                        "det": ood_aleatoric,
                        "ens": ens_ood_uncerts,
                        "ens_ll": ens_ll_ood_uncerts,
                        "sgmcmc": sgmcmc_ood_uncerts}
    
    process_all(ood_uncerts_dict=ood_uncerts_dict, 
                results_folder=results_folder,
                all_splits=all_splits,
                quam_temperature=temperature,
                quam_num_samples=num_samples,
                plot_dict=plot_dict)

    # ---------------------------------------------------------------------------- imagenet-o ----------------------------------------------------------------------------
    print("imagenet-o")
    num_samples = 50
    temperature = 0.05
    results_folder = "results_o"
    all_splits=[list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]

    plot_dict["quam"]["auc"] = 0.75
    plot_dict["ens"]["auc"] = 0.71
    plot_dict["ens_ll"]["auc"] = 0.56
    plot_dict["mcd"]["auc"] = 0.68
    plot_dict["sgmcmc"]["auc"] = 0.68
    plot_dict["det"]["auc"] = 0.63

    # ood data
    quam_ood_uncerts = process_preds_ood(os.path.join(base_path, "imagenet-o-projected"), results_folder, num_samples, temperature, splits=all_splits[0])
    mcd_ood_uncerts, ood_aleatoric, _, _ = mc_dropout(dataset="imagenet-o-projected", version="s", start_samples=0, end_samples=2000)
    ens_ood_uncerts, _, _ = ensembles(dataset="imagenet-o", start_samples=0, end_samples=2000, ll=False)
    ens_ll_ood_uncerts, _, _ = ensembles(dataset="imagenet-o-projected", start_samples=0, end_samples=2000, ll=True)
    sgmcmc_ood_uncerts, _, _ = sgmcmc(dataset="imagenet-o-projected", version="s", start_samples=0, end_samples=2000)

    ood_uncerts_dict = {"quam": quam_ood_uncerts,
                        "mcd": mcd_ood_uncerts,
                        "det": ood_aleatoric,
                        "ens": ens_ood_uncerts,
                        "ens_ll": ens_ll_ood_uncerts,
                        "sgmcmc": sgmcmc_ood_uncerts}
    
    process_all(ood_uncerts_dict=ood_uncerts_dict, 
                results_folder=results_folder,
                all_splits=all_splits,
                quam_temperature=temperature,
                quam_num_samples=num_samples,
                plot_dict=plot_dict)

    # ------------------------------------------------------------------ misclassification + retention ------------------------------------------------------------------
    print("imagenet-misclassification")
    num_samples = 100
    temperature = 0.0005
    results_folder = "results_mis"
    all_splits = [list(range(0, 35)), list(range(35, 70)), list(range(70, 105))]

    pth_path = os.path.join('final_experiments', '42', 'mis-ret')
    load_pth_files = False
    
    plot_dict["quam"]["auc"] = 0.90
    plot_dict["ens"]["auc"] = 0.78
    plot_dict["ens_ll"]["auc"] = 0.66
    plot_dict["mcd"]["auc"] = 0.80
    plot_dict["sgmcmc"]["auc"] = 0.77
    plot_dict["det"]["auc"] = 0.87

    if load_pth_files:
        ind_uncerts_dict = torch.load(os.path.join(pth_path, 'ind_uncerts_dict.pth'))
        ind_classes_dict = torch.load(os.path.join(pth_path, 'ind_classes_dict.pth'))
        avg_net_preds_dict = torch.load(os.path.join(pth_path, 'avg_net_preds_dict.pth'))
        target_dict = torch.load(os.path.join(pth_path, 'target_dict.pth'))

    else:
        ind_uncerts_dict = {"quam": list(),
                            "mcd": list(), 
                            "det": list(),
                            "rand": list(),
                            "ens": list(), 
                            "ens_ll": list(), 
                            "sgmcmc": list()}
        ind_classes_dict = {"quam": list(),
                            "mcd": list(), 
                            "det": list(),
                            "rand": list(),
                            "ens": list(), 
                            "ens_ll": list(), 
                            "sgmcmc": list(),}
        avg_net_preds_dict = {"quam": list(),
                              "mcd": list(), 
                              "det": list(), 
                              "rand": list(),
                              "ens": list(), 
                              "ens_ll": list(), 
                              "sgmcmc": list()}
        target_dict = {"quam": list(),
                        "mcd": list(), 
                        "det": list(), 
                        "rand": list(),
                        "ens": list(), 
                        "ens_ll": list(), 
                        "sgmcmc": list()}

        for s, ind_split in enumerate(all_splits):
            start_samples = ind_split[0] * 200
            end_samples = (ind_split[-1] + 1) * 200

            for method in ind_uncerts_dict.keys():
                if  method == "quam":
                    id_uncerts, targets, average_net_preds = process_preds_miss(os.path.join(base_path, "imagenet-projected"), results_folder, num_samples, temperature, splits=ind_split)
                elif method == "mcd":
                    id_uncerts, id_aleatoric, targets, average_net_preds = mc_dropout(dataset="imagenet-projected", version="s", start_samples=start_samples, end_samples=end_samples)
                elif method == "det":
                    id_uncerts = id_aleatoric
                elif method == "rand":
                    id_uncerts = list(range(len(id_uncerts)))
                    random.shuffle(id_uncerts)
                    id_uncerts = torch.tensor(id_uncerts)
                elif method == "ens":
                    id_uncerts, targets, average_net_preds = ensembles(dataset="imagenet", start_samples=start_samples, end_samples=end_samples, ll=False)
                elif method == "ens_ll":
                    id_uncerts, targets, average_net_preds = ensembles(dataset="imagenet-projected", start_samples=start_samples, end_samples=end_samples, ll=True)
                elif method == "sgmcmc":
                    id_uncerts, targets, average_net_preds = sgmcmc(dataset="imagenet-projected", version="s", start_samples=start_samples, end_samples=end_samples)
                else:
                    raise ValueError(f"Method '{method}' not defined!")

                auroc = evaluate_retention(id_uncerts=id_uncerts, id_uncerts_target=targets, id_uncerts_average_net_pred=average_net_preds)
                log_results(path=csv_results_path, dataset= method + "_" + results_folder, split=ind_split, results=auroc)

                uncerts_true = id_uncerts[torch.argmax(average_net_preds, dim=-1) == targets]
                uncerts_false = id_uncerts[torch.argmax(average_net_preds, dim=-1) != targets]

                ind_uncerts_dict[method].append(id_uncerts.float())
                ind_classes_dict[method].append((torch.argmax(average_net_preds, dim=-1) != targets).float())
                avg_net_preds_dict[method].append(average_net_preds.float())
                target_dict[method].append(targets.float())

                plot_histograms(id_uncerts=uncerts_true.float(), 
                                id_label="ImageNet (TP+TN)", 
                                ood_uncerts=uncerts_false.float(), 
                                ood_label="ImageNet (FP+FN)", 
                                dir=method + '_auroc_' + results_folder + f'_split{s}.png')

        torch.save(ind_uncerts_dict,  os.path.join(pth_path, 'ind_uncerts_dict.pth'))
        torch.save(ind_classes_dict, os.path.join(pth_path, 'ind_classes_dict.pth'))
        torch.save(avg_net_preds_dict, os.path.join(pth_path, 'avg_net_preds_dict.pth'))
        torch.save(target_dict, os.path.join(pth_path, 'target_dict.pth'))

    for method in plot_dict.keys():
        all_auc_values = plot_retention(id_uncerts=ind_uncerts_dict[method],
                                        targets=target_dict[method], 
                                        average_net_preds=avg_net_preds_dict[method], 
                                        plot_dict=plot_dict[method])
        # a bit hacky
        for i, a in enumerate(all_auc_values):
            log_results(path=csv_results_path, dataset= method + "_" + results_folder, split=all_splits[i], results=a)

    plt.tight_layout()
    plt.savefig("results_ret_roc_final.pdf")
    plt.close()

    for method in plot_dict.keys():
        plot_roc(ind_uncerts_dict[method],
                ind_classes_dict[method], 
                plot_dict=plot_dict[method])    
    plt.tight_layout()
    plt.savefig(results_folder + "_roc_final.pdf")
    plt.close()
