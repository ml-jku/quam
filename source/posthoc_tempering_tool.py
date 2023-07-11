import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .evaluation.uncertainty import combine_sample_softmax, calculate_uncertainty_setting_b
from .evaluation.ood import evaluate_ood


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='produce ood value for different temperings and dump them into a third directory')
    parser.add_argument("--path_a", dest="a", action='store',
                        metavar="id dump path")
    parser.add_argument("--path_b", dest="b", action='store',
                        metavar="ood dump path")
    parser.add_argument("--dump_to", dest="to", action='store',
                        metavar="path to save the plot to", default='./temp.png')
    args = parser.parse_args()
    path_a = args.a
    path_b = args.b
    dump_to = args.to

    # load path a
    with open(path_a, "rb") as f:
        d_in = pickle.load(f)
    # load path a
    with open(path_b, "rb") as f:
        d_out = pickle.load(f)

    # now define the range of temperatures
    temps = np.logspace(start=-10, stop=1, num=30)

    temps_dict = {}
    for temp in tqdm(temps):
        results = evaluate_ood(
            **{k: calculate_uncertainty_setting_b(**combine_sample_softmax(**v, temperature=temp))['epistemic']
               for k, v in {'id_uncerts': d_in, 'ood_uncerts': d_out}.items()})
        for metric_type, metric in results.items():
            if metric_type not in temps_dict:
                temps_dict[metric_type] = []
            temps_dict[metric_type].append(metric)

    # make a plot
    valorder = [0.0, 0.05, 0.1]
    for i, (mtype, mres) in enumerate(temps_dict.items()):
        line, = plt.plot(temps, mres, label=mtype, alpha = 0.5)
        for x, y in zip(temps, mres):
            txt = plt.annotate(f"{y:4.3f}", xy=(x, valorder[i]), color=line.get_color(), rotation=45)
            txt.set_fontsize(5)

    plt.xlabel('softmax temperature')
    plt.ylabel('quality')
    plt.ylim((0, 1.))
    plt.grid(True)
    plt.xscale('log')
    legend = plt.legend(loc='center left', shadow=True,  bbox_to_anchor=(1.04, 0.5))

    plt.savefig(fname=dump_to, dpi=300, bbox_inches="tight")
    print(f"Figure saved at {dump_to}")
