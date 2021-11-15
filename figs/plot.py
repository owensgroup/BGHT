import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import scipy.stats


def remove_failed_experiments(df):
    df = df.applymap(lambda x: float('nan') if x < 0 else x)
    return df



def plot_rates_fixed_lf(results_dir, min_find, max_find, min_insert, max_insert, load_factor, probing = 'BCHT'):
    df = pd.DataFrame()
    svg_name=''
    bucket_sizes = []
    if probing == 'BCHT':
        subdir = '/rates_fixed_lf/bcht_rates_lfeq'
        fmt = '.csv'
        df = pd.read_csv(results_dir +  subdir + str(load_factor) + fmt)
        svg_name='bcht_rates_lfeq' + str(load_factor)
        bucket_sizes = [1, 8, 16, 32]
    elif probing == 'P2BHT':
        subdir = '/rates_fixed_lf/p2bht_rates_lfeq'
        fmt = '.csv'
        df = pd.read_csv(results_dir + subdir + str(load_factor) + fmt)
        svg_name='p2bht_rates_lfeq' + str(load_factor)
        bucket_sizes = [16, 32]
    elif probing == 'IHT':
        subdir = '/rates_fixed_lf/iht_rates_lfeq'
        fmt = '.csv'
        df = pd.read_csv(results_dir + subdir+ str(load_factor) + fmt)
        svg_name='iht_rates_lfeq' + str(load_factor)
        bucket_sizes = [16, 32]
    else:
        print("Uknown probing scheme")
        sys.exit()
    df = remove_failed_experiments(df)
    df['num_keys'] = df['num_keys'].divide(1.0e6)

    scale=5
    subplots=[]
    fig = plt.figure(figsize=(4*scale,1*scale))
    ax = fig.add_subplot(111)
    subplots.append(fig.add_subplot(141))
    subplots.append(fig.add_subplot(142))
    subplots.append(fig.add_subplot(143))
    subplots.append(fig.add_subplot(144))

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    markers =['s', 'o', '^', 'D']
    print(probing + ': Fixed load factor = 0.' + str(load_factor) + ' summary:')
    print('Probing scheme | HMean insertion | HMean find 100 | 50 | 0')

    if probing == 'BCHT' or probing == 'P2BHT':
        for b, m in zip(bucket_sizes, markers):
            insert_c = 'insert_' + str(b)
            find_c = 'find_' + str(b) + '_'
            l = probing + ', b=' + str(b)
            subplots[0].plot(df['num_keys'], df[insert_c], marker = m, label=l)
            subplots[1].plot(df['num_keys'], df[find_c + str(100)], marker = m)
            subplots[2].plot(df['num_keys'], df[find_c + str(50)], marker = m)
            subplots[3].plot(df['num_keys'], df[find_c + str(0)], marker = m)
            mean_insertion_rate = scipy.stats.hmean(df[insert_c].dropna())
            mean_find100_rate = scipy.stats.hmean(df[find_c + str(100)].dropna())
            mean_find50_rate = scipy.stats.hmean(df[find_c + str(50)].dropna())
            mean_find0_rate = scipy.stats.hmean(df[find_c + str(0)].dropna())
            print(l + ' |', "{:.2f}".format(mean_insertion_rate)\
                 + ' | ' + "{:.3f}".format(mean_find100_rate)\
                 + ' | ' + "{:.3f}".format(mean_find50_rate)\
                 + ' | ' + "{:.3f}".format(mean_find0_rate))
    elif probing == 'IHT':
        thresholds = [20, 40, 60, 80]
        for b in bucket_sizes:
            for t, m in zip(thresholds, markers):
                insert_c = 'insert_' + str(b) + '_' + str(t)
                find_c = 'find_' + str(b) + '_' + str(t) + '_'
                l = probing + ', b=' + str(b) + ', t=' + str(t) + '%'
                subplots[0].plot(df['num_keys'], df[insert_c], marker = m, label=l)
                subplots[1].plot(df['num_keys'], df[find_c + str(100)], marker = m)
                subplots[2].plot(df['num_keys'], df[find_c + str(50)], marker = m)
                subplots[3].plot(df['num_keys'], df[find_c + str(0)], marker = m)
                mean_insertion_rate = scipy.stats.hmean(df[insert_c].dropna())
                mean_find100_rate = scipy.stats.hmean(df[find_c + str(100)].dropna())
                mean_find50_rate = scipy.stats.hmean(df[find_c + str(50)].dropna())
                mean_find0_rate = scipy.stats.hmean(df[find_c + str(0)].dropna())
                print(l + ' |', "{:.2f}".format(mean_insertion_rate)\
                    + ' | ' + "{:.3f}".format(mean_find100_rate)\
                    + ' | ' + "{:.3f}".format(mean_find50_rate)\
                    + ' | ' + "{:.3f}".format(mean_find0_rate))
    print('--------------------------------------------------------')
    ax.set_xlabel('Millions of keys')
    subplots[0].set_ylabel('Insert Rate (MKey/s) ' + 'load factor = 0.' + str(load_factor))

    for p in subplots:
        p.spines["right"].set_visible(False)
        p.spines["top"].set_visible(False)

    if min_insert != -1 and max_insert != 1:
        subplots[0].set_ylim([min_insert, max_insert])
    if min_find != -1 and max_find != 1:
        subplots[1].set_ylim([min_find, max_find])
        subplots[2].set_ylim([min_find, max_find])
        subplots[3].set_ylim([min_find, max_find])


    fig.tight_layout()
    fig.legend()
    fig.show()
    fig.savefig(svg_name + '.svg',bbox_inches='tight')


def plot_rates_fixed_keys(results_dir, min_find, max_find, min_insert, max_insert, probing = 'BCHT'):
    df = pd.DataFrame()
    svg_name=''
    bucket_sizes = []
    if probing == 'BCHT':
        df = pd.read_csv(results_dir + '/rates_fixed_keys/bcht_rates_fixed_keys.csv')
        svg_name='bcht_rates_fixed_keys'
        bucket_sizes = [1, 8, 16, 32]
    elif probing == 'P2BHT':
        df = pd.read_csv(results_dir + '/rates_fixed_keys/p2bht_rates_fixed_keys.csv')
        svg_name='p2bht_rates_fixed_keys'
        bucket_sizes = [16, 32]
    elif probing == 'IHT':
        df = pd.read_csv(results_dir + '/rates_fixed_keys/iht_rates_fixed_keys.csv')
        svg_name='iht_rates_fixed_keys'
        bucket_sizes = [16, 32]
    else:
        print("Uknown probing scheme")
        sys.exit()
    df = remove_failed_experiments(df)

    scale=5
    subplots=[]
    fig = plt.figure(figsize=(4*scale,1*scale))
    ax = fig.add_subplot(111)
    subplots.append(fig.add_subplot(141))
    subplots.append(fig.add_subplot(142))
    subplots.append(fig.add_subplot(143))
    subplots.append(fig.add_subplot(144))

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    markers =['s', 'o', '^', 'D']
    min_x_axis = 0.6
    max_x_axis = 1.0

    print(probing + ': Fixed number of keys summary:')
    print('Probing scheme | HMean insertion | HMean find 100 | 50 | 0')
    if probing == 'BCHT' or probing == 'P2BHT':
        for b, m in zip(bucket_sizes, markers):
            insert_c = 'insert_' + str(b)
            find_c = 'find_' + str(b) + '_'
            l = probing + ', b=' + str(b)
            subplots[0].plot(df['load_factor'], df[insert_c], marker = m, label=l)
            subplots[1].plot(df['load_factor'], df[find_c + str(100)], marker = m)
            subplots[2].plot(df['load_factor'], df[find_c + str(50)], marker = m)
            subplots[3].plot(df['load_factor'], df[find_c + str(0)], marker = m)
            mean_insertion_rate = scipy.stats.hmean(df[insert_c].dropna())
            mean_find100_rate = scipy.stats.hmean(df[find_c + str(100)].dropna())
            mean_find50_rate = scipy.stats.hmean(df[find_c + str(50)].dropna())
            mean_find0_rate = scipy.stats.hmean(df[find_c + str(0)].dropna())
            print(l + ' |', "{:.2f}".format(mean_insertion_rate)\
                 + ' | ' + "{:.3f}".format(mean_find100_rate)\
                 + ' | ' + "{:.3f}".format(mean_find50_rate)\
                 + ' | ' + "{:.3f}".format(mean_find0_rate))
    elif probing == 'IHT':
        thresholds = [20, 40, 60, 80]
        for b in bucket_sizes:
            for t, m in zip(thresholds, markers):
                insert_c = 'insert_' + str(b) + '_' + str(t)
                find_c = 'find_' + str(b) + '_' + str(t) + '_'
                l = probing + ', b=' + str(b) + ', t=' + str(t) + '%'
                subplots[0].plot(df['load_factor'], df[insert_c], marker = m, label=l)
                subplots[1].plot(df['load_factor'], df[find_c + str(100)], marker = m)
                subplots[2].plot(df['load_factor'], df[find_c + str(50)], marker = m)
                subplots[3].plot(df['load_factor'], df[find_c + str(0)], marker = m)
                mean_insertion_rate = scipy.stats.hmean(df[insert_c].dropna())
                mean_find100_rate = scipy.stats.hmean(df[find_c + str(100)].dropna())
                mean_find50_rate = scipy.stats.hmean(df[find_c + str(50)].dropna())
                mean_find0_rate = scipy.stats.hmean(df[find_c + str(0)].dropna())
                print(l + ' |', "{:.2f}".format(mean_insertion_rate)\
                    + ' | ' + "{:.3f}".format(mean_find100_rate)\
                    + ' | ' + "{:.3f}".format(mean_find50_rate)\
                    + ' | ' + "{:.3f}".format(mean_find0_rate))

    print('--------------------------------------------------------')

    ax.set_xlabel('Load factor')
    subplots[0].set_ylabel('Insert Rate (MKey/s)')

    for p in subplots:
        p.spines["right"].set_visible(False)
        p.spines["top"].set_visible(False)

    if min_insert != -1 and max_insert != 1:
        subplots[0].set_ylim([min_insert, max_insert])
    if min_find != -1 and max_find != 1:
        subplots[1].set_ylim([min_find, max_find])
        subplots[2].set_ylim([min_find, max_find])
        subplots[3].set_ylim([min_find, max_find])

    for ax in fig.get_axes():
        ax.set_xlim([min_x_axis, max_x_axis])

    fig.tight_layout()
    fig.legend()
    fig.show()
    fig.savefig(svg_name + '.svg',bbox_inches='tight')


def plot_avg_probes_fixed_keys(results_dir, probing = 'BCHT'):
    df = pd.DataFrame()
    svg_name=''
    bucket_sizes = []
    if probing == 'BCHT':
        df = pd.read_csv(results_dir + '/avg_probes/bcht_probes.csv')
        svg_name='bcht_probes'
        bucket_sizes = [1, 8, 16, 32]
    elif probing == 'P2BHT':
        df = pd.read_csv(results_dir + '/avg_probes/p2bht_probes.csv')
        svg_name='p2ht_probes'
        bucket_sizes = [16, 32]
    elif probing == 'IHT':
        df = pd.read_csv(results_dir + '/avg_probes/iht_probes.csv')
        svg_name='iht_probes'
        bucket_sizes = [16, 32]
    else:
        print("Uknown probing scheme")
        sys.exit()
    df = remove_failed_experiments(df)

    scale=5
    subplots=[]
    fig = plt.figure(figsize=(4*scale,1*scale))
    ax = fig.add_subplot(111)
    subplots.append(fig.add_subplot(141))
    subplots.append(fig.add_subplot(142))
    subplots.append(fig.add_subplot(143))
    subplots.append(fig.add_subplot(144))

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    markers =['s', 'o', '^', 'D']
    min_x_axis = 0.6
    max_x_axis = 1.0
    min_y_axis = 1.0
    max_y_axis = 4.0

    print(probing + ': average probes summary:')
    print('Probing scheme | HMean insertion | HMean find 100 | 50 | 0')

    if probing == 'BCHT' or probing == 'P2BHT':
        for b, m in zip(bucket_sizes, markers):
            insert_c = 'insert_' + str(b)
            find_c = 'find_' + str(b) + '_'
            l = probing + ', b=' + str(b)
            subplots[0].plot(df['load_factor'], df[insert_c], marker = m, label=l)
            subplots[1].plot(df['load_factor'], df[find_c + str(100)], marker = m)
            subplots[2].plot(df['load_factor'], df[find_c + str(50)], marker = m)
            subplots[3].plot(df['load_factor'], df[find_c + str(0)], marker = m)

            mean_insertion_probes = scipy.stats.hmean(df[insert_c].dropna())
            mean_find100_probes = scipy.stats.hmean(df[find_c + str(100)].dropna())
            mean_find50_probes = scipy.stats.hmean(df[find_c + str(50)].dropna())
            mean_find0_probes = scipy.stats.hmean(df[find_c + str(0)].dropna())
            print(l + ' |', "{:.2f}".format(mean_insertion_probes)\
                 + ' | ' + "{:.3f}".format(mean_find100_probes)\
                 + ' | ' + "{:.3f}".format(mean_find50_probes)\
                 + ' | ' + "{:.3f}".format(mean_find0_probes))
    elif probing == 'IHT':
        thresholds = [20, 40, 60, 80]
        for b in bucket_sizes:
            for t, m in zip(thresholds, markers):
                insert_c = 'insert_' + str(b) + '_' + str(t)
                find_c = 'find_' + str(b) + '_' + str(t) + '_'
                l = probing + ', b=' + str(b) + ', t=' + str(t) + '%'
                subplots[0].plot(df['load_factor'], df[insert_c], marker = m, label=l)
                subplots[1].plot(df['load_factor'], df[find_c + str(100)], marker = m)
                subplots[2].plot(df['load_factor'], df[find_c + str(50)], marker = m)
                subplots[3].plot(df['load_factor'], df[find_c + str(0)], marker = m)

                mean_insertion_probes = scipy.stats.hmean(df[insert_c].dropna())
                mean_find100_probes = scipy.stats.hmean(df[find_c + str(100)].dropna())
                mean_find50_probes = scipy.stats.hmean(df[find_c + str(50)].dropna())
                mean_find0_probes = scipy.stats.hmean(df[find_c + str(0)].dropna())
                print(l + ' |', "{:.2f}".format(mean_insertion_probes)\
                    + ' | ' + "{:.3f}".format(mean_find100_probes)\
                    + ' | ' + "{:.3f}".format(mean_find50_probes)\
                    + ' | ' + "{:.3f}".format(mean_find0_probes))

    print('--------------------------------------------------------')

    ax.set_xlabel('Load factor')
    subplots[0].set_ylabel('Insert Rate (MKey/s)')

    for p in subplots:
        p.spines["right"].set_visible(False)
        p.spines["top"].set_visible(False)

    for ax in fig.get_axes():
        ax.set_xlim([min_x_axis, max_x_axis])
        ax.set_ylim([min_y_axis, max_y_axis])

    fig.tight_layout()
    fig.legend()
    fig.show()
    fig.savefig(svg_name + '.svg',bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    parser.add_argument('-mf','--min-find-throughput', default=-1,type=int)
    parser.add_argument('-xf','--max-find-throughput', default=-1,type=int)
    parser.add_argument('-mi','--min-insert-throughput', default=-1,type=int)
    parser.add_argument('-xi','--max-insert-throughput', default=-1,type=int)
    parser.add_argument('-p','--probing-scheme', default='all')

    probing_schemes =['BCHT', 'P2BHT', 'IHT']
    args = parser.parse_args()
    print("Reading results from: ", args.dir)
    # Plotting rates vs. load factor
    if args.probing_scheme == 'all':
        for p in probing_schemes:
            plot_rates_fixed_keys(args.dir, args.min_find_throughput, args.max_find_throughput,\
                args.min_insert_throughput, args.max_insert_throughput, p)
    else:
        plot_rates_fixed_keys(args.dir, args.min_find_throughput, args.max_find_throughput,\
            args.min_insert_throughput, args.max_insert_throughput, args.probing_scheme)

    # Plotting probes count vs. load factor
    if args.probing_scheme == 'all':
        for p in probing_schemes:
            plot_avg_probes_fixed_keys(args.dir, p)
    else:
        plot_avg_probes_fixed_keys(args.dir, args.probing_scheme)

    # Plotting rates vs. number of keys
    load_factors = [80, 90]
    if args.probing_scheme == 'all':
        for p in probing_schemes:
            for lf in load_factors:
                plot_rates_fixed_lf(args.dir, args.min_find_throughput, args.max_find_throughput,\
                    args.min_insert_throughput, args.max_insert_throughput, lf, p)
    else:
        for lf in load_factors:
            plot_rates_fixed_lf(args.dir, args.min_find_throughput, args.max_find_throughput,\
                args.min_insert_throughput, args.max_insert_throughput, lf, args.probing_scheme)


