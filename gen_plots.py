import argparse
import pandas as pd
import pandas.errors as pd_errors
import seaborn as sns
import os


def get_params(p: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Gets the parameters passed when the script is called.
    :param p: the argument parser
    :return: a argparse.Namespace obj with the parsed arguments
    """
    p.add_argument('-f', '--folder', help='Folder relative to script location with CSVs to parse.', default='outputs')
    p.add_argument('-m', '--merge', help='Set the flag if you want to merge all the plots using the hidden state size.', action='store_true')
    return p.parse_args()


def parse_and_gen_plots(rel_folder_path: str) -> None:
    """
    Parses the CSVs in a given folder and outputs a plot for each CSV.

    :param rel_folder_path: folder path with respect to the script location
    :return: None
    """
    sns.set()
    sns.set_context("notebook")

    for output in os.listdir(rel_folder_path):
        file = os.path.join(rel_folder_path, output)

        try:
            output_df = pd.read_csv(file, names=['iterations', 'train loss', 'test loss', 'accuracy'])
        except pd_errors.ParserError:
            continue

        g = sns.relplot(x='iterations', y='accuracy', kind='line', data=output_df, aspect=1.5)
        g.fig.autofmt_xdate()
        g.savefig(os.path.join(rel_folder_path, 'figures', '{fh}.png'.format(fh=output.split('.')[0])))


def parse_and_gen_single_plot(rel_folder_path: str) -> None:
    """
    Parses the CSVs in the given folder and outputs a single plot for each CSV based off the hidden dimension.
    TODO: pass in a param to determine which variable to use for defining different lines.


    :param rel_folder_path: folder path with respect to the script location
    :return: None
    """
    sns.set()
    sns.set_context("notebook")

    merged_dfs = None
    for output in os.listdir(rel_folder_path):
        file = os.path.join(rel_folder_path, output)

        try:
            output_df = pd.read_csv(file, names=['iterations', 'train loss', 'test loss', 'accuracy'])
        except pd_errors.ParserError:
            continue

        _, _, hidden_state_size, _ = output.split('-')

        output_df['hidden state size'] = hidden_state_size

        if merged_dfs is None:
            merged_dfs = output_df
        else:
            merged_dfs = pd.concat([merged_dfs, output_df])

    merged_dfs = merged_dfs.reset_index()
    merged_dfs = merged_dfs.drop(columns=['index'])
    merged_dfs = merged_dfs.sort_values(by=['hidden state size'], ascending=False)
    merged_dfs['hidden state size'] = merged_dfs['hidden state size'].astype(str)

    g = sns.relplot(x='iterations', y='accuracy', kind='line', hue='hidden state size',
                    palette=sns.color_palette("Set1", 7), data=merged_dfs, aspect=2.5)
    g.fig.autofmt_xdate()
    g.savefig(os.path.join(rel_folder_path, 'figures', '{fh}.png'.format(fh='output')))


parser = argparse.ArgumentParser(description="Output plots for a folder of CSVs.")
args = get_params(parser)

if args.merge:
    parse_and_gen_single_plot(args.folder)
else:
    parse_and_gen_plots(args.folder)

