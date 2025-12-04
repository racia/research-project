import ast
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def normalize_intervention_ix(x):
    """
    Normalise an entry in the dataframe to be an integer or a list of integers.

    :param x: entry
    :return: normalised entry
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, int):
        return [x]
    elif isinstance(x, str):
        x = x.strip()
        if x == "" or x == "[]":
            return []
        return ast.literal_eval(x)
    else:
        return []


def parse_string_tensor_representation(s):
    """
    Parse the string representation of teacher_prob_approved_tokens into a dictionary of token -> float.
    :param s: string, the string representation
    :return:
    """
    if not s or s == "{}":
        return {}
    pattern = r"'([^']+)': tensor\(([\d\.]+), device='cuda:\d+'\)"
    matches = re.findall(pattern, s)

    return {token: float(val) for token, val in matches}


def get_eval_dicts(path: str):
    """
    Get evaluation dictionaries for a given setting.

    :param path: str, the path to the joined results for one setting
    :return: list of evaluation dictionaries paths
    """
    eval_paths = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            # naming: eval_dict_{setting}_hh-mm-ss_task_{number}.json
            task = "_".join(file.split("_")[-2:] if file else "unknown_task")
            # task: task_{number}.json
            task = task.split(".")[0]
            if file.startswith("eval_dict") and file.endswith(".json"):
                if task not in eval_paths:
                    eval_paths[task] = [os.path.join(dirpath, file)]
                else:
                    eval_paths[task].append(os.path.join(dirpath, file))

    print(
        f"Found tasks: {eval_paths.keys()} \n with number of eval dicts: {[len(v) for v in eval_paths.values()]}"
    )
    print(f"Found eval paths: {eval_paths}")
    return eval_paths


def read_eval_dicts(eval_paths: dict) -> dict[str, pd.DataFrame]:
    """
    Process the evaluation dictionaries.

    :param eval_paths: dict, keys are task names, values are lists of eval dict paths
    :return: dict[str, pd.DataFrame], keys are task names, values are dataframes
    """
    eval_dfs = {}
    for task, paths in eval_paths.items():
        print(f"Task: {task}, Number of eval dicts: {len(paths)}")
        for path in paths:
            df = pd.read_csv(path)
            if task not in eval_dfs:
                eval_dfs[task] = df
            else:
                eval_dfs[task] = pd.concat([eval_dfs[task], df], ignore_index=True)
    return eval_dfs


def clean_eval_dfs(eval_dfs: dict) -> dict[str, pd.DataFrame]:
    """
    Process the evaluation dataframes.

    :param eval_dfs: dict, keys are task names, values are dataframes
    :return: dict[str, pd.DataFrame], keys are task names, values are cleaned dataframes
    """
    for task, df in eval_dfs.items():
        print(f"Processing task: {task}, dataframe: {df.head()}")
        df.drop_duplicates(inplace=True, subset=["task_id", "sample_id", "part_id"])
    return eval_dfs


def process_eval_dicts(path: str) -> dict[str, pd.DataFrame]:
    """
    Get the paths for all evaluation dictionaries at the given paths, read them, and clean them.

    :param path: str, path to the joined results for one setting
    :return: dict[str, pd.DataFrame], keys are task names, values are cleaned dataframes
    """
    paths = get_eval_dicts(path)
    eval_dfs = read_eval_dicts(paths)
    clean_dfs = clean_eval_dfs(eval_dfs)

    return clean_dfs


def overall_stats(dfs: dict[str, pd.DataFrame], result_path: str):
    """
    Compute overall statistics across all tasks and save the results.

    :param dfs: dict[str, pd.DataFrame], keys are task names, values are cleaned dataframes
    :param result_path: str, path to save the overall statistics results
    """
    pass


def analyse_iterations(task: str, df: pd.DataFrame, result_path: str):
    """
    Analyse the amount of iterations per part.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :return:
    """
    # histogram
    plt.hist(
        df["iterations"],
        bins=range(1, df["iterations"].max() + 2),
        align="left",
        rwidth=0.8,
    )
    plt.title("Histogram of Iterations per Part")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(result_path, "iterations_hist.png"))
    plt.close()

    # boxplot
    plt.boxplot(df["iterations"])
    plt.title("Boxplot of Iterations per Part")
    plt.ylabel("Number of Iterations")
    plt.savefig(os.path.join(result_path, "iterations_boxplot.png"))
    plt.close()

    with open(os.path.join(result_path, "iterations_stats.txt"), "w") as f:
        f.write(f"Task: {task}\n")
        f.write(f"Mean iterations: {df['iterations'].mean()}\n")
        f.write(f"Median iterations: {df['iterations'].median()}\n")
        f.write(f"Max iterations: {df['iterations'].max()}\n")
        f.write(f"Min iterations: {df['iterations'].min()}\n")
        f.write(f"Distribution:\n")
        dist = df["iterations"].value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} -> {v}\n")


def analyse_interventions(task: str, df: pd.DataFrame, result_path: str):
    """
    Analyse the amount of interventions per part.
    TODO: should be normalised by length of part?

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :return:
    """
    # safety
    df = df.copy()

    df["intervention_ix"] = df["intervention_ix"].apply(normalize_intervention_ix)
    df["num_interventions"] = df["intervention_ix"].apply(len)

    no_interventions = df["num_interventions"] == 0
    has_interventions = df["num_interventions"] > 0

    # Histogram: number of interventions per part
    max_interv = df["num_interventions"].max()
    bins = range(0, max_interv + 2)  # one bin per integer count

    plt.hist(df["num_interventions"], bins=bins, align="left", rwidth=0.8)
    plt.title("Histogram of Number of Interventions per Part")
    plt.xlabel("Number of Interventions")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(result_path, "num_interventions_hist.png"))
    plt.close()

    # Stats
    with open(os.path.join(result_path, "interventions_stats.txt"), "w") as f:
        f.write(f"Task: {task}\n")
        f.write(f"Number of parts without interventions: {no_interventions.sum()}\n")
        f.write(f"Number of parts with interventions: {has_interventions.sum()}\n")

        if has_interventions.any():
            vals = df.loc[has_interventions, "num_interventions"]
            f.write(f"Mean interventions (with interventions): {vals.mean():.3f}\n")
            f.write(f"Median interventions: {vals.median():.3f}\n")
            f.write(f"Max interventions: {vals.max()}\n")
            f.write(f"Distribution:\n")
            dist = vals.value_counts().sort_index()
            for k, v in dist.items():
                f.write(f"  {k} -> {v}\n")
        else:
            f.write("No interventions found.\n")


def analyse_early_stops(task: str, df: pd.DataFrame, result_path: str):
    """
    Analyse how often early stopping occurred.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :return:
    """
    df = df.copy()
    early_stops = df["early_stop"] == True
    no_early_stops = df["early_stop"] == False

    with open(os.path.join(result_path, "early_stops_stats.txt"), "w") as f:
        f.write(f"Task: {task}\n")
        f.write(f"Number of parts with early stops: {early_stops.sum()}\n")
        f.write(f"Number of parts without early stops: {no_early_stops.sum()}\n")
        if len(df) > 0:
            f.write(f"Proportion of early stops: {early_stops.sum() / len(df):.3f}\n")
        else:
            f.write("No parts found.\n")


def analyse_empty_suggestions(task: str, df: pd.DataFrame, result_path: str):
    """
    Analyse how often empty suggestions were made.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :return:
    """
    df = df.copy()

    empty_suggestions = df["teacher_empty_suggestion"] > 0
    no_empty_suggestions = df["teacher_empty_suggestion"] == 0

    with open(os.path.join(result_path, "empty_suggestions_stats.txt"), "w") as f:
        f.write(f"Task: {task}\n")
        f.write(f"Number of parts with empty suggestions: {empty_suggestions.sum()}\n")
        f.write(
            f"Number of parts without empty suggestions: {no_empty_suggestions.sum()}\n"
        )
        if len(df) > 0:
            f.write(
                f"Proportion of empty suggestions: {empty_suggestions.sum() / len(df):.3f}\n"
            )
        else:
            f.write("No parts found.\n")


def analyse_approved_tokens(task: str, df: pd.DataFrame, result_path: str):
    """
    Analyse the probabilities of the approved tokens.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :return:
    """
    df = df.copy()

    # parse the teacher_prob_approved_tokens column
    df["teacher_prob_floats"] = df["teacher_prob_approved_tokens"].apply(
        parse_string_tensor_representation
    )

    all_probs = []
    for prob_dict in df["teacher_prob_floats"]:
        all_probs.extend(prob_dict.values())

    plt.hist(all_probs, bins=100)
    plt.title("Histogram of Approved Token Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(result_path, "approved_token_probs_hist.png"))
    plt.close()

    plt.boxplot(all_probs)
    plt.title("Boxplot of Approved Token Probabilities")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(result_path, "approved_token_probs_boxplot.png"))
    plt.close()


def analyse_intervention_probs(task: str, df: pd.DataFrame, result_path: str):
    """
    Analyse the probabilities of the intervention tokens.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :return:
    """
    df = df.copy()

    # parse the teacher_prob_intervention_tokens column
    df["intervention_with_prob_floats"] = df["intervention_with_prob"].apply(
        parse_string_tensor_representation
    )

    all_intervention_probs = []
    for prob_dict in df["intervention_with_prob_floats"]:
        all_intervention_probs.extend(prob_dict.values())

    plt.hist(all_intervention_probs, bins=100)
    plt.title("Histogram of Intervention Token Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(result_path, "intervention_token_probs_hist.png"))
    plt.close()

    plt.boxplot(all_intervention_probs)
    plt.title("Boxplot of Intervention Token Probabilities")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(result_path, "intervention_token_probs_boxplot.png"))
    plt.close()


def run_stats(dfs: dict[str, pd.DataFrame], result_path: str):
    """
    Run statistics on the processed evaluation dataframes and save the results.

    :param dfs: dict[str, pd.DataFrame], keys are task names, values are cleaned dataframes
    :param result_path: str, path to save the statistics results
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # do overall stats
    # TODO

    # do task-specific stats
    for task, df in dfs.items():
        task_result_path = os.path.join(result_path, f"{task}")
        if not os.path.exists(task_result_path):
            os.makedirs(task_result_path)
        analyse_iterations(task, df, task_result_path)
        analyse_interventions(task, df, task_result_path)
        analyse_early_stops(task, df, task_result_path)
        analyse_empty_suggestions(task, df, task_result_path)
        analyse_approved_tokens(task, df, task_result_path)
        analyse_intervention_probs(task, df, task_result_path)


def main():
    data_path = "/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/v1/all_tasks_joined"
    dfs = process_eval_dicts(data_path)
    result_path = f"/pfs/work9/workspace/scratch/hd_nc326-research-project/SD/test/reasoning/v1/all_tasks_stats"
    run_stats(dfs=dfs, result_path=result_path)


if __name__ == "__main__":
    main()
