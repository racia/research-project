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


def read_eval_dicts(eval_paths: dict) -> pd.DataFrame:
    """
    Process the evaluation dictionaries.

    :param eval_paths: dict, keys are task names, values are lists of eval dict paths
    :return: pd.DataFrame, concatenated dataframe
    """
    dfs = []
    for task, paths in eval_paths.items():
        print(f"Task: {task}, Number of eval dicts: {len(paths)}")
        for path in paths:
            df = pd.read_csv(path)
            df["task"] = task
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined_df


def clean_eval_dfs(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the evaluation dataframe.

    :param eval_df: pd.DataFrame, concatenated dataframe
    :return: pd.DataFrame, cleaned dataframe
    """
    eval_df.drop_duplicates(inplace=True, subset=["task_id", "sample_id", "part_id"])
    return eval_df


def process_eval_dicts(path: str) -> pd.DataFrame:
    """
    Get the paths for all evaluation dictionaries at the given paths, read them, and clean them.

    :param path: str, path to the joined results for one setting
    :return: pd.DataFrame, cleaned concatenated dataframe
    """
    paths = get_eval_dicts(path)
    eval_df = read_eval_dicts(paths)
    clean_df = clean_eval_dfs(eval_df)
    return clean_df


def analyse_iterations(
    task: str, df: pd.DataFrame, result_path: str, setting: str
) -> pd.Series:
    """
    Analyse the amount of iterations per part.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :return: pd.Series, the amount of iterations
    """
    # histogram
    fig, ax = plt.subplots()
    ax.hist(
        df["iterations_eval"],
        bins=range(1, df["iterations_eval"].max() + 2),
        align="left",
        rwidth=0.8,
    )
    ax.set_title("Histogram of Iterations per Part")
    plt.suptitle(f"{setting}: {task}")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Frequency")
    plt.savefig(os.path.join(result_path, "iterations_hist.png"))
    plt.close()

    # boxplot
    fig, ax = plt.subplots()
    ax.boxplot(df["iterations_eval"], label=[task])
    ax.set_title("Boxplot of Iterations per Part")
    plt.suptitle(f"{setting}: {task}")
    ax.set_ylabel("Number of Iterations")
    plt.savefig(os.path.join(result_path, "iterations_boxplot.png"))
    plt.close()

    with open(os.path.join(result_path, "iterations_stats.txt"), "w") as f:
        f.write(f"Task: {task}\n")
        f.write(f"Mean iterations: {df['iterations_eval'].mean()}\n")
        f.write(f"Median iterations: {df['iterations_eval'].median()}\n")
        f.write(f"Max iterations: {df['iterations_eval'].max()}\n")
        f.write(f"Min iterations: {df['iterations_eval'].min()}\n")
        f.write(f"Distribution:\n")
        dist = df["iterations_eval"].value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} -> {v}\n")

    return df["iterations_eval"]


def analyse_interventions(
    task: str, df: pd.DataFrame, result_path: str, setting: str
) -> (pd.Series, pd.Series):
    """
    Analyse the amount of interventions per part.
    TODO: should be normalised by length of part?

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :return: (pd.Series, pd.Series), the intervention indices and the amount of interventions
    """
    # safety
    df = df.copy()

    df["intervention_ix"] = df["intervention_ix"].apply(normalize_intervention_ix)

    flat_ix = [ix for seq in df["intervention_ix"] for ix in seq]
    flat_ix = pd.Series(flat_ix)

    fig, ax = plt.subplots()
    ax.hist(flat_ix, bins=50)
    ax.set_title("Histogram of Intervention Indices")
    plt.suptitle(f"{setting}: {task}")
    ax.set_xlabel("Intervention Index")
    ax.set_ylabel("Frequency")
    plt.savefig(os.path.join(result_path, "intervention_ix_hist.png"))
    plt.close()

    fig, ax = plt.subplots()
    ax.boxplot(flat_ix)
    ax.set_title("Boxplot of Intervention Indices")
    plt.suptitle(f"{setting}: {task}")
    ax.set_ylabel("Intervention Index")
    plt.savefig(os.path.join(result_path, "intervention_ix_boxplot.png"))
    plt.close()

    with open(os.path.join(result_path, "intervention_ix_stats.txt"), "w") as f:
        f.write(f"Mean: {flat_ix.mean():.3f}\n")
        f.write(f"Median: {flat_ix.median():.3f}\n")
        f.write(f"Max: {flat_ix.max()}\n")
        f.write(f"Min: {flat_ix.min()}\n")
        f.write(f"Distribution:\n")
        dist = flat_ix.value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} -> {v}\n")

    return df["intervention_ix"], flat_ix


def analyse_early_stops(
    task: str, df: pd.DataFrame, result_path: str, setting: str
) -> pd.Series:
    """
    Analyse how often early stopping occurred.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :return: pd.Series, the amount of early stops
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

    return df["early_stop"]


def analyse_empty_suggestions(
    task: str, df: pd.DataFrame, result_path: str, setting: str
) -> pd.Series:
    """
    Analyse how often empty suggestions were made.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :return: pd.Series, the empty suggestion counts
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

    return df["teacher_empty_suggestion"]


def analyse_approved_tokens(
    task: str, df: pd.DataFrame, result_path: str, setting: str
) -> pd.Series:
    """
    Analyse the probabilities of the approved tokens.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :return: pd.Series, the approved token probabilities
    """
    df = df.copy()

    # parse the teacher_prob_approved_tokens column
    df["teacher_prob_floats"] = df["teacher_prob_approved_tokens"].apply(
        parse_string_tensor_representation
    )

    all_probs = []
    for prob_dict in df["teacher_prob_floats"]:
        all_probs.extend(prob_dict.values())

    all_probs = pd.Series(all_probs)

    fig, ax = plt.subplots()
    ax.hist(all_probs, bins=100)
    ax.set_title("Histogram of Approved Student Token Probabilities")
    plt.suptitle(f"{setting}: {task}")
    ax.set_xlabel("Token Probability of Student")
    ax.set_ylabel("Frequency")
    plt.savefig(os.path.join(result_path, "approved_token_probs_hist.png"))
    plt.close()

    fig, ax = plt.subplots()
    ax.boxplot(all_probs, label=[task])
    ax.set_title("Boxplot of Approved Student Token Probabilities")
    plt.suptitle(f"{setting}: {task}")
    ax.set_ylabel("Token Probability of Student")
    plt.savefig(os.path.join(result_path, "approved_token_probs_boxplot.png"))
    plt.close()

    with open(os.path.join(result_path, "approved_token_probs_stats.txt"), "w") as f:
        f.write(f"Mean: {all_probs.mean():.6f}\n")
        f.write(f"Median: {all_probs.median():.6f}\n")
        f.write(f"Max: {all_probs.max()}\n")
        f.write(f"Min: {all_probs.min()}\n")
        f.write("Distribution:\n")
        dist = all_probs.value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} -> {v}\n")

    return df["teacher_prob_floats"]


def analyse_intervention_probs(
    task: str, df: pd.DataFrame, result_path: str, setting: str
) -> pd.Series:
    """
    Analyse the probabilities of the intervention tokens.

    :param task: str, the task name
    :param df: pd.DataFrame, the evaluation dataframe
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :return: pd.Series, the intervention probabilities per part
    """
    df = df.copy()

    # parse the teacher_prob_intervention_tokens column
    df["intervention_with_prob_floats"] = df["intervention_with_prob"].apply(
        parse_string_tensor_representation
    )

    all_intervention_probs = []
    for prob_dict in df["intervention_with_prob_floats"]:
        all_intervention_probs.extend(prob_dict.values())

    all_intervention_probs = pd.Series(all_intervention_probs)

    fig, ax = plt.subplots()
    ax.hist(all_intervention_probs, bins=100)
    ax.set_title("Histogram of Teacher Intervention Token Probabilities")
    plt.suptitle(f"{setting}: {task}")
    ax.set_xlabel("Token Probability of Teacher")
    ax.set_ylabel("Frequency")
    plt.savefig(os.path.join(result_path, "intervention_token_probs_hist.png"))
    plt.close()

    fig, ax = plt.subplots()
    ax.boxplot(all_intervention_probs, label=[task])
    ax.set_title("Boxplot of Teacher Intervention Token Probabilities")
    plt.suptitle(f"{setting}: {task}")
    ax.set_ylabel("Token Probability of Teacher")
    plt.savefig(os.path.join(result_path, "intervention_token_probs_boxplot.png"))
    plt.close()

    with open(
        os.path.join(result_path, "intervention_token_probs_stats.txt"), "w"
    ) as f:
        f.write(f"Mean: {all_intervention_probs.mean():.6f}\n")
        f.write(f"Median: {all_intervention_probs.median():.6f}\n")
        f.write(f"Max: {all_intervention_probs.max()}\n")
        f.write(f"Min: {all_intervention_probs.min()}\n")
        f.write("Distribution:\n")
        dist = all_intervention_probs.value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} -> {v}\n")

    return df["intervention_with_prob_floats"]


def analyse_overall_iterations(
    overall_iterations: dict[str, pd.Series],
    result_path: str,
    setting: str,
    colourmap="tab10",
):
    """
    Analyse overall interactions across all tasks.

    :param overall_iterations: dict[str, pd.Series], keys are task names, values are iterations
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :param colourmap: str, the colourmap, default "tab10"
    """
    vals_hist = pd.concat(overall_iterations.values(), ignore_index=True)
    vals_box = list(overall_iterations.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals_hist, bins=range(1, int(vals_hist.max()) + 2), align="left", log=True)
    ax.set_title("Overall Histogram of Iterations per Part")
    plt.suptitle(f"{setting}: Overall")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Frequency (log)")
    plt.set_cmap(colourmap)
    plt.savefig(os.path.join(result_path, "overall_iterations_hist.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(vals_box, tick_labels=list(overall_iterations.keys()), manage_ticks=True)
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Overall Boxplot of Iterations per Part")
    plt.suptitle(f"{setting}: Overall")
    ax.set_xlabel("Task")
    ax.set_ylabel("Number of Iterations")
    plt.savefig(os.path.join(result_path, "overall_iterations_boxplot.png"))
    plt.close()

    with open(os.path.join(result_path, "overall_iterations_stats.txt"), "w") as f:
        f.write(f"Overall Mean iterations: {vals_hist.mean()}\n")
        f.write(f"Overall Median iterations: {vals_hist.median()}\n")
        f.write(f"Overall Max iterations: {vals_hist.max()}\n")
        f.write(f"Overall Min iterations: {vals_hist.min()}\n")
        f.write(f"Overall Distribution:\n")
        dist = vals_hist.value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} -> {v}\n")


def analyse_overall_interventions(
    overall_interventions: dict[str, pd.Series],
    result_path: str,
    setting: str,
    colourmap="tab10",
):
    """
    Analyse overall interventions across all tasks.

    :param overall_interventions: dict[str, pd.Series], keys are task names, values are interventions
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :param colourmap: str, the colourmap, default "tab10"
    """
    vals_hist = pd.concat(overall_interventions.values(), ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals_hist, bins=range(0, int(vals_hist.max()) + 2), align="left", log=True)
    ax.set_title("Overall Histogram of Intervention Indices")
    plt.suptitle(f"{setting}: Overall")
    ax.set_xlabel("Intervention Index")
    ax.set_ylabel("Frequency (log)")
    plt.set_cmap(colourmap)
    plt.savefig(os.path.join(result_path, "overall_num_interventions_hist.png"))
    plt.close()

    with open(
        os.path.join(result_path, "overall_num_interventions_stats.txt"), "w"
    ) as f:
        f.write(
            f"Overall Mean interventions (with interventions): {vals_hist.mean():.3f}\n"
        )
        f.write(f"Overall Median interventions: {vals_hist.median():.3f}\n")
        f.write(f"Overall Max interventions: {vals_hist.max()}\n")
        f.write(f"Overall Distribution:\n")
        dist = vals_hist.value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} -> {v}\n")


def analyse_overall_approved_tokens(
    overall_approved_tokens: dict[str, pd.Series],
    result_path: str,
    setting: str,
    colourmap="tab10",
):
    """
    Analyse overall approved token probabilities across all tasks.

    :param overall_approved_tokens: dict[str, pd.Series], keys are task names, values are approved token probabilities
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :param colourmap: str, the colourmap, default "tab10"
    """
    vals_hist = []
    vals_box = []
    for task in overall_approved_tokens:
        probs_list = overall_approved_tokens[task].apply(
            lambda d: list(d.values()) if isinstance(d, dict) else []
        )
        flattened = [prob for sublist in probs_list for prob in sublist]
        vals_hist.extend(flattened)
        vals_box.append(flattened)

    vals_hist = pd.Series(vals_hist)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals_hist, bins=100, log=True)
    ax.set_title("Overall Histogram of Approved Student Token Probabilities")
    plt.suptitle(f"{setting}: Overall")
    ax.set_xlabel("Token Probability of Student")
    ax.set_ylabel("Frequency (log)")
    plt.set_cmap(colourmap)
    plt.savefig(os.path.join(result_path, "overall_approved_token_probs_hist.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        vals_box, tick_labels=list(overall_approved_tokens.keys()), manage_ticks=True
    )
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Overall Boxplot of Approved Token Probabilities")
    plt.suptitle(f"{setting}: Overall")
    ax.set_xlabel("Task")
    ax.set_ylabel("Token Probability of Student")
    plt.set_cmap(colourmap)
    plt.savefig(os.path.join(result_path, "overall_approved_token_probs_boxplot.png"))
    plt.close()

    with open(
        os.path.join(result_path, "overall_approved_token_probs_stats.txt"), "w"
    ) as f:
        f.write(f"Mean: {vals_hist.mean():.6f}\n")
        f.write(f"Median: {vals_hist.median():.6f}\n")
        f.write(f"Max: {vals_hist.max()}\n")
        f.write(f"Min: {vals_hist.min()}\n")
        f.write("Distribution:\n")
        dist = vals_hist.value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} -> {v}\n")


def analyse_overall_intervention_probs(
    overall_intervention_probs: dict[str, pd.Series],
    result_path: str,
    setting: str,
    colourmap="tab10",
):
    """
    Analyse overall intervention token probabilities across all tasks.

    :param overall_intervention_probs: dict[str, pd.Series], keys are task names, values are intervention token probabilities
    :param result_path: str, path to save the results
    :param setting: str, the setting
    :param colourmap: str, the colourmap, default "tab10"
    """
    vals_hist = []
    vals_box = []
    for task in overall_intervention_probs:
        probs_list = overall_intervention_probs[task].apply(
            lambda d: list(d.values()) if isinstance(d, dict) else []
        )
        flattened = [prob for sublist in probs_list for prob in sublist]
        vals_hist.extend(flattened)
        vals_box.append(flattened)

    vals_hist = pd.Series(vals_hist)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals_hist, bins=100, log=True)
    ax.set_title("Overall Histogram of Intervention Token Probabilities")
    plt.suptitle(f"{setting}: Overall")
    ax.set_xlabel("Token Probability of Teacher")
    ax.set_ylabel("Frequency (log)")
    plt.set_cmap(colourmap)
    plt.savefig(os.path.join(result_path, "overall_intervention_token_probs_hist.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        vals_box, tick_labels=list(overall_intervention_probs.keys()), manage_ticks=True
    )
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Overall Boxplot of Intervention Token Probabilities")
    plt.suptitle(f"{setting}: Overall")
    ax.set_xlabel("Task")
    ax.set_ylabel("Token Probability of Teacher")
    plt.set_cmap(colourmap)
    plt.savefig(
        os.path.join(result_path, "overall_intervention_token_probs_boxplot.png")
    )
    plt.close()

    with open(
        os.path.join(result_path, "overall_intervention_token_probs_stats.txt"), "w"
    ) as f:
        f.write(f"Mean: {vals_hist.mean():.6f}\n")
        f.write(f"Median: {vals_hist.median():.6f}\n")
        f.write(f"Max: {vals_hist.max()}\n")
        f.write(f"Min: {vals_hist.min()}\n")
        f.write("Distribution:\n")
        dist = vals_hist.value_counts().sort_index()
        for k, v in dist.items():
            f.write(f"  {k} -> {v}\n")


def run_stats(df: pd.DataFrame, result_path: str, setting: str):
    """
    Run statistics on the processed evaluation dataframe and save the results.

    :param df: pd.DataFrame, cleaned concatenated dataframe
    :param result_path: str, path to save the statistics results
    :param setting: str, the setting
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    overall_iterations = {}
    if setting in ["Speculative decoding", "SD"]:
        overall_interventions = {}
        overall_early_stops = {}
        overall_empty_suggestions = {}
        overall_approved_tokens = {}
        overall_intervention_probs = {}

    for task, task_df in df.groupby("task_id"):
        task_result_path = os.path.join(result_path, f"{task}")
        if not os.path.exists(task_result_path):
            os.makedirs(task_result_path)

        iterations = analyse_iterations(task, task_df, task_result_path, setting)
        overall_iterations[task] = iterations

        if setting in ["Speculative decoding", "SD"]:
            task_df["intervention_ix"], flat_ix = analyse_interventions(
                task, task_df, task_result_path, setting
            )
            overall_interventions[task] = flat_ix
            early_stops = analyse_early_stops(task, task_df, task_result_path, setting)
            overall_early_stops[task] = early_stops
            empty_suggestions = analyse_empty_suggestions(
                task, task_df, task_result_path, setting
            )
            overall_empty_suggestions[task] = empty_suggestions
            approved_tokens = analyse_approved_tokens(
                task, task_df, task_result_path, setting
            )
            overall_approved_tokens[task] = approved_tokens
            intervention_probs = analyse_intervention_probs(
                task, task_df, task_result_path, setting
            )
            overall_intervention_probs[task] = intervention_probs

    analyse_overall_iterations(
        dict(sorted(overall_iterations.items())),
        result_path,
        setting,
    )

    if setting in ["Speculative decoding", "SD"]:
        analyse_overall_interventions(
            dict(sorted(overall_interventions.items())),
            result_path,
            setting,
        )
        analyse_overall_approved_tokens(
            dict(sorted(overall_approved_tokens.items())),
            result_path,
            setting,
        )
        analyse_overall_intervention_probs(
            dict(sorted(overall_intervention_probs.items())),
            result_path,
            setting,
        )

    df.to_csv(
        os.path.join(result_path, "complete_evaluation_dataframe.csv"),
        index=False,
        sep=",",
    )


def combine_eval_dfs(eval_df: pd.DataFrame, result_df: pd.DataFrame, result_path: str):
    """
    Create a complete evaluation dataframe by merging the eval dataframe with the result dataframe.

    :param eval_df: pd.DataFrame, cleaned concatenated dataframe
    :param result_df: pd.DataFrame, the result dataframe
    :param result_path: str, path to save the complete evaluation dataframe
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    merged_df = pd.merge(
        eval_df,
        result_df,
        on=["task_id", "sample_id", "part_id"],
        suffixes=("_eval", "_result"),
    )

    merged_df.to_csv(
        os.path.join(result_path, "complete_evaluation_with_results_dataframe.csv"),
        index=False,
        sep=",",
    )

    print(f"Merged dataframe: {merged_df}")
    return merged_df


def analyse_effects(df: pd.DataFrame, res_path: str):
    """
    Analyse the effects of interventions on the results.

    :param df: pd.DataFrame, the complete evaluation dataframe
    :param res_path: str, path to save the effects analysis
    """
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    df = df.copy()

    df["intervention_present"] = df["iterations_result"] > 0

    grouped = (
        df.groupby(["answer_before_correct_result", "intervention_present"])[
            "answer_correct_after_result"
        ]
        .agg(["count", "mean"])
        .reset_index()
    )

    labels = ["Incorrect", "Correct"]
    x = [0, 1]

    means_no_int = []
    counts_no_int = []
    means_int = []
    counts_int = []

    for prev_corr in [0, 1]:
        no_int_row = grouped[
            (grouped["answer_before_correct_result"] == prev_corr)
            & (grouped["intervention_present"] == False)
        ]
        int_row = grouped[
            (grouped["answer_before_correct_result"] == prev_corr)
            & (grouped["intervention_present"] == True)
        ]

        means_no_int.append(no_int_row["mean"].values[0] if not no_int_row.empty else 0)
        counts_no_int.append(
            no_int_row["count"].values[0] if not no_int_row.empty else 0
        )

        means_int.append(int_row["mean"].values[0] if not int_row.empty else 0)
        counts_int.append(int_row["count"].values[0] if not int_row.empty else 0)

    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        [p - width / 2 for p in x],
        means_no_int,
        width,
        label="No Intervention",
        color="skyblue",
    )
    ax.bar(
        [p + width / 2 for p in x],
        means_int,
        width,
        label="Intervention",
        color="salmon",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion Correct After Intervention")
    ax.set_xlabel("Previous Answer Correct")
    ax.set_title(
        "Correctness After Intervention\nGrouped by Previous Correctness & Intervention Presence"
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(res_path, "correctness_by_intervention_and_prev_correct.png")
    )
    plt.close()

    with open(os.path.join(res_path, "intervention_effects_raw_numbers.txt"), "w") as f:
        f.write("Intervention Effects (Raw Numbers)\n")
        f.write(
            f"{'Prev Correct':<15}{'Intervention':<15}{'Count':<10}{'Mean Correctness':<20}\n"
        )
        for idx in range(2):
            for intervention_state in [False, True]:
                count = counts_int[idx] if intervention_state else counts_no_int[idx]
                mean_corr = means_int[idx] if intervention_state else means_no_int[idx]
                f.write(
                    f"{labels[idx]:<15}{str(intervention_state):<15}{count:<10}{mean_corr:<20.3f}\n"
                )


def analyse_iterations_vs_correctness(
    df: pd.DataFrame, result_path: str
) -> pd.DataFrame:
    """
    Analyse the relationship between the number of iterations and correctness after intervention.

    :param df: pd.DataFrame, the evaluation dataframe containing 'iterations' and 'answer_correct_after'
    :param result_path: str, path to save the results
    :return: pd.DataFrame with iteration counts as index and columns ['count', 'mean_correctness']
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    df = df.copy()

    stats = (
        df.groupby("iterations_result")["answer_correct_after_result"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "mean_correctness"})
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stats.index, stats["mean_correctness"], marker="o", linestyle="-")
    ax.set_title(f"Correctness After Intervention vs Number of Iterations")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Mean Correctness After Intervention")
    ax.set_xticks(stats.index)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "iterations_vs_correctness.png"))
    plt.close()

    with open(
        os.path.join(result_path, "iterations_vs_correctness_stats.txt"), "w"
    ) as f:
        f.write(f"Correctness After Intervention vs Number of Iterations\n")
        f.write(f"{'Iterations':<12}{'Count':<10}{'Mean Correctness':<20}\n")
        for iterations, row in stats.iterrows():
            f.write(
                f"{iterations:<12}{row['count']:<10}{row['mean_correctness']:<20.3f}\n"
            )

    return stats


def main():
    setting = "SD"
    data_path = "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/all_tasks_joined_old"
    result_path = f"/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/all_tasks_stats"
    df = process_eval_dicts(data_path)
    res_df = pd.read_csv(
        os.path.join(data_path, "joined__results_task_results.csv"), sep="\t"
    )
    merged_df = combine_eval_dfs(eval_df=df, result_df=res_df, result_path=result_path)
    run_stats(df=merged_df, result_path=result_path, setting=setting)
    analyse_effects(merged_df, res_path=result_path)
    analyse_iterations_vs_correctness(merged_df, result_path=result_path)

    setting = "Feedback"
    data_path = "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v2/all_tasks_joined"
    result_path = f"/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v2/all_tasks_stats"
    df = process_eval_dicts(data_path)
    res_df = pd.read_csv(
        os.path.join(data_path, "joined_reasoning_results.csv"), sep="\t"
    )
    merged_df = combine_eval_dfs(eval_df=df, result_df=res_df, result_path=result_path)
    run_stats(df=merged_df, result_path=result_path, setting=setting)
    analyse_effects(merged_df, res_path=result_path)
    analyse_iterations_vs_correctness(merged_df, result_path=result_path)


if __name__ == "__main__":
    main()
