# This script jointly evaluates two same runs of different settings: one with the reasoning and one without.
# It compares the results and outputs a summary of the differences in the answers to detect "Toxic-CoT" problems.
# It can also be used to evaluate interventional methods by comparing the results of before and after the intervention.
import logging
from pathlib import Path

import pandas as pd

from plots.Plotter import Plotter

ID_COLS = ["task_id", "sample_id", "part_id"]
ATTRS = [
    "max_supp_attn",
    "attn_on_target",
    "there",
    "verbs",
    "pronouns",
    "not_mentioned",
    "context_sents_hall",
    "answer_not_mentioned",
    "empty_attn_scores",
]


def load_eval_table(run_dir: Path, label: str) -> pd.DataFrame:
    """
    Load the evaluation table from a run directory.
    Assumes a single *_upd.csv file containing the results.

    :param run_dir: Path to the evaluation directory of the run.
    :param label: A label to identify the run (e.g., "reas" or "da") to suffix the columns with.
    :return: A DataFrame containing the evaluation results with an added 'run' column for the label.
    """
    if logging.getLogger().level == logging.DEBUG:
        breakpoint()
    csv_files = list(run_dir.glob("*_upd.csv"))
    if len(csv_files) != 1:
        raise ValueError(
            f"Expected exactly one *_upd.csv in {run_dir}, found {csv_files}"
        )
    df = pd.read_csv(csv_files[0], sep="\t")
    df["run"] = label
    return df


def build_wide_table(df_reas: pd.DataFrame, df_da: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the two DataFrames from the reasoning and direct answer runs into a single wide-format DataFrame.
    This allows for direct comparison of the results on a per-example basis.
    :param df_reas: DataFrame from the reasoning run, with columns suffixed by "_reas".
    :param df_da: DataFrame from the direct answer run, with columns suffixed by "_da".
    :return: A merged DataFrame containing columns from both runs, with suffixes to distinguish them,
    and only rows that exist in both runs.
    """
    if logging.getLogger().level == logging.DEBUG:
        breakpoint()
    # Suffix columns to distinguish runs
    df_da = df_da.add_suffix("_da")
    df_reas = df_reas.add_suffix("_reas")

    # Restore ID columns (they got suffixed)
    for col in ID_COLS:
        df_da[col] = df_da[f"{col}_da"]
        df_da.drop(columns=f"{col}_da", inplace=True, errors="ignore", axis=1)
        df_reas[col] = df_reas[f"{col}_reas"]
        df_reas.drop(columns=f"{col}_reas", inplace=True, errors="ignore", axis=1)

    # Inner join: keep only examples that exist in both runs
    merged = pd.merge(df_reas, df_da, on=ID_COLS, how="inner")
    return merged


def add_toxic_cot_flags(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """
    Add columns to the DataFrame to indicate whether each example falls into one of the following categories:
        - both_correct: *answers* in both reasoning and direct answer runs are correct
        - both_incorrect: *answers* in both reasoning and direct answer runs are incorrect
        - reas_only_correct: answer is correct only in the reasoning run
        - da_only_correct: answer is correct only the direct answer run
        - toxic_cot: cases for which reasoning flips a correct answer into an incorrect one

    :param df: The merged DataFrame, containing results from both runs.
    :param version: The result version for the columns ("before" or "after") to identify which answers to compare.
    :return: The DataFrame with added boolean columns for each of the categories described above.
    """
    if logging.getLogger().level == logging.DEBUG:
        breakpoint()
    assert version in ("before", "after"), "Version must be 'before' or 'after'"

    # Adjust column names to your real ones
    correct_da = df[f"answer_correct_{version}_da"]
    correct_reas = df[f"answer_correct_{version}_reas"]

    df[f"both_correct_{version}"] = correct_da & correct_reas
    df[f"both_incorrect_{version}"] = (~correct_da) & (~correct_reas)
    df[f"reas_only_correct_{version}"] = (~correct_da) & correct_reas
    df[f"da_only_correct_{version}"] = correct_da & (~correct_reas)

    # cases where reasoning flips a correct answer into an incorrect one.
    df[f"toxic_cot_{version}"] = correct_da & (~correct_reas)

    return df


def summarize_global(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """
    Summarize the counts and ratios of each category
    (both_correct, both_incorrect, reas_only_correct, da_only_correct, toxic_cot)
    across the entire dataset. Correctness is defined only by answer itself, without the reasoning.

    :param df: The DataFrame containing the merged results with the added category columns.
    :param version: The result version for the columns ("before" or "after").
    :return: A DataFrame summarizing the count and ratio of each category across the whole
    """
    if logging.getLogger().level == logging.DEBUG:
        breakpoint()
    counts = df[
        [
            f"both_correct_{version}",
            f"both_incorrect_{version}",
            f"da_only_correct_{version}",
            f"reas_only_correct_{version}",
            f"toxic_cot_{version}",
        ]
    ].sum()
    total = len(df)
    summary = (
        pd.DataFrame({"count": counts, "ratio": counts / total})
        .reset_index()
        .rename(columns={"index": "category"})
    )
    return summary


def summarize_by_task(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """
    Summarize the counts and ratios of each category by task_id,
    to see if certain tasks are more prone to reasoning errors or toxic CoT issues.

    :param df: The DataFrame containing the merged results with the added category columns.
    :param version: The result version for the columns ("before" or "after").
    :return: A DataFrame summarizing the count and ratio of each category for each task_id.
    """
    if logging.getLogger().level == logging.DEBUG:
        breakpoint()
    group = df.groupby("task_id")
    metrics = group[
        [
            f"both_correct_{version}",
            f"both_incorrect_{version}",
            f"da_only_correct_{version}",
            f"reas_only_correct_{version}",
            f"toxic_cot_{version}",
        ]
    ].mean()
    metrics["n_examples"] = group.size()
    return metrics.reset_index()


def summarize_attributes_global(df: pd.DataFrame, multi_system: bool) -> pd.DataFrame:
    """
    Global summary for each attribute and version, for both runs.
    Rows: attribute
    Columns: mean_reas_before, mean_reas_after, mean_da_before, mean_da_after, diff_before, diff_after
    """
    if logging.getLogger().level == logging.DEBUG:
        breakpoint()
    records = []

    for attr in ATTRS:
        col_da_before = f"{attr}_before_da"
        col_da_after = f"{attr}_after_da"
        col_reas_before = f"{attr}_before_reas"
        col_reas_after = f"{attr}_after_reas"

        # Skip attributes that aren't fully present
        needed = [col_da_before, col_reas_before]
        if multi_system:
            needed += [col_reas_after, col_da_after]
        if any(c not in df.columns for c in needed):
            continue

        m_da_before = df[col_da_before].mean()
        m_reas_before = df[col_reas_before].mean()
        row = {
            "attribute": attr,
            "mean_da_before": m_da_before,
            "mean_reas_before": m_reas_before,
            "diff_before": m_reas_before - m_da_before,
        }
        if multi_system:
            m_reas_after = df[col_reas_after].mean()
            m_da_after = df[col_da_after].mean()
            row.update(
                {
                    "mean_da_after": m_da_after,
                    "mean_reas_after": m_reas_after,
                    "diff_after": m_reas_after - m_da_after,
                }
            )

        records.append(row)

    return pd.DataFrame.from_records(records)


def summarize_attributes_by_task(df: pd.DataFrame, multi_system: bool) -> pd.DataFrame:
    """
    Per-task summary for each attribute.
    Rows: (task_id, attribute)
    """
    if logging.getLogger().level == logging.DEBUG:
        breakpoint()
    rows = []

    for attr in ATTRS:
        col_da_before = f"{attr}_before_da"
        col_da_after = f"{attr}_after_da"
        col_reas_before = f"{attr}_before_reas"
        col_reas_after = f"{attr}_after_reas"

        needed = [col_da_before, col_reas_before]
        if multi_system:
            needed += [col_da_after, col_reas_after]
        if any(c not in df.columns for c in needed):
            continue

        grouped = df.groupby("task_id")

        m_da_before = grouped[col_da_before].mean()
        m_reas_before = grouped[col_reas_before].mean()
        m_reas_after, m_da_after = None, None
        if multi_system:
            m_da_after = grouped[col_da_after].mean()
            m_reas_after = grouped[col_reas_after].mean()

        for task_id in m_reas_before.index:
            row = {
                "task_id": task_id,
                "attribute": attr,
                "mean_da_before": m_da_before.loc[task_id],
                "mean_reas_before": m_reas_before.loc[task_id],
                "diff_before": m_reas_before.loc[task_id] - m_da_before.loc[task_id],
            }
            if multi_system:
                row.update(
                    {
                        "mean_da_after": m_da_after.loc[task_id],
                        "mean_reas_after": m_reas_after.loc[task_id],
                        "diff_after": m_reas_after.loc[task_id]
                        - m_da_after.loc[task_id],
                    }
                )
            rows.append(row)

    return pd.DataFrame.from_records(rows)


def run(
    reasoning_path: Path, direct_answer_path: Path, multi_system: bool, out_dir: Path
):
    logging.info("Running evaluation of direct answers and reasoning...")

    out_dir = out_dir / "da_reasoning_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_da = load_eval_table(direct_answer_path, label="da")
    df_reas = load_eval_table(reasoning_path, label="reas")
    merged = build_wide_table(df_reas, df_da)

    merged_before = add_toxic_cot_flags(merged, version="before")
    if multi_system:
        merged_after = add_toxic_cot_flags(merged, version="after")
        merged_by_versions = [merged_before, merged_after]
        merged_together = pd.merge(
            merged_before,
            merged_after,
            on=["task_id", "sample_id", "part_id"],
            suffixes=("_before", "_after"),
        )
    else:
        merged_by_versions = [merged_before]
        merged_together = merged_before

    plotter = Plotter(results_path=out_dir, color_map="tab20")
    VERSIONS = ["before", "after"] if multi_system else ["before"]

    for version, df in zip(VERSIONS, merged_by_versions):
        summary_global = summarize_global(df, version=version)
        summary_by_task = summarize_by_task(df, version=version)
        summary_global.to_csv(out_dir / f"summary_{version}_global.csv", index=False)
        summary_by_task.to_csv(out_dir / f"summary_{version}_by_task.csv", index=False)

        for group in ("task_id", "sample_id", "part_id"):
            plotter.plot_acc_two_runs_per(
                group,
                df,
                y_label="Accuracy",
                file_name=f"acc_two_runs_per_{group}.png",
                version=version,
                plot_name_add=[
                    version,
                    "comparison",
                    "multi_system=" + str(multi_system),
                ],
            )
            if logging.getLogger().level == logging.DEBUG:
                breakpoint()

            plotter.plot_toxic_cot_per(group, df, version, plot_name_add=[version])

            if logging.getLogger().level == logging.DEBUG:
                breakpoint()

            plotter.plot_acc_and_toxic_cot(group, df, version, plot_name_add=[version])

        if logging.getLogger().level == logging.DEBUG:
            breakpoint()

        for attr in ATTRS:
            col_da = f"{attr}_{version}_da"
            col_reas = f"{attr}_{version}_reas"

            # Skip gracefully if an attribute is missing in the merged table
            if col_reas not in df.columns or col_da not in df.columns:
                continue

            mean_df = df.groupby("task_id")[[col_reas, col_da]].mean().reset_index()
            mean_df["diff"] = mean_df[col_reas] - mean_df[col_da]

            label = f"{attr.replace('_', ' ').title()} Difference (reas - da)"

            plotter.plot_diff_two_runs_per_task(
                mean_df,
                label,
                file_name=f"{attr}_diff_two_runs_per_task.png",
                plot_name_add=[version],
            )

    if logging.getLogger().level == logging.DEBUG:
        breakpoint()

    if multi_system:
        plotter.plot_toxic_cot_transition_overview(
            merged_together,
            file_name="toxic_cot_overview_between_versions.pdf",
            plot_name_add=["comparison"],
        )
    plotter.plot_correctness_agreement(
        merged_together,
        VERSIONS,
    )
    plotter.plot_attr_agreement(
        merged_together,
        VERSIONS,
    )
    plotter.plot_attrs_by_runs_versions_toxicity(merged_together, ATTRS, multi_system)

    if logging.getLogger().level == logging.DEBUG:
        breakpoint()

    for attr in ATTRS:
        col_da_before = f"{attr}_before_da"
        col_da_after = f"{attr}_after_da"
        col_reas_before = f"{attr}_before_reas"
        col_reas_after = f"{attr}_after_reas"

        # Skip gracefully if any of the four columns are missing
        needed = [col_da_before, col_da_after, col_reas_before, col_reas_after]
        logging.debug(f"Attr: {attr}")
        logging.debug(f"Needed columns: {needed}")
        logging.debug(f"Present columns: {merged.columns}")
        if any(col not in merged.columns for col in needed):
            logging.warning(
                f"Skipping attribute '{attr}' because not all needed columns are present in the merged DataFrame."
            )
            continue

        mean_da_before = merged.groupby("task_id")[col_da_before].mean()
        mean_reas_before = merged.groupby("task_id")[col_reas_before].mean()
        mean_reas_after, mean_da_after = None, None
        if multi_system:
            mean_da_after = merged.groupby("task_id")[col_da_after].mean()
            mean_reas_after = merged.groupby("task_id")[col_reas_after].mean()

        pretty_name = attr.replace("_", " ").capitalize()
        vals_reas = (
            [mean_reas_before.to_dict(), mean_reas_after.to_dict()]
            if multi_system
            else [mean_reas_before.to_dict()]
        )
        vals_da = (
            [mean_da_before.to_dict(), mean_da_after.to_dict()]
            if multi_system
            else [mean_da_before.to_dict()]
        )
        plotter.plot_attr_before_after_two_runs_per_task(
            vals_da=vals_da,
            vals_reas=vals_reas,
            y_label=f"{pretty_name}",
            file_name=(
                f"{attr}_before_after_two_runs_per_task.png"
                if multi_system
                else f"{attr}_before_two_runs_per_task.png"
            ),
            plot_name_add=["comparison", attr],
        )

    attr_global = summarize_attributes_global(merged, multi_system=multi_system)
    attr_global.to_csv(out_dir / "summary_attributes_global.csv", index=False)
    attr_by_task = summarize_attributes_by_task(merged, multi_system=multi_system)
    attr_by_task.to_csv(out_dir / "summary_attributes_by_task.csv", index=False)

    merged_together.to_csv(
        out_dir / "results_from_both_runs.csv", index=False, sep="\t"
    )

    print("Comparison completed. Results saved to:", out_dir)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Compare two runs of the same task.")
    parser.add_argument(
        "--run_da",
        type=Path,
        required=True,
        help="Path to the evaluation directory of the run without reasoning.",
    )
    parser.add_argument(
        "--run_with_reas",
        type=Path,
        required=True,
        help="Path to the evaluation directory of the run with reasoning.",
    )
    parser.add_argument(
        "--multi_system",
        action="store_true",
        help="Whether the runs are from a multi-system evaluation ('before' and 'after' an intervention).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save the comparison results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # import pdb; pdb.set_trace()

    # python3 evaluate_da_reasoning.py \
    #   --run_with_reas /path/to/reasoning_run/eval \
    #   --run_without_reas /path/to/direct_run/eval \
    #   --out_dir /path/to/comparison

    # logging.basicConfig(level=logging.INFO)
    # args = parse_args()
    # run(
    #     direct_answer_path=args.run_da,
    #     reasoning_path=args.run_with_reas,
    #     multi_system=args.multi_system,
    #     out_dir=args.out_dir,
    # )

    logging.basicConfig(level=logging.INFO)
    run(
        reasoning_path=Path(
            "/workspace/students/reasoning/results/basic-baseline/test/reasoning/v1/all_tasks_joined/eval"
        ),
        direct_answer_path=Path(
            "/workspace/students/reasoning/results/basic-baseline/test/da/v1/all_tasks_joined/eval"
        ),
        multi_system=False,
        out_dir=Path(
            "/workspace/students/reasoning/results/basic-baseline/test/toxic_eval_test"
        ),
    )
