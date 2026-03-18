import pandas as pd


def create_master_file(data_paths: dict[str, str], output_path: str):
    """
    Create a master result file that contains all the setting results.

    :param data_paths: dict[str, str], keys: setting names, values: paths to the result file
    :param output_path: str, path to save the master result file
    """
    master_df = pd.DataFrame()

    for setting, path in data_paths:
        df = pd.read_csv(path)
        master_df = pd.merge(
            [master_df, df],
            on=["task_id", "sample_id", "part_id"],
            ignore_index=True,
            suffixes=[None, f"_{setting}"],
        )

    print(f"Saving master result file to {output_path}...")
    master_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    data_paths = {
        "baseline": "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/baseline/test/reasoning/v1/all_tasks_joined/joined_reasoning_results.csv",
        # "basic_baseline": "",
        "skyline": "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/skyline/test/reasoning/v1/all_tasks_joined/joined_reasoning_results.csv",
        "feedback": "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/feedback/test/reasoning/v2/all_tasks_joined/joined_reasoning_results.csv",
        "sd": "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/SD/test/reasoning/v1/all_tasks_joined/joined_reasoning_results.csv",
    }

    output_path = "/pfs/work9/workspace/scratch/hd_mr338-research-results-2/master_reasoning_results.zip"

    create_master_file(data_paths=data_paths, output_path=output_path)
