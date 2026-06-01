import pandas as pd


def main():
    """
    Create separate CSV files for each task_id from the joined silver_reasoning_claude.csv file.
    :return:
    """
    input_csv = "silver_reasoning_claude.csv"

    # Read the joined CSV
    df = pd.read_csv(input_csv)

    # Group by task_id and save each group
    for task_id, group in df.groupby("task_id"):
        output_file = f"silver_reasoning_test_{task_id}.csv"
        group.to_csv(output_file, index=False)

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
