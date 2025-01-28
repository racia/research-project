import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data.DataSaver import DataSaver
from data.DataLoader import DataLoader
from data.DataProcessor import DataProcessor
from settings.feedback.Feedback import Feedback
from prompts.Prompt import Prompt
from sklearn.metrics import accuracy_score
from data.Statistics import Statistics as St
from settings.feedback.Feedback import Feedback



@hydra.main(version_base=None)
def run(cfg: DictConfig):
    """
    Run the feedback experiment.

    :return:
    """
    OmegaConf.resolve(cfg)

    print("Config data for the run:", OmegaConf.to_yaml(cfg), end="\n\n", flush=True)
    print("Running the script...")

    saver = DataSaver(project_dir=cfg.repository.path, save_to_repo=True)
    loader = DataLoader()
    processor = DataProcessor()

    path = "../../data/tasks_1-20_v1-2/en-valid"

    #results_path = cfg.repository.path + cfg.results.path
    results_path = os.path.join(cfg.repository.path + cfg.results.path, "results.log")

    log_file = sys.stdout

    if cfg.results.print_to_file:
         log_file = saver.redirect_printing_to_log_file(results_path)

    saver.set_results_details(results_path=results_path, headers=cfg.results.headers)

    # Get the prompts
    init_prompt = Prompt(prompt_path=cfg.init_prompt.path)
    feedback_prompt = Prompt(prompt_path=cfg.feedback_prompt.path)
    refine_prompt = Prompt(prompt_path=cfg.refine_prompt.path)

    # Initialize the speculative decoding setting
    feedback = Feedback(
        teacher=cfg.teacher.name,
        student=cfg.student.name,
        init_prompt=init_prompt,
        feedback_prompt=feedback_prompt,
        refine_prompt=refine_prompt,
        teacher_max_new_tokens=cfg.teacher.max_new_tokens,
        student_max_new_tokens=cfg.student.max_new_tokens,
        logfile=log_file,
    )

    # Run the speculative decoding setting
    feedback.total_tasks = 0
    data_in_splits = {}

    for split, to_fetch in cfg.data.splits.items():
        if to_fetch:
            data_tasks = loader.load_data(
                path=cfg.data.path, split=split, tasks=cfg.data.task_ids
            )
            processed_data = processor.process_data(data=data_tasks)
            feedback.total_tasks += len(data_tasks)
            data_in_splits[split] = processed_data

    print("The data is loaded successfully", end="\n\n", file=log_file)
    print("Starting to query the model", end="\n\n", file=log_file)

    for split, tasks in data_in_splits.items():
        for task_id, task in tasks.items():
            task_result = feedback.iterate_task(
                task_id=task_id,
                task_data=task,
                no_samples=cfg.data.samples_per_task,
                to_enumerate=cfg.data.to_enumerate,
                parse_output=cfg.results.parse,
            )
            saver.save_output(data=task_result)
            print("______________________________", end="\n\n", file=log_file)

    print("The run is finished successfully", file=log_file)

    print("\n- RUN RESULTS -", end="\n\n", file=log_file)

    print(
        "Processed",
        feedback.total_tasks,
        "tasks in total with",
        cfg.data.samples_per_task,
        "samples in each",
        file=log_file,
    )
    print(
        "Total samples processed",
        feedback.total_tasks * cfg.data.samples_per_task,
        end="\n\n",
        file=log_file,
    )

    feedback.accuracy = round(accuracy_score(feedback.y_true, feedback.y_pred), 2)
    print("General accuracy:", feedback.accuracy, file=log_file)

    feedback.soft_match_accuracy = round(
        St.soft_match_accuracy_score(feedback.y_true, feedback.y_pred), 2
    )
    print("General soft match accuracy:", feedback.soft_match_accuracy, file=log_file)

    row = [
        {
            "accuracy": feedback.accuracy,
            "soft_match_accuracy": feedback.soft_match_accuracy,
        }
    ]
    saver.save_output(data=row)

    if cfg.results.print_to_file:
        # console printing must be returned
        # if printing was redirected to logs created at the beginning of the script
        # 'log_file' will exist in that case as well
        saver.return_console_printing(log_file)


if __name__ == "__main__":
    run()
