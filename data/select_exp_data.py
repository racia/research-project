"""Script to select a portion of data from a given data subset of the bAbI tasks"""

bAbI_small_data_path = '../../bAbI_tasks/en-valid/'
dataset_type = 'valid'
samples_per_task_type = 10

import os
import csv


def process_line(line, context, questions):
    line = line.strip()

    if '\t' in line:
        parts = line.split('\t')
        context.append(parts[0])
        questions.append('\t'.join(parts))
    else:
        context.append(line)


def into_table(path, base_path, sample_id=1, table='validation.csv'):
    all_contexts = []
    all_questions = []
    with open(base_path+path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        context = []
        questions = []
        process_line(lines[0], context, questions)
        for line in lines[1:]:
            if line.startswith('1 '):
                all_contexts.append('\n'.join(context))
                all_questions.append('\n'.join(questions))
                context.clear()
                questions.clear()

            process_line(line, context, questions)

        all_contexts.append('\n'.join(context))
        all_questions.append('\n'.join(questions))

    print(f'File "{path}" is read.')

    with open(table, 'a+', encoding='utf-8') as result:
        writer = csv.writer(result, delimiter=';')
        task = path.split('.')[0]

        for cont, q in zip(all_contexts[:samples_per_task_type], all_questions[:samples_per_task_type]):
            writer.writerow([sample_id, task, cont, q])
            sample_id += 1

    print(f'File "{path}" is saved into the table "{table}".')
    return sample_id


task_files = [file for file in os.listdir(bAbI_small_data_path) if dataset_type in file]
sample_id = 1

for file in task_files:
    sample_id = into_table(file, bAbI_small_data_path, sample_id)
