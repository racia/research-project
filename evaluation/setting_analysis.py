from pathlib import Path
from typing import Dict, List
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import evaluate
from scipy.stats import pearsonr


# ============================================================
# LOADING FUNCTIONS
# ============================================================

def parse_feedback_content(file_path: Path, file_type: str = 'teacher') -> Dict:
    """Parse feedback file content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    if not lines:
        return {'is_malformed': True, 'error_type': 'Empty_file', 'full_content': content}

    first_line = lines[0].strip()
    pattern = r'\b(?:correct|incorrect|reason(?:ing)?|answer)\b' if file_type == 'teacher' \
        else r'(?:\b(?:correct|incorrect|reason(?:ing)?|answer)\b|here is|i apologize|i see|there is|my mistake|corrected answer)'

    match = re.match(pattern, first_line, flags=re.IGNORECASE)
    if not match:
        return {'is_malformed': True, 'error_type': 'Hallucination', 'full_content': content}

    first_word_lower = first_line.lower()
    if first_word_lower.startswith(('correct.', 'incorrect.')):
        is_correct = first_word_lower.startswith('correct.')
        body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''
    else:
        is_correct = None
        body = content.strip()

    return {'is_malformed': False, 'is_correct': is_correct, 'body': body,
            'full_content': content, 'error_type': None}


def get_feedback_files(path: str, file_type: str = 'teacher'):
    """Get feedback file paths organized by task."""
    feedback_paths = {}
    pattern = rf'(\d+)_{file_type}_(\d+)-(\d+)-(\d+)\.txt'

    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            match = re.match(pattern, file)
            if match:
                iteration, task_id, sample_id, part_id = match.groups()
                task_key = f"task_{task_id}"
                file_path = os.path.join(dirpath, file)

                if task_key not in feedback_paths:
                    feedback_paths[task_key] = []
                feedback_paths[task_key].append(file_path)

    print(f"Found tasks: {feedback_paths.keys()}")
    print(f"Found {file_type} files: {sum(len(v) for v in feedback_paths.values())}")
    return feedback_paths


def read_feedback_files(feedback_paths: dict, file_type: str = 'teacher') -> dict[str, pd.DataFrame]:
    """Read feedback files into dataframes organized by task."""
    feedback_dfs = {}

    for task, paths in feedback_paths.items():
        records = []
        pattern = rf'(\d+)_{file_type}_(\d+)-(\d+)-(\d+)'

        for file_path in paths:
            file_name = Path(file_path).stem
            match = re.match(pattern, file_name)

            if match:
                iteration, task_id, sample_id, part_id = match.groups()
                parsed = parse_feedback_content(Path(file_path), file_type=file_type)

                record = {
                    'file_name': file_name,
                    'file_type': file_type,
                    'iteration': int(iteration),
                    'task_id': int(task_id),
                    'sample_id': int(sample_id),
                    'part_id': int(part_id),
                    'is_correct': parsed.get('is_correct'),
                    'is_malformed': parsed.get('is_malformed'),
                    'body': parsed.get('body'),
                    'error_type': parsed.get('error_type'),
                    'full_content': parsed.get('full_content'),
                }
                records.append(record)

        if records:
            feedback_dfs[task] = pd.DataFrame(records)

    return feedback_dfs


def clean_feedback_dfs(feedback_dfs: dict) -> dict[str, pd.DataFrame]:
    """Clean feedback dataframes."""
    for task, df in feedback_dfs.items():
        df.drop_duplicates(inplace=True, subset=['iteration', 'task_id', 'sample_id', 'part_id'])
    return feedback_dfs


def process_feedback_files(path: str, file_type: str = 'teacher') -> dict[str, pd.DataFrame]:
    """Get paths, read, and clean feedback files."""
    paths = get_feedback_files(path, file_type)
    feedback_dfs = read_feedback_files(paths, file_type)
    clean_dfs = clean_feedback_dfs(feedback_dfs)
    return clean_dfs


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyse_correction_success(teacher_dfs: dict[str, pd.DataFrame],
                               student_dfs: dict[str, pd.DataFrame],
                               result_path: str) -> pd.DataFrame:
    """Analyse correction success rate per iteration."""
    all_tasks = set(teacher_dfs.keys()) & set(student_dfs.keys())
    success_by_iteration = defaultdict(lambda: {'success': 0, 'total': 0})

    for task in all_tasks:
        teacher_df = teacher_dfs[task]
        student_df = student_dfs[task]

        for (task_id, sample_id, part_id), group in teacher_df.groupby(['task_id', 'sample_id', 'part_id']):
            group_sorted = group.sort_values('iteration')

            for i, row in group_sorted.iterrows():
                if row['is_correct'] == False:
                    iteration = row['iteration']
                    next_teacher = teacher_df[
                        (teacher_df['task_id'] == task_id) &
                        (teacher_df['sample_id'] == sample_id) &
                        (teacher_df['part_id'] == part_id) &
                        (teacher_df['iteration'] == iteration + 1)
                        ]
                    if not next_teacher.empty:
                        success = next_teacher.iloc[0]['is_correct'] == True
                        success_by_iteration[iteration]['total'] += 1
                        if success:
                            success_by_iteration[iteration]['success'] += 1

    results = []
    for iteration in sorted(success_by_iteration.keys()):
        stats = success_by_iteration[iteration]
        rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        results.append({'iteration': iteration, 'success_rate': rate, 'n_samples': stats['total']})

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['iteration'], df['success_rate'], marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Correction Success Rate (%)')
    ax.set_title('Student Correction Success Rate by Iteration')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(result_path, 'correction_success_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    df.to_csv(os.path.join(result_path, 'correction_success_stats.csv'), index=False)
    return df




def analyse_semantic_similarity(student_dfs: dict[str, pd.DataFrame], result_path: str) -> pd.DataFrame:
    """Analyse semantic similarity."""
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    results = []

    for task, student_df in student_dfs.items():
        student_df = student_df.query('not is_malformed')

        for (task_id, sample_id, part_id), group in student_df.groupby(['task_id', 'sample_id', 'part_id']):
            group_sorted = group.sort_values('iteration')

            for i in range(len(group_sorted) - 1):
                current = group_sorted.iloc[i]
                next_row = group_sorted.iloc[i + 1]

                if current['body'] and next_row['body']:
                    results.append({
                        'task_id': task_id, 'sample_id': sample_id, 'part_id': part_id,
                        'iteration': current['iteration'],
                        'bleu': bleu.compute(predictions=[next_row['body']], references=[[current['body']]])['bleu'],
                        'rouge': rouge.compute(predictions=[next_row['body']], references=[current['body']])['rougeL'],
                        'meteor': meteor.compute(predictions=[next_row['body']], references=[current['body']])['meteor']
                    })

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, metric in enumerate(['bleu', 'rouge', 'meteor']):
        axes[idx].hist(df[metric], bins=50, alpha=0.7)
        axes[idx].set_xlabel(f'{metric.upper()} Score')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{metric.upper()} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'semantic_similarity_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

    summary = df[['bleu', 'rouge', 'meteor']].describe()
    summary.to_csv(os.path.join(result_path, 'semantic_similarity_stats.csv'))

    return df


def analyse_task_difficulty(teacher_dfs: dict[str, pd.DataFrame],
                            student_dfs: dict[str, pd.DataFrame],
                            result_path: str) -> pd.DataFrame:
    """Rank tasks by difficulty."""
    results = []

    all_tasks = sorted(set(teacher_dfs.keys()) | set(student_dfs.keys()))

    for task in all_tasks:
        task_id = int(task.split('_')[1])
        task_teacher = teacher_dfs.get(task, pd.DataFrame())
        task_student = student_dfs.get(task, pd.DataFrame())

        if not task_teacher.empty:
            avg_iterations = task_teacher.groupby(['sample_id', 'part_id'])['iteration'].max().mean()
            teacher_hall_rate = (task_teacher['is_malformed'].sum() / len(task_teacher) * 100)
        else:
            avg_iterations = 0
            teacher_hall_rate = 0

        student_hall_rate = (task_student['is_malformed'].sum() / len(task_student) * 100) \
            if not task_student.empty else 0

        difficulty_score = avg_iterations + (teacher_hall_rate + student_hall_rate) / 20

        results.append({
            'task_id': task_id,
            'avg_iterations': avg_iterations,
            'teacher_hallucination_rate': teacher_hall_rate,
            'student_hallucination_rate': student_hall_rate,
            'difficulty_score': difficulty_score
        })

    df = pd.DataFrame(results).sort_values('difficulty_score', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    ax.bar(x, df['difficulty_score'])
    ax.set_xticks(x)
    ax.set_xticklabels(df['task_id'])
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Difficulty Score')
    ax.set_title('Task Difficulty Ranking')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'task_difficulty_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()

    df.to_csv(os.path.join(result_path, 'task_difficulty_ranking.csv'), index=False)
    return df


def analyse_feedback_effectiveness(teacher_dfs: dict[str, pd.DataFrame], result_path: str) -> pd.DataFrame:
    """Analyse feedback effectiveness: correct after 0, 1, or multiple hints."""
    results = []

    for task, teacher_df in teacher_dfs.items():
        for (task_id, sample_id, part_id), group in teacher_df.groupby(['task_id', 'sample_id', 'part_id']):
            group_sorted = group.sort_values('iteration')

            hints_given = 0
            for i, row in group_sorted.iterrows():
                if row['is_correct'] == False:
                    hints_given += 1
                elif row['is_correct'] == True:
                    results.append({
                        'task_id': task_id, 'sample_id': sample_id, 'part_id': part_id,
                        'hints_to_success': hints_given, 'total_iterations': row['iteration'] + 1
                    })
                    break

    df = pd.DataFrame(results)

    effectiveness = {
        '0 hints': (df['hints_to_success'] == 0).sum(),
        '1 hint': (df['hints_to_success'] == 1).sum(),
        '2-3 hints': ((df['hints_to_success'] >= 2) & (df['hints_to_success'] <= 3)).sum(),
        '4+ hints': (df['hints_to_success'] >= 4).sum()
    }

    total = sum(effectiveness.values())
    percentages = {k: v / total * 100 for k, v in effectiveness.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(percentages.keys(), percentages.values())
    ax.set_ylabel('Percentage of Parts (%)')
    ax.set_title('Feedback Effectiveness: Hints to Success')
    plt.savefig(os.path.join(result_path, 'feedback_effectiveness.png'), dpi=300, bbox_inches='tight')
    plt.close()

    with open(os.path.join(result_path, 'feedback_effectiveness_stats.txt'), 'w') as f:
        f.write(f"Mean hints to success: {df['hints_to_success'].mean():.2f}\n")
        f.write(f"Median hints to success: {df['hints_to_success'].median():.2f}\n")
        f.write(
            f"Success after 0 hints: {effectiveness['0 hints']} ({effectiveness['0 hints'] / len(df) * 100:.1f}%)\n")
        f.write(f"Success after 1 hint: {effectiveness['1 hint']} ({effectiveness['1 hint'] / len(df) * 100:.1f}%)\n")

    df.to_csv(os.path.join(result_path, 'feedback_effectiveness_stats.csv'), index=False)
    return df


def analyse_correlations(task_difficulty_df: pd.DataFrame, result_path: str):
    """Analyse correlations."""
    corr_data = task_difficulty_df[['avg_iterations', 'teacher_hallucination_rate',
                                    'student_hallucination_rate', 'difficulty_score']]
    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix: Task Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    corr_matrix.to_csv(os.path.join(result_path, 'correlation_matrix.csv'))

    with open(os.path.join(result_path, 'correlation_stats.txt'), 'w') as f:
        for col1, col2 in [('avg_iterations', 'teacher_hallucination_rate'),
                           ('avg_iterations', 'student_hallucination_rate'),
                           ('teacher_hallucination_rate', 'student_hallucination_rate')]:
            r, p = pearsonr(corr_data[col1], corr_data[col2])
            f.write(f"{col1} vs {col2}: r={r:.3f}, p={p:.4f}\n")


def analyse_hallucination_comparison(teacher_dfs: dict[str, pd.DataFrame],
                                     student_dfs: dict[str, pd.DataFrame],
                                     result_path: str) -> pd.DataFrame:
    """Compare hallucinations per task."""
    results = []
    all_tasks = sorted(set(teacher_dfs.keys()) | set(student_dfs.keys()))

    for task in all_tasks:
        task_id = int(task.split('_')[1])
        task_teacher = teacher_dfs.get(task, pd.DataFrame())
        task_student = student_dfs.get(task, pd.DataFrame())

        teacher_hall_rate = (task_teacher['is_malformed'].sum() / len(task_teacher) * 100) \
            if not task_teacher.empty else 0
        student_hall_rate = (task_student['is_malformed'].sum() / len(task_student) * 100) \
            if not task_student.empty else 0

        results.append({
            'task_id': task_id,
            'teacher_hallucination_rate': teacher_hall_rate,
            'student_hallucination_rate': student_hall_rate,
            'teacher_count': task_teacher['is_malformed'].sum() if not task_teacher.empty else 0,
            'student_count': task_student['is_malformed'].sum() if not task_student.empty else 0,
            'teacher_total': len(task_teacher),
            'student_total': len(task_student)
        })

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df['teacher_hallucination_rate'], width, label='Teacher', alpha=0.8)
    ax.bar(x + width / 2, df['student_hallucination_rate'], width, label='Student', alpha=0.8)
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Hallucination Rate (%)')
    ax.set_title('Teacher vs Student Hallucination Rates by Task')
    ax.set_xticks(x)
    ax.set_xticklabels(df['task_id'])
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'hallucination_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    df.to_csv(os.path.join(result_path, 'hallucination_comparison.csv'), index=False)
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_feedback_analysis(data_path: str, result_path: str):
    """Run complete feedback analysis."""

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    teacher_dfs = process_feedback_files(data_path, 'teacher')
    student_dfs = process_feedback_files(data_path, 'student')


    print("\n" + "=" * 80)
    print("SEMANTIC SIMILARITY")
    print("=" * 80)
    similarity_df = analyse_semantic_similarity(student_dfs, result_path)
    print(similarity_df[['bleu', 'rouge', 'meteor']].describe())

    print("\n" + "=" * 80)
    print("TASK DIFFICULTY RANKING")
    print("=" * 80)
    difficulty_df = analyse_task_difficulty(teacher_dfs, student_dfs, result_path)
    print(difficulty_df)

    print("\n" + "=" * 80)
    print("FEEDBACK EFFECTIVENESS")
    print("=" * 80)
    effectiveness_df = analyse_feedback_effectiveness(teacher_dfs, result_path)
    print(effectiveness_df.describe())

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    analyse_correlations(difficulty_df, result_path)

    print("\n" + "=" * 80)
    print("HALLUCINATION COMPARISON")
    print("=" * 80)
    hall_comparison_df = analyse_hallucination_comparison(teacher_dfs, student_dfs, result_path)
    print(hall_comparison_df)



    print(f"\nAll results saved to: {result_path}")


if __name__ == "__main__":
    data_path = "/pfs/work9/workspace/scratch/hd_nc326-research-project/feedback/test/reasoning/v1/all_tasks_joined/iterations"
    result_path = "/pfs/work9/workspace/scratch/hd_nc326-research-project/feedback/test/reasoning/v1/all_tasks_plots"

    run_feedback_analysis(data_path, result_path)
