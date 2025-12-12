from pathlib import Path
from typing import Dict
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
#import evaluate
#from scipy.stats import pearsonr

class FeedbackAnalysis:
    def __init__(self, teacher_dfs, student_dfs, result_path):
        self.teacher_dfs = teacher_dfs
        self.student_dfs = student_dfs
        self.result_path = result_path

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


    def analyse_correction_success(teacher_dfs: dict[str, pd.DataFrame],
                                   student_dfs: dict[str, pd.DataFrame],
                                   result_path: str) -> pd.DataFrame:
        """Analyse correction success rate per iteration."""
        all_tasks = set(teacher_dfs.keys()) & set(student_dfs.keys())
        success_by_iteration = defaultdict(lambda: {'success': 0, 'total': 0})

        for task in all_tasks:
            teacher_df = teacher_dfs[task]

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


    def analyse_results_without_hallucinations(
            teacher_dfs: dict[str, pd.DataFrame],
            student_dfs: dict[str, pd.DataFrame],
            result_path: str
    ) -> pd.DataFrame:
        """
        Compare results with and without teacher hallucinations.
        Shows how removing hallucinated feedback affects student performance.

        :param teacher_dfs: Teacher feedback dataframes by task
        :param student_dfs: Student feedback dataframes by task
        :param result_path: Path to save results
        :return: Comparison dataframe
        """
        results = []

        for task in teacher_dfs.keys():
            teacher_df = teacher_dfs[task]
            student_df = student_dfs.get(task, pd.DataFrame())

            if student_df.empty:
                continue

            # Identify parts with teacher hallucinations
            hallucinated_parts = teacher_df[teacher_df['is_malformed'] == True][
                ['task_id', 'sample_id', 'part_id']
            ].drop_duplicates()

            # Calculate metrics for ALL parts
            total_parts = len(student_df.groupby(['task_id', 'sample_id', 'part_id']))

            # Calculate metrics WITHOUT hallucinated parts
            merged = student_df.merge(
                hallucinated_parts,
                on=['task_id', 'sample_id', 'part_id'],
                how='left',
                indicator=True
            )
            clean_student_df = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)

            clean_parts = len(clean_student_df.groupby(['task_id', 'sample_id', 'part_id']))

            # Success rates
            all_success = (student_df['is_correct'] == True).sum() / len(student_df) * 100
            clean_success = (clean_student_df['is_correct'] == True).sum() / len(clean_student_df) * 100 if len(
                clean_student_df) > 0 else 0

            # Average iterations
            all_avg_iter = student_df.groupby(['task_id', 'sample_id', 'part_id'])['iteration'].max().mean()
            clean_avg_iter = clean_student_df.groupby(['task_id', 'sample_id', 'part_id'])['iteration'].max().mean() if len(
                clean_student_df) > 0 else 0

            results.append({
                'task': task,
                'total_parts': total_parts,
                'clean_parts': clean_parts,
                'removed_parts': total_parts - clean_parts,
                'all_success_rate': all_success,
                'clean_success_rate': clean_success,
                'success_rate_diff': clean_success - all_success,
                'all_avg_iterations': all_avg_iter,
                'clean_avg_iterations': clean_avg_iter,
                'iterations_diff': clean_avg_iter - all_avg_iter
            })

        df = pd.DataFrame(results)

        # Plot 1: Success rate comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        x = np.arange(len(df))
        width = 0.35

        ax1.bar(x - width / 2, df['all_success_rate'], width, label='All Parts', alpha=0.8)
        ax1.bar(x + width / 2, df['clean_success_rate'], width, label='Without Hallucinations', alpha=0.8)
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate: With vs Without Teacher Hallucinations')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['task'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Difference in success rate
        colors = ['green' if x > 0 else 'red' for x in df['success_rate_diff']]
        ax2.bar(x, df['success_rate_diff'], color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Success Rate Difference (%)')
        ax2.set_title('Impact of Removing Hallucinations\n(Positive = Improvement)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['task'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(result_path, 'results_without_hallucinations.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return df


    def analyse_never_agreed_cases(
            teacher_dfs: dict[str, pd.DataFrame],
            result_path: str
    ) -> pd.DataFrame:
        """
        Identify and analyze cases where teacher NEVER agreed with student across all iterations.

        :param teacher_dfs: Teacher feedback dataframes by task
        :param result_path: Path to save results
        :return: DataFrame of never-agreed cases
        """
        never_agreed = []

        for task, teacher_df in teacher_dfs.items():
            # Group by part
            for (task_id, sample_id, part_id), group in teacher_df.groupby(['task_id', 'sample_id', 'part_id']):
                # Filter out malformed feedback
                valid_group = group[~group['is_malformed']]

                if valid_group.empty:
                    continue

                # Check if teacher ever marked as correct
                ever_correct = (valid_group['is_correct'] == True).any()

                if not ever_correct:
                    never_agreed.append({
                        'task': task,
                        'task_id': task_id,
                        'sample_id': sample_id,
                        'part_id': part_id,
                        'total_iterations': group['iteration'].max() + 1,
                        'valid_iterations': len(valid_group),
                        'hallucination_count': group['is_malformed'].sum()
                    })

        df = pd.DataFrame(never_agreed)

        if df.empty:
            print("No cases where teacher never agreed!")
            return df

        # Plot 1: Count by task
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        task_counts = df['task'].value_counts().sort_index()
        ax1.bar(range(len(task_counts)), task_counts.values)
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Count')
        ax1.set_title('Cases Where Teacher Never Agreed (by Task)')
        ax1.set_xticks(range(len(task_counts)))
        ax1.set_xticklabels(task_counts.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distribution of iterations
        ax2.hist(df['total_iterations'], bins=range(1, df['total_iterations'].max() + 2),
                 align='left', rwidth=0.8, edgecolor='black')
        ax2.set_xlabel('Number of Iterations')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Iterations\n(Never Agreed Cases)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(result_path, 'never_agreed_cases.png'), dpi=300, bbox_inches='tight')
        plt.close()


        return df


    def analyse_disagreement_patterns(
            teacher_dfs: dict[str, pd.DataFrame],
            student_dfs: dict[str, pd.DataFrame],
            result_path: str
    ) -> pd.DataFrame:
        """
        Analyze patterns in teacher-student disagreements.
        Plot when and how often teacher disagrees with student.

        :param teacher_dfs: Teacher feedback dataframes by task
        :param student_dfs: Student feedback dataframes by task
        :param result_path: Path to save results
        :return: DataFrame of disagreement patterns
        """
        disagreements = []

        all_tasks = set(teacher_dfs.keys()) & set(student_dfs.keys())

        for task in all_tasks:
            teacher_df = teacher_dfs[task]
            student_df = student_dfs[task]

            for (task_id, sample_id, part_id), teacher_group in teacher_df.groupby(['task_id', 'sample_id', 'part_id']):
                # Filter out malformed
                teacher_group = teacher_group[~teacher_group['is_malformed']].sort_values('iteration')

                if teacher_group.empty:
                    continue

                # Count disagreements (teacher says incorrect)
                disagreement_count = (teacher_group['is_correct'] == False).sum()
                total_feedback = len(teacher_group)

                # Find when first agreement happened (if ever)
                first_correct_iter = None
                if (teacher_group['is_correct'] == True).any():
                    first_correct_iter = teacher_group[teacher_group['is_correct'] == True]['iteration'].min()

                disagreements.append({
                    'task': task,
                    'task_id': task_id,
                    'sample_id': sample_id,
                    'part_id': part_id,
                    'disagreement_count': disagreement_count,
                    'total_feedback': total_feedback,
                    'disagreement_rate': disagreement_count / total_feedback * 100 if total_feedback > 0 else 0,
                    'first_correct_iteration': first_correct_iter,
                    'ever_agreed': first_correct_iter is not None
                })

        df = pd.DataFrame(disagreements)

        # Plot 1: Disagreement rate distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Histogram of disagreement counts
        axes[0, 0].hist(df['disagreement_count'], bins=range(0, df['disagreement_count'].max() + 2),
                        align='left', rwidth=0.8, edgecolor='black')
        axes[0, 0].set_xlabel('Number of Disagreements')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Disagreement Counts')
        axes[0, 0].grid(True, alpha=0.3)

        # Disagreement rate by task
        task_avg_disagreement = df.groupby('task')['disagreement_rate'].mean().sort_values(ascending=False)
        axes[0, 1].barh(range(len(task_avg_disagreement)), task_avg_disagreement.values)
        axes[0, 1].set_yticks(range(len(task_avg_disagreement)))
        axes[0, 1].set_yticklabels(task_avg_disagreement.index)
        axes[0, 1].set_xlabel('Average Disagreement Rate (%)')
        axes[0, 1].set_title('Average Disagreement Rate by Task')
        axes[0, 1].grid(True, alpha=0.3)

        # First agreement iteration distribution
        df_agreed = df[df['ever_agreed']]
        if not df_agreed.empty:
            axes[1, 0].hist(df_agreed['first_correct_iteration'],
                            bins=range(0, int(df_agreed['first_correct_iteration'].max()) + 2),
                            align='left', rwidth=0.8, edgecolor='black')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('When First Agreement Occurred')
            axes[1, 0].grid(True, alpha=0.3)

        # Agreement vs never agreed
        agreement_counts = df['ever_agreed'].value_counts()
        axes[1, 1].pie(agreement_counts.values, labels=['Eventually Agreed', 'Never Agreed'],
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Agreement Status')

        plt.tight_layout()
        plt.savefig(os.path.join(result_path, 'disagreement_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()


        return df

    def run_all(self):
        """Run all feedback analyses."""
        self.analyse_correction_success()
        self.analyse_feedback_effectiveness()
        self.analyse_results_without_hallucinations()
        self.analyse_never_agreed_cases()
        self.analyse_disagreement_patterns()