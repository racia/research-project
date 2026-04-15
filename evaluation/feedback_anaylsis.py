from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


class FeedbackAnalysis:
    def __init__(self, teacher_dfs, student_dfs, result_path):
        self.teacher_dfs = teacher_dfs
        self.student_dfs = student_dfs
        self.result_path = result_path

    def analyse_correction_success(self) -> pd.DataFrame:
        """Analyse correction success rate per iteration."""
        all_tasks = set(self.teacher_dfs.keys()) & set(self.student_dfs.keys())
        success_by_iteration = defaultdict(lambda: {'success': 0, 'total': 0})

        for task in all_tasks:
            teacher_df = self.teacher_dfs[task]

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
        plt.savefig(os.path.join(self.result_path, 'correction_success_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()

        df.to_csv(os.path.join(self.result_path, 'correction_success_stats.csv'), index=False)
        return df

    def analyse_feedback_effectiveness(self) -> pd.DataFrame:
        """Analyse feedback effectiveness: correct after 0, 1, or multiple hints."""
        results = []

        for task, teacher_df in self.teacher_dfs.items():
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

        if df.empty:
            print("No data for feedback effectiveness analysis")
            return df

        effectiveness = {
            '0 hints': (df['hints_to_success'] == 0).sum(),
            '1 hint': (df['hints_to_success'] == 1).sum(),
            '2-3 hints': ((df['hints_to_success'] >= 2) & (df['hints_to_success'] <= 3)).sum(),
            '4+ hints': (df['hints_to_success'] >= 4).sum()
        }

        total = sum(effectiveness.values())
        percentages = {k: v / total * 100 if total > 0 else 0 for k, v in effectiveness.items()}

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(percentages.keys(), percentages.values())
        ax.set_ylabel('Percentage of Parts (%)')
        ax.set_title('Feedback Effectiveness: Hints to Success')
        plt.savefig(os.path.join(self.result_path, 'feedback_effectiveness.png'), dpi=300, bbox_inches='tight')
        plt.close()

        with open(os.path.join(self.result_path, 'feedback_effectiveness_stats.txt'), 'w') as f:
            f.write(f"Mean hints to success: {df['hints_to_success'].mean():.2f}\n")
            f.write(f"Median hints to success: {df['hints_to_success'].median():.2f}\n")

        df.to_csv(os.path.join(self.result_path, 'feedback_effectiveness_stats.csv'), index=False)
        return df

    def analyse_disagreement_by_task(self) -> pd.DataFrame:
        """Analyze disagreement rates by task."""
        disagreements = []
        all_tasks = set(self.teacher_dfs.keys()) & set(self.student_dfs.keys())

        for task in all_tasks:
            teacher_df = self.teacher_dfs[task]

            for (task_id, sample_id, part_id), teacher_group in teacher_df.groupby(['task_id', 'sample_id', 'part_id']):
                teacher_group = teacher_group[~teacher_group['is_malformed']].sort_values('iteration')

                if teacher_group.empty:
                    continue

                disagreement_count = (teacher_group['is_correct'] == False).sum()
                total_feedback = len(teacher_group)

                disagreements.append({
                    'task': task,
                    'task_id': task_id,
                    'sample_id': sample_id,
                    'part_id': part_id,
                    'disagreement_count': disagreement_count,
                    'total_feedback': total_feedback,
                    'disagreement_rate': disagreement_count / total_feedback * 100 if total_feedback > 0 else 0,
                })

        df = pd.DataFrame(disagreements)

        if df.empty:
            return df

        # Single plot: disagreement rate by task
        fig, ax = plt.subplots(figsize=(12, 6))
        task_avg_disagreement = df.groupby('task')['disagreement_rate'].mean().sort_values(ascending=False)
        ax.barh(range(len(task_avg_disagreement)), task_avg_disagreement.values)
        ax.set_yticks(range(len(task_avg_disagreement)))
        ax.set_yticklabels(task_avg_disagreement.index)
        ax.set_xlabel('Average Disagreement Rate (%)')
        ax.set_title('Disagreement Rate by Task')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'disagreement_by_task.png'), dpi=300, bbox_inches='tight')
        plt.close()

        df.to_csv(os.path.join(self.result_path, 'disagreement_patterns.csv'), index=False)
        return df


    def run_all(self):
        """Run all feedback analyses."""
        print("Running correction success analysis...")
        self.analyse_correction_success()

        print("Running feedback effectiveness analysis...")
        self.analyse_feedback_effectiveness()

        print("Running disagreement analysis...")
        self.analyse_disagreement_by_task()



        print(f"\nAll feedback analyses complete. Results saved to: {self.result_path}")