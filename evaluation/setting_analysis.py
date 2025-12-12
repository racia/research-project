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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class SettingAnalysis:

    def __init__(self, student_dfs, result_path, teacher_dfs=None):
        self.student_dfs = student_dfs
        self.teacher_dfs = teacher_dfs
        self.result_path = result_path

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


    def cluster_errors(
            student_dfs: dict[str, pd.DataFrame],
            teacher_dfs: Optional[dict[str, pd.DataFrame]] = None,
            result_path: str = ".",
            n_clusters: int = 5
    ) -> tuple[pd.DataFrame, dict]:
        """
        Cluster errors to identify patterns.

        :param student_dfs: Student/model dataframes by task
        :param teacher_dfs: Optional teacher dataframes
        :param result_path: Output directory
        :param n_clusters: Number of clusters
        :return: Clustered dataframe and cluster descriptions
        """
        errors = []

        # Collect student errors
        for task, df in student_dfs.items():
            if 'is_malformed' in df.columns:
                task_errors = df[df['is_malformed'] == True].copy()
            else:
                task_errors = df.copy()

            if len(task_errors) > 0:
                task_errors['source'] = 'student'
                task_errors['task'] = task
                errors.append(task_errors)

        # Collect teacher errors (if provided)
        if teacher_dfs:
            for task, df in teacher_dfs.items():
                if 'is_malformed' in df.columns:
                    task_errors = df[df['is_malformed'] == True].copy()
                else:
                    task_errors = df.copy()

                if len(task_errors) > 0:
                    task_errors['source'] = 'teacher'
                    task_errors['task'] = task
                    errors.append(task_errors)

        if not errors:
            print("No errors found to cluster!")
            return pd.DataFrame(), {}

        error_df = pd.concat(errors, ignore_index=True)
        print(f"Found {len(error_df)} errors to cluster")

        # Adjust clusters if needed
        if len(error_df) < n_clusters:
            n_clusters = max(2, len(error_df) // 2)
            print(f"Adjusted to {n_clusters} clusters")

        # Find content column
        content_col = 'full_content' if 'full_content' in error_df.columns else 'body'
        if content_col not in error_df.columns:
            text_cols = error_df.select_dtypes(include=['object']).columns
            content_col = text_cols[0] if len(text_cols) > 0 else None

        if not content_col:
            print("No text column found!")
            return error_df, {}

        # Vectorize and cluster
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=1)
        X = vectorizer.fit_transform(error_df[content_col].fillna(''))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        error_df['cluster'] = kmeans.fit_predict(X)

        # Analyze clusters
        cluster_descriptions = {}
        for cluster_id in range(n_clusters):
            cluster_data = error_df[error_df['cluster'] == cluster_id]

            # Extract keywords
            cluster_texts = ' '.join(cluster_data[content_col].fillna(''))
            vec = TfidfVectorizer(max_features=10, stop_words='english')
            try:
                vec.fit_transform([cluster_texts])
                keywords = vec.get_feature_names_out()
            except:
                keywords = []

            desc = {
                'size': len(cluster_data),
                'keywords': list(keywords)[:5],
                'sample_texts': cluster_data[content_col].head(3).tolist()
            }

            # Add source breakdown if teacher exists
            if teacher_dfs:
                desc['student_count'] = (cluster_data['source'] == 'student').sum()
                desc['teacher_count'] = (cluster_data['source'] == 'teacher').sum()

            # Add error types if available
            if 'error_type' in cluster_data.columns:
                desc['error_types'] = cluster_data['error_type'].value_counts().to_dict()

            cluster_descriptions[cluster_id] = desc

        # Plot
        _plot_clusters(error_df, result_path, teacher_dfs is not None)
        _save_cluster_analysis(error_df, cluster_descriptions, result_path, n_clusters, teacher_dfs is not None)

        error_df.to_csv(os.path.join(result_path, 'error_clusters.csv'), index=False)
        return error_df, cluster_descriptions


    def _plot_clusters(error_df, result_path, has_teacher):
        """Create cluster plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Cluster sizes
        cluster_sizes = error_df['cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values)
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Error Cluster Sizes')
        axes[0, 0].grid(True, alpha=0.3)

        # Source comparison
        if has_teacher:
            cluster_source = error_df.groupby(['cluster', 'source']).size().unstack(fill_value=0)
            cluster_source.plot(kind='bar', ax=axes[0, 1], stacked=True)
            axes[0, 1].set_title('Student vs Teacher by Cluster')
            axes[0, 1].legend(title='Source')
        else:
            cluster_pct = (cluster_sizes / cluster_sizes.sum() * 100).sort_index()
            axes[0, 1].bar(cluster_pct.index, cluster_pct.values, color='coral')
            axes[0, 1].set_ylabel('Percentage (%)')
            axes[0, 1].set_title('Cluster Distribution')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].grid(True, alpha=0.3)

        # Error types or heatmap
        if 'error_type' in error_df.columns and not error_df['error_type'].isna().all():
            error_type_counts = error_df['error_type'].value_counts()
            if len(error_type_counts) > 0:
                axes[1, 0].pie(error_type_counts.values, labels=error_type_counts.index,
                               autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Error Types')
        else:
            task_cluster = pd.crosstab(error_df['task'], error_df['cluster'])
            sns.heatmap(task_cluster, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 0])
            axes[1, 0].set_title('Errors by Task and Cluster')

        # Errors by task
        task_counts = error_df['task'].value_counts().sort_index()
        axes[1, 1].barh(range(len(task_counts)), task_counts.values, color='mediumseagreen')
        axes[1, 1].set_yticks(range(len(task_counts)))
        axes[1, 1].set_yticklabels(task_counts.index)
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_title('Errors by Task')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(result_path, 'error_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()


    def _save_cluster_analysis(error_df, descriptions, result_path, n_clusters, has_teacher):
        """Save cluster analysis text."""
        with open(os.path.join(result_path, 'error_cluster_analysis.txt'), 'w') as f:
            f.write("ERROR CLUSTER ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total errors: {len(error_df)}\n")
            f.write(f"Clusters: {n_clusters}\n\n")

            for cluster_id, desc in descriptions.items():
                f.write(f"\nCLUSTER {cluster_id}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Size: {desc['size']} ({desc['size'] / len(error_df) * 100:.1f}%)\n")

                if has_teacher:
                    f.write(f"Student: {desc.get('student_count', 0)}, Teacher: {desc.get('teacher_count', 0)}\n")

                f.write(f"Keywords: {', '.join(desc['keywords']) if desc['keywords'] else 'N/A'}\n")

                if 'error_types' in desc:
                    f.write(f"Error types: {desc['error_types']}\n")

                f.write(f"\nSamples:\n")
                for i, text in enumerate(desc['sample_texts'], 1):
                    preview = text[:200] if len(text) > 200 else text
                    f.write(f"  {i}. {preview}{'...' if len(text) > 200 else ''}\n")
                f.write("\n")


if __name__ == "__main__":
    data_path = "/pfs/work9/workspace/scratch/hd_nc326-research-project/feedback/test/reasoning/v1/all_tasks_joined/iterations"
    result_path = "/pfs/work9/workspace/scratch/hd_nc326-research-project/feedback/test/reasoning/v1/all_tasks_plots"

    run_feedback_analysis(data_path, result_path)
