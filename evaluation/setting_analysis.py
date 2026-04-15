"""
Cross-setting analysis functions (shared between feedback, SD, etc.)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import evaluate


class SettingAnalysis:
    def __init__(self, student_dfs, result_path, teacher_dfs=None):
        """
        Initialize SettingAnalysis.

        :param student_dfs: Student/model dataframes by task
        :param result_path: Output directory
        :param teacher_dfs: Optional teacher dataframes (for feedback setting)
        """
        self.student_dfs = student_dfs
        self.teacher_dfs = teacher_dfs
        self.result_path = result_path




    def analyse_semantic_similarity(self, source_name: str = "student") -> pd.DataFrame:
        """Analyze semantic similarity between consecutive iterations."""
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
        results = []

        for task, df in self.student_dfs.items():
            df = df.query('not is_malformed') if 'is_malformed' in df.columns else df

            for (task_id, sample_id, part_id), group in df.groupby(['task_id', 'sample_id', 'part_id']):
                group_sorted = group.sort_values('iteration') if 'iteration' in group.columns else group

                for i in range(len(group_sorted) - 1):
                    current = group_sorted.iloc[i]
                    next_row = group_sorted.iloc[i + 1]

                    current_text = current.get('body', str(current.get('full_content', '')))
                    next_text = next_row.get('body', str(next_row.get('full_content', '')))

                    if current_text and next_text:
                        results.append({
                            'task': task,
                            'task_id': task_id,
                            'sample_id': sample_id,
                            'part_id': part_id,
                            'iteration': current.get('iteration', i),
                            'bleu': bleu.compute(predictions=[next_text], references=[[current_text]])['bleu'],
                            'rouge': rouge.compute(predictions=[next_text], references=[current_text])['rougeL'],
                            'meteor': meteor.compute(predictions=[next_text], references=[current_text])['meteor']
                        })

        df = pd.DataFrame(results)

        if df.empty:
            print(f"No data for semantic similarity ({source_name})")
            return df

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, metric in enumerate(['bleu', 'rouge', 'meteor']):
            axes[idx].hist(df[metric], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(f'{metric.upper()} Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{metric.upper()} Distribution')
            axes[idx].axvline(df[metric].mean(), color='red', linestyle='--',
                              label=f'Mean: {df[metric].mean():.3f}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(f'Semantic Similarity: {source_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, f'semantic_similarity_{source_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        df.to_csv(os.path.join(self.result_path, f'semantic_similarity_{source_name}.csv'), index=False)
        return df






    def cluster_errors(self, n_clusters: int = 5) -> tuple[pd.DataFrame, dict]:
        """
        Cluster errors to identify patterns.

        :param n_clusters: Number of clusters
        :return: Clustered dataframe and cluster descriptions
        """
        errors = []

        # Collect student errors
        for task, df in self.student_dfs.items():
            if 'is_malformed' in df.columns:
                task_errors = df[df['is_malformed'] == True].copy()
            else:
                task_errors = df.copy()

            if len(task_errors) > 0:
                task_errors['source'] = 'student'
                task_errors['task'] = task
                errors.append(task_errors)

        # Collect teacher errors (if provided)
        if self.teacher_dfs:
            for task, df in self.teacher_dfs.items():
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
            if self.teacher_dfs:
                desc['student_count'] = (cluster_data['source'] == 'student').sum()
                desc['teacher_count'] = (cluster_data['source'] == 'teacher').sum()

            # Add error types if available
            if 'error_type' in cluster_data.columns:
                desc['error_types'] = cluster_data['error_type'].value_counts().to_dict()

            cluster_descriptions[cluster_id] = desc

        # Plot and save
        self._plot_clusters(error_df, self.teacher_dfs is not None)
        self._save_cluster_analysis(error_df, cluster_descriptions, n_clusters, self.teacher_dfs is not None)

        error_df.to_csv(os.path.join(self.result_path, 'error_clusters.csv'), index=False)
        return error_df, cluster_descriptions

    def _plot_clusters(self, error_df, has_teacher):
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
        plt.savefig(os.path.join(self.result_path, 'error_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def analyse_edge_cases(self) -> dict:
        """Analyze edge cases (hallucinations, never agreed) and save summary."""
        stats = {
            'hallucinations': {'total': 0, 'by_task': {}},
            'never_agreed': {'total': 0, 'by_task': {}}
        }

        # Count hallucinations from both sources
        for source_name, dfs in [('teacher', self.teacher_dfs), ('student', self.student_dfs)]:
            if not dfs:
                continue
            for task, df in dfs.items():
                if 'is_malformed' not in df.columns:
                    continue
                hallucinated = df[df['is_malformed'] == True][
                    ['task_id', 'sample_id', 'part_id']
                ].drop_duplicates()
                count = len(hallucinated)
                if count > 0:
                    key = f"{task}_{source_name}"
                    stats['hallucinations']['by_task'][key] = count
                    stats['hallucinations']['total'] += count

        # Count never agreed (teacher only)
        if self.teacher_dfs:
            for task, teacher_df in self.teacher_dfs.items():
                never_agreed_count = 0
                for (task_id, sample_id, part_id), group in teacher_df.groupby(['task_id', 'sample_id', 'part_id']):
                    valid_group = group[~group['is_malformed']]
                    if valid_group.empty:
                        continue
                    if not (valid_group['is_correct'] == True).any():
                        never_agreed_count += 1

                if never_agreed_count > 0:
                    stats['never_agreed']['by_task'][task] = never_agreed_count
                    stats['never_agreed']['total'] += never_agreed_count

        # Save summary
        with open(os.path.join(self.result_path, 'edge_cases_summary.txt'), 'w') as f:
            f.write("EDGE CASES SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Hallucinations: {stats['hallucinations']['total']} parts\n")
            if stats['hallucinations']['by_task']:
                for task, count in stats['hallucinations']['by_task'].items():
                    f.write(f"  {task}: {count}\n")
            f.write(f"\nNever Agreed: {stats['never_agreed']['total']} parts\n")
            if stats['never_agreed']['by_task']:
                for task, count in stats['never_agreed']['by_task'].items():
                    f.write(f"  {task}: {count}\n")

        return stats

    def _save_cluster_analysis(self, error_df, descriptions, n_clusters, has_teacher):
        """Save cluster analysis text."""
        with open(os.path.join(self.result_path, 'error_cluster_analysis.txt'), 'w') as f:
            f.write("ERROR CLUSTER ANALYSIS\n")
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