import nltk
import numpy as np
import pandas as pd
from data.DataLoader import DataLoader
from evaluation.Statistics import Statistics
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
#from rouge_score import rouge_score
try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    # Download required nltk data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    meteor = meteor_score
    print("Successfully loaded METEOR from nltk")
except ImportError:
    print("Warning: nltk not available for METEOR score. Install with: pip install nltk")
    meteor = None
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import evaluate


class StudentAnswerLoader:
    """
        Helper class to load student answer files with specific naming convention.
        """

    def __init__(self, data_loader=None):
        """
        Initialize the loader for student answer files.

        data_loader : DataLoader,
        """
        self.data_loader = data_loader if data_loader else DataLoader()
    def load_student_answers_files(self,directory_path: str) -> Dict:
        """
        Load student answer files with format where file name follows:
        "{iteration}_student_{task_id}-{sample_id}-{part_id}.txt"

        :param directory_path :Path to directory containing student answer files
        return :Dictionary with loaded data organized by iteration, task, sample.
        """
        data = {
            'by_iteration': defaultdict(list),
            'by_task': defaultdict(list),
            'by_file': {},
            'all_answers': [],
            'all_reasonings': [],
            'metadata': []
        }
        # Get all text files in directory
        directory = Path(directory_path)
        files = list(directory.glob('*.txt'))

        print(f"Found {len(files)} student answer files")
        for file_path in files:
            file_name = file_path.stem
            # Parse filename to extract metadata
            # Format: "{iteration}_student_{task_id}-{sample_id}-{part_id}"
            match= re.match(r'(\d+)_student_(\d+)-(\d+)-(\d+)', file_name)
            if not match:
                print(f"Warning: File {file_name} doesn't match expected naming pattern. Skipping.")
                continue
            iteration, task_id, sample_id, part_id= match.groups()
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()

            #extract reasoning and answers
            reasoning = ""
            answer = ""

            reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=\n\nAnswer:|$)',content,re.DOTALL)
            answer_match = re.search(r'Answer:\s*(.*?)$', content, re.DOTALL)

            if reasoning_match:
                reasoning = reasoning_match.groups(1).strip()
            else:
                # If no explicit reasoning section, look for content before "Answer:"
                answer_pos = content.find("Answer:")
                if answer_pos > 0:
                    potential_reasoning = content[:answer_pos].strip()
                    reasoning = potential_reasoning

            if answer_match:
                answer = answer_match.groups(1).strip()
            else:
                lines = content.strip().split('\n')
                if len(lines) >= 1:
                    last_line = lines[-1].strip()
                    # If the last line is short, it's likely just the answer
                    if len(last_line) < 50 and not last_line.startswith("Reasoning:"):
                        answer = last_line
            metadata = {
                'file_name': file_name,
                'iteration': int(iteration),
                'task_id': int(task_id),
                'sample_id': int(sample_id),
                'part_id': int(part_id),
                'reasoning': reasoning,
                'answer': answer,
                'full_content': content  # Store full content as backup
            }
            data['by_file'][file_name] = metadata
            data['by_iteration'][int(iteration)].append(metadata)
            data['by_task'][int(task_id)].append(metadata)

            data['all_answers'].append(answer)
            data['all_reasonings'].append(reasoning)

        # Some statistics to verify data loading
        answers_found = len([a for a in data['all_answers'] if a.strip()])
        reasonings_found = len([r for r in data['all_reasonings'] if r.strip()])

        print(f"Processed {len(data['metadata'])} student answer files")
        print(f"Successfully extracted {answers_found} answers and {reasonings_found} reasonings")
        print(f"Found {len(set(int(m['task_id']) for m in data['metadata']))} unique tasks")
        print(f"Found {len(set(int(m['iteration']) for m in data['metadata']))} iterations")

        return data

    def load_golden_answers(self,results_path: str, student_data: Dict) -> List[str]:
        """
        Load golden answers from results CSV using DataLoader and match them to student answers.

        :param results_path : Path to the CSV file with results including golden answers
        :param student_data : Student data dictionary from load_student_answer_files
        returns: List of golden answers matching student answers order

        """
        golden_data = self.data_loader.load_results(
            results_path=results_path,
            list_output=True,
            sep=','
        )
        golden_answers = []

        for metadata in student_data['metadata']:
            matched = False
            for row in golden_data:
                if (int(row['task_id']) == metadata['task_id'] and
                    int(row['sample_id']) == metadata['sample_id'] and
                    int(row['part_id']) == metadata['part_id']):

                    golden_answers.append(row['golden_answer'])
                    matched = True
                    break
            if not matched:
                print(f"Warning: No golden answer found for task={metadata['task_id']}, "
                  f"sample={metadata['sample_id']}, part={metadata['part_id']}")
                golden_answers.append("")

        print(f"Matched {len(golden_answers)} golden answers to student answers")

        return golden_answers

class AnswerClusterer:
    """
    Class for clustering student answers and analyzing error patterns.
    """

    def __init__(self, embedding_method='sentence-transformer', data_loader=None):
        """
        Initialize the AnswerClusterer.
        :param embedding_method : Method to use for embedding answers:
            - 'sentence-transformer': Use SentenceTransformer for embeddings (recommended)
            - 'tfidf': Use TF-IDF for vectorizing answers
        :param data_loader : DataLoader instance
        """
        self.stats = Statistics()
        self.embedding_method = embedding_method
        self.data_loader = data_loader if data_loader else DataLoader()
        self.student_loader = StudentAnsewerLoader(self.data_loader)

        if embedding_method == 'sentence-transformer':
            self.model = SentenceTransformer('all-mpnet-base-v2')
        else:
            self.embedding_method = 'tfidf'
            self.model = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1,2)
            )
            # Data storage
            self.student_answers = []
            self.student_answers_refined = []
            self.student_reasonings = []
            self.golden_answers = []
            self.teacher_feedback = []
            self.embedding_vectors = None
            self.clusters = None
            self.error_types = defaultdict(list)
            self.student_data = None

        return
    def load_student_asnwers_directory(self, directory_path: str, golden_answers_path: str = None) -> Dict:
        """
        Load student answer files from directory and match with golden answers if provided.

        :param directory_path : Path to directory containing student answer files
        :param golden_answers_path : Path to CSV file with golden answers
        return:Student data dictionary
        """

        self.student_data = self.student_loader.load_student_answers_files(directory_path)
        self.student_answers = self.student_data['all_answers']
        self.student_reasonings = self.student_data['all_reasonings']

        if golden_answers_path:
            self.golden_answers = self.student_loader.load_golden_answers(golden_answers_path, self.student_answers)
        else:
            print("No golden answers path provided.")

        return self.student_data
    def calculate_correctness(self) -> List:
        """
        Calculate correctness for each student answer using the Statistics class.

        :return list : Boolean list indicating if each answer is correct
        """
        if not self.golden_answers:
            print("Warning: No golden answers available. Cannot calculate correctness")
            return [False] * len(self.student_answers)

        correctness = []
        for student_answer, golden_answer in zip(self.student_answers,self.golden_answers):
            correctness.append(self.stats.are_identical(golden_answer,student_answer))
        return correctness

    def extract_errors(self):
        """
        Extract features from student reasoning without requiring golden reasoning.
        This analyzes reasoning quality using various linguistic and structural metrics.

        :return list: List of dictionaries containing reasoning features for each answer
        """
        reasoning_features = []

        for i, (reasoning,answer) in enumerate(zip(self.student_reasonings,self.student_answers)):
            if not reasoning:
                reasoning = ""
            if not answer:
                answer = ""

            reasoning_sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
            reasoning_words = reasoning.split()
            answer_words = answer.split()

            # Logical connectors that indicate reasoning structure

            logical_connectors = [
                'because', 'since', 'therefore', 'thus', 'hence', 'so', 'consequently',
                'implies', 'suggests', 'means', 'indicates', 'shows', 'proves',
                'if', 'then', 'when', 'given', 'assuming'
            ]
            # Confidence indicators
            confidence_indicators = [
                'clearly', 'obviously', 'definitely', 'certainly', 'surely',
                'might', 'maybe', 'possibly', 'probably', 'likely', 'perhaps',
                'appears', 'seems', 'suggests'
            ]

            # Question addressing words
            question_words = ['what', 'where', 'when', 'who', 'why', 'how', 'which']

            # Calculate features
            features = {
                # Basic length and structure
                'reasoning_length': len(reasoning),
                'reasoning_word_count': len(reasoning_words),
                'reasoning_sentence_count': len(reasoning_sentences),
                'answer_length': len(answer),
                'answer_word_count': len(answer_words),

                # Reasoning complexity
                'avg_sentence_length': len(reasoning_words) / len(reasoning_sentences) if reasoning_sentences else 0,
                'word_diversity': len(set(reasoning_words)) / len(reasoning_words) if reasoning_words else 0,

                # Logical structure indicators
                'logical_connector_count': sum(1 for word in reasoning_words if word.lower() in logical_connectors),
                'logical_connector_ratio': sum(
                    1 for word in reasoning_words if word.lower() in logical_connectors) / len(
                    reasoning_words) if reasoning_words else 0,

                # Confidence and uncertainty
                'confidence_indicator_count': sum(
                    1 for word in reasoning_words if word.lower() in confidence_indicators),
                'has_confidence_indicators': any(word.lower() in confidence_indicators for word in reasoning_words),

                # Question addressing
                'addresses_question_words': sum(1 for word in reasoning_words if word.lower() in question_words),

                # Answer-reasoning alignment
                'reasoning_answer_overlap': len(set(reasoning_words) & set(answer_words)) / max(
                    len(set(reasoning_words)), 1),
                'answer_in_reasoning': answer.lower() in reasoning.lower() if reasoning and answer else False,

                # Reasoning patterns
                'starts_with_explanation': reasoning.lower().startswith(('since', 'because', 'given', 'as')),
                'ends_with_conclusion': reasoning.lower().endswith(('therefore', 'thus', 'so', 'hence')),
                'has_conditional_logic': any(word in reasoning.lower() for word in ['if', 'then', 'when']),

                # Metadata if available
                'has_reasoning': len(reasoning.strip()) > 0,
                'reasoning_to_answer_ratio': len(reasoning) / max(len(answer), 1) if answer else 0,
            }

            reasoning_features.append(features)

        return reasoning_features



    def calculate_reasoning_distances(self, use_embeddings=True, skyline_reasoning=None) -> dict:
        """
        Calculate distance metrics between reasoning texts to identify patterns.
        Now includes ROUGE, BLEU, METEOR scores and skyline reasoning comparison.

        :params use_embeddings: Whether to use semantic embeddings or lexical similarity
        :params skyline_reasoning:  Reference/optimal reasoning texts to compare against

        :return Dictionary with distance matrices and similarity scores
        """
        if not self.student_reasonings:
            print("Warning: No reasoning data available.")
            return {}

        # Import evaluation metrics with proper error handling


        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        print("Successfully loaded ROUGE and BLEU from evaluate library")



        results = {}

        # 1. Calculate pairwise similarities between student reasoning
        if use_embeddings and self.embedding_method == 'sentence-transformer':
            reasoning_embeddings = self.model.encode(self.student_reasonings)

            # Calculate cosine similarity
            cosine_sim = cosine_similarity(reasoning_embeddings)

            # Find most similar reasoning pairs
            similarity_pairs = []
            n = len(reasoning_embeddings)
            for i in range(n):
                for j in range(i + 1, n):
                    similarity_pairs.append({
                        'idx1': i,
                        'idx2': j,
                        'cosine_similarity': cosine_sim[i][j],
                        'reasoning1': self.student_reasonings[i][:100] + "..." if len(
                            self.student_reasonings[i]) > 100 else self.student_reasonings[i],
                        'reasoning2': self.student_reasonings[j][:100] + "..." if len(
                            self.student_reasonings[j]) > 100 else self.student_reasonings[j]
                    })

            # Sort by similarity
            similarity_pairs.sort(key=lambda x: x['cosine_similarity'], reverse=True)

            results.update({
                'cosine_similarity_matrix': cosine_sim,
                'most_similar_pairs': similarity_pairs[:20],  # Top 20 most similar
                'average_similarity': np.mean(cosine_sim[np.triu_indices_from(cosine_sim, k=1)]),
            })
        else:
            # Use lexical similarity
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(self.student_reasonings)
            cosine_sim = cosine_similarity(tfidf_matrix)

            results.update({
                'lexical_similarity_matrix': cosine_sim,
                'average_lexical_similarity': np.mean(cosine_sim[np.triu_indices_from(cosine_sim, k=1)])
            })

        # 2. Calculate ROUGE, BLEU, METEOR scores between reasoning pairs
        if rouge or bleu or meteor:
            print("Calculating evaluation metrics between reasoning pairs...")

            # Pairwise evaluation metrics
            rouge_scores = []
            bleu_scores = []
            meteor_scores = []

            # Calculate for top similar pairs to avoid too many computations
            n_samples = min(100, len(self.student_reasonings))  # Limit for performance

            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    reasoning1 = self.student_reasonings[i]
                    reasoning2 = self.student_reasonings[j]

                    if not reasoning1.strip() or not reasoning2.strip():
                        continue

                    try:
                        # ROUGE score
                        if rouge:
                            rouge_result = rouge.compute(predictions=[reasoning1], references=[reasoning2])
                            rouge_scores.append({
                                'idx1': i, 'idx2': j,
                                'rouge1': rouge_result.get('rouge1', 0),
                                'rouge2': rouge_result.get('rouge2', 0),
                                'rougeL': rouge_result.get('rougeL', 0)
                            })

                        # BLEU score
                        if bleu:
                            bleu_result = bleu.compute(predictions=[reasoning1], references=[[reasoning2]])
                            bleu_scores.append({
                                'idx1': i, 'idx2': j,
                                'bleu': bleu_result.get('bleu', 0)
                            })

                        # METEOR score (from nltk)
                        if meteor:
                            # METEOR expects tokenized sentences
                            reasoning1_tokens = nltk.word_tokenize(reasoning1.lower())
                            reasoning2_tokens = nltk.word_tokenize(reasoning2.lower())
                            meteor_result = meteor(reasoning2_tokens, reasoning1_tokens)
                            meteor_scores.append({
                                'idx1': i, 'idx2': j,
                                'meteor': meteor_result
                            })

                    except Exception as e:
                        print(f"Warning: Error calculating metrics for pair {i},{j}: {e}")
                        continue

            # Calculate averages
            if rouge_scores:
                avg_rouge1 = np.mean([s['rouge1'] for s in rouge_scores])
                avg_rouge2 = np.mean([s['rouge2'] for s in rouge_scores])
                avg_rougeL = np.mean([s['rougeL'] for s in rouge_scores])
                results['rouge_scores'] = rouge_scores
                results['average_rouge1'] = avg_rouge1
                results['average_rouge2'] = avg_rouge2
                results['average_rougeL'] = avg_rougeL

            if bleu_scores:
                avg_bleu = np.mean([s['bleu'] for s in bleu_scores])
                results['bleu_scores'] = bleu_scores
                results['average_bleu'] = avg_bleu

            if meteor_scores:
                avg_meteor = np.mean([s['meteor'] for s in meteor_scores])
                results['meteor_scores'] = meteor_scores
                results['average_meteor'] = avg_meteor

        # 3. Compare against skyline reasoning if provided
        if skyline_reasoning:
            print("Comparing against skyline reasoning...")

            # Handle different skyline reasoning formats
            if isinstance(skyline_reasoning, dict):
                # If skyline_reasoning is a dict mapping task_id to reasoning
                skyline_comparisons = []

                for i, reasoning in enumerate(self.student_reasonings):
                    if not reasoning.strip():
                        continue

                    # Get task info for this reasoning
                    task_id = None
                    if self.student_data and i < len(self.student_data['metadata']):
                        task_id = self.student_data['metadata'][i]['task_id']

                    # Find corresponding skyline reasoning
                    skyline_text = skyline_reasoning.get(task_id, None)
                    if not skyline_text:
                        continue

                    # Calculate similarity metrics
                    comparison = {'student_idx': i, 'task_id': task_id}

                    # Cosine similarity
                    if use_embeddings and self.embedding_method == 'sentence-transformer':
                        student_emb = self.model.encode([reasoning])
                        skyline_emb = self.model.encode([skyline_text])
                        comparison['cosine_similarity'] = cosine_similarity(student_emb, skyline_emb)[0][0]

                    # ROUGE, BLEU, METEOR
                    try:
                        if rouge:
                            rouge_result = rouge.compute(predictions=[reasoning], references=[skyline_text])
                            comparison.update({
                                'rouge1': rouge_result.get('rouge1', 0),
                                'rouge2': rouge_result.get('rouge2', 0),
                                'rougeL': rouge_result.get('rougeL', 0)
                            })

                        if bleu:
                            bleu_result = bleu.compute(predictions=[reasoning], references=[[skyline_text]])
                            comparison['bleu'] = bleu_result.get('bleu', 0)

                        if meteor:
                            import nltk
                            reasoning_tokens = nltk.word_tokenize(reasoning.lower())
                            skyline_tokens = nltk.word_tokenize(skyline_text.lower())
                            comparison['meteor'] = meteor(skyline_tokens, reasoning_tokens)

                    except Exception as e:
                        print(f"Warning: Error calculating skyline metrics for {i}: {e}")

                    skyline_comparisons.append(comparison)

                results['skyline_comparisons'] = skyline_comparisons

                # Calculate averages
                if skyline_comparisons:
                    for metric in ['cosine_similarity', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']:
                        scores = [c[metric] for c in skyline_comparisons if metric in c]
                        if scores:
                            results[f'avg_skyline_{metric}'] = np.mean(scores)

            elif isinstance(skyline_reasoning, list):
                # If skyline_reasoning is a list of reference texts
                skyline_comparisons = []

                for i, reasoning in enumerate(self.student_reasonings):
                    if not reasoning.strip():
                        continue

                    # Compare against all skyline reasonings and take the best match
                    best_similarity = -1
                    best_comparison = {'student_idx': i}

                    for j, skyline_text in enumerate(skyline_reasoning):
                        if not skyline_text.strip():
                            continue

                        comparison = {'student_idx': i, 'skyline_idx': j}

                        # Cosine similarity
                        if use_embeddings and self.embedding_method == 'sentence-transformer':
                            student_emb = self.model.encode([reasoning])
                            skyline_emb = self.model.encode([skyline_text])
                            similarity = cosine_similarity(student_emb, skyline_emb)[0][0]
                            comparison['cosine_similarity'] = similarity

                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_comparison = comparison.copy()

                        # Calculate other metrics for best match only
                        if similarity == best_similarity:
                            try:
                                if rouge:
                                    rouge_result = rouge.compute(predictions=[reasoning], references=[skyline_text])
                                    best_comparison.update({
                                        'rouge1': rouge_result.get('rouge1', 0),
                                        'rouge2': rouge_result.get('rouge2', 0),
                                        'rougeL': rouge_result.get('rougeL', 0)
                                    })

                                if bleu:
                                    bleu_result = bleu.compute(predictions=[reasoning], references=[[skyline_text]])
                                    best_comparison['bleu'] = bleu_result.get('bleu', 0)

                                if meteor:
                                    import nltk
                                    reasoning_tokens = nltk.word_tokenize(reasoning.lower())
                                    skyline_tokens = nltk.word_tokenize(skyline_text.lower())
                                    best_comparison['meteor'] = meteor(skyline_tokens, reasoning_tokens)

                            except Exception as e:
                                print(f"Warning: Error calculating skyline metrics: {e}")

                    if best_similarity > -1:
                        skyline_comparisons.append(best_comparison)

                results['skyline_comparisons'] = skyline_comparisons

                # Calculate averages
                if skyline_comparisons:
                    for metric in ['cosine_similarity', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']:
                        scores = [c[metric] for c in skyline_comparisons if metric in c]
                        if scores:
                            results[f'avg_skyline_{metric}'] = np.mean(scores)

        return results



    def embed_answers(self, use_reasoning=False) -> np.ndarray:
        """
        Create vector embeddings of student answers.

        Parameters:
        -----------
        use_reasoning : bool
            If True, use reasoning text for embeddings instead of answers

        Returns:
        --------
        numpy.ndarray : Matrix of embeddings
        """
        texts_to_embed = self.student_reasonings if use_reasoning else self.student_answers

        if self.embedding_method == 'sentence-transformer':
            print("Using sentence-transformer for embeddings...")
            self.embedding_vectors = self.model.encode(texts_to_embed)
        else:
            print("Using TF-IDF for embeddings...")
            self.embedding_vectors = self.model.fit_transform(texts_to_embed).toarray()

        print(f"Created embeddings with shape: {self.embedding_vectors.shape}")
        return self.embedding_vectors

    def cluster_answers(self, n_clusters=None, method='kmeans', min_samples=5) -> np.ndarray:
        """
        Cluster student answers based on their embeddings.

        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters to create. If None, will use silhouette analysis to find optimal.
        method : str
            Clustering method to use. Options:
            - 'kmeans': K-means clustering
            - 'hierarchical': Hierarchical (Agglomerative) clustering
            - 'dbscan': DBSCAN clustering
        min_samples : int
            Minimum samples per cluster (used for DBSCAN)

        Returns:
        --------
        numpy.ndarray : Array of cluster labels
        """
        if self.embedding_vectors is None:
            self.embed_answers()

        # If n_clusters is not specified, find optimal number using silhouette score
        if n_clusters is None and method != 'dbscan':
            print("Finding optimal number of clusters...")
            max_silhouette = -1
            best_n = 2  # Default to at least 2 clusters

            # Try different numbers of clusters
            for n in range(2, min(20, len(self.student_answers) // 5)):
                # Use KMeans for speed during optimization
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.embedding_vectors)

                # Skip if only one cluster remains
                if len(np.unique(labels)) < 2:
                    continue

                # Calculate silhouette score
                silhouette = silhouette_score(self.embedding_vectors, labels)
                print(f"  {n} clusters: silhouette score = {silhouette:.4f}")

                if silhouette > max_silhouette:
                    max_silhouette = silhouette
                    best_n = n

            n_clusters = best_n
            print(f"Optimal number of clusters: {n_clusters}")

        # Perform clustering with chosen method
        if method == 'kmeans':
            print(f"Clustering with KMeans (n_clusters={n_clusters})...")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = clusterer.fit_predict(self.embedding_vectors)

        elif method == 'hierarchical':
            print(f"Clustering with Agglomerative Clustering (n_clusters={n_clusters})...")
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            self.clusters = clusterer.fit_predict(self.embedding_vectors)

        elif method == 'dbscan':
            print("Clustering with DBSCAN...")
            # For DBSCAN we need to find a good eps value
            # A rule of thumb is to use the knee in the k-distance graph

            nn = NearestNeighbors(n_neighbors=min_samples)
            nn.fit(self.embedding_vectors)
            distances, _ = nn.kneighbors(self.embedding_vectors)

            # Sort distances to find knee point
            distances = np.sort(distances[:, -1])

            # Approximate knee point (could be improved)
            knee_idx = np.argmax(distances[1:] - distances[:-1]) + 1
            eps = distances[knee_idx]

            print(f"Using DBSCAN with eps={eps:.4f}, min_samples={min_samples}")
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.clusters = clusterer.fit_predict(self.embedding_vectors)

            # -1 means noise in DBSCAN
            n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
            print(f"DBSCAN found {n_clusters} clusters and {np.sum(self.clusters == -1)} noise points")

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        return self.clusters

    def visualize_clusters(self, method='tsne') -> None:
        """
        Visualize the clusters in 2D using dimensionality reduction.

        Parameters:
        -----------
        method : str
            Method to use for dimensionality reduction. Options:
            - 'tsne': t-SNE
            - 'pca': PCA
            - 'umap': UMAP
        """
        if self.clusters is None:
            print("You need to run cluster_answers() first!")
            return

        # Reduce dimensions for visualization
        if method == 'tsne':
            print("Reducing dimensions with t-SNE...")
            reducer = TSNE(n_components=2, random_state=42)
            reduced = reducer.fit_transform(self.embedding_vectors)

        elif method == 'pca':
            print("Reducing dimensions with PCA...")
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(self.embedding_vectors)

        elif method == 'umap':
            if not HAVE_UMAP:
                print("UMAP not available. Using t-SNE instead.")
                reducer = TSNE(n_components=2, random_state=42)
            else:
                print("Reducing dimensions with UMAP...")
                reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(self.embedding_vectors)

        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

        # Plot results
        plt.figure(figsize=(12, 10))

        # If we have correctness information, use that for colors
        correctness = self.calculate_correctness()

        # Create scatter plot
        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.clusters,
            cmap='tab20',
            alpha=0.7,
            s=100
        )

        # Highlight correct/incorrect answers with different markers
        for i, (x, y, correct) in enumerate(zip(reduced[:, 0], reduced[:, 1], correctness)):
            if correct:
                plt.scatter(x, y, marker='o', s=120, facecolors='none', edgecolors='green', linewidths=2)
            else:
                plt.scatter(x, y, marker='x', s=120, color='red', linewidths=2)

        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Student Answer Clusters ({method.upper()} projection)')
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')

        # Add legend for correctness
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='g',
                   markersize=10, label='Correct Answer'),
            Line2D([0], [0], marker='x', color='r', markersize=10, label='Incorrect Answer')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(f'student_answer_clusters_{method}.png', dpi=300)
        plt.show()

    def analyze_clusters(self) -> Dict[str, Any]:
        """
        Analyze the clusters to identify common patterns and error types.

        Returns:
        --------
        Dict : Dictionary with cluster analysis results
        """
        if self.clusters is None:
            print("You need to run cluster_answers() first!")
            return {}

        # Calculate correctness for each answer
        correctness = self.calculate_correctness()

        # Extract reasoning features instead of error features
        reasoning_features = self.extract_reasoning_features()

        # Calculate answer-reasoning alignment
        alignment_features = self.analyze_answer_reasoning_alignment()

        # Analysis results
        cluster_analysis = {
            'cluster_sizes': {},
            'error_rates': {},
            'common_patterns': {},
            'representative_examples': {},
            'reasoning_feature_averages': {},
            'alignment_feature_averages': {}
        }

        # Get unique cluster labels and iterate
        unique_clusters = sorted(set(self.clusters))

        for cluster in unique_clusters:
            # Get indices of answers in this cluster
            indices = np.where(self.clusters == cluster)[0]
            cluster_size = len(indices)

            # Skip empty clusters
            if cluster_size == 0:
                continue

            # Store cluster size
            cluster_analysis['cluster_sizes'][str(cluster)] = cluster_size

            # Calculate error rate
            correct_count = sum(correctness[i] for i in indices)
            error_rate = 1.0 - (correct_count / cluster_size)
            cluster_analysis['error_rates'][str(cluster)] = error_rate

            # Get student answers in this cluster
            cluster_answers = [self.student_answers[i] for i in indices]

            # Get golden answers for this cluster
            cluster_golden = [self.golden_answers[i] for i in indices] if self.golden_answers else []

            # Get reasoning for this cluster
            cluster_reasoning = [self.student_reasonings[i] for i in indices] if self.student_reasonings else []

            # Get teacher feedback if available
            cluster_feedback = []
            if self.teacher_feedback:
                cluster_feedback = [self.teacher_feedback[i] for i in indices]

            # Get student metadata
            cluster_metadata = []
            if self.student_data:
                cluster_metadata = [self.student_data['metadata'][i] for i in indices]

            # Get most common words in this cluster's answers
            all_words = []
            for answer in cluster_answers:
                all_words.extend([t for t in answer.strip("-,.:;!?").lower().split(" ") if t != "and"])

            word_counts = defaultdict(int)
            for word in all_words:
                word_counts[word] += 1

            common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Get most common reasoning patterns
            reasoning_words = []
            for reasoning in cluster_reasoning:
                reasoning_words.extend([t for t in reasoning.lower().split() if len(t) > 3])

            reasoning_word_counts = defaultdict(int)
            for word in reasoning_words:
                reasoning_word_counts[word] += 1

            common_reasoning_words = sorted(reasoning_word_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Store common patterns
            cluster_analysis['common_patterns'][str(cluster)] = {
                'common_answer_words': common_words,
                'common_reasoning_words': common_reasoning_words,
                'average_answer_length': np.mean([len(a.strip("-,.:;!?").lower()) for a in cluster_answers]),
                'average_reasoning_length': np.mean([len(r) for r in cluster_reasoning]) if cluster_reasoning else 0
            }

            # Find most representative example (closest to centroid)
            if len(indices) > 1:
                cluster_vectors = self.embedding_vectors[indices]
                centroid = np.mean(cluster_vectors, axis=0)

                # Calculate distances to centroid
                distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
                representative_idx = indices[np.argmin(distances)]

                # Store representative example
                example = {
                    'student_answer': self.student_answers[representative_idx],
                }

                if self.student_reasonings:
                    example['reasoning'] = self.student_reasonings[representative_idx]

                if self.golden_answers:
                    example['golden_answer'] = self.golden_answers[representative_idx]
                    example['is_correct'] = correctness[representative_idx]

                if self.teacher_feedback:
                    example['teacher_feedback'] = self.teacher_feedback[representative_idx]

                if self.student_data:
                    metadata = self.student_data['metadata'][representative_idx]
                    example['task_id'] = metadata['task_id']
                    example['sample_id'] = metadata['sample_id']
                    example['part_id'] = metadata['part_id']
                    example['iteration'] = metadata['iteration']

                cluster_analysis['representative_examples'][str(cluster)] = example
            else:
                # Only one example in cluster
                example = {
                    'student_answer': self.student_answers[indices[0]],
                }

                if self.student_reasonings:
                    example['reasoning'] = self.student_reasonings[indices[0]]

                if self.golden_answers:
                    example['golden_answer'] = self.golden_answers[indices[0]]
                    example['is_correct'] = correctness[indices[0]]

                if self.teacher_feedback:
                    example['teacher_feedback'] = self.teacher_feedback[indices[0]]

                if self.student_data:
                    metadata = self.student_data['metadata'][indices[0]]
                    example['task_id'] = metadata['task_id']
                    example['sample_id'] = metadata['sample_id']
                    example['part_id'] = metadata['part_id']
                    example['iteration'] = metadata['iteration']

                cluster_analysis['representative_examples'][str(cluster)] = example

            # Calculate average reasoning features for this cluster
            if reasoning_features:
                cluster_reasoning_features = {
                    feature: np.mean([reasoning_features[i][feature] for i in indices])
                    for feature in reasoning_features[0].keys()
                    if isinstance(reasoning_features[0][feature], (int, float))
                }
                cluster_analysis['reasoning_feature_averages'][str(cluster)] = cluster_reasoning_features

            # Calculate average alignment features for this cluster
            if alignment_features:
                cluster_alignment_features = {
                    feature: np.mean([alignment_features[i][feature] for i in indices])
                    for feature in alignment_features[0].keys()
                    if isinstance(alignment_features[0][feature], (int, float))
                }
                cluster_analysis['alignment_feature_averages'][str(cluster)] = cluster_alignment_features

        return cluster_analysis

    def analyze_iterations(self) -> Dict[str, Any]:
        """
        Analyze how student answers evolve across iterations.
        This is specifically for the teacher-student refinement process.

        Important: Iteration 0 represents initial answers, while iterations > 0
        represent refined answers after teacher feedback.

        Returns:
        --------
        Dict : Dictionary with iteration analysis results
        """
        if not self.student_data:
            raise ValueError("No student data loaded. Run load_student_answer_directory() first")

        if not self.golden_answers:
            print("Warning: No golden answers loaded. Iteration analysis will be limited.")

        # Group data by task-sample-part and sort by iteration
        grouped_data = defaultdict(list)

        # Track initial vs refined answers
        initial_answers = []  # iteration 0
        refined_answers = []  # iteration > 0

        for metadata in self.student_data['metadata']:
            key = (metadata['task_id'], metadata['sample_id'], metadata['part_id'])
            grouped_data[key].append(metadata)

            # Separate initial from refined answers
            if metadata['iteration'] == 0:
                initial_answers.append(metadata)
            else:
                refined_answers.append(metadata)

        print(f"Found {len(initial_answers)} initial answers and {len(refined_answers)} refined answers")

        # Sort each group by iteration
        for key in grouped_data:
            grouped_data[key].sort(key=lambda x: x['iteration'])

        # Analyze improvements across iterations
        improvements = defaultdict(list)
        error_persistence = defaultdict(int)
        common_error_transitions = defaultdict(int)

        # Track first-to-last improvement (initial to final refinement)
        first_to_last_improvements = {
            'initial_correct': 0,
            'final_correct': 0,
            'improved': 0,
            'regressed': 0,
            'unchanged_correct': 0,
            'unchanged_incorrect': 0,
            'total': 0
        }

        for key, iterations in grouped_data.items():
            if len(iterations) <= 1:
                continue

            # Get corresponding golden answer if available
            golden_answer = None
            if self.golden_answers:
                for i, metadata in enumerate(self.student_data['metadata']):
                    if (metadata['task_id'], metadata['sample_id'], metadata['part_id']) == key:
                        if i < len(self.golden_answers):
                            golden_answer = self.golden_answers[i]
                            break

            # Check first and last iterations
            first_iteration = iterations[0]
            last_iteration = iterations[-1]

            first_to_last_improvements['total'] += 1

            if golden_answer:
                first_correct = self.stats.are_identical(golden_answer, first_iteration['answer'])
                last_correct = self.stats.are_identical(golden_answer, last_iteration['answer'])

                if first_correct:
                    first_to_last_improvements['initial_correct'] += 1
                if last_correct:
                    first_to_last_improvements['final_correct'] += 1

                if first_correct and last_correct:
                    first_to_last_improvements['unchanged_correct'] += 1
                elif not first_correct and not last_correct:
                    first_to_last_improvements['unchanged_incorrect'] += 1
                elif not first_correct and last_correct:
                    first_to_last_improvements['improved'] += 1
                elif first_correct and not last_correct:
                    first_to_last_improvements['regressed'] += 1

            # Analyze each consecutive pair of iterations
            for i in range(len(iterations) - 1):
                current = iterations[i]
                next_iter = iterations[i + 1]

                # Check if correct (if golden answer available)
                current_correct = False
                next_correct = False

                if golden_answer:
                    current_correct = self.stats.are_identical(golden_answer, current['answer'])
                    next_correct = self.stats.are_identical(golden_answer, next_iter['answer'])

                # Determine transition type
                if not current_correct and next_correct:
                    transition = "incorrect_to_correct"
                elif current_correct and next_correct:
                    transition = "remained_correct"
                elif not current_correct and not next_correct:
                    transition = "remained_incorrect"
                    # Analyze persistent error patterns
                    error_persistence[key] += 1
                    # Compare error transitions
                    error_pair = (current['answer'], next_iter['answer'])
                    common_error_transitions[error_pair] += 1
                else:
                    transition = "correct_to_incorrect"

                improvements[transition].append({
                    'key': key,
                    'iterations': (current['iteration'], next_iter['iteration']),
                    'answers': (current['answer'], next_iter['answer']),
                    'reasonings': (current['reasoning'], next_iter['reasoning']),
                    'golden_answer': golden_answer
                })

        # Calculate improvement statistics
        total_transitions = sum(len(v) for v in improvements.values())
        improvement_stats = {
            'total_question_iterations': total_transitions,
            'incorrect_to_correct_rate': len(
                improvements['incorrect_to_correct']) / total_transitions if total_transitions else 0,
            'remained_correct_rate': len(
                improvements['remained_correct']) / total_transitions if total_transitions else 0,
            'remained_incorrect_rate': len(
                improvements['remained_incorrect']) / total_transitions if total_transitions else 0,
            'correct_to_incorrect_rate': len(
                improvements['correct_to_incorrect']) / total_transitions if total_transitions else 0,
            'persistent_error_count': sum(error_persistence.values()),
            'questions_with_persistent_errors': len(error_persistence),
            # Add first-to-last improvement stats
            'initial_accuracy': first_to_last_improvements['initial_correct'] / first_to_last_improvements['total'] if
            first_to_last_improvements['total'] else 0,
            'final_accuracy': first_to_last_improvements['final_correct'] / first_to_last_improvements['total'] if
            first_to_last_improvements['total'] else 0,
            'improvement_rate': first_to_last_improvements['improved'] / first_to_last_improvements['total'] if
            first_to_last_improvements['total'] else 0,
            'regression_rate': first_to_last_improvements['regressed'] / first_to_last_improvements['total'] if
            first_to_last_improvements['total'] else 0,
        }

        # Find common error patterns that persist
        common_errors = sorted(dict(common_error_transitions).items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'improvement_stats': improvement_stats,
            'improvements': improvements,
            'persistent_errors': error_persistence,
            'common_error_transitions': common_errors,
            'first_to_last_improvements': first_to_last_improvements
        }

    def compare_initial_vs_refined(self) -> Dict[str, Any]:
        """
        Compare initial answers (iteration 0) vs refined answers (iterations > 0)
        to analyze the impact of teacher feedback.

        Returns:
        --------
        Dict : Dictionary with comparison results
        """
        if not self.student_data:
            raise ValueError("No student data loaded. Run load_student_answer_directory() first")

        if not self.golden_answers:
            print("Warning: No golden answers loaded. Comparison will be limited.")

        # Separate initial from refined answers
        initial_data = {
            'answers': [],
            'reasonings': [],
            'metadata': [],
            'indices': []
        }

        refined_data = {
            'answers': [],
            'reasonings': [],
            'metadata': [],
            'indices': [],
            'by_iteration': defaultdict(list)
        }

        for i, metadata in enumerate(self.student_data['metadata']):
            if metadata['iteration'] == 0:
                initial_data['answers'].append(metadata['answer'])
                initial_data['reasonings'].append(metadata['reasoning'])
                initial_data['metadata'].append(metadata)
                initial_data['indices'].append(i)
            else:
                refined_data['answers'].append(metadata['answer'])
                refined_data['reasonings'].append(metadata['reasoning'])
                refined_data['metadata'].append(metadata)
                refined_data['indices'].append(i)
                refined_data['by_iteration'][metadata['iteration']].append((i, metadata))

        print(
            f"Found {len(initial_data['answers'])} initial answers and {len(refined_data['answers'])} refined answers")

        # Calculate accuracy for initial and refined answers
        if self.golden_answers:
            initial_correct = []
            refined_correct = []

            for i in initial_data['indices']:
                if i < len(self.golden_answers):
                    golden = self.golden_answers[i]
                    student = self.student_answers[i]
                    initial_correct.append(self.stats.are_identical(golden, student))

            for i in refined_data['indices']:
                if i < len(self.golden_answers):
                    golden = self.golden_answers[i]
                    student = self.student_answers[i]
                    refined_correct.append(self.stats.are_identical(golden, student))

            initial_accuracy = sum(initial_correct) / len(initial_correct) if initial_correct else 0
            refined_accuracy = sum(refined_correct) / len(refined_correct) if refined_correct else 0

            # Calculate accuracy by iteration
            accuracy_by_iteration = {}
            for iteration, items in refined_data['by_iteration'].items():
                correct = []
                for i, metadata in items:
                    if i < len(self.golden_answers):
                        golden = self.golden_answers[i]
                        student = self.student_answers[i]
                        correct.append(self.stats.are_identical(golden, student))

                accuracy_by_iteration[iteration] = sum(correct) / len(correct) if correct else 0
        else:
            initial_accuracy = 0
            refined_accuracy = 0
            accuracy_by_iteration = {}
            initial_correct = []
            refined_correct = []

        # Compare common errors
        initial_errors = [self.student_answers[i] for i, correct in zip(initial_data['indices'], initial_correct) if
                          not correct]
        refined_errors = [self.student_answers[i] for i, correct in zip(refined_data['indices'], refined_correct) if
                          not correct]

        # Find common error words in initial vs refined
        initial_error_words = []
        for error in initial_errors:
            words = error.strip('-,.:;!?').lower().split()
            initial_error_words.extend(words)

        refined_error_words = []
        for error in refined_errors:
            words = error.strip('-,.:;!?').lower().split()
            refined_error_words.extend(words)

        initial_word_counts = Counter(initial_error_words)
        refined_word_counts = Counter(refined_error_words)

        # Results
        return {
            'initial_count': len(initial_data['answers']),
            'refined_count': len(refined_data['answers']),
            'initial_accuracy': initial_accuracy,
            'refined_accuracy': refined_accuracy,
            'accuracy_improvement': refined_accuracy - initial_accuracy,
            'accuracy_by_iteration': accuracy_by_iteration,
            'initial_errors': len(initial_errors),
            'refined_errors': len(refined_errors),
            'initial_common_error_words': initial_word_counts.most_common(20),
            'refined_common_error_words': refined_word_counts.most_common(20),
        }

    def identify_skyline_reasoning(self, method='best_answers') -> dict:
        """
        Identify skyline (optimal/reference) reasoning from the dataset.

        Parameters:
        -----------
        method : str
            Method to identify skyline reasoning:
            - 'best_answers': Use reasoning from correct answers
            - 'longest': Use longest reasoning for each question type
            - 'highest_similarity': Use reasoning with highest avg similarity to others

        Returns:
        --------
        dict : Dictionary mapping task_id to skyline reasoning
        """
        if not self.student_data:
            print("Warning: No student data available for skyline identification.")
            return {}

        skyline_reasoning = {}

        if method == 'best_answers' and self.golden_answers:
            # Use reasoning from correct answers as skyline
            correctness = self.calculate_correctness()

            # Group by task_id
            task_groups = defaultdict(list)
            for i, metadata in enumerate(self.student_data['metadata']):
                if i < len(correctness) and correctness[i]:  # Only correct answers
                    task_groups[metadata['task_id']].append({
                        'reasoning': self.student_reasonings[i],
                        'answer': self.student_answers[i],
                        'index': i
                    })

            # Select best reasoning for each task
            for task_id, correct_items in task_groups.items():
                if correct_items:
                    # Use longest reasoning among correct answers
                    best_item = max(correct_items, key=lambda x: len(x['reasoning']))
                    skyline_reasoning[task_id] = best_item['reasoning']

        elif method == 'longest':
            # Use longest reasoning for each task
            task_groups = defaultdict(list)
            for i, metadata in enumerate(self.student_data['metadata']):
                task_groups[metadata['task_id']].append({
                    'reasoning': self.student_reasonings[i],
                    'length': len(self.student_reasonings[i]),
                    'index': i
                })

            for task_id, items in task_groups.items():
                if items:
                    best_item = max(items, key=lambda x: x['length'])
                    skyline_reasoning[task_id] = best_item['reasoning']

        elif method == 'highest_similarity':
            # Use reasoning with highest average similarity to others in same task
            task_groups = defaultdict(list)
            for i, metadata in enumerate(self.student_data['metadata']):
                task_groups[metadata['task_id']].append({
                    'reasoning': self.student_reasonings[i],
                    'index': i
                })

            for task_id, items in task_groups.items():
                if len(items) < 2:
                    if items:
                        skyline_reasoning[task_id] = items[0]['reasoning']
                    continue

                # Calculate similarity matrix for this task
                reasonings = [item['reasoning'] for item in items]
                if self.embedding_method == 'sentence-transformer':
                    embeddings = self.model.encode(reasonings)
                    sim_matrix = cosine_similarity(embeddings)
                else:
                    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(reasonings)
                    sim_matrix = cosine_similarity(tfidf_matrix)

                # Find reasoning with highest average similarity
                avg_similarities = np.mean(sim_matrix, axis=1)
                best_idx = np.argmax(avg_similarities)
                skyline_reasoning[task_id] = reasonings[best_idx]

        print(f"Identified skyline reasoning for {len(skyline_reasoning)} tasks using method '{method}'")
        return skyline_reasoning

    def load_skyline_reasoning_from_file(self, file_path: str, format='csv') -> dict:
        """
        Load skyline reasoning from external file.

        Parameters:
        -----------
        file_path : str
            Path to file containing skyline reasoning
        format : str
            File format: 'csv', 'json', or 'txt'

        Returns:
        --------
        dict : Dictionary mapping task_id to skyline reasoning
        """
        skyline_reasoning = {}

        try:
            if format == 'csv':
                df = pd.read_csv(file_path)
                # Assuming columns: task_id, skyline_reasoning
                for _, row in df.iterrows():
                    skyline_reasoning[row['task_id']] = row['skyline_reasoning']

            elif format == 'json':
                import json
                with open(file_path, 'r') as f:
                    skyline_reasoning = json.load(f)

            elif format == 'txt':
                # Simple format: task_id:reasoning per line
                with open(file_path, 'r') as f:
                    for line in f:
                        if ':' in line:
                            task_id, reasoning = line.split(':', 1)
                            skyline_reasoning[int(task_id.strip())] = reasoning.strip()

            print(f"Loaded skyline reasoning for {len(skyline_reasoning)} tasks from {file_path}")

        except Exception as e:
            print(f"Error loading skyline reasoning from {file_path}: {e}")

        return skyline_reasoning

    def save_results(self, output_dir: str = 'cluster_results') -> None:
        """
        Save all analysis results to files.

        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        import os
        import json

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save cluster analysis if available
        if self.clusters is not None:
            cluster_analysis = self.analyze_clusters()
            with open(os.path.join(output_dir, 'cluster_analysis.json'), 'w') as f:
                json.dump(cluster_analysis, f, indent=2, default=str)
            print(f"Saved cluster analysis to {output_dir}/cluster_analysis.json")

        # Save iteration analysis if available
        if self.student_data:
            try:
                iteration_analysis = self.analyze_iterations()
                with open(os.path.join(output_dir, 'iteration_analysis.json'), 'w') as f:
                    json.dump(iteration_analysis, f, indent=2, default=str)
                print(f"Saved iteration analysis to {output_dir}/iteration_analysis.json")

                initial_vs_refined = self.compare_initial_vs_refined()
                with open(os.path.join(output_dir, 'initial_vs_refined.json'), 'w') as f:
                    json.dump(initial_vs_refined, f, indent=2, default=str)
                print(f"Saved initial vs refined comparison to {output_dir}/initial_vs_refined.json")
            except Exception as e:
                print(f"Error saving iteration analysis: {e}")

        # Save reasoning analysis if available
        if self.student_reasonings:
            try:
                reasoning_features = self.extract_reasoning_features()
                with open(os.path.join(output_dir, 'reasoning_features.json'), 'w') as f:
                    json.dump(reasoning_features, f, indent=2, default=str)
                print(f"Saved reasoning features to {output_dir}/reasoning_features.json")

                alignment_features = self.analyze_answer_reasoning_alignment()
                with open(os.path.join(output_dir, 'alignment_features.json'), 'w') as f:
                    json.dump(alignment_features, f, indent=2, default=str)
                print(f"Saved alignment features to {output_dir}/alignment_features.json")

                consistency_analysis = self.analyze_reasoning_consistency()
                with open(os.path.join(output_dir, 'consistency_analysis.json'), 'w') as f:
                    json.dump(consistency_analysis, f, indent=2, default=str)
                print(f"Saved consistency analysis to {output_dir}/consistency_analysis.json")
            except Exception as e:
                print(f"Error saving reasoning analysis: {e}")

        print(f"All available results saved to {output_dir}/")


# Usage example:
if __name__ == "__main__":
    # Initialize the clusterer
    clusterer = AnswerClusterer(embedding_method='sentence-transformer')

    # Load student answers from directory
    student_data = clusterer.load_student_answer_directory(
        directory_path='path/to/student/answers',
        golden_answers_path='path/to/golden_answers.csv'
    )

    # Perform clustering
    clusterer.embed_answers()
    clusterer.cluster_answers(method='kmeans')

    # Analyze results
    cluster_analysis = clusterer.analyze_clusters()
    iteration_analysis = clusterer.analyze_iterations()
    reasoning_distances = clusterer.calculate_reasoning_distances()

    # Visualize clusters
    clusterer.visualize_clusters(method='tsne')

    # Save results
    clusterer.save_results('results/')

    print("Analysis complete!")

