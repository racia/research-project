import nltk
import numpy as np

from data.DataLoader import DataLoader
from evaluation.Statistics import Statistics

from pathlib import Path
import re
from typing import Dict, List
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import evaluate

from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
#from rouge_score import rouge_score


import evaluate

class StudentAnsewerLoader:
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


    def calculate_reasoning_distances(self, use_embeddings=True, skyline_reasoning=None)->dict:
        """
        Calculate distance metrics between reasoning texts to identify patterns.
        Now includes ROUGE, BLEU, METEOR scores and skyline reasoning comparison.

        :params use_embeddings: Whether to use semantic embeddings or lexical similarity
        :params skyline_reasoning:  Reference/optimal reasoning texts to compare against

        :return Dictionary with distance matrices and similarity scores
        """
        if not self.student_reasonings:
            print("Warning: No reasoning data available. ")
            return {}

        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        meteor = evaluate("meteor")

        results = {}

        # 1. Calculate pairwise similarities between student reasoning
        if use_embeddings and self.embedding_method == 'sentence-trasnformer':
            reasoning_embeddings = self.model.encode(self.student_reasonings)

            cosine_sim = cosine_similarity(reasoning_embeddings)

            #find most similar reasoning pairs
            similarity_pairs = []
            n = len(reasoning_embeddings)
            for i in range(n):
                for j in range(i+1,n):
                    similarity_pairs.append({
                        'idx1':i,
                        'idx2':j,
                        'cosine_similarity': cosine_sim[i][j],
                        'reasoning1': self.student_reasonings[i][:100] + "..." if len(self.student_reasonings[i]) > 100 else self.student_reasonings[i],
                        'reasoning2': self.student_reasonings[j][:100] + "..." if len( self.student_reasonings[j]) > 100 else self.student_reasonings[j]

                    })

            similarity_pairs.sort(key=lambda  x:x['cosine_similarity'], reverse=True)
            results.update({
                'cosine_similarity_maitrix':cosine_sim,
                'most_similar_pairs': similarity_pairs[:20],
                'average_similarity': np.mean(cosine_similarity[np.triu_indices_from(cosine_sim, k=1)]),
            })
        else:

        #Use Lexical similarity
        vectorize= TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix= vectorize.fit_transform(self.student_reasonings)
        cosine_sim = cosine_similarity(tfidf_matrix)

        results.update({
            'lexcial_similarity_matrix': cosine_sim,
            'average_lexcial_similarity': np.mean(cosine_sim[np.triu_indices_from(cosine_sim, k=1)])

        })
        if rouge and bleu and meteor:
            print("Calculating ROUGE, BLEU, METEOR scores...")

            # Pairwise evaluation metrics
            rouge_scores = []
            bleu_scores = []
            meteor_scores = []

            # Calculate for top similar pairs to avoid too many computations
            n_samples = min(100, len(self.student_reasonings))

            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    reasoning1 = self.student_reasonings[i]
                    reasoning2 = self.student_reasonings[j]

                    if not reasoning1.strip() or not reasoning2.strip():
                        continue
                    try:
                        #ROUGE score
                        rouge_result = rouge.compute(predictions=[reasoning1], references=[reasoning2])
                        rouge_scores.append({
                            'idx1':i,
                            'idx2':j,
                            'rouge1': rouge_result.get('rouge1', 0),
                            'rouge2': rouge_result.get('rouge2', 0),
                            'rougeL': rouge_result.get('rougeL', 0)

                        })
                        #BLEU score
                        bleu_result = bleu.compute(predictions=[reasoning1], refrences=[[reasoning2]])
                        bleu_scores.append({
                            'idx1': i,
                            'idx2': j,
                            'bleu': bleu_result.get('bleu',0)

                        })
                        #meteor score
                        reasoning1_tokens = nltk.word_tokenize(reasoning1)
                        reasoning2_tokens = nltk.word_tokenize(reasoning2)
                        #results.update()




        return


    def emeb_answers(self):
        return



    def cluster(self):
        return

