from dataclasses import dataclass

import en_core_web_sm

nlp = en_core_web_sm.load()

from data.DataLoader import DataLoader


@dataclass
class SentenceScenery:
    sentence: str
    human_subjects: list[str]
    non_human_subjects: list[str]
    direct_objects: list[str]
    indirect_objects: list[str]
    locations: list[str]
    relations: list[str]
    subj_attributes: list[str]
    obj_attributes: list[str]

    def __repr__(self):
        return (
            f"Sentence: {self.sentence}"
            f"Human Subjects: {self.human_subjects}\n"
            f"Non-Human Subjects: {self.non_human_subjects}\n"
            f"Direct Objects: {self.direct_objects}\n"
            f"Indirect Objects: {self.indirect_objects}\n"
            f"Locations: {self.locations}\n"
            f"Relations: {self.relations}\n"
            f"Subjects Attributes: {self.subj_attributes}\n"
            f"Object Attributes: {self.obj_attributes}\n"
        )

    def get(self):
        return {
            "human_subjects": self.human_subjects,
            "non_human_subjects": self.non_human_subjects,
            "direct_objects": self.direct_objects,
            "indirect_objects": self.indirect_objects,
            "locations": self.locations,
            "relations": self.relations,
            "subj_attributes": self.subj_attributes,
            "obj_attributes": self.obj_attributes,
        }


class Scenery:
    """
    This class is responsible for extracting the agents, locations, objects and relations that occur in the data.
    """

    base_phrasal_verbs = (
        "base",
        "beat",
        "break",
        "bring",
        "call",
        "calm",
        "catch",
        "check",
        "cheer",
        "count",
        "cut",
        "deal",
        "do",
        "drop",
        "eat",
        "end",
        "figure",
        "fill",
        "find",
        "get",
        "give",
        "go",
        "grow",
        "hang",
        "hit",
        "hold",
        "keep",
        "look",
        "make",
        "mess",
        "pass",
        "pick",
        "point",
        "put",
        "rip",
        "run",
        "set",
        "show",
        "shut",
        "sleep",
        "speak",
        "stand",
        "take",
        "think",
        "tie",
        "work",
    )

    def __init__(self):
        self.task_sceneries = {}

        self.agents = set()
        self.locations = set()
        self.objects = set()
        self.relations = set()

        self.pos_tags = set()
        self.lemmas = set()

        self.data = None

    def get_DO_NP(self, DO_head):
        """
        Extracts the whole noun phrase of a direct object.

        :param DO_head: the head of the direct object
        """
        children = [
            [child.text]
            + [child.text for child in child.children if child.dep_ != "det"]
            for child in DO_head.children
            if child.pos_ not in ["ADJ"]
            and child.lemma_
            not in ["before", "after", "to", "right", "left", "below", "above"]
        ]
        children = [child for child in children if child[0] != child[-1]]
        if children:
            longest_child = max(children, key=lambda x: len(x))
            if len(longest_child) >= 2:
                return longest_child

    def extract_from_line(self, line: str) -> SentenceScenery:
        """
        Extracts the agents, locations, objects and relations from a context line or a question.

        :param line: the line to extract the scenery from
        :return: a dictionary containing the extracted scenery
        """
        parsed_line = nlp(line)

        human_subjects = []
        non_human_subjects = []
        direct_objects = []
        indirect_objects = []
        relations = []
        locations = []
        subj_attributes = []
        obj_attributes = []

        for token in parsed_line:
            self.pos_tags.update([token.pos_])
            self.lemmas.update([token.lemma_])

            # Extract the human subjects: Jason, Mary, etc.
            # As an exception, "what", a direct objects, is also caught here.
            if token.dep_ in ["nsubj", "agent", "conj"] and token.pos_ in [
                "PROPN",
                "PRON",
            ]:
                if token.text.lower() == "what":
                    direct_objects.append(token.lemma_)
                else:
                    if token.pos == "PROPN":
                        human_subjects.append(token.lemma_.capitalize())
                    else:
                        human_subjects.append(token.lemma_)
            # Catch human subjects mistakes when they look like adverbs: Emily, Lily, etc.
            elif (token.dep_ == "nsubj" and token.pos_ == "ADV") or (
                token.dep_ == "advmod" and token.text in ["emily", "lily"]
            ):
                human_subjects.append(token.lemma_.capitalize())
            # Extract the other human and non-human subjects.
            elif token.dep_ == "nsubj" and token.pos_ == "NOUN":
                # Catch non-capitalized human subjects as exceptions: antoine, yann, etc.
                if token.lemma_ in ["antoine", "yann"]:
                    human_subjects.append("Antoine")
                    continue
                # check if a non-human subject is a noun phrase: box of chocolates, etc.
                children = self.get_DO_NP(token)
                if children:
                    non_human_subjects.append(tuple([token.lemma_] + children))
                else:
                    non_human_subjects.append(token.lemma_.lower())
            # Extract the relations: go, bring, etc.
            elif token.dep_ == "ROOT" and token.pos_ == "VERB":
                particle = [
                    child
                    for child in token.children
                    if child.dep_ in ["prt", "advmod", "neg"]
                    and child.pos_ not in ["SCONJ", "ADV"]
                ]
                # Extract the particles of phrasal verbs: go up, bring down, etc.
                if token.lemma_ in Scenery.base_phrasal_verbs and particle:
                    relations.append((token.lemma_, particle[0].text))
                else:
                    # Otherwise add a normal relation.
                    relations.append(token.lemma_)
            # Extract the to-be relations: be, is, etc.
            elif token.dep_ == "ROOT" and token.lemma_ == "be":
                relations.append(token.lemma_)
            # Extract the attributes.
            elif token.dep_ in "amod" and token.pos_ in "ADJ":
                # Analyse their dependencies: is their head a subject or an object?
                if token.head.dep_ == "pobj":
                    obj_attributes.append(token.lemma_)
                if token.head.dep_ == "nsubj":
                    subj_attributes.append(token.lemma_)
            # Extract various parts associated with attribution.
            elif token.dep_ in ["attr", "acomp"] and token.pos_ in ["NOUN", "ADJ"]:
                # Catch exceptional relations: bigger, smaller, north, south, east, west, etc.
                if token.text in [
                    "bigger",
                    "smaller",
                    "north",
                    "south",
                    "east",
                    "west",
                ]:
                    relations.append(token.text)
                # Additionally catch "to be" as a relation.
                elif (
                    token.head.head.lemma_ == "be"
                    and token.head.head.lemma_ == parsed_line[0].lemma_
                ):
                    relations.append(token.text)
                else:
                    # Otherwise, it is a subject attribute: tired, hungry etc.
                    subj_attributes.append(token.lemma_)
            # Extract the direct objects: the book, the box of chocolates, etc.
            elif token.dep_ == "dobj" and token.pos_ == "NOUN":
                children = self.get_DO_NP(token)
                if children:
                    direct_objects.append(tuple([token.lemma_] + children))
                else:
                    direct_objects.append(token.lemma_)
            # Extract the indirect objects that are usually humans here: Mary etc.
            elif (
                token.dep_ in ["dobj", "pobj", "nsubjpass", "conj"]
                and token.pos_ == "PROPN"
            ):
                indirect_objects.append(token.lemma_)
            # Catch various parts as objects of prepositions
            # and conjunctions (could be a second part of a compound subject).
            elif token.dep_ in ["pobj", "conj"] and token.pos_ == "NOUN":
                # Catch exceptional relations: right, left, below, above, etc.
                if token.text in ["right", "left", "below", "above"]:
                    relations.append(token.text)
                    continue
                # Catch indirect objects with their prepositions: to, for, etc.
                if token.head.pos_ == "ADP" and token.head.text == "of":
                    indirect_objects.append(token.lemma_)
                # Escape if it's a subject (we have already caught it).
                elif token.head.head.dep_ == "nsubj":
                    continue
                # Otherwise, it is a location.
                # Ex: Then he went back _to_ the *bathroom*.
                else:
                    locations.append(token.lemma_)
            # Catch abstract deictic locations.
            elif token.dep_ == "advmod" and token.text in ["here", "there", "where"]:
                locations.append(token.lemma_.lower())

        return SentenceScenery(
            sentence=line,
            human_subjects=human_subjects,
            non_human_subjects=non_human_subjects,
            direct_objects=direct_objects,
            indirect_objects=indirect_objects,
            relations=relations,
            locations=locations,
            subj_attributes=subj_attributes,
            obj_attributes=obj_attributes,
        )

    def extract_scenery(self):
        """
        Extracts the agents, locations, objects and relations that occur in the data.
        :return:
        """
        for task in sorted(list(self.data.keys())):
            self.task_sceneries[task] = []
            for sample in self.data[task].values():
                sample_sentences = []
                for part in sample.values():
                    sample_sentences.extend(
                        list(part["context"].values())
                        + list(part["question"].values())
                        + list(part["answer"].values())
                    )
                for line in sample_sentences:
                    if type(line) is list:
                        line = " ".join(line)
                    self.task_sceneries[task].append(self.extract_from_line(line))

    def get_unique_scenery(self, task, scenery_type) -> list:
        """
        Get the unique scenery elements for a given task and scenery type.

        :param task: the task to get the unique scenery elements from
        :param scenery_type: the type of scenery to get the unique elements from
        :return: a list of unique scenery elements
        """
        unique_scenery = set()
        for sample in self.task_sceneries[task]:
            unique_scenery.update(sample[scenery_type])

        unique_scenery = [
            " ".join(list(scenery)) if type(scenery) is tuple else scenery
            for scenery in unique_scenery
        ]
        sorted_unique_scenery = sorted(list(unique_scenery))
        return sorted_unique_scenery

    def save_scenery(self, save_path: str):
        """
        Saves the extracted agents, locations, objects and relations to a file.

        :param save_path: the path to save the extracted scenery to
        """
        scenery_dict = {
            "Human Subjects:\n": "human_subjects",
            "Non-Human Subjects:\n": "non_human_subjects",
            "Direct Objects:\n": "direct_objects",
            "Indirect Objects:\n": "indirect_objects",
            "Relations:\n": "relations",
            "Locations:\n": "locations",
            "Subjects Attributes:\n": "subj_attributes",
            "Object Attributes:\n": "obj_attributes",
        }
        with open(save_path, "wt", encoding="UTF-8") as file:
            for task in self.task_sceneries:
                file.write(str(task) + ":\n")
                file.write("-----------------\n")

                for header, scenery_type in scenery_dict.items():
                    file.write(header)
                    unique_scenery = self.get_unique_scenery(task, scenery_type)
                    if unique_scenery:
                        [file.write(subject + "\n") for subject in unique_scenery]
                    else:
                        file.write("-\n")
                    file.write("\n")
                file.write("\n")

            file.write("\n\nPOS Tags:\n")
            for pos in self.pos_tags:
                file.write(pos + ", ")
            file.write("\n\nLemmas:\n")
            for lemma in self.lemmas:
                file.write(lemma + ", ")
            file.write("\n\n\n")


if __name__ == "__main__":
    scenery = Scenery()
    loader = DataLoader()
    scenery.data = loader.load_task_data(
        path="../../tasks_1-20_v1-2/en-valid", split="train"
    )
    scenery.extract_scenery()
    scenery.save_scenery("../evaluation/scenery_train.txt")
