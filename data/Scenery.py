import nltk
from DataHandler import DataHandler
from nltk.tokenize import word_tokenize


class Scenery:
    """
    This class is responsible for extracting the agents, locations, objects and relations that occur in the data.
    """

    def __init__(self, path: str):
        self.task_sceneries = {}

        self.agents = set()
        self.locations = set()
        self.objects = set()
        self.relations = set()

        self.pos_tags = set()
        self.tokens = set()

        dh = DataHandler()
        data = dh.read_data(path=path, split="train")
        self.data = dh.process_data(data)

    def extract_from_word(self, answer: str) -> str:
        """
        Extracts the agents, locations, objects and relations from a word, e.g. the answer.

        :param line: the context line

        :return str: the token type
        """
        pos_tuple = nltk.pos_tag(answer)

        to_or_in = False

        token_type = None

        for token, pos in pos_tuple:
            self.pos_tags.add(pos)
            self.tokens.add(token)

            if pos == "NNP" or pos == "NNPS":
                token_type = "agent"
                self.agents.add(token)
            elif pos == "NN" or pos == "NNS":
                if to_or_in:
                    token_type = "location"
                    self.locations.add(token)
                else:
                    token_type = "object"
                    self.objects.add(token)
            elif pos == "IN":
                token_type = "location"
                self.locations.add(token)
            elif "VB" in pos:
                token_type = "relation"
                self.relations.add(token)

            if pos == "TO" or pos == "IN":
                to_or_in = True

        return token_type

    def extract_from_line(self, line: str) -> dict:
        """
        Extracts the agents, locations, objects and relations from a context line or a question.

        :param line: the context line

        :return dict: a dictionary containing the agents, locations, objects and relations
        """
        tokens = word_tokenize(line)
        pos_tuple = nltk.pos_tag(tokens)

        tagged_sent = []

        agents = []
        locations = []
        objects = []
        relations = []

        for token, pos in pos_tuple:
            self.pos_tags.add(pos)
            self.tokens.add(token)
            tagged_sent.append(pos)

            if pos == "NNP" or pos == "NNPS":
                agents.append(token)
                self.agents.add(token)
            elif pos == "NN" or pos == "NNS":
                # check if to or in occurred within the last two tokens
                if "TO" in tagged_sent[-3:] or "IN" in tagged_sent[-3:]:
                    locations.append(token)
                    self.locations.add(token)
                else:
                    objects.append(token)
                    self.objects.add(token)
            elif pos == "IN":
                locations.append(token)
                self.locations.add(token)
            elif "VB" in pos:
                relations.append(token)
                self.relations.add(token)

        return {
            "agents": agents,
            "locations": locations,
            "objects": objects,
            "relations": relations,
        }

    def extract_scenery(self):
        """
        Extracts the agents, locations, objects and relations that occur in the data.
        :return:
        """
        for task in self.data.keys():
            for sample in self.data[task].values():
                for context_line in sample["context"].values():
                    self.extract_from_line(context_line)
                for answer in sample["answer"].values():
                    self.extract_from_word(answer)
                for question in sample["question"].values():
                    self.extract_from_line(question)
            self.task_sceneries[task] = {
                "agents": self.agents,
                "locations": self.locations,
                "objects": self.objects,
                "relations": self.relations,
            }

            # reset the sets
            self.agents = set()
            self.locations = set()
            self.objects = set()
            self.relations = set()

    def save_scenery(self, save_path: str):
        """
        Saves the extracted agents, locations, objects and relations to a file.
        """
        with open(save_path, "wt", encoding="UTF-8") as file:
            for task in self.task_sceneries:
                file.write(str(task) + ":\n")
                file.write("-----------------\n")
                file.write("Agents:\n")
                for agent in self.task_sceneries[task]["agents"]:
                    file.write(agent + "\n")
                file.write("\nLocations:\n")
                for location in self.task_sceneries[task]["locations"]:
                    file.write(location + "\n")
                file.write("\nObjects:\n")
                for obj in self.task_sceneries[task]["objects"]:
                    file.write(obj + "\n")
                file.write("\nRelations:\n")
                for relation in self.task_sceneries[task]["relations"]:
                    file.write(relation + "\n")

                file.write("\nPOS Tags:\n")
                for pos in self.pos_tags:
                    file.write(pos + ", ")
                file.write("\nTokens:\n")
                for token in self.tokens:
                    file.write(token + ", ")
                file.write("\n\n")


if __name__ == "__main__":
    scenery = Scenery(path="tasks_1-20_v1-2/en")
    scenery.extract_scenery()
    scenery.save_scenery("scenery.txt")
