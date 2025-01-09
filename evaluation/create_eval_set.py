from pathlib import Path

from Scenery import Scenery
from data.DataLoader import DataLoader
from data.DataSaver import DataSaver


general_relations = {
    "be": "be",
    "be\nbigger": "bigger",
    "be\nsmaller": "smaller",
    "be\nleft": "left",
    "be\nright": "right",
    "be\neast": "east",
    "be\nnorth": "north",
    "be\nsouth": "south",
    "be\nwest": "west",
    "carry": "have",
    "discard": "put",
    "drop": "put",
    "fit": "fit",
    "follow": "move",
    "get": "take",
    "give": "give",
    "go": "move",
    "grab": "take",
    "hand": "give",
    "journey": "move",
    "leave": "put",
    "move": "move",
    "pass": "give",
    "pick up": "take",
    "put down": "put",
    "receive": "have",
    "take": "take",
    "travel": "move",
}

result_relations = {
    "be": "be",
    "have": "have",
    "put": "not have",
    "move": "be",
    "take": "have",
    "give": "not have",
}

implicit_relations = {
    "give": "have",
}


def create_dataset(split="valid"):
    scenery = Scenery()
    loader = DataLoader()
    current_path = Path.cwd() / ".." / "evaluation"
    saver = DataSaver(results_path=current_path)

    # Create the evaluation set
    data_path = f"../../tasks_1-20_v1-2/en-{split}"
    data = loader.load_data(path=data_path, split=split, tasks=None)

    print("Data is read.")
    print("Example:", list(data.items())[0])
    print("Data is processed.")
    print("Example:", list(data.items())[0])

    headers = ["entry_id", "line_id", "part_id", "task_id", "sample_id",
               "context/question",
               # extracted scenery
               "human_subjects", "non_human_subjects", "subj_attributes",
               "relations", "generalized_relations", "locations",
               "direct_objects", "obj_attributes", "indirect_objects",
               # to annotate
               "result_relation", "reasoning",
               # implicit knowledge to annotate
               "implicit_object", "implicit_relation", "implicit_location",
               # golden data
               "golden_answer", "golden_supporting_facts"]

    saver.results_headers = headers
    saver.results_name = "evalset_to_annotate.csv"

    entry_id = 0

    print("Starting to process data.")
    question = True

    for task_id, task in data.items():
        print("\n")
        print("Task:", task_id)

        for sample_id, sample in list(task.items())[0:5]:
            sample_id_ = sample_id + 1
            print("Sample:", sample_id_)

            part_id = 1

            line_ids = list(sample["context"].keys()) + list(sample["question"].keys())
            line_ids = sorted([int(line_id) for line_id in line_ids])

            for line_id in line_ids:
                if question is True:
                    question = False
                    print("Part:", part_id)

                entry_id += 1
                line = {
                    "entry_id": entry_id,
                    "task_id": task_id,
                    "sample_id": sample_id_,
                    "part_id": part_id,
                    "line_id": line_id
                }
                sentence = ""

                if line_id in sample["context"].keys():
                    sentence = sample["context"][line_id]

                elif line_id in sample["question"].keys():
                    question = True
                    sentence = sample["question"][line_id]

                    line["golden_answer"] = " ".join(sample["answer"][line_id])

                    supporting_facts = [str(i) for i in sample["supporting_fact"][part_id - 1]]
                    line["golden_supporting_facts"] = " ".join(supporting_facts)

                    part_id += 1

                line["context/question"] = sentence
                print(sentence)

                line_scenery = scenery.extract_from_line(line=sentence).items()
                print(line_scenery)

                line_scenery = {type_: "\n".join(located)
                                if not located or type(located[0]) is str
                                else "\n".join([" ".join(parts) for parts in located])
                                for type_, located in line_scenery}

                line_scenery["generalized_relations"] = general_relations.get(
                    line_scenery.get("relations", ""), ""
                )
                line_scenery["result_relation"] = result_relations.get(
                    line_scenery.get("generalized_relations", ""), ""
                )
                line_scenery["implicit_relation"] = implicit_relations.get(
                    line_scenery.get("generalized_relations", ""), ""
                )

                line.update(line_scenery)
                saver.save_output(data=[line])
                print(line)

    print("Data processing complete.")

    print("Data is saved in", saver.results_path / saver.results_name)


if __name__ == '__main__':
    create_dataset()
