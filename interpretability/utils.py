import os


def get_scenery_words():
    """
    Get scenery words from the scenery_words folder.

    :return: list of scenery words for filtering attention scores
    """
    scenery_words = []
    for entry in os.scandir("interpretability/scenery_words"):
        if entry.is_file and entry.name.endswith(".txt"):
            with open(entry.path, "r", encoding="UTF-8") as f:
                scenery_words.extend(f.read().splitlines())
    return scenery_words
