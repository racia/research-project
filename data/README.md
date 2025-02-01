# Data

This README will contain information about how we handle the data.
We will probably describe the process from reading the data in to processing it so the models can actually use it.
Furthermore, we should probably give a brief overview of the structure of this folder/what each of the classes does.
Note that this is also already contained in the docstrings of the classes, so the README should focus more on how these
different classes work together and how a pipeline would look like.

## The text below is proobably outdated and should be updated

### Data

The data can be read and preprocessed using the datahandler. The preprocessing includes:

* Splitting the text files into samples. The ID of each sample is used as a key in the data dictionary.
* For each sample, split the sample in context lines, questions, answers, and supporting facts.
* For each line, remove newlines as well as trailing and leading whitespaces.

The preprocessed data is saved as a dictionary of the following format:

```
{sample_id:
    {"context:
        {line_number: line,
         line_number: line, ... }
     "question":
        {line_number: line,
         line_number: line, ... }
    "answer":
        {line_number: answer,
         line_number: answer, ... }
    "supporting_fact": [[line_number_first_answer, ...], [line_number_second_answer, ...], ...]
    }
}
```

Below is an example:

```
{0: 
    {
        'context': {
            1: 'Mary moved to the bathroom.', 
            2: 'John went to the hallway.', 
            4: 'Daniel went back to the hallway.', 
            5: 'Sandra moved to the garden.', 
            7: 'John moved to the office.', 
            8: 'Sandra journeyed to the bathroom.', 
            10: 'Mary moved to the hallway.', 
            11: 'Daniel travelled to the office.', 
            13: 'John went back to the garden.', 
            14: 'John moved to the bedroom.'
            }, 
        'question': {
            3: 'Where is Mary?', 
            6: 'Where is Daniel?', 
            9: 'Where is Daniel?', 
            12: 'Where is Daniel?', 
            15: 'Where is Sandra?'
            }, 
        'answer': {
            3: 'bathroom', 
            6: 'hallway', 
            9: 'hallway', 
            12: 'office', 
            15: 'bathroom'
            }, 
        'supporting_fact': [1, 4, 4, 11, 8]
    }
}
```