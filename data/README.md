# Data

For the project, it is necessary to read, process and save data. These processes are handles with the files within this
directory.

## Pipeline

The DataLoader is responsible for reading in the data from the the files that it was provided in.
The DataProcessor then handles everything related to proceesing it.
The preprocessed data is saved as dictionary of the following format:

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

Finally, the DataSaver is used to save all the generated data into some files.

