# Prompts

This project uses multiple different prompts. These prompts are stored as text files in this directory.

## Types of prompts

### Setting overlapping

#### init_prompt

The "init" prompt is used in all of our settings. It is used to prompt the model to generate an initial chain-of-thought
and answer the provided question.
It is possible to append some demonstrations to the prompt.

### Feedback

#### feedback_prompt

The "feedback" prompt is given to the teacher model in the feedback setting. It is used to prompt the teacher to look at
the current chain-of-thought of the student model and provide some feedback in order to improve it.

#### refine_prompt

The "refine" prompt is given to the student model after the teacher has provided some feedback. It is used to prompt the
student to refine its chain-of-thought based on the feedback that was provided by the teacher.

### Speculative Decoding

#### eval_prompt

The "eval" prompt is given to the teacher model. It tells the teacher to generate its own answer and reasoning to the
question. This answer is then used to evaluate the student's answer.

#### continue_prompt

The "continue" prompt is given to the student model after its initial chain-of-thought has been evaluated with the
teacher model. It prompts the student to continue its chain-of thought from the corrected token on that was provided by
the teacher.