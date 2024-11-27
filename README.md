# Research Project

## Running the baseline

After git cloning the repository:
1. Create a virtual environment: `python3 -m venv venv`
2. Install all dependencies: `pip install -r requirements.txt`
3. Activate environment: `source /venv/bin/activate`

Running models from the Hugging Face hub requires an access token, which you can obtain via the website on your https://huggingface.com profile.  
4. Save your token as an environment variable in bash:
```
export HUGGINGFACE="<<your-token>>"
```

5. Submit batch job `sbatch initial_baseline.sh`, which will run script.py and save the outputs to "init_bl.txt"
