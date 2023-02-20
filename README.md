# SemEval2023-Task5-clickbait-spoiling

#### Only codes are provided, models only have config files but not the .bin file.

Training spoiler type classify model, read file from dataset folder, and initialize with deberta model.
Hyperparameters are not read by args, but edited in beginning of code. Such as data_from_pickle = False when first time running.  

`python run_classification.py --input tmp_path --output tmp_path --model models/deberta`

Training spoiler extract models, read file from dataset folder, and initialize with deberta model.

`python run_training.py --input tmp_path --output tmp_path --model models/deberta`

Inferencing for task 1, using the actual input and output filepath:
Models loaded is set in the LOADING_MODEL constant. 
if gt_path in the code is provided with a not None value, evaluation of BLEU-n and meteor metrics would be performed. 

`python clickbait_spoiling_task_1.py --input dataset/test.jsonl --output output/run.jsonl`

Inferencing for task 2, using the actual input and output filepath:
Models loaded is set in the LOADING_BUCKET_MODELS constant mapping.
if gt_path in the code is provided with a not None value, evaluation of BLEU-n and meteor metrics would be performed. 

`python clickbait_spoiling_task_2.py --input dataset/test.jsonl --output output/run.jsonl`

Known spoiler type evaluation and threshold gridsearch are performed by running "oracle_eval.py" and "confidence_search.py" similar as the inference scripts.