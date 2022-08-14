# DL4NLP

Repository containing the code for the project "NLI as evaluation metric"

Install the required modules using pip install -r requirements.txt

Then, there are two scripts for evaluation:

- evaluate_metrics.py --metric --path
- evaluate_correlation.py --metric --path

A list of metrics:
[BartScore, BertScore, NLI1Score, NLI2Score, SummaCZS, SummaCConv, SummaQA, Blanc, MoverScore, CHRF, Meteor]

Argument "path" denotes the path to the data, i.e. the adversarial test suite (https://drive.google.com/drive/folders/1GO83X04cBsZlmct92NdjUW9E8WUJleAC?usp=sharing) or the human annotated data (already in the "data" folder)

The output of the model is written into the data folder
