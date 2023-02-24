# Gateway Extraction (Master Thesis)

## Information

Title: *Extraction of Gateways in Process Model Generation from Text*

Author: Janek Putz

Supervisor: [Prof. Dr. Han van der Aa](https://hanvanderaa.com/) ([Data and Web Science Group](https://www.uni-mannheim.de/dws/), University of Mannheim)

Date: Sep 2022 - Feb 2023

Keywords: Business Process Management, Process Discovery, Process Model Extraction, Gateway Extraction, Keyword Search, Sequence Flow Detection, NLP, Token Classification, Activity Relations, Relation Classification

This thesis investigates the extraction of gateways in the scope of extracting process models from unstructured documents. The repository contains code for implementation and evaluation of the following components:
1. Traditional rule-based approach to extract gateways by keyword search and derive sequence flows
2. Token classification model to filter false positive gateway extractions from the rule-based approach
3. A model to support the rule-based approach in detecting if two gateway entities describe branches of the same gateway construct
4. A novel approach how to extract gateways by analyzing the relation between activities

The work is based on a new dataset for process extraction from natural language texts called *PET*. It contains extensive annotations for various process elements.
+ Huggingface: [patriziobellan/PET](https://huggingface.co/datasets/patriziobellan/PET)
+ [Website](https://pdi.fbk.eu/pet-dataset/) and [Paper](https://arxiv.org/abs/2203.04860)


## Example: Process Model Extraction
Gateway extraction and subsequent sequence flow derivation are demonstrated with the following example:

**Input** (doc-3.2, PET dataset):
````
Each morning, the files which have yet to be processed need to be checked, to make sure they are in order for the court hearing that day.
If some files are missing, a search is initiated, otherwise the files can be physically tracked to the intended location.
Once all the files are ready, these are handed to the Associate, and meantime the Judgeis Lawlist is distributed to the relevant people.
Afterwards, the directions hearings are conducted.
````

**Output**
![doc-3.2](/other/doc-3.2.png)


## Usage Instructions
Detailed usage and parameterization options are documented in each class and function.

### 1) Token-based Approaches
[/token_approaches](/token_approaches) contains all code for the rule-based approach and its extension of token and same gateway classification.
In order to ...
+ evaluate the rule-based approach on all documents, test single documents or rerun the paper statistics, run [RuleApproach.py](/token_approaches/RuleApproach.py)
+ train a false positive filter model, adjust the namespace arguments in [GatewayTokenClassifier.py](/token_approaches/GatewayTokenClassifier.py) and execute
+ use the false positive filter extension in the rule-based approach, set the path to the saved model in [config.json](config.json), adjust the run section of [RuleTokenFilteredApproach.py](token_approaches/RuleTokenFilteredApproach.py) and run
+ train a same gateway classification model, adjust the namespace arguments in [SameGatewayClassifier.py](/token_approaches/SameGatewayClassifier.py) and execute
+ use the same gateway classification extension in the rule-based approach, set the path to the saved model in [config.json](config.json), adjust the run section of [RuleSGCApproach.py](token_approaches/RuleSGCApproach.py) and execute

The training of different architecture refinements for both models can be executed automatically via scripts in [../run_scripts/same_gateway_cls](/token_approaches/run_scripts/same_gateway_cls) and [../run_scripts/token_cls](/token_approaches/run_scripts/token_cls) that set the respective paramterizations and initiates the training.

### 2) Activity-relation Approach
[/relation_approaches](/relation_approaches) contains all code for the extraction of gateways analyzing activity relations.
In order to ...
+ train a activity relation classification model, adjust the namespace arguments in [RelationClassifier.py](/relation_approaches/RelationClassifier.py) and execute
+ evaluate a activity relation classification model, adjust the dummy namespace arguments in ``get_dummy_args`` in [RelationClassificationBenchmark.py](/relation_approaches/RelationClassificationBenchmark.py) to the architecture values used during training and execute
+ test gateway extraction on single documents, configure the main section in [GatewayExtractor.py](/relation_approaches/GatewayExtractor.py) with a relation classifier instance and desired document and execute
+ evaluate the gateway extraction on all or desired test documents, configure the main section in [GatewayExtractionBenchmark.py](/relation_approaches/GatewayExtractionBenchmark.py) with a relation classifier instance and execute

The training of different architecture refinements for the relation classification model can be executed automatically via scripts in [../run_scripts](/relation_approaches/run_scripts) that set the respective paramterizations and initiates the training.

### 3) Rest
Additional runnable code:
+ [/important_analysis](/important_analysis) contains additional scripts to analyze data in results to generate stats used in the paper
+ [/notebooks](/notebooks) contains various Jupyter notebooks used during development

More classes and scripts exist that are not intended for stand alone execution.
+ [PetReader.py](/PetReader.py) wraps the ``petdatasetreader`` module to read [patriziobellan/PET](https://huggingface.co/datasets/patriziobellan/PET) with customized reading functions
+ [training.py](/training.py) and [Ensemble.py](/Ensemble.py) provide structures used to train models of both token and activity relation approaches
+ [utils.py](/utils.py) collects various helper methods and functionality to read statc information from files in [/data](/data)

## Requirements
+ matplotlib~=3.6.3
+ numpy~=1.23.0
+ openpyxl~=3.0.10
+ pandas~=1.4.3
+ petbenchmarks~=0.0.1a3
+ petdatasetreader~=0.0.1b2
+ scikit-learn~=1.1.3
+ tensorflow~=2.8.0
+ tensorflow-addons~=0.18.0
+ transformers~=4.20.1