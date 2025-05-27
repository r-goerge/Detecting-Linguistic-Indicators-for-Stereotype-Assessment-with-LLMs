# Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models

**Rebekka Görge, Michael Mock, Héctor Allende-Cid**


[[PDF]]() [[arXiv]](https://arxiv.org/pdf/2502.19160)


This is the official repository for our FAccT 2025 paper "Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models". See `paper/DetectingLinguisticIndicatorsForStereotypeAssessmentWithLargeLanguageModels.pdf`.

To detect linguistic indicators described in the SCSC categorization scheme, and to calculate the score_scsc for these stereotypes, please perform the following steps: 

## Setup

Create a virtual environment with Python 3.9.2. and run

```
pip install -r requirements.txt
pip install -e .
```

The code uses OpenAI Chat API as endpoint. To use GPT-4 or other OPEN AI models, rename `.env.sample` to `.env` and insert your API keys. 

## Evaluate the stereotypes in a given dataset

### Query an LLM to evaluate linguistic indicators 

Run 

```bash
python stereotype_assessment.py 
```

to query the LLM for the prompt that implements the SCSC categorization scheme. This function has the parameters  `model_name` (str), `prompt_id` (str), `evaluate_linguistic_indcators` (boolean), and `score scsc`(boolean). By default model_name is set to GPT-4.1, prompt_id is P_F_01, evaluate_linuistuic_indicators is TRUE (meaning the linguistic indicators in a sentence are detected by the model) and the score_scsc is TRUE (meaning the score_scsc is determined for each sentence based on the linguistic indicators). To calculate the score_scsc, our linear regression model is used. The linear regression model is stored in `model/2025_02_linear_regression_model.joblib`.

Two output files are created as csv-file  in `output/postprocess` (containing the linguisitc indicators per sentence) and `output/score_scsc` (containing the linguisitc indicators per sentence and the score scsc).

### Adaptions

Adapt `src/utils.py` to change the prompt and evaluation dataset path. By default, as evaluation dataset the annotated CrowS-Pairs sample presented in our paper is evaluated. 

### Custom experiments 

#### Other models 

To use other models, adapt `src\setup.py` and change the base_url (e.g., TogetherAI provides LLama familiy using the OpenAI Endpoint (https://docs.together.ai/docs/openai-api-compatibility)). 

#### Other prompts 

If you want to use your own prompt you can replace `prompts\Prompts_to_detect_linguistic_indicators.csv`. You might need to adjust `src\query_models.py` and `src\processing.py`, to ensure that your custom prompts are handled correctly. 

#### Other data 
To use another dataset, change the evaluation dataset path. To run the code without adaption, the dataset should be in a .csv file containing a column `input`, `del=','`. 


## Reference
If you use or discuss our survey in your work, please use the following citation:
```
@article{gorge2025detecting,
  title={Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models},
  author={G{\"o}rge, Rebekka and Mock, Michael and Allende-Cid, H{\'e}ctor},
  journal={arXiv preprint arXiv:2502.19160},
  year={2025}
}
```

## License

Copyright (c) 2021 Fraunhofer IAIS. All rights reserved.

This repository is licensed under the Apache-2.0 license. See LICENSE.txt for the full license text.

This code was developed as part of ZERTIFZIERTE KI (Fraunhofer IAIS).
