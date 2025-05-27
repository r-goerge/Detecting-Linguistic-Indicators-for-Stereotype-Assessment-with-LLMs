import datetime
import os
import logging
from fire import Fire
import pandas as pd
from setup import setup_client

import processing
from query_models import process_prompts
from scsc_scoring import define_score_scsc
from utils import BASE_PATH, prompt_path, evaluation_dataset


def main(model_name='gpt-4.1', prompt_id='P_F_01', evaluate_linguistic_indicators=True, score_scsc=True):
    """
    Run stereotype assessment based on a given model and dataset. As output a score per sentence is returned.

    Args:
        model_name: str, optional
            Name of the model that should be called (by default: 'gpt-4')    
        prompt_id: str, optional
            ID of the model prompt, by default P_F_01
        evaluate_linguistic_indcators: Boolean, optional
            Boolean weather linguistic indicator should be assessed, by default yes 
        score_scsc: Boolean, optional
            Boolean weather score_scsc should be updated, by default yes 
    """ 

    client= setup_client()
    
    #get time to store results with the current time
    ct = datetime.datetime.now().replace(second=0, microsecond=0)
    logging.basicConfig(filename='stereotype_assessment.log',level=logging.INFO)

    #load dataframe with  sentences amd prompts
    prompts=pd.read_csv(os.path.join(BASE_PATH, prompt_path),sep=';')
    prompts=prompts.set_index('prompt_id')
    prompts = prompts.astype(str)
    prompts.fillna(" ", inplace=True)

    df_stereotype_asssessment= processing.load_data_frame(os.path.join(BASE_PATH, evaluation_dataset),sample=False, n=2, sep=',')

    #all evaluations in one prompt
    if evaluate_linguistic_indicators:
        prompt=prompts.loc[prompt_id]
        output_path=('output/postprocessed/')
        os.makedirs('src/'+output_path, exist_ok=True) 
            
        df_stereotype_asssessment=process_prompts(df_stereotype_asssessment, prompt, prompt_id, model_name, client)
        logging.info('Extraction of linguistic indicators by '+model_name+' done.')
        print('Extraction of linguistic indicators done.')

        #save to csv
        output_file=os.path.join(BASE_PATH,output_path+'df_stereotype_asssessment_postprocessed'+str(ct)+model_name+'.csv')
        df_stereotype_asssessment.to_csv(output_file, sep=";")

    if score_scsc:    
        #load trained models 
        df_stereotype_asssessment_scored=define_score_scsc(df_stereotype_asssessment)
        #save to csv
        output_path=('output/score_scsc/')
        os.makedirs('src/'+output_path, exist_ok=True) 
        output_file=os.path.join(BASE_PATH,output_path+'df_stereotype_asssessment_score_scsc'+str(ct)+model_name+'.csv')
        df_stereotype_asssessment_scored.to_csv(output_file, sep=";")
        logging.info('Scoring done.')
        print('Scoring done.')

if __name__ == "__main__":
    Fire(main)