import time
import random
import pandas as pd
import processing

def process_prompts(df, prompt,prompt_id, model_name, client):
    """
    Prepare prompts by adding the prompt with each sentence that should be evaluated and call the model.

    Args:
        df (dataframe): Evaluation dataset.
        prompts (dataframe): List with specific prompts that should be evaluated 
        model_name (string): Name of the model that should be called.
        client (GenAIclient): Client to call for model evaluation. 
        client_open_AI (OpenAIclient): Client to call for model evaluation. 
        open_AI_models (list): models for which to use openAI plattform

    Returns:
        dataframe: dataframe with model evaluation for each sentence.
    """ 
 
    # Construct user prompt
    user_prompt = prompt['explanation']+prompt['instruction']+prompt['examples']+" Sentence: " 
    output_file = f'output_{model_name}_{prompt_id}'
    # Call appropriate request function

    df[output_file] = df['input'].apply(lambda sample: send_request(sample, user_prompt, client, model_name, prompt['system_role']))
     
    json_fields = df[output_file].apply(processing.extract_json_fields).apply(pd.Series)
    df = pd.concat([df, json_fields], axis=1)
    return df

def send_request(sample,user_prompt, client,model_name, system_role):
    """
    Prompt model.

    Args:
        sample (string): sentence to evaluate.
        user prompt (string): prompt.  
        client (GenAIclient): Client to call for model evaluation. 
        model_name (string): Name of the model that should be called.
        system_role (string): The role of the system
    Returns:
        output (string): model output.
    """ 
    llm_request=user_prompt+ sample
    time.sleep(random.randint(0,3))
    
    messages = [{"role": "system", "content": system_role},
                {"role": "user", "content": llm_request}]
    try:
        completion = client.chat.completions.create(
           model=model_name,
           messages=messages,
           max_tokens=400
        )
        return completion.choices[0].message.content       
    
    except Exception as e: 
          print("Failed")
          return None
    


