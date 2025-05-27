import os
import pandas as pd
import json

def load_data_frame(path_to_data, sample=False, n=50, sep=','):
    """
    Load the dataframe.

    Args:
        sample (boolean): Whether to use a sample of the dataset.
        n (int): size of the sample

    Returns:
        dataframe: loaded dataframe.
    """ 
    df_preproccesed=pd.read_csv(os.path.join(path_to_data),sep=sep)
    if sample:
        df_preproccesed=df_preproccesed.sample(n)

    return df_preproccesed

def change_delimiter(input_file_name, output_file_name):
    """
    Change delimiter of csv file and put input into qutes

    Args:
        input_path (str): Input File path to csv file 
        Output_path (str): Output File path to csv file

    """ 
    current_delimiter=','
    new_delimiter=';'
    df= pd.read_csv(input_file_name, sep=current_delimiter)
    df['input'] = '"' + df['input'].astype(str) + '"'  # Enclose values in quotes

    # Save the modified DataFrame to a new CSV file with the new delimiter
    df.to_csv(output_file_name, sep=new_delimiter, index=False)



def extract_json_fields(json_str):
    """
    Extract json fields of full output.

    Args:
        json_str (str): Output of the model.

    Returns:
        Dict[str, Any]: Extracted fields.
    """ 
    try:
        json_data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        result = {
        'output_person_related':  None,
        'output_label': None,
        'output_target': None,
        'output_connotation': None,
        'output_gram_form': None,
        'output_ling_form': None,
        'output_information': None,
        'output_situation': None,
        'output_generalization': None,
        'output_explanation': None,
        'output_signal_word': None
    }
        return result
    

    
    result = {
        'output_person_related': json_data.get('has_category_label',{}),
        'output_label': 'not-applicable',
        'output_target': 'not-applicable',
        'output_connotation': 'not-applicable',
        'output_gram_form': 'not-applicable',
        'output_ling_form': 'not-applicable',
        'output_information': 'not-applicable',
        'output_situation': 'not-applicable',
        'output_generalization': 'not-applicable',
        'output_explanation': 'not-applicable',
        'output_signal_word': 'not-applicable'
    }

  
    
    if json_data.get('has_category_label') == "yes":
        result.update({
            'output_label': json_data.get('full_label', 'not-applicable'),
            'output_target': json_data.get('target_type', 'not-applicable'),
            'output_connotation': json_data.get('connotation', 'not-applicable'),
            'output_gram_form': json_data.get('grammatical_form', 'not-applicable'),
            'output_ling_form': json_data.get('linguistic_form', 'not-applicable'),
            'output_information': json_data.get('information', 'not-applicable'),
            'output_situation': json_data.get('situation', 'not-applicable'),
            'output_generalization': json_data.get('generalization', 'not-applicable'),
            'output_explanation': json_data.get('explanation', 'not-applicable'),
            'output_signal_word': json_data.get('signal_word', 'not-applicable')
        })

    return result

# Function to parse JSON and extract relevant fields
def extract_json_fields_label(json_str):
    """
    Extract json fields of label part.

    Args:
        json_str (str): Output of the model.

    Returns:
        Dict[str, Any]: Extracted fields.
    """ 
    try:
        json_data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        result = {
        'output_person_related':  None,
        'output_label': None,
        'output_target': None,
        'output_connotation': None,
        'output_gram_form': None,
        'output_ling_form': None
    }
        return result
    
    
    result = {
        'output_person_related': json_data.get('has_category_label',{}),
        'output_label': 'not-applicable',
        'output_target': 'not-applicable',
        'output_connotation': 'not-applicable',
        'output_gram_form': 'not-applicable',
        'output_ling_form': 'not-applicable',
    }
    
    if json_data.get('has_category_label') == "yes":
        result.update({
            'output_label': json_data.get('full_label', 'not-applicable'),
            'output_target': json_data.get('target_type', 'not-applicable'),
            'output_connotation': json_data.get('connotation', 'not-applicable'),
            'output_gram_form': json_data.get('grammatical_form', 'not-applicable'),
            'output_ling_form': json_data.get('linguistic_form', 'not-applicable'),
        })
    
    return result

def extract_json_fields_content(json_str):
    """
    Extract json fields of content part.

    Args:
        json_str (str): Output of the model.

    Returns:
        Dict[str, Any]: Extracted fields.
    """ 
    try:
       json_data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
       result = {
        'output_information': None,
        'output_situation': None,
        'output_generalization': None,
        'output_explanation': None,
        'output_signal_word': None
    }
       return result
    
    result = {
        'output_information': 'not-applicable',
        'output_situation': 'not-applicable',
        'output_generalization': 'not-applicable',
        'output_explanation': 'not-applicable',
        'output_signal_word': 'not-applicable'
    }
    # Update result with relevant fields if they exist in the JSON data
    if 'information' in json_data:
        result['output_information'] = json_data['information']
    
    if 'situation' in json_data:
        result['output_situation'] = json_data['situation']
    
    if 'generalization' in json_data:
        result['output_generalization'] = json_data['generalization']
    
    if 'explanation' in json_data:
        result['output_explanation'] = json_data['explanation']
    
    if 'signal_word' in json_data:
        result['output_signal_word'] = json_data['signal_word']

    return result


def prepare_scoring(df_output):
    """
    Postprocess dataframe for scoring.

    Args:
        dataframe: output dataframe of the model.

    Returns:
        dataframe: postprocessed dataframe.
    """ 
    df_output = df_output[~df_output.isin(['fail']).any(axis=1)].copy()
    #postprocess
    df_output.loc[:,'output_connotation']= df_output['output_connotation'].apply(lambda x: 'neutral' if x=='positive' else x)
    # add feature that are combined to one categorical variable 
    df_output.loc[:,'output_generalization_category_label'] = df_output['output_ling_form'].astype(str) + '_' + df_output['output_target'].astype(str)
    df_output.loc[:,'output_generalization_situation'] = df_output['output_situation'].astype(str) + '_' + df_output['output_generalization'].astype(str)
    
    
    return df_output


