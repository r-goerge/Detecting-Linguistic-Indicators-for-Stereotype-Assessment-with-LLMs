import numpy as np
import pandas as pd
from joblib import load

from processing import prepare_scoring


def define_score_scsc(df_stereotype_asssessment): 
    trained_model= load_regression_model('src/model/2025_02_14_linear_regression_model.joblib')
    loaded_encoder=load_encoder_model('src/model/2025_02_14_one_hot_encoder.joblib')

    # predict only on person-related sentences 
    df_stereotype_asssessment.loc[(df_stereotype_asssessment['output_person_related'] == 'no') | (df_stereotype_asssessment['output_situation'] == 'not-applicable') | (df_stereotype_asssessment['output_situation'] == 'other'), 'score_scsc'] = 0.0
        
    #constraints (will be done as prefiltering!)
    df_stereotype_asssessment_to_be_scored = df_stereotype_asssessment[(df_stereotype_asssessment['output_person_related'] != 'no') &  (df_stereotype_asssessment['output_situation'] != 'not-applicable') &  (df_stereotype_asssessment['output_situation'] != 'other')]
    
    if len(df_stereotype_asssessment_to_be_scored)!=0:
        df_stereotype_asssessment_to_be_scored=prepare_scoring(df_stereotype_asssessment_to_be_scored)

        df_predicted_stereotypes_x=pd.DataFrame()
        df_predicted_stereotypes_x[['person_related','generalization_category_label','connotation','gram_form','generalization_situation','explanation']] = df_stereotype_asssessment_to_be_scored[['output_person_related','output_generalization_category_label','output_connotation','output_gram_form', 'output_generalization_situation', 'output_explanation']]

        # Transform the features
        X_transformed_features = loaded_encoder.transform(df_predicted_stereotypes_x)

        # Predictions on the original dataset 
        y_pred = trained_model.predict(X_transformed_features)

        df_stereotype_asssessment.loc[(df_stereotype_asssessment['output_person_related'] != 'no') & (df_stereotype_asssessment['output_situation'] != 'not-applicable') & (df_stereotype_asssessment['output_situation'] != 'other'), 'score_scsc'] = y_pred

    return df_stereotype_asssessment


def load_regression_model(linear_regression_model_path):
    trained_model= load(linear_regression_model_path)
    
    return trained_model

def load_encoder_model(encoder_model_path):
    encoder_model= load(encoder_model_path)
    
    return encoder_model




            
    
