'''
This script loads the speed dating dataset from PostgreSQL into AWS EC2.
Then it creates 2 separate dataframes with feature selection and engineering to
optimize for modeling.
'''
from sqlalchemy import create_engine
import pandas as pd

def get_dataframes_from_aws():
    '''
    This script connects to PostgreSQL from an AWS EC2 instanace and returns 2
    datafames that have null values removed. You will need to add change and add your
    specific EC2 IP address to the script in the cnx variable below.
    '''
    cnx = create_engine('postgresql://ubuntu@' + 'ec2_ip' + ':5432/loves')

    date_data = pd.read_sql_query('''select match, attr, sinc, intel, fun, amb, shar, like_1, prob,
                                    met, attr_o, sinc_o, intel_o, fun_o, amb_o, shar_o, like_o, prob_o, met_o
                                    from speeddating6
                                    where shar_o is not null
                                    and shar is not null
                                    and amb_o is not null
                                    and amb is not null
                                    and met_o is not null
                                    and met is not null
                                    and fun is not null
                                    and prob is not null
                                    and prob_o is not null
                                    and intel is not null
                                    and fun_o is not null
                                    and attr is not null
                                    and intel_o is not null
                                    and sinc is not null
                                    and attr_o is not null
                                    and sinc_o is not null
                                    and like_o is not null
                                    and like_1 is not null ''',
                                  cnx)

    participant_data = pd.read_sql_query('''select avg(date_3) as date_after, iid as participant,
                                                avg(date) as date_freq, avg(go_out) as go_out_freq,
                                                avg(imprace) as race_importance, avg(imprelig) as religious_importance,
                                                avg(sports) as sports, avg(tvsports) as tvsports, avg(exercise) as exercise,
                                                avg(dining) as dining, avg(museums) as museums, avg(art) as art,
                                                avg(hiking) as hiking, avg(gaming) as gaming, avg(clubbing) as clubbing,
                                                avg(tv) as tv, avg(theater) as theater, avg(concerts) as concerts,
                                                avg(music) as music, avg(yoga) as yoga, avg(attr3_1) as self_attractive,
                                                avg(sinc3_1) as self_sincere, avg(intel3_1) as self_intelligent,
                                                avg(fun3_1) as self_fun, avg(reading) as reading
                                                from speeddating6 s
                                                where date_3 is not null
                                                and exercise is not null
                                                and attr3_1 is not null
                                                group by iid
                                                order by avg(match) desc;''',
                                         cnx)

    return date_data, participant_data

def date_data_feature_engineering(date_data):
    '''
    Returns a dataframe with additional engineered features and drops features that
    do not have an impact on the performance of the final model of predicting a match in
    a round of speed dating.
    '''

    # Feature engineering
    date_data['diff_attr'] = date_data['attr'] - date_data['attr_o']
    date_data['diff_sinc'] = date_data['sinc'] - date_data['sinc_o']
    date_data['diff_intel'] = date_data['intel'] - date_data['intel_o']
    date_data['diff_fun'] = date_data['fun'] - date_data['fun_o']
    date_data['diff_amb'] = date_data['amb'] - date_data['amb_o']
    date_data['diff_shar'] = date_data['shar'] - date_data['shar_o']
    date_data['diff_like'] = date_data['like_1'] - date_data['like_o']
    date_data['diff_prob'] = date_data['prob'] - date_data['prob_o']
    date_data['diff_met'] = date_data['met'] - date_data['met_o']

    date_data['abs_val_sum_differences'] = (date_data['diff_attr'].abs()
                                            + date_data['diff_sinc'].abs()
                                            + date_data['diff_intel'].abs()
                                            + date_data['diff_shar'].abs()
                                            + date_data['diff_fun'].abs()
                                            + date_data['diff_amb'].abs()
                                            + date_data['diff_like'].abs()
                                            + date_data['diff_prob'].abs()
                                            + date_data['diff_met'].abs())

    date_data['attr_times_like'] = date_data['attr'] * date_data['like_1']
    date_data['partner_attr_times_like'] = date_data['attr_o'] * date_data['like_o']

    # Drop unnecessary features
    date_data.drop(columns=['attr_o', 'sinc', 'sinc_o', 'intel', 'intel_o', 'fun_o',
                            'amb', 'amb_o', 'shar_o', 'like_o', 'prob', 'prob_o', 'met',
                            'met_o'], inplace=True)

    return date_data

def participant_data_feature_engineering(participant_data):
    '''
    Returns a dataframe with additional engineered features and drops features that
    do not have an impact on the performance of the final model of predicting a date after
    a speed dating event.
    '''

    # Feature engineering
    participant_data['race_squared'] = participant_data['race_importance'] **2
    participant_data['confidence'] = (participant_data['self_intelligent']
                                      + participant_data['self_fun']
                                      + participant_data['self_attractive'])
    participant_data['sociableness'] = (participant_data['date_freq']
                                        + participant_data['go_out_freq'])
    participant_data['tv_oriented_activities'] = (participant_data['tv']
                                                  + participant_data['tvsports']
                                                  + participant_data['gaming'])
    participant_data['culture_activities'] = (participant_data['museums']
                                              + participant_data['art']
                                              + participant_data['reading']
                                              + participant_data['theater']
                                              + participant_data['music'])
    participant_data['social_outings'] = (participant_data['clubbing'] + participant_data['dining']
                                          + participant_data['concerts'])
    participant_data['active_activities'] = (participant_data['sports']
                                             + participant_data['exercise']
                                             + participant_data['hiking']
                                             + participant_data['yoga'])

    # Drop unnecessary features
    participant_data.drop(columns=['date_freq', 'go_out_freq', 'sports', 'tvsports',
                                   'exercise', 'clubbing', 'tv', 'self_attractive',
                                   'dining', 'museums', 'art', 'hiking', 'gaming',
                                   'theater', 'concerts', 'music', 'yoga', 'self_fun'],
                          inplace=True)

    return participant_data

def main():
    '''
    Loads in the date data and participant dataframes, performs data cleaning and
    feature engineering and saves the datasets as csv files.
    '''

    # Call internal functions to this script
    date_data, participant_data = get_dataframes_from_aws()
    date_data = date_data_feature_engineering(date_data)
    participant_data = participant_data_feature_engineering(participant_data)

    # Save dataframes as csv files
    date_data.to_csv(r'date_data.csv', index=False)
    participant_data.to_csv(r'participant_data.csv', index=False)

main()
