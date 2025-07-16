"""
Stanford Center for Narcolpesy
Sleep Inventory
Digital version of the questionnaire docs/Old Questionnaire - All Pages.pdf
"""
import numpy as np
ethnic_group = {
    0: 'black',
    1: 'caucasian',
    2: 'latino',
    3: 'asian',
    4: 'pacific islander',
    5: 'american indian',
    6: 'other',
    7: 'unkown'
}
binary = {
    0: 'No',
    1: 'Yes',
    9: np.nan
}

binary_not_sure = {
    0: 'No',
    1: 'Yes',
    2: 'Not Sure',
    9: np.nan
}
doze = {
    0: 'Would never doze',
    1: 'Slight chance of dozing',
    2: 'Moderate chance of dozing',
    3: 'High chance of dozing',
    9: np.nan
}

likert_frequency_scale = {
    0: 'never',
    1: 'rarely',
    2: 'often',
    3: 'usually',
    4: 'always'
}

likert_increase = {
    3: 'increased',
    2: 'decreased',
    0: 'no change',
    1: 'not sure',
}

likert_within = {
    4: "within the past 24 hours",
    3: "within the past week",
    2: "within the past month",
    1: "within the past year",
    0: "more than a year ago"
}

likert_episodes = {
    4: 'always',
    3: 'sometines',
    2: 'rarely',
    0: 'never',
    1: 'not sure'
}

episodes_muscle_weakness = {
    1: 'both knees/legs',
    2: 'either one or both knees/legs',
    3: 'only one knee/leg'
}

time_ranges = {
    1: "5 seconds - 30 seconds",
    2: "30 seconds - 2 minutes",
    3: "2 minutes - 10 minutes",
    4: "More than 10 minutes"
}

time_periods = {
    1: "within the past 24 hours",
    2: "within the past week",
    3: "within the past month",
    4: "within the past year",
    5: "more than a year ago"
}

frequency = {
    1: "Once or more per day",
    2: "Several times per week",
    3: "Once per week",
    4: "Once per month",
    5: "Once per year or less"
}

other_person = {
    1: 'family',
    2: 'friend',
    3: 'acquaintance',
    4: 'physician',
    5: 'stranger'
}

# muscle_weakness_experiences (questions 54, ... sparse format)
mw_experiences = {
    1: "Do you currently experience, or have you ever experienced episodes of muscle weakness in the legs and/or buckling of your knees when you...",
    2: "Have you ever experienced a sagging or dropping of your jaw when you...",
    3: "Have you ever experienced an abrupt dropping of your head and/or shoulders when you...",
    4: "Have you abruptly dropped objects from your hand or felt weakness in your arm when you...",
    5: "Has your speech ever become slurred when you...",
    6: "Have you ever fallen to the ground and found yourself unable to move (paralyzed) when you..."
}
responses_dict = {
    1: 'Where were you born',
    2: {'Major Ethnic Group ID': ethnic_group},
    3: 'Where Mother  Born',
    4: {'Mother Major Ethnic Group ID': ethnic_group},
    5: 'Mother"s father born',
    6: {'Mother"s father Major Ethnic Group ID': ethnic_group},
    7: 'where father"s mother born',
    8: {'Father"s mother Major Ethnic Group ID': ethnic_group},
    9: 'where father"s father born',
    10: {'father"s father Major Ethnic Group ID': ethnic_group},
    11: {'Do you sleep well at night': binary},
    12: {'Do you have difficulty staying awake during the day': binary},
    # Epworth
    13: {"Sitting and reading": doze},
    14: {"Watching TV": doze},
    15: {"Sitting inactive in a public place (e.g., a theater or meeting)": doze},
    16: {"As a passenger in a car for an hour without a break": doze},
    17: {"Lying down to rest in the afternoon when circumstances permit": doze},
    18: {"Sitting and talking to someone": doze},
    19: {"Sitting quietly after a lunch without alcohol": doze},
    20: {"In a car, while stopped for a few minutes in traffic": doze},
    # Epworth - end
    21: "At what time do you usually get into bed at night?",
    22: "At what time do you usually get out of bed in the morning?",
    23: "How long after going to bed do you usually turn out the lights?",
    24: "How long does it usually take you to fall asleep after the lights are off?",
    25: {"Have you ever had difficulty falling asleep at night?": binary},
    26: {"Do you currently have difficulty falling asleep at night?": binary},
    27: {"How many times do you wake up during a typical night's sleep? If 0 times, please go to question 29.": binary},
    28: {"How long does your longest nighttime awakening typically last?": binary},
    29: {"Do you usually feel refreshed after a typical night of sleep?": binary},
    30: {"Do you experience muscle twitches during your sleep?": binary},
    31: {"Do your legs kick during your sleep?": binary},
    32: {"Do you sweat excessively during your sleep?": binary},
    33: {"Do you sleep restlessly?": binary},
    34: {"Do you snore? If YES, please go to question 36.": binary},
    35: {"Does your bedpartner say that you snore? If NO, please go to question 37.": binary},
    36: {"Do you snore loudly or irregularly?": binary},
    37: {"Do you or your partner notice that you sometimes stop breathing during your sleep?": binary},
    38: {"Do you nap during the day?": binary},
    39: "If yes, how many times per week do you take a nap?",
    40: "How long does a typical nap last?",
    41: {"Do you usually feel refreshed after napping?": binary},
    42: {"How often do you dream during your naps?": likert_frequency_scale},
    43: {"Do you believe that you are sleepier than other individuals your age?": binary},
    44: "At what age did you begin to believe or become aware that you were sleepier than other individuals your age?",
    45: {"Since the age you indicated in question 44 above, what would you say has happened to the severity of your sleepiness?": likert_increase},
    46: "At what age was your sleepiness the most severe ever?",
    47: {'When was the last time you were excessively sleepy?': likert_within},
    48: {'Has physician ever told you were excessively sleepy or sleep probelms': binary},
    49: 'OpenQuestion',
    50: {'Have you ever been excessively sleepy': binary},
    51: 'OpenQuestion',
    52: 'medications',
    53: 'OpenQuestion',
    54: {"laugh": binary_not_sure},
    55: {"are angry": binary_not_sure},
    56: {"are excited": binary_not_sure},
    57: {"are surprised": binary_not_sure},
    58: {"remember a happy moment": binary_not_sure},
    59: {"remember an emotional event": binary_not_sure},
    60: {
        "are required to make a quick verbal response in a playful or funny context (e.g., a witty repartee)": binary_not_sure},
    61: {"are embarrassed": binary_not_sure},
    62: {"discipline children": binary_not_sure},
    63: {"During sexual intercourse": binary_not_sure},
    64: {"During athletic activities": binary_not_sure},
    65: {"After athletic activities": binary_not_sure},
    66: {"are elated": binary_not_sure},
    67: {"are stressed": binary_not_sure},
    68: {"are startled": binary_not_sure},
    69: {"are tense": binary_not_sure},
    70: {"While you are playing an exciting game": binary_not_sure},
    71: {"have a romantic thought or moment": binary_not_sure},
    72: {"tell or hear a joke":binary_not_sure },
    73: {"are moved by something emotional":binary_not_sure},
    74: "Other (please describe)",
    75: 'OpenQuestion',
    76: {'Statement best describes episodes of muscle weakness': episodes_muscle_weakness},
    77: {'Experienced sagging or dropping of jaw': binary},
    78 : {'Experienced abrupt dropping of head and/or shoulder': binary},
    79: {'Abruptly dropped objects from your hand or felt weakness in your arms': binary},
    80: {'Statement best describes episodes of muscle weakness in your hands or arms': episodes_muscle_weakness},
    81 : {'Has speech ever become slurred during situations': binary},
    82: {'Fallen to the fround and unable to move': binary},
    83: '',
    84: {'How long does muscle weakness last': time_ranges},
    85: {'How frequently experience one of these episodes of muscle weakness': frequency},
    86: {"During the episodes of muscle weakness, can you hear?": likert_episodes},
    87: {"During the episodes of muscle weakness, can you see?": likert_episodes},
    88: {"During the episodes of muscle weakness, is your vision blurred?": likert_episodes},
    89: {"During the episodes of muscle weakness, do you dream?": likert_episodes},
    90: {"During the episodes of muscle weakness, do you fall asleep?": likert_episodes},
    91: {"During the episodes of muscle weakness, do you lose control of your bladder (lose urine)?": likert_episodes},
    92: {"During the episodes of muscle weakness, do you lose control of your bowels?": likert_episodes},
    93: {"During the episodes of muscle weakness, do you faint?": likert_episodes},
    94: {"During the episodes of muscle weakness, do you have time to sit down or hold onto something to break a fall?": likert_episodes},
    95: {"How old were you the first time you experienced one of these episodes of muscle weakness?": likert_episodes},
    96: {'Did the episodres of muscles wekaness peak at a certain age': binary},
    97: '',
    98: {'How long ago was last episode of muscle weakness': time_periods},
    99: '',
    100: 'OpenQuestion',
    101: {'Injured during episode': binary},
    102: {'Observe another person': other_person},
    103: 'medications'
}

# rename the keys in the dictionary with more meaningful name
# the key is the column name (given as a number in the raw dataset) and the value is the label of the column
key_mapping = {
    1: 'birth_place',
    2: 'ethnic_group_id',
    3: 'mother_birth_place',
    4: 'mother_ethnic_group_id',
    5: 'mother_father_birth_place',
    6: 'mother_father_ethnic_group_id',
    7: 'father_mother_birth_place',
    8: 'father_mother_ethnic_group_id',
    9: 'father_father_birth_place',
    10: 'father_father_ethnic_group_id',
    11: 'sleep_quality',
    12: 'epworth_daytime_sleepiness',
    13: 'epworth_reading',
    14: 'epworth_tv',
    15: 'epworth_public_place',
    16: 'epworth_car_passenger',
    17: 'epworth_afternoon_rest',
    18: 'epworth_talking',
    19: 'epworth_lunch_rest',
    20: 'epworth_traffic',
    21: 'bedtime',
    22: 'wake_time',
    23: 'lights_out_time',
    24: 'fall_asleep_time',
    25: 'difficulty_falling_asleep_ever',
    26: 'current_difficulty_falling_asleep',
    27: 'night_wake_count',
    28: 'longest_awake_duration',
    29: 'feel_refreshed_after_sleep',
    30: 'muscle_twitches_during_sleep',
    31: 'legs_kick_during_sleep',
    32: 'excessive_sweating_during_sleep',
    33: 'restless_sleep',
    34: 'snoring',
    35: 'partner_says_snoring',
    36: 'snoring_loud_irregular',
    37: 'stop_breathing_during_sleep',
    38: 'daytime_naps',
    39: 'nap_frequency',
    40: 'nap_duration',
    41: 'feel_refreshed_after_nap',
    42: 'dream_frequency_during_naps',
    43: 'sleepier_than_peers',
    44: 'age_aware_sleepiness',
    45: 'sleepiness_severity_since_age',
    46: 'most_severe_sleepiness_age',
    47: 'last_excessive_sleepiness',
    48: 'physician_diagnosed_sleepiness',
    49: 'open_question_1',
    50: 'ever_excessively_sleepy',
    51: 'open_question_2',
    52: 'medications_1',
    53: 'open_question_3',
    54: 'laughing-cataplexy',
    55: 'anger-cataplexy',
    56: 'excitement-cataplexy',
    57: 'surprise-cataplexy',
    58: 'happy_memory-cataplexy',
    59: 'emotional_memory-cataplexy',
    60: 'quick_response-cataplexy',
    61: 'embarrassment-cataplexy',
    62: 'disciplining_children-cataplexy',
    63: 'sexual_intercourse-cataplexy',
    64: 'athletic_activities-cataplexy',
    65: 'post_athletic_activities-cataplexy',
    66: 'elation-cataplexy',
    67: 'stress-cataplexy',
    68: 'startle-cataplexy',
    69: 'tension-cataplexy',
    70: 'exciting_game-cataplexy',
    71: 'romantic_moment-cataplexy',
    72: 'joke-cataplexy',
    73: 'emotional_moment-cataplexy',
    74: 'other_description',
    75: 'open_question_4',
    76: 'muscle_weakness_legs_knees',
    77: 'jaw_sagging',
    78: 'head_shoulder_dropping',
    79: 'hand_arm_weakness',
    80: 'hand_arm_weakness_description',
    81: 'slurred_speech',
    82: 'fell_to_ground',
    83: 'OpenQuestion_83',
    84: 'muscle_weakness_duration',
    85: 'muscle_weakness_frequency',
    86: 'muscle_weakness_hearing',
    87: 'muscle_weakness_vision',
    88: 'muscle_weakness_blurred_vision',
    89: 'muscle_weakness_dreaming',
    90: 'muscle_weakness_sleeping',
    91: 'muscle_weakness_bladder_control',
    92: 'muscle_weakness_bowel_control',
    93: 'muscle_weakness_fainting',
    94: 'muscle_weakness_fall_prevention',
    95: 'first_muscle_weakness_age',
    96: 'muscle_weakness_peak_age',
    97: 'OpenQuestion_97',
    98: 'last_muscle_weakness_episode',
    99: 'OpenQuestion_99',
    100: 'open_question_5',
    101: 'injured_during_episode',
    102: 'observed_another_person',
    103: 'medications_2'

}

# %% Make a pandas object
# import pandas as pd
# df_responses = {}
# for num, value in responses_dict.items():
#     if isinstance(value, dict):
#         df_responses[num] = [*value.keys()][0]
#     else:
#         df_responses[num] = value
#
# df_responses = pd.DataFrame(list(df_responses.items()), columns=['id', 'question'])
# df_key_mapping = pd.DataFrame(list(key_mapping.items()), columns=['id', 'alias'])
#
# df_questionnaire = pd.merge(on='id', left=df_responses, right=df_key_mapping, how='left')
# df_questionnaire.to_csv('questionnaire.csv', index=False)


sss_questionnaire = {
            'sss1': {
                'label': 'Over the past two weeks, how likely is it that you would unintentionally fall asleep or doze off?',
                'levels': []},
            'sss2': {
                'label': 'Sitting at a desk/table working on a computer or tablet',
                'levels': []},
            'sss3': {
                'label': 'Talking to someone on the phone',
                'levels': []},
            'sss4': {
                'label': 'In a meeting with several people',
                'levels': []},
            'sss5': {
                'label': 'Listening to someone talking in a class, lecture or at church',
                'levels': []},
            'sss6': {
                'label': 'Playing cards or a board game with others',
                'levels': []},
            'sss7': {
                'label': 'Driving a car',
                'levels': []},
            'sss8': {
                'label': 'Playing a videogame',
                'levels': []},
            'sss9': {
                'label': 'Lying down to rest',
                'levels': []},
            'sss10': {
                'label': 'Traveling as a passenger in a bus, train or car for more than 30 minutes',
                'levels': []},
            'sss11': {
                'label': 'Watching a film at home or the cinema',
                'levels': []},
            'sss_score': {
                'label': 'SSS Score ',
                'levels': []}
        }
#
# df_sss = {}
# for key, value in sss_questionnaire.items():
#     df_sss[key] = value.get('label')
#
# df_sss = pd.DataFrame(list(df_sss.items()), columns=['id', 'question'])
# df_sss.to_csv('sss_questionnaire.csv', index=False)


variable_definitions = {
    "AFTATHLETIC": "After athletic activities",
    "ANGER": "Angry",
    "DISCIPLINE": "Discipline children",
    "DURATHLETIC": "During athletic activities",
    "ELATED": "Elated",
    "EMBARRAS": "Embarrassed",
    "EMOTIONAL": "Remember an emotional moment",
    "EXCITED": "Excited",
    "HAPPY": "Remember a happy moment",
    "JOKING": "Hear or tell a joke",
    "LAUGHING": "Laugh",
    "MOVEDEMOT": "Moved by something emotional",
    "PLAYGAME": "Playing an exciting game",
    "QUICKVERBAL": "Quick response cataplexy",
    "ROMANTIC": "Have a romantic thought or moment",
    "SEX": "During sexual intercourse",
    "SPEECH": "Muscle weakness, speech becomes slurred",
    "STARTLED": "Startled",
    "STRESSED": "Stressed",
    "SURPRISED": "Surprised",
    "TENSE": "Tense",
    "sex": "Gender (Male)",
    "ONSET": "Muscle weakness age onset",
    "DURATION": "Muscle weakness duration",
    "SP": "Sleep paralysis",
    "SPSEVER": "Sleep paralysis severity",
    "SPONSET": "Sleep paralysis age onset",
    "SLEEPIONSET": "Age sleep complaints",
    "SE": "Sleep latency",
    "MEDCATA": "Cataplexy medication",
    "REMLAT": "REM latency",
    'DISNOCSLEEP': 'Disturbed nocturnal sleep',
    'DQB10602': 'HLA-DQB1*06:02',

    "HAND": "Muscle weakness in hand and arms",
    "HEAD": "Muscle weakness head and shoulder dropping",
    "JAW": "Muscle weakness jaw sagging",
    "KNEES": "Muscle weakness legs and knees",


    'Race': 'Race',
    'AGE': 'Age',
    'BMI': 'BMI',

    'ESS': 'ESS Score',

    'HALLUC': 'Hallucinations',
    'HHONSET':'Hallucinations Age Onset',

    'NAPS': 'Naps',
    'MSLT':' MSLT',
    'MSLTAGE':'MSLT Age',
    'SOREMP': 'Sleep-Onset Rapid Eye Movement Period',

}
