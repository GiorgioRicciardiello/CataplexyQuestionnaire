"""
Create a dictionary to mal the resposnes of the UK Biobank and the local dataset
"""
import pandas as pd
import numpy as np
from config import config
from typing import Dict, List, Union

nar_questions = {
    "Nar1a": {
        "description": "Buckling of the knees?",
        "responses": {
            1: "When I tell or hear a joke",
            2: "When I laugh",
            3: "When I am angry",
            4: "When I am making a quick verbal response in a playful context",
            5: "In a different situation",
            6: "I used to but not currently",
            -99: "I have never experienced this",
            -88: "Prefer not to answer"
        }
    },
    "Nar1b": {
        "description": "Sagging or dropping of your jaw?",
        "responses": {
            1: "When I tell or hear a joke",
            2: "When I laugh",
            3: "When I am angry",
            4: "When I am making a quick verbal response in a playful context",
            5: "In a different situation",
            6: "I used to but not currently",
            -99: "I have never experienced this",
            -88: "Prefer not to answer"
        }
    },
    "Nar1c": {
        "description": "Abrupt dropping of your head and/or shoulders?",
        "responses": {
            1: "When I tell or hear a joke",
            2: "When I laugh",
            3: "When I am angry",
            4: "When I am making a quick verbal response in a playful context",
            5: "In a different situation",
            6: "I used to but not currently",
            -99: "I have never experienced this",
            -88: "Prefer not to answer"
        }
    },
    "Nar1d": {
        "description": "Weakness in your arms?",
        "responses": {
            1: "When I tell or hear a joke",
            2: "When I laugh",
            3: "When I am angry",
            4: "When I am making a quick verbal response in a playful context",
            5: "In a different situation",
            6: "I used to but not currently",
            -99: "I have never experienced this",
            -88: "Prefer not to answer"
        }
    },
    "Nar1e": {
        "description": "Slurring of speech?",
        "responses": {
            1: "When I tell or hear a joke",
            2: "When I laugh",
            3: "When I am angry",
            4: "When I am making a quick verbal response in a playful context",
            5: "In a different situation",
            6: "I used to but not currently",
            -99: "I have never experienced this",
            -88: "Prefer not to answer"
        }
    },
    "Nar1f": {
        "description": "Falling to the ground, unable to move?",
        "responses": {
            1: "When I tell or hear a joke",
            2: "When I laugh",
            3: "When I am angry",
            4: "When I am making a quick verbal response in a playful context",
            5: "In a different situation",
            6: "I used to but not currently",
            -99: "I have never experienced this",
            -88: "Prefer not to answer"
        }
    }
}


def map_muscle_weakness_and_emotions(df_data: pd.DataFrame,
                                     df_ukbb: pd.DataFrame,
                                     emotion_mapper: Dict[int, str],
                                     muscle_weakness_mapper: Dict[str, str]) -> pd.DataFrame:
    """
    Standardizes `df_ukbb` to match the structure of `df_data` and applies binary mappings
    based on muscle weakness and emotions using `emotion_mapper` and `muscle_weakness_mapper`.

    Parameters:
    -----------
    df_data : pd.DataFrame
        The primary reference dataset for narcolepsy (df_data) containing columns related to emotions and muscle weakness.

    df_ukbb : pd.DataFrame
        The dataset to be modified to match the structure of `df_data`, adding columns for emotions and muscle weakness as needed.

    emotion_mapper : dict
        A dictionary mapping numerical responses to emotions, where each key represents a response that triggers a specific emotion.

    muscle_weakness_mapper : dict
        A dictionary mapping muscle weakness indicators (e.g., 'Nar1a') to specific muscle columns (e.g., 'KNEES').

    Returns:
    --------
    pd.DataFrame
        The modified `df_ukbb` DataFrame with binary indicators for muscle weaknesses and emotions,
        matching the column structure of `df_data`.
    """
    # Step 1: Ensure `df_ukbb` has the same columns as `df_data`, initializing missing ones to zero
    for col in df_data.columns:
        if col not in df_ukbb.columns:
            df_ukbb[col] = 0  # Initialize missing columns to zero for binary mapping

    # Step 2: Apply mappings for muscle weaknesses and emotions
    for weakness_key, muscle_column in muscle_weakness_mapper.items():
        if weakness_key in df_ukbb.columns:
            # Create a binary column indicating presence of muscle weakness
            df_ukbb[muscle_column] = df_ukbb[weakness_key].apply(lambda x: 1 if x > 0 else 0)

            # Map the numerical response to emotion labels, storing it in a temporary column
            df_ukbb[weakness_key + '_tmp'] = df_ukbb[weakness_key].map(emotion_mapper)

    # Step 3: Select all temporary columns ending with '_tmp' for one-hot encoding
    columns_to_dummy = [col for col in df_ukbb.columns if col.endswith('_tmp')]

    # Step 4: Apply one-hot encoding to the temporary columns, creating binary columns for each emotion
    df_dummies = pd.get_dummies(df_ukbb[columns_to_dummy], prefix=columns_to_dummy, dummy_na=False)

    # Initialize a DataFrame to store the grouped results for each emotion
    unique_values = [*emotion_mapper.values()]
    df_grouped = pd.DataFrame()

    # Step 5: For each unique emotion, group columns and aggregate across rows
    for value in unique_values:
        # Identify columns representing the specific emotion across all '_tmp' columns
        columns_with_value = [col for col in df_dummies.columns if f"_{value}" in col]

        # Sum these columns and convert to boolean (1 if present, 0 if absent), then cast to integer
        df_grouped[value] = (df_dummies[columns_with_value].sum(axis=1) > 0).astype(int)

    # Step 6: Add the grouped binary emotion columns back to `df_ukbb`
    df_ukbb[unique_values] = df_grouped

    # Step 7: Remove the temporary '_tmp' columns as they are no longer needed
    df_ukbb.drop(columns=columns_to_dummy, inplace=True)

    return df_ukbb

