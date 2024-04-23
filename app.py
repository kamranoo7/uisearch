from flask import Flask, render_template, request, jsonify
import openai 
import numpy as np
import json
app = Flask(__name__)

items = {
    "system.": ["nn1.jpeg", "Please search what is bothering you."],
}

import ast
import pandas as pd

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')

# Use the API key with the OpenAI client
import openai
lm_client = openai.OpenAI(api_key=api_key)

custom_functions_paitent = [
    {
        'name': 'process_paitent_query',
        'description': 'Function to be used in order to return the python list related to the query made',
        'parameters': {
            'type': 'object',
            'properties': {
                'tags_list': {
                    'type': 'array',
                    'description': 'List to be returned',
                    'items': {'type': 'integer'}
                },
                "reason": {
                    "type": "string",
                    "description": "This should be the reason how the values were chosen.",
                },
            },
            "required": ["tags_list", "reason"]
        }
    }
]

system_message_paitent = """

You will be given a search query from a paitent.

based on the query, you must return a list of 30 elements. where each element contains a value ranging from 0-1, indicating how relevant that index is to the paitent query.

The refrenece list is:

[
    "Cardiology",
    "Dermatology",
    "Emergency Medicine",
    "Endocrinology",
    "Gastroenterology",
    "Hematology",
    "Infectious Disease",
    "Nephrology",
    "Neurology",
    "Obstetrics & Gynecology",
    "Oncology",
    "Ophthalmology",
    "Orthopedics",
    "Otorhinolaryngology (ENT)",
    "Pediatrics",
    "Psychiatry",
    "Pulmonology",
    "Rheumatology",
    "Surgery",
    "Urology",
    "Anesthesiology",
    "Pathology",
    "Radiology",
    "Geriatrics",
    "Sports Medicine",
    "Physical Medicine and Rehabilitation (PM&R)",
    "Allergy and Immunology",
    "Neonatology",
    "Plastic Surgery",
    "Preventive Medicine"
]
The numeric value in each index signifies how relevant each index is to the query.
Things to return: A list of size 30, and the reason adressed to the paitent.

""" 

def generate_chat_response(user_message, system_message, func_json):

    msg = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg, max_tokens=4000, temperature=0.0,  functions=func_json, function_call='auto')
    reply = response.choices[0].message.content
    tags = response.choices[0].message.content

    try:
        print(response.choices[0].message.function_call.arguments)
        if reply is None:
            print("----")
            tags = json.loads(response.choices[0].message.function_call.arguments)["tags_list"]
            reason = json.loads(response.choices[0].message.function_call.arguments)["reason"]
            print(type(tags))
        else:
            tags = [0]*30
            reason = "Error during processing."
    except Exception as e:
        print(e)
        tags = [0]*30
        reason = "Error during processing."
    return tags, reason


def extract_columns_from_excel(file_path, col1, col2, col3):
    df = pd.read_excel(file_path)
    if col1 not in df.columns or col2 not in df.columns or col3 not in df.columns:
        raise ValueError("One or more specified columns do not exist in the Excel file.")
    
    list1 = df[col1].tolist()
    list2 = df[col2].tolist()
    list3 = df[col3].tolist()
    
    return list1, list2, list3

doctor_names,doctor_descriptions,hits = extract_columns_from_excel('sheet.xlsx', 'name_en', 'treatment', 'hits')

import random

saved_docs = np.load('doc_list_new.npy')

column_31 = saved_docs[:, 30]

# Find the minimum and maximum of the 31st column
min_val = np.min(column_31)
max_val = np.max(column_31)

# Normalize the 31st column
normalized_column_31 = (column_31 - min_val) / (max_val - min_val)

# Replace the 31st column in the original matrix with the normalized values
saved_docs[:, 30] = normalized_column_31


def random_true_60():
    return random.random() < 0.6

def return_scores(docs, lst):
    lst = np.array(lst).T
    lst = lst[:, np.newaxis]
    scores = np.dot(docs, lst)  
    return scores

def sort_indices_by_values_desc_np(array):
    if array is None:
        raise ValueError("Input array is None")
    sorted_indices = np.argsort(-array.ravel())
    return sorted_indices

@app.route('/')
def home():
    return render_template('index.html', items=items)

items1 = {
    "Tea": ["tea.jpeg", "An aromatic beverage commonly prepared by pouring hot or boiling water over cured leaves."]
}

items1 = {}

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '').lower()
    lst, reason = generate_chat_response(query, system_message_paitent, custom_functions_paitent)
    prioritize = random_true_60()
    if prioritize: lst.append(1)
    else: lst.append(-1)
    
    print(lst)
    scores = return_scores(saved_docs, lst)
    argsscores = sort_indices_by_values_desc_np(scores)

    items1 = [{'name': 'system.', 'image':'nn1.jpeg', 'description':reason}]
    for idx in argsscores:
        items1.append({'name': doctor_names[idx], 'image': 'doc.png', 'description': doctor_descriptions[idx]})
    

    return jsonify(items1)

if __name__ == '__main__':
    app.run(debug=True)
