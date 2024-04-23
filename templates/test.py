import openai
import json

key = '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'
lm_client = openai.OpenAI(api_key=key)

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
                    'items': {
                        'type': 'integer'
                    }
                },
            },
            "required": ["tags_list"]
        }
    }
]


custom_functions_doctor = [
    {
        'name': 'process_doctor_description',
        'description': 'Function to be used in order to return the python list',
        'parameters': {
            'type': 'object',
            'properties': {
                'tags_list': {
                    'type': 'array',
                    'description': 'List to be returned',
                    'items': {
                        'type': 'integer'
                    }
                },
            },
            "required": ["tags_list"]
        }
    }
]




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
    # reply = None
    if reply is None:
       print("----")
       tags = json.loads(response.choices[0].message.function_call.arguments)["tags_list"]
       print(type(tags))
    else:
        tags = tags.strip()
        tags = ast.literal_eval(tags)
    return tags

categories = ['Orthopedics', 'Gastroenterology', 'Cardiology', 'Neurology', 'Dermatology', 'Psychiatry', 'Pediatrics', 'Obstetrics and Gynecology (OB/GYN)', 'Oncology', 'Endocrinology']

system_message_doctor = """

You will be given a description of a doctor. I want to create a vector representing their skills. Use this list: 

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

Based on the description return a list of 30 elements, where each index referes to the item in the list above. mark each position with either a 1 or a 0, depending on if the item is relevant to the decsription.

Based on it, you must return a list where each index is either marked with a 1 or a 0 integer. return only the list and nothing else.

"""

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

"""

import numpy as np

doctor_descriptions = [
    "Specializes in Orthopedics, providing expert care in musculoskeletal system disorders, ensuring improved mobility and pain relief.",
    "An authority in Gastroenterology, adept at diagnosing and treating a wide range of digestive health issues with advanced techniques.",
    "Renowned in Cardiology for innovative heart health management and treatment strategies, enhancing patient cardiovascular wellness.",
    "Expert in Neurology, focusing on cutting-edge treatments for brain and nervous system disorders, improving patient quality of life.",
    "A Dermatology specialist with a knack for addressing complex skin conditions using the latest in dermatological therapies.",
    "In Psychiatry, provides compassionate mental health care, employing a holistic approach to patient well-being.",
    "Dedicated to Pediatrics, ensuring the health and development of children through preventative care and tailored treatments.",
    "A leader in Obstetrics and Gynecology (OB/GYN), offering comprehensive women's health services from reproductive health to prenatal care.",
    "Skilled in Oncology, utilizing groundbreaking cancer therapies to offer hope and healing to patients facing cancer diagnoses.",
    "Specializes in Endocrinology, expertly managing hormonal imbalances and endocrine disorders with personalized treatment plans.",
    "Known for urgent care in Orthopedics, adept at addressing acute musculoskeletal injuries with swift and effective interventions.",
    "A Gastroenterology expert, specializing in important but non-urgent digestive surgeries, providing meticulous and planned care.",
    "In Cardiology, renowned for handling critical heart conditions with urgency, ensuring timely and life-saving interventions.",
    "A Neurology specialist, focusing on urgent neurological issues, providing rapid response to prevent long-term damage.",
    "A Dermatologist known for urgent skin condition treatments, offering immediate relief for acute dermatological issues.",
    "Psychiatrist specializing in crisis intervention, providing immediate support for acute mental health emergencies.",
    "Pediatrician skilled in urgent care for children, ensuring rapid response to pediatric emergencies for the youngest patients.",
    "OB/GYN expert in high-risk pregnancies, providing urgent care to ensure the safety and health of both mother and child.",
    "Oncologist specializing in urgent cancer care, swiftly implementing treatment plans for aggressive or advanced-stage cancers.",
    "Endocrinologist known for urgent care in hormonal crises, quickly stabilizing patients with acute endocrine disorders."
]

def euclidean_distance_np(locs, point2):
    lst = []
    for point1 in locs:
        lst.append(np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2)))
    return lst


docs = []
i = 0
ids = []
complete_array = np.empty((0,19))
locations = [np.random.rand(2) for _ in range(5)]
distances = []
import ast

import pandas as pd

def extract_columns_from_excel(file_path, col1, col2, col3):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Check if the specified columns exist in the DataFrame
    if col1 not in df.columns or col2 not in df.columns or col3 not in df.columns:
        raise ValueError("One or more specified columns do not exist in the Excel file.")
    
    # Extract the columns into lists
    list1 = df[col1].tolist()
    list2 = df[col2].tolist()
    list3 = df[col3].tolist()
    
    return list1, list2, list3



for desc in doctor_descriptions:
    desc = "Doctor decsription: " + desc
    lst = generate_chat_response(system_message_doctor, desc, custom_functions_doctor)
    complete_list = lst
    complete_array = np.append(complete_array, [complete_list], axis=0)
    i += 1

print(docs)
print(ids)
print(locations)
print(distances)

np.save('doc_list.npy', complete_array)


