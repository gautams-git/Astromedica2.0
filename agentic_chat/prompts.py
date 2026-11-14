"""The only two prompts the model sees."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "label":    {"type": "string",
                     "enum": ["yes_strong", "yes_but", "no_but", "no_strong"]},
        "symptoms": {"type": "array", "items": {"type": "string"}},
        "duration": {"type": "string"},
        "severity": {"type": "string",
                     "enum": ["none", "mild", "moderate", "severe"]},
        "red_flag": {"type": "boolean"},
    },
    "required": ["label", "symptoms", "red_flag"],
}

EXTRACT_SYSTEM = """You read ONE answer from a patient in a health check-up chat in India.
The patient may write English, Hindi or a mix.
Question that was asked: {question}

{rag_examples}

Return JSON only.
label: yes_strong = clearly yes or happens often; yes_but = yes but only sometimes or with a condition;
       no_but = unsure or rarely; no_strong = clearly no.
symptoms: short phrases the patient actually mentioned. Do not add any.
red_flag: true only if the patient describes chest pain at rest, fainting, blood in vomit, stool or urine,
          sudden weakness of one side, or trouble breathing at rest.
Never diagnose. Never advise."""

PHRASE_SYSTEM = """Reword ONE health check-up question for a patient in India.
Keep the meaning. One question only, at most 170 characters, ends with a question mark.
You may begin with a few words that acknowledge the patient's last answer.
Do not name any condition or internal organ. Do not give advice."""