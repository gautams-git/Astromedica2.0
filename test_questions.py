import json
from rules import apply_rule_1, generate_chat_questions

with open('chart_data.json') as f:
    chart = json.load(f)

rule1  = apply_rule_1(chart)
result = generate_chat_questions(rule1)

print(f"Total questions: {result['total']}")
print(f"Context keys: {list(result['context'].keys())}")
print()
for q in result['questions']:
    print(f"Q{q['id']} [{q['category']}]")
    print(f"   {q['question']}")
    print(f"   Options: {q['options']}")
    print()
