import json
with open('chart_data.json') as f:
    d = json.load(f)

print('ASCENDANT:', d['ascendant'])
print()
print('PLANETS:')
for p in d['planets']:
    print(f"  {p['name']} -> house {p['house']} sign {p['sign']['name']}")
print()
print('HOUSES:')
for h in d['houses']:
    print(f"  House {h['house']} = {h['sign']['name']}")
