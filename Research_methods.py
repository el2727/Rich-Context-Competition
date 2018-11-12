
# coding: utf-8

import json

# Extract research methods list from SAGE JSON file

with open('sage_research_methods.json') as file:
    data = json.load(file)

value_list = []
for i in data["@graph"]:
    value_list.append(i['skos:prefLabel']['@value'])

# Open participants' research methods file

with open('output_research_methods.json') as file:
    data_output = json.load(file)

results = []
for i in data_output['output']:
    results.append(i)
    
# Compare output files with research methods JSON file

matches = []
for i in results:
    for y in value_list:
        if i == y:
            matches.append(i)
            
new_methods = []
for i in results:
    for y in value_list:
        if i != y:
            new_methods.append(i)

# Export text file with new methods for qualitative assessment

with open('new_methods_output.txt', 'w') as file:
    file.write(new_methods)
    
# Compute percentages of exact matches with existing research methods JSON and a percentage of new methods identified

total_methods_identified = matches + new_methods
matches_percentage = len(matches)/len(value_list)
new_methods_percentage = len(total_methods_identified) - len(matches)

print("Percentage of existing methods identified:", matches_percentage)
print("Percentage of new methods identified:", new_methods_percentage)

