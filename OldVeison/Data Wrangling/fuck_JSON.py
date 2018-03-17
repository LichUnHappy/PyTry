import json

json_data = open('*.json').read()
data = json.loads(json_data)

for item in data:
	print(data)