import csv

csvfile = open('*.csv', 'rb')
# reader = csv.reader(csvfile)
reader = csv.DictReader(csvfile)

for row in reader:
	print(row)