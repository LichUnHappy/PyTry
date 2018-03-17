# import random
# f = open("integers", 'w')
# for count in range(500):
# 	number = random.randint(1, 500)
# 	f.write(str(number) + "\n")
# f.close()

f = open("integers", 'r')
# text = f.read()
# print(text)
# for line in f:
# 	print(line)
# while 1:
# 	line = f.readline()
# 	if line == "":
# 		break
# 	print(line)
# sum = 0
# for line in f:
# 	line = line.strip()
# 	number = int(line)
# 	sum += number
# print("The sum is:"+ str(sum))

import pickle

# lyst = [60, "A string object", 1997]
# fileObj = open("item.dat", 'wb')
# for item in lyst:
# 	pickle.dump(item, fileObj)
# fileObj.close()
lyst = list()
fileObj = open("item.dat", 'rb')
while 1:
	try:
		item = pickle.load(fileObj)
		lyst.append(item)
	except EOFError:
		fileObj.close()
		break
print(lyst)
