import random
import csv

year = 1987
while year <= 2008:
	reader = open("{}.csv".format(year), "r")
	r = csv.reader(reader)
	next(reader, None) #skip header
	file_length=0
	for row in r:
		file_length+=1
	print file_length
	rl1 = random.sample(range(1, file_length), file_length/10)
	rl=sorted(rl1)
	reader.close()

	f = open('{}.csv'.format(year), 'r')
	fout = open('{}_r.csv'.format(year), 'w')
	i=0
	j=0
	for line in f:
		if i == 0:
			fout.write(line)
		elif i == rl[j]:
			fout.write(line)
			j += 1
		if j==len(rl):
			break
		i+=1
		
		if i%10000==0:
			print i, year
	fout.close() 
	year += 1