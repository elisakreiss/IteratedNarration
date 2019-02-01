from wordfreq import word_frequency
import csv

csvfile = open('word_freq_raw.csv', 'r')

# fieldnames = ("reproduction","id")
reader = csv.DictReader( csvfile, delimiter=";")
output_file = open('freqs_raw.csv', 'w+')

output_file.write("num,Freq\n")
freqs = []
for row in reader:
	print("")
	print(row)
	repro_freqs = []
	for word in row['repro_nostopwords'].split():
		repro_freqs.append(word_frequency(word,'en'))

	avg_freq = sum(repro_freqs)/len(repro_freqs)
	freqs.append(avg_freq)
	output_file.write(row['num']+","+str(avg_freq)+"\n")

print(len(freqs))
