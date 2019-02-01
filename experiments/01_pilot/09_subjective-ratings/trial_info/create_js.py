import csv
import json

csvfile = open('pilot-reproduction-data.csv', 'r')
jsfile = open('stories.js', 'w+')

fieldnames = ("chain","generation","reproduction","story_title")
reader = csv.DictReader( csvfile, fieldnames, delimiter=";")

i=0
jsfile.write("var story = [ \n")
for row in reader:
	if i != 0: 
		print row
		row["reproduction"] = row["reproduction"].replace("\"","\\\"")
		print "{generation: " + row["generation"] + ", story_title: \"" + row["story_title"] + "\", reproduction: \"" + row["reproduction"] + "\", chain: \"" + row["chain"] + "\"}\n"
		jsfile.write("{generation: " + row["generation"] + ", story_title: \"" + row["story_title"] + "\", reproduction: \"" + row["reproduction"] + "\", chain: \"" + row["chain"] + "\"},\n")
	else:
		i =+ 1
jsfile.write("]")