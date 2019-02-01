from graphviz import Digraph
import pandas as pd

# csv_path = "/Users/elisakreiss/Documents/business/projects/iterated_narration/study/data/01_pilot/01_psychStudents/flowchart_data.csv"
csv_path = "./../../../data/01_pilot/07_revised06/flowchart_data.csv"

def create_graph(data_dict,story,slice_nr,chains):

	f = Digraph('repro_chain', filename='flowcharts/'+story+str(slice_nr)+'_repro_chain.gv')
	f.attr(rankdir='TB', size='30')
	f.attr('node', shape='rectangle')

	for chain in chains:
		# arr collects all reproductions from one chain
		arr = []
		print("***")
		print(chain)
		
		for row in range(0,len(data_dict["story_title"])):
			# go through all data of the currently looked at story (e.g., bees)
			if (data_dict["story_title"][row] == story):
				# create a node with the original seed (taken from first reproduction) and call it "original" (not chain specific)
				if (data_dict["generation"][row] == 1):
					f.node('original',label=data_dict["story_text"][row])

			# select data of current chain and put it in arr
			if chain == data_dict["chain"][row]:
				arr.append({'chain': chain, 'reproduction': data_dict["reproduction"][row], 'generation': data_dict["generation"][row], 'deadend': data_dict["deadend"][row], 'guilty': data_dict["guilty"][row], 'row': row})
		
		# after all rows of one chain have been saved in arr, create graph
		for repro in range(0,len(arr)):
			# create new node for current reproduction
			if arr[repro]['deadend'] == True:
				f.node(arr[repro]['chain']+str(arr[repro]['generation'])+str(arr[repro]['deadend'])+str(arr[repro]['row']),label=str(arr[repro]['reproduction'])+"\n\n How likely that the main character(s) committed the crime?: "+str(arr[repro]['guilty']))
				f.edge((arr[repro]['chain']+str(arr[repro]['generation']-1)+'False'),arr[repro]['chain']+str(arr[repro]['generation'])+str(arr[repro]['deadend'])+str(arr[repro]['row']))
			else:
				f.node(arr[repro]['chain']+str(arr[repro]['generation'])+str(arr[repro]['deadend']),label=str(arr[repro]['reproduction'])+"\n\n How likely that the main character(s) committed the crime?: "+str(arr[repro]['guilty']))
				if arr[repro]['generation'] > 1:
					# print('arr[repro]')
					# print(arr[repro])
					f.edge((arr[repro]['chain']+str(arr[repro]['generation']-1)+'False'),arr[repro]['chain']+str(arr[repro]['generation'])+str(arr[repro]['deadend']))
				else:
					f.edge('original',arr[repro]['chain']+str(arr[repro]['generation'])+str(arr[repro]['deadend']))
	f.view()


def new_graph(story):

	data = pd.read_csv(csv_path)
	data_dict = {col: list(data[col]) for col in data.columns}

	chains = []
	for chain in range(0,len(data_dict["chain"])):
		if data_dict["story_title"][chain] == story:
			if data_dict["chain"][chain] not in chains:
				chains.append(data_dict["chain"][chain])

	i = 0
	slice_nr = 1
	sliced_chain = []
	for chain in chains:
		if ((i == 0) | (i == 1)):
			sliced_chain.append(chain)
			i += 1
		else:
			sliced_chain.append(chain)
			create_graph(data_dict,story,slice_nr,sliced_chain)
			sliced_chain = []
			slice_nr += 1
			i = 0
	if sliced_chain:
		create_graph(data_dict,story,slice_nr,sliced_chain)

new_graph('arson_free')
new_graph('arson_jail')
new_graph('bees_free')
new_graph('bees_jail')
new_graph('smuggler_free')
new_graph('smuggler_jail')

