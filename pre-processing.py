# Select a subset for training and testing purpose
import os
import pickle
from subprocess import call

files=[i for i in os.listdir(os.getcwd()) if 'split' in i]
data={}
key='P17-R04-ContinentalBreakfast'
data['train']=[]
data['test']=[]
for fname in files:
	with open(fname,'r+') as nfile:
		for line in nfile:
			if key in line:
				data[fname.split('_')[0]].append(line.strip())
with open('subset.pkl','wb+')as f:
	pickle.dump(data,f)
with open('subset.pkl','rb+') as ff:
	d=pickle.load(ff)

#Generate images from the videos for the subset
labels={}
for key in data:
	print(key)
	labels[key]={}
	for val in data[key]:
		labels[key][val[:-4]]=val.split(" ")[-3]
for key in data:
	print(key)
	try:
		os.mkdir(key)
	except Exception as e:
		print(e)
	for file in data[key]:
		fn='-'.join(file.split("-")[:3])
		try:
			os.mkdir(os.path.join(key,labels[key][file[:-4]]))
		except Exception as e:
			print(e)
		print('after')
		print(" before Doing for file "+fn)
		call(['ffmpeg','-i',os.path.join(os.getcwd(),fn,' '.join(file.split(" ")[:-3])+'.mp4')
			,os.path.join(os.getcwd(),key,labels[key][file[:-4]],fn[:22]+'%04d.jpg')])

	print("Done")
# To make sure keras flow_from_directory faces same number of folders for train and test.
# otherwise it doesn't work since it takes the number of folders as the total number
# of classes separately for train as well as for test.
directory_tr=os.listdir(os.path.join(os.getcwd(),'train'))
directory_te=os.listdir(os.path.join(os.getcwd(),'test'))
for folder in directory_tr:
	if folder not in directory_te:
		try:
			os.mkdir(os.path.join(os.getcwd(),'test',folder))
		except Exception as e:
			print(e)

for folder in directory_te:
	if folder not in directory_tr:
		try:
			os.mkdir(os.path.join(os.getcwd(),'train',folder))
		except Exception as e:
			print(e)


