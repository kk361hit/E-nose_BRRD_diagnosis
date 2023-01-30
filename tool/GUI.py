import PySimpleGUI as sg
import os
from pathlib import Path
import pandas as pd
import matplotlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def Dia(Mtype,inputcsv, inputmodelP, resultcsv):

	dataset=pd.read_csv(inputcsv,index_col=0,header=0)
	sonser = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32']
	mydata = dataset[sonser]
	
	if Mtype == 'LDA':
		inputmodel  = str(inputmodelP) + '\\LDA.m'
		enlda = joblib.load(inputmodel)
		prelda = enlda.predict(mydata)
		dataset['Predict'] = prelda
		ldascore = enlda.predict_proba(mydata)
		samplescore = []
		for scores in ldascore:
			samplescore.append(scores[0])
		dataset['Score'] = samplescore
		dataset.to_csv(resultcsv)
	
		
	if Mtype == 'NCA':
		inputmodelP1 = str(inputmodelP) + '\\NCA.1.m'
		inputmodelP2 = str(inputmodelP) + '\\NCA.2.m'
		nca = joblib.load(inputmodelP1)
		knn = joblib.load(inputmodelP2)
		prenca = knn.predict(nca.transform(mydata))
		dataset['Predict'] = prenca
		NCAscore = knn.predict_proba(nca.transform(mydata))
		print(NCAscore)
		samplescore = []
		for scores in NCAscore:
			samplescore.append(scores[0])
		dataset['Score'] = samplescore
		dataset.to_csv(resultcsv)
	
	if Mtype == 'PCA':
		inputmodelP1 = str(inputmodelP) + '\\PCA.1.m'
		inputmodelP2 = str(inputmodelP) + '\\PCA.2.m'
		enpca = joblib.load(inputmodelP1)
		knn = joblib.load(inputmodelP2)
		prepca = knn.predict(enpca.transform(mydata))
		dataset['Predict'] = prepca
		PCAscore = knn.predict_proba(enpca.transform(mydata))
		print(PCAscore)
		samplescore = []
		for scores in PCAscore:
			samplescore.append(scores[0])
		dataset['Score'] = samplescore
		dataset.to_csv(resultcsv)

def MTrain(Mtype,Traincsv,outputmodelP):

	datasize = 0
	for datasize , line in enumerate(open(Traincsv)):
		datasize  += 1

	dataset=pd.read_csv(Traincsv,index_col=0,header=0)
	sonser = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32']
	mydata = dataset[sonser]
	classification = dataset['Status']
	
	if Mtype == 'LDA':
		outputmodel  = str(outputmodelP) + '\\LDA.m'
		outputmodelP3 = str(outputmodelP) + '\\info'
		enlda = LDA(n_components=1,store_covariance=False)
		plotD = enlda.fit(mydata,classification).transform(mydata)
		prelda = enlda.predict(mydata)
		if not os.path.exists(outputmodelP):
			os.makedirs(outputmodelP)
		joblib.dump(enlda,outputmodel)
		acc = enlda.score(mydata,classification)
		
	if Mtype == 'NCA':
		outputmodelP1 = str(outputmodelP) + '\\NCA.1.m'
		outputmodelP2 = str(outputmodelP) + '\\NCA.2.m'
		outputmodelP3 = str(outputmodelP) + '\\info'
		nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, max_iter=5000,random_state=42))
		knn = KNeighborsClassifier(n_neighbors=3)
		nca.fit(mydata,classification).transform(mydata)
		plotD = nca.fit(mydata,classification)
		knn.fit(nca.transform(mydata), classification)
		if not os.path.exists(outputmodelP):
			os.makedirs(outputmodelP)
		joblib.dump(nca,outputmodelP1)
		joblib.dump(knn,outputmodelP2)
		acc = knn.score(nca.transform(mydata), classification)
		
	if Mtype == 'PCA':
		outputmodelP1 = str(outputmodelP) + '\\PCA.1.m'
		outputmodelP2 = str(outputmodelP) + '\\PCA.2.m'
		outputmodelP3 = str(outputmodelP) + '\\info'
		enpca = make_pipeline(StandardScaler(),PCA(n_components=2))
		knn = KNeighborsClassifier(n_neighbors=3)
		plotD = enpca.fit(mydata,classification).transform(mydata)
		knn.fit(enpca.transform(mydata), classification)
		if not os.path.exists(outputmodelP):
			os.makedirs(outputmodelP)
		joblib.dump(enpca,outputmodelP1)
		joblib.dump(knn,outputmodelP2)
		acc = knn.score(enpca.transform(mydata), classification)
		
	'''
	i = 0
	tags = {}
	for p in plotD:
		if not (classification[i] in tags.keys()):
			tags[classification[i]] = {'x':[],'y':[],'z':[]}
		for tag in tags:
			if classification[i] == tag:
				tags[tag]['x'].append(p[0])
				tags[tag]['y'].append(p[1])
				#tags[tag]['z'].append(p[2])
		i = i+1 
	
	plt.figure()
	mc = ['navy', 'turquoise','red','orange','green','blue']
	i=0
	for tag in tags:
		plt.scatter(tags[tag]['x'], tags[tag]['y'], s=20, c=mc[i], label=tag )
		i += 1
	plt.legend(loc='best', shadow=True, scatterpoints=1)
	plt.title(Mtype + 'of E-Nose dataset')
	plt.show()
	'''
	
	
	modelinfo = open(outputmodelP3,"w")
	modelinfo.write("This " + Mtype + " model is trained with " + str(datasize) + " E-nose data. Test accuracy of this model is " + str(acc) + ". ")
	
	return datasize,acc

def main():

	defultmodelpath = "../Model/LDA"
	inputmodelpath = defultmodelpath
	model_type = "LDA"
	Tmodel_type = "LDA"
	matplotlib.use("TkAgg")
	sg.ChangeLookAndFeel('Reddit')
	menu_def = [['Help', ['About tool','Exit']]]
	Minfo = "Please chose one model."
	Tinfo = "Please train a model."

	Welcome_layout = [
		[sg.Text('E-nose BRRD diagnose tool is developed by Hong Kong Polytechnic University and supported by Highways Department of the Hong Kong Government.', size=(60, 3),font=('Arial', 15))]
		]
	
	Diagnose_layout = [
		[sg.Frame(layout=[[sg.Radio('LDA', "RADIO1", default=True, size=(18,1),key="LDA_m",enable_events=True), sg.Radio('NCA', "RADIO1",size=(18,1),key="NCA_m",enable_events=True), sg.Radio('PCA', "RADIO1",size=(18,1),key="PCA_m",enable_events=True)]],title='Model',title_location='n',relief=sg.RELIEF_SUNKEN, tooltip='Chose one model' )],
		[sg.Text('Input E-nose data', size=(14, 1), auto_size_text=False, justification='right'),sg.InputText(enable_events=True,key="inputfile"),sg.FileBrowse(file_types=(("Enose data", "*.csv"),))],
		[sg.Text('Trained model folder', size=(14, 1), auto_size_text=False, justification='right'),sg.InputText(defultmodelpath,enable_events=True,key="inputmodelpath"),sg.FolderBrowse()],
		[sg.Text('Result', size=(14, 1), auto_size_text=False, justification='right'),sg.InputText(enable_events=True,key="outputfile"),sg.FileSaveAs(file_types=(("Results", "*.csv"),))],
		[sg.Button('Model info',key = 'Minfo'),sg.Submit('Diagnose',key = 'diagnose')]
		]
	
	Train_layout = [
		[sg.Frame(layout=[[sg.Radio('LDA', "RADIO1", default=True, size=(18,1),key="LDA_T",enable_events=True), sg.Radio('NCA', "RADIO1",size=(18,1),key="NCA_T",enable_events=True), sg.Radio('PCA', "RADIO1",size=(18,1),key="PCA_T",enable_events=True)]],title='Model',title_location='n',relief=sg.RELIEF_SUNKEN, tooltip='Chose one model' )],
		[sg.Text('Training data set', size=(18, 1), auto_size_text=False, justification='right'),sg.InputText(enable_events=True,key="training data file"),sg.FileBrowse(file_types=(("Training data set", "*.csv"),))],
		[sg.Text('Save trained model as', size=(18, 1), auto_size_text=False, justification='right'),sg.InputText(enable_events=True,key="outputmodelpath"),sg.FileSaveAs(file_types=(("Model File", "*"),))],
		[sg.Button('Info',key = 'look'),sg.Submit('Train',key = 'Train')]
		]
	
	layout = [
		[sg.Menu(menu_def, tearoff=False)],[sg.Text('E-nose BRRD diagnose tool',font=('Arial', 20, 'bold'))],[sg.Image('./logo.png')],[sg.TabGroup([[sg.Tab('Welcome', Welcome_layout),sg.Tab('Diagnose', Diagnose_layout), sg.Tab('Train', Train_layout)]])]
		]
		
	
	window = sg.Window('E-nose BRRD diagnose tool', layout, default_element_size=(50,60), grab_anywhere=False, enable_close_attempted_event=True)

	while True:
		event, values = window.read()
		if event == "About tool":
			sg.popup('About the tool','E-nose BRRD diagnose tool is developed by Hong Kong Polytechnic University and supported by Highways Department of the Hong Kong Government.')
		if event == "LDA_m":
			model_type = "LDA"
			defultmodelpath = "../Model/LDA"
			inputmodelpath = defultmodelpath
			window['inputmodelpath'].update(defultmodelpath)
		if event == "NCA_m":
			model_type = "NCA"
			defultmodelpath = "../Model/NCA"
			inputmodelpath = defultmodelpath
			window['inputmodelpath'].update(defultmodelpath)
		if event == "PCA_m":
			model_type = "PCA"
			defultmodelpath = "../Model/PCA"
			inputmodelpath = defultmodelpath
			window['inputmodelpath'].update(defultmodelpath)
		if event == "LDA_T":
			Tmodel_type = "LDA"
		if event == "NCA_T":
			Tmodel_type = "NCA"
		if event == "PCA_T":
			Tmodel_type = "PCA"
		if event == "inputfile":
			inputfile = values["inputfile"]
		if event == "inputmodelpath":
			inputmodelpath = values["inputmodelpath"]
		if event == "outputfile":
			outputfile = values["outputfile"]
		if event == "training data file":
			trainingdatafile = values["training data file"]
		if event == "outputmodelpath":
			outputmodelpath = values["outputmodelpath"]
		if event == "Minfo":
			pmi = str(Path(inputmodelpath)) + "\\info"
			MI = open(pmi)
			Minfo = ""
			for line in MI:
				Minfo = Minfo + line
			sg.popup("Model information",Minfo)
		if event == "look":
			sg.popup(Tinfo)
		if event == "diagnose":
			Dia(model_type, Path(inputfile),Path(inputmodelpath),Path(outputfile))
			sg.popup('Finished!\nThe result file was saved as:', outputfile)
		if event == "Train":
			Dsize = 0
			Macc = 0
			Dsize, Macc = MTrain(Tmodel_type,Path(trainingdatafile),Path(outputmodelpath))
			Tinfo = "This " + Tmodel_type + " model is trained with " + str(Dsize) + " E-nose data. Test accuracy of this model is " + str(Macc) + ". "
			sg.popup('Finished!\n' + Tinfo + '\nThe trained model was saved as:', outputmodelpath)
		if (event == "Exit" or event == None or event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT) and sg.popup_yes_no('Do you really want to exit?') == 'Yes': #窗口关闭事件
			break
	window.close()

if __name__ == '__main__':
	main()