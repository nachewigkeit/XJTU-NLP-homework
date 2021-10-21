projectPath = r"E:\AI\program\XJTU-NLP-homework\NewsClassify"
dataPath = r"E:\AI\dataset\20_newsgroups"
clsDict = {'alt.atheism': 0, 'comp.graphics': 1, 'comp.os.ms-windows.misc': 2, 'comp.sys.ibm.pc.hardware': 3,
           'comp.sys.mac.hardware': 4, 'comp.windows.x': 5, 'misc.forsale': 6, 'rec.autos': 7, 'rec.motorcycles': 8,
           'rec.sport.baseball': 9, 'rec.sport.hockey': 10, 'sci.crypt': 11, 'sci.electronics': 12, 'sci.med': 13,
           'sci.space': 14, 'soc.religion.christian': 15, 'talk.politics.guns': 16, 'talk.politics.mideast': 17,
           'talk.politics.misc': 18, 'talk.religion.misc': 19}

numThres = 5
fileNum = 19997
fold = 5