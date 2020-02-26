#!/usr/bin/python3
# coding: utf-8

# =======================================================================================================================
# NORMATIVE MODELING and FEATURE INFLUENCE ANALYSIS using SUPER LEARNER
# Written by Taylor Keding (tjkeding@gmail.com)
# Last Updated: 11.16.19
# =======================================================================================================================
# -----------------------------------------------------------------------------------------------------------------------
# IMPORTS:
# -----------------------------------------------------------------------------------------------------------------------
# ---------- STOP ALL WARNINGS ----------
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
# ---------------------------------------
# Python 3:
import sys
import string
import os
import re
import csv
import random
import copy
from random import randint, shuffle
from datetime import datetime
from time import sleep
from functools import partial
import multiprocessing as mp

# Joblib:
import joblib as joblib

# Scikit-Learn:
import sklearn
from sklearn import datasets,svm,metrics,tree,linear_model,preprocessing
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVC,SVR
from sklearn.model_selection import cross_val_score,KFold,cross_val_predict,RandomizedSearchCV
from sklearn.metrics import make_scorer,mean_squared_error,mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor

# Numpy:
import numpy as np

# Pandas:
import pandas as pd

# SciPy:
import scipy.stats as stats

# Statsmodels:
import statsmodels.stats
from statsmodels.stats import multitest

# Start the timer
startTime = datetime.now()

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS:
# -----------------------------------------------------------------------------------------------------------------------

# ============================================ READ, SAMPLE, & FORMAT DATA ==============================================

def check_args_read_data(args):
    
    print("")
    print("   CHECK ARGUMENTS, READ-IN DATA")
    print("   ---------------------------------------------------------")

    # Check for the correct number of args
    if(len(sys.argv)!= 12):
        print("Incorrect number of args! 11 Required:")
        printUsage()
        sys.exit()

    # Check args[1]: read-in CSV training set file
    try:
        trainDF = pd.read_csv(args[1])
    except:
        print("Training data file could not be found!")
        printUsage()
        sys.exit()

    # Check args[2]: read-in CSV evalutation set file
    try:
        evalDF = pd.read_csv(args[2])
    except:
        print("Evaluation data file could not be found!")
        printUsage()
        sys.exit()

    # Check args[3]: read-in CSV prediction set file
    try: 
        predDF = pd.read_csv(args[3])
    except:
        print("Prediction data file could not be found!")
        printUsage()
        sys.exit()

    # Check args[4]: read-in model choice
    if(str(args[4])!="randForest" and str(args[4])!="mlp" and str(args[4])!="svm" and str(args[4])!="glm" \
        and str(args[4])!="gradBoost" and str(args[4])!="all"):
        print("Invalid model option, must be one of the following:")
        print("{ randForest, mlp, svm, glm, gradBoost, all }")
        printUsage()
        sys.exit()
    else:
        model = str(args[4])

    # Check args[5]: Check if kCV_paramSearch is in correct format
    try:
        kCV_paramSearch = int(args[5])
    except:
        print("<kCV_paramSearch> is incorrect (must be an int)")
        printUsage()
        sys.exit()

    # Check args[6]: Check if numIter_paramSearch is in correct format
    try:
        numIter_paramSearch = int(args[6])
    except:
        print("<numIter_paramSearch> is incorrect (must be an int)")
        printUsage()
        sys.exit()

    # Check args[7]: Check if numSims_MB is in correct format
    try:
        numSims_MB = int(args[7])
    except:
        print("<numSims_MB> is incorrect (must be an int)")
        printUsage()
        sys.exit()

    # Check args[8]: Check if numSims_NPS is in correct format
    try:
        numSims_NPS = int(args[8])
    except:
        print("<numSims_NPS> is incorrect (must be an int)")
        printUsage()
        sys.exit()

    # Check args[9]: Check if pThresh is in correct format
    try:
        pThresh = float(args[9])
        if(pThresh <=0.0 and pThresh>=1.0):
            print("<pThresh> is incorrect (must be an float between 0 and 1)")
            printUsage()
            sys.exit()
    except:
        print("<pThresh> is incorrect (must be an float between 0 and 1)")
        printUsage()
        sys.exit()

    # Check args[10]: check if the number of cores is correct
    try:
        numCores = int(args[10])
        if(numCores > mp.cpu_count() or (numCores < 1 and numCores != -1)):
            print(str("Specifed too many (or too few) cores! 0 < Integer <= "+str(mp.cpu_count())))
            print(str("If you want to use all available, input '-1"))
            printUsage()
            sys.exit()
    except:
        print("Number of cores is not an integer!")
        printUsage()
        sys.exit()

    # Check args[10]: get prefix for output files
    prefix = str(args[11])

    return {'trainDF':trainDF,'evalDF':evalDF,'predDF':predDF,'model':model,'kCV_paramSearch':kCV_paramSearch,'numIter_paramSearch':numIter_paramSearch, \
        'numSims_MB':numSims_MB,'numSims_NPS':numSims_NPS,'pThresh':pThresh,'numCores':numCores,'prefix':prefix}

# --------------------

def vectorizeDataFrame(dataframe,labelCol,startOfFeats):

    x = dataframe[dataframe.columns[startOfFeats:len(dataframe.columns)]].values
    y = dataframe[dataframe.columns[labelCol]]

    return {'x':x,'y':y}

# --------------------

def saveDFtoFile(DF,filename):

    if os.path.exists(filename):
        os.remove(filename)
    DF.to_csv(filename,index=False)

# --------------------

def compileGroupErrs(SL,groupDFs,evalDF,trainDF,errors,labelCol,startOfFeats):
        
    errorsOut = {}

    # Add normative instances to errors table
    errorsOut['SUBJ']=list(evalDF[evalDF.columns[0]])
    errorsOut['GROUP']=list(["NORM_EVAL"]*evalDF.shape[0])
    errorsOut['LABEL']=list(evalDF[evalDF.columns[labelCol]])
    errorsOut['ERR']=list(errors)

    # Add predict group instances to errors table
    for key in groupDFs:
        errorsOut['SUBJ'].extend(list(groupDFs[key]['SUBJ']))
        errorsOut['GROUP'].extend(list([str(key)]*groupDFs[key].shape[0]))
        errorsOut['LABEL'].extend(list(groupDFs[key][groupDFs[key].columns[labelCol]]))
        std_groupDF = normalizeDFs(trainDF,groupDFs[key],labelCol,startOfFeats)['testDF']
        groupVec = vectorizeDataFrame(std_groupDF,labelCol,startOfFeats)
        groupErrs = getErrors(groupVec['y'],predictWithSuperLearner(SL,groupVec))
        errorsOut['ERR'].extend(groupErrs)

    return errorsOut

# ========================================= PROCESSING, SCALING, & STATS TOOLS ==========================================

def getScore(true,pred):
    return mean_absolute_error(true,pred)

# --------------------

def getErrors(true,preds):

    errs = np.subtract(preds,true)
    return list(errs)

# --------------------

def compareDists(nullDist,testDist,numInstances):

    # Use Wilcoxon two-sample test to compare original error and group error distributions
    # Calculate effect size based on z-approximation to R2 (for n > 20)
    stat, pVal = stats.wilcoxon(x=nullDist,y=testDist,zero_method='zsplit',correction=True)
    z = stat/np.sqrt((numInstances*(numInstances+1)*(2*numInstances+1))/6)
    effectSize = np.power(z,2)/numInstances

    return [stat,pVal,effectSize]

# --------------------

def MCC(statsDF,groups,pThresh):

    outDF = pd.DataFrame()

    # Iterate through each group
    for i in range(0,len(groups)):

        # Correct both sets of p-values
        currGrp = statsDF.loc[statsDF[statsDF.columns[1]]==groups[i]]
        neg_reject, currGrp['NEG_P_FDR'] = multitest.fdrcorrection(list(currGrp['NEG_P']),alpha=pThresh, \
            method='indep',is_sorted=False)
        pos_reject, currGrp['POS_P_FDR'] = multitest.fdrcorrection(list(currGrp['POS_P']),alpha=pThresh, \
            method='indep',is_sorted=False)

        # Append corrected p-values to the output
        if i==0:
            outDF = currGrp
        else:
            outDF = outDF.append(currGrp)

    # Only output the rows that survive correction
    outDF = outDF.loc[outDF['NEG_P']<pThresh]
    outDF = outDF.loc[outDF['POS_P']<pThresh]

    return outDF 

# --------------------

def normalizeDFs(trainDF,testDF,labelCol,startOfFeats):
    
    copyTrain = copy.deepcopy(trainDF)
    if not isinstance(testDF,str): 
        copyTest = copy.deepcopy(testDF)
    else:
        copyTest = "none"

    for j in range(startOfFeats,len(copyTrain.columns)):
        currMean = np.mean(copyTrain[copyTrain.columns[j]])
        currSD = np.std(copyTrain[copyTrain.columns[j]])

        copyTrain[copyTrain.columns[j]]=(copyTrain[copyTrain.columns[j]]-currMean)/currSD
        if not isinstance(copyTest,str): 
            copyTest[copyTest.columns[j]]=(copyTest[copyTest.columns[j]]-currMean)/currSD

    return {'trainDF':copyTrain,'testDF':copyTest}

# --------------------

def multithreadProcess(partialFunc,start,jobs,numCores):
        
    pool = mp.Pool(processes=numCores)
    output_from_pool = pool.map(partialFunc,range(start,jobs))
    pool.close()
    pool.join()

    return output_from_pool

# ================================================= MODEL OUTPUT TOOLS ==================================================

def printUsage():
    print("python3 normativeModel_FLEP.py <trainFile> <evalFile> <predFile> <model> <kCV_paramSearch> <numIter_paramSearch> <numSims_MB> <numSims_NPS> <pThresh> <numCores> <prefix>")

# --------------------

def printModelDescription(modelName):
    if modelName == "glm":
        outString = str("Applying GENERAL LINEAR MODEL:\n" + \
                        ".....Implemented with SciKit-Learn's SGDClassifier/Regressor with l2 Regularization \n")

    elif modelName == "randForest":
        outString = str("Applying RANDOM FORESTS:\n" + \
                        ".....Implemented with SciKit-Learn's RandomForestClassifier/Regressor \n") 

    elif modelName == "svm":
        outString = str("Applying SUPPORT VECTOR MACHINE:\n" + \
                        ".....Implemented with SciKit-Learn's SVC/R\n")

    elif modelName == "mlp":
        outString = str("Applying MULTILAYER PERCEPTRON (NEURAL NETWORK):\n" + \
                        ".....Implemented with SciKit-Learn's MLPClassifier/Regressor \n")

    elif modelName == "kNN":
        outString = str("Applying k-NEAREST NEIGHBORS:\n" + \
                        ".....Implemented with SciKit-Learn's kNeighborsClassifier/Regressor \n")

    elif modelName == "gradBoost":
        outString = str("Applying GRADIENT BOOSTING MACHINE:\n" + \
                        ".....Implemented with SciKit-Learn's GradientBoostingClassifier/Regressor \n")

    print(outString)

# --------------------

def printProgressBar(iteration,total,prefix='',suffix='',decimals=1,length=100,fill='█'):
    """
    Print iterations progress
    Adapted from work by Greenstick (https://stackoverflow.com/users/2206251/greenstick)
    and Chris Luengo (https://stackoverflow.com/users/7328782/cris-luengo)
    in response to https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/30740258

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# =========================================== FEATURE INFLUENCE & EVALUATION ============================================

def runFeatureInfluence(SL,groupDFs,trainVec,trainDF,evalDF,numSims_NPS,numCores,labelCol,startOfFeats):
        
    print("")
    print("   CALCULATING GROUP x FEATURE INFLUENCE ON ERRORS")
    print("   ---------------------------------------------------------")
    # Feature influence
    # Simulations: sample each feature with replacement from the normative population,
    # Get distribution of new errors and see if it's sig diff than original errors
    # Then, sample each feature with replacement from each group and look for significant
    # differences from both the original errors AND from the feature influence of model

    groups = list(groupDFs.keys())
    origErrs = getErrors(trainVec['y'],predictWithSuperLearner(SL,trainVec))
    origMedian = np.median(origErrs)
    endOfFeats = trainDF.shape[1]
    rawOutDF = pd.DataFrame()

    # Iteratively process each feature
    counts = 0
    printProgressBar(0,(endOfFeats-startOfFeats)*len(groups),prefix='   Progress:',suffix='',length=50)
    for featIndx in range(startOfFeats,endOfFeats):

        # Process the norm feature's influence using multi-core parallel processing if available
        featInfNorm_func = partial(calcGroupNPS,featIndx=featIndx,SL=SL,currGroupDF=evalDF,\
            trainDF=trainDF,labelCol=labelCol,startOfFeats=startOfFeats)
        normSims = multithreadProcess(featInfNorm_func,0,numSims_NPS,numCores)
        normErrDist = np.mean(normSims,axis=0)

        # Process each group
        for grpIndx in range(0,len(groups)):

            # Process the groups' feature's influence using multi-core parallel processing if available
            featInfNorm_func = partial(calcGroupNPS,featIndx=featIndx,SL=SL,currGroupDF=groupDFs[groups[grpIndx]],\
                trainDF=trainDF,labelCol=labelCol,startOfFeats=startOfFeats)
            grpSims = multithreadProcess(featInfNorm_func,0,numSims_NPS,numCores)
            grpErrDist = np.mean(grpSims,axis=0)
            grpMedian = np.median(grpErrDist)
            if origMedian <= grpMedian:
                effectDir = "INCREASE"
            else:
                effectDir = "DECREASE"

            # Use Wilcoxon two-sample test to compare original error and group error distributions
            negStats = compareDists(origErrs,grpErrDist,trainDF.shape[0])

            # Use Wilcoxon two-sample test to compare normative error and group error distributions
            posStats = compareDists(normErrDist,grpErrDist,trainDF.shape[0])

            # Create new record for this feature/group combination
            newRec = pd.DataFrame({'FEAT':str(trainDF.columns[featIndx]),'GROUP':str(groups[grpIndx]), \
                'DIR':effectDir,'MEDIAN':grpMedian,'DIFF':grpMedian-origMedian,\
                'STAT':negStats[0],'RSQUARED':negStats[2],'NEG_P':negStats[1],'POS_P':posStats[1]},index=[0])
            rawOutDF = rawOutDF.append(newRec,ignore_index=True)

            # Update Progress Bar
            sleep(0.1)
            counts=counts+1
            printProgressBar(counts,(endOfFeats-startOfFeats)*len(groups),prefix='   Progress:',suffix='',length=50)
    
    # Calculate multiple comparison corrections on the compiled dataframe
    outDF = MCC(rawOutDF,groups,pThresh)

    # Output stats for this feature
    print("   COMPLETE!")
    return outDF

# --------------------

def calcGroupNPS(numSims_NPS,featIndx,SL,currGroupDF,trainDF,labelCol,startOfFeats):

    # Run NPS using group features
    # Make a copy of the trainDF as to not overwrite data
    npsDF = copy.deepcopy(trainDF)

    # Replace the current feature in the training set with a bootstrap sample of group's feature
    npsDF[npsDF.columns[featIndx]] = list(np.random.choice(currGroupDF[currGroupDF.columns[featIndx]], \
        size=trainDF.shape[0],replace=True))

    # Standardize and vectorize the copied dataframe
    std_npsDF = normalizeDFs(trainDF,npsDF,labelCol,startOfFeats)['testDF']
    npsVec = vectorizeDataFrame(std_npsDF,labelCol,startOfFeats)

    # Evaluate with super learner, get errors, and return
    preds = predictWithSuperLearner(SL,npsVec)
    return getErrors(npsVec['y'],preds)

# --------------------

def modelEvaluation(stats_SL,evalDF,evalVec,labelCol,startOfFeats):
        
    print("")
    print("   CALCULATING ABOVE-CHANCE PERFORMANCE OF MODELS")
    print("   ---------------------------------------------------------")

    # Output for model stats
    models = list(stats_SL['superLearner']['models'].keys())
    modelOutStats = {'MODEL':list(["superLearner"])+list(models),\
    'EVAL_MAE':list([0.0]*len(models)),'DIFF_FROM_CHANCE':list([0.0]*(len(models)+1)),\
    'PEARSON_R':list([0.0]*len(models)),'R_P':list([0.0]*len(models))}

    printProgressBar(0,len(models)+1,prefix='   Progress:',suffix='',length=50)
    for i in range(0,len(models)+1):
        
        # Get the name of the model current being evaluated
        # and the MAE score to compare to
        if i == 0:
            currName = "superLearner"
            bestMAE = stats_SL['MAE']
            modelOutStats['EVAL_MAE'].insert(i,bestMAE)
            modelOutStats['PEARSON_R'].insert(i,stats_SL['R'])
            modelOutStats['R_P'].insert(i,stats_SL['R_p'])
        else:
            currModel = stats_SL['superLearner']['models'][models[i-1]]
            currName = models[i-1]
            currStats = errorDiagnostics(stats_SL['superLearner'],currModel.predict(evalVec['x']),evalVec['y'])
            bestMAE = currStats['bestMAE']
            modelOutStats['EVAL_MAE'][i]=bestMAE
            modelOutStats['PEARSON_R'][i]=currStats['bestR']
            modelOutStats['R_P'][i]=currStats['bestRSig']

        # Evaluate the current model versus chance performance
        # Simulations: sample labels with replacement from the normative population
        # Process using multi-core parallel processing if available
        modelEval_func = partial(calcAboveChancePerf,SL=stats_SL['superLearner'],modelName=currName,\
            testDF=evalDF,bestScore=bestMAE,labelCol=labelCol,startOfFeats=startOfFeats)
        output_from_pool = multithreadProcess(modelEval_func,0,numSims_NPS,numCores)
        
        # Add model statistics to table
        modelOutStats['DIFF_FROM_CHANCE'][i] = np.mean(output_from_pool)

        # Update Progress Bar
        sleep(0.1)
        printProgressBar(i+1,len(models)+1,prefix='   Progress:',suffix='',length=50)

    print("   COMPLETE!")
    return modelOutStats

# --------------------

def calcAboveChancePerf(thisSim,SL,modelName,testDF,bestScore,labelCol,startOfFeats):

    # Make copy of the data
    bootstrapDF = testDF.copy()

    # Bootstrap sample labels
    bootstrapDF[bootstrapDF.columns[labelCol]] = np.random.choice(bootstrapDF[bootstrapDF.columns[labelCol]], \
        size=bootstrapDF.shape[0],replace=True)

    # Get vectorized data and evaluate bootsrap
    perSets = vectorizeDataFrame(bootstrapDF,labelCol,startOfFeats)

    # Return difference in model performance
    if modelName == "superLearner":
        return getScore(perSets['y'],predictWithSuperLearner(SL,perSets)) - bestScore
    else:       
        return getScore(perSets['y'],SL['models'][modelName].predict(perSets['x'])) - bestScore

# ================================================ MODELS & MODEL TOOLS =================================================

def getModelParams(choice,modelName,numFeats,trainingSize):

    # TODO: UPDATE THIS
    # ADD MODELS
    # ADD ADDITIONAL PARAMETER OPTIONS
    
    if modelName=="glm":
    
        # Define hyperparameter sampling distributions
        alphas_raw = stats.norm.rvs(loc=0.0,scale=0.1,size=5000)
        tols_raw = stats.norm.rvs(loc=0.001,scale=0.1,size=5000)
        eta0_raw = stats.norm.rvs(loc=0.0,scale=0.1,size=5000)
        power_t_raw = stats.norm.rvs(loc=0.5,scale=0.1,size=5000)
    
        # Make sure hyperparameters are in the correct range
        alphas = [i for i in alphas_raw if (i < 1 and i > 0)]
        tols = [i for i in tols_raw if i > 0]
        eta0s = [i for i in eta0_raw if i > 0]
        power_ts = [i for i in power_t_raw if (i > 0 and i < 1)]
        learning_rates = ["optimal", "invscaling", "adaptive"]

        # Combine parameters
        parameters = {'alpha':alphas,'tol':tols,'power_t':power_ts,'learning_rate':learning_rates,'eta0':eta0s}

        # Define model based on classification or regression
        if(choice==0):
            model = SGDClassifier(loss="log",penalty="l2",fit_intercept=False,max_iter=100000)
        else:
            model = SGDRegressor(loss="epsilon_insensitive",epsilon=0,penalty="l2",max_iter=100000,fit_intercept=True)

    elif modelName=="mlp":

        # Define hyperparameter sampling distributions
        alphas_raw = stats.norm.rvs(loc=0.0,scale=0.1,size=5000)
        learning_rate_init_raw = stats.norm.rvs(loc=0.0,scale=0.1,size=5000)
        power_t_raw = stats.norm.rvs(loc=0.5,scale=0.1,size=5000)
        tols_raw = stats.norm.rvs(loc=0.001,scale=0.1,size=5000)

        # Make sure hyperparameters are in the correct range
        alphas = [i for i in alphas_raw if (i < 1 and i > 0)]
        learning_rate_inits = [i for i in learning_rate_init_raw if i > 0]
        power_ts = [i for i in power_t_raw if (i > 0 and i < 1)]
        tols = [i for i in tols_raw if i > 0]
        solvers = ["lbfgs", "adam"]

        # Combine parameters
        parameters = {'alpha':alphas,'tol':tols,'learning_rate_init':learning_rate_inits,'power_t':power_ts, \
            'solver':solvers}

        # Define model based on classification or regression
        if(choice==0):
            model = MLPClassifier(hidden_layer_sizes = (int(round(trainingSize/2)),), \
                activation="relu",shuffle=True,max_iter=100000)
        else:
            model = MLPRegressor(hidden_layer_sizes = (int(round(trainingSize/2)),), \
                activation="relu",shuffle=True,max_iter=100000)

    elif modelName=="kNN":

        # Define hyperparameter sampling distributions
        n_neighbors_raw = stats.uniform.rvs(loc=1,scale=round(trainingSize/2),size=5000)

        # Make sure hyperparameters are in the correct range
        n_neighbors = [int(round(x)) for x in n_neighbors_raw]
        ps = [1,2,3,4,5]

        # Combine parameters
        parameters = {'n_neighbors':n_neighbors,'p':ps}

        # Define model based on classification or regression
        if(choice==0):
            model = KNeighborsClassifier(algorithm="auto")
        else:
            model = KNeighborsRegressor(algorithm="auto")

    elif modelName=="randForest":

        # Define hyperparameter sampling distributions
        n_estimators_raw = stats.norm.rvs(loc=500,scale=300,size=5000)
        max_features_raw = stats.norm.rvs(loc=0.25*(numFeats),scale=0.25*(numFeats),size=5000)
        max_depths_raw = stats.norm.rvs(loc=0.01*(numFeats),scale=0.1*(numFeats),size=5000)

        # Make sure hyperparameters are in the correct range
        n_estimators = [int(round(i)) for i in n_estimators_raw if (int(round(i)) < numFeats and int(round(i)) > 0)]
        max_features = [int(round(i)) for i in max_features_raw if (int(round(i)) < numFeats and int(round(i)) > 0)]
        max_depths = [int(round(i)) for i in max_depths_raw if (int(round(i)) < numFeats and int(round(i)) > 0)]
        criterions = ["gini","entropy"]

        # Combine parameters
        parameters = {'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depths}

        # Define model based on classification or regression
        if(choice==0):
            model = RandomForestClassifier(bootstrap=True)
            parameters['criterion'] = criterions
        else:
            model = RandomForestRegressor(bootstrap=True,criterion="mae")
    
    elif modelName=="svm":

        # Define hyperparameter sampling distributions
        Cs_raw = stats.norm.rvs(loc=0.0,scale=1,size=5000)
        tols_raw = stats.norm.rvs(loc=0.0,scale=0.1,size=5000)
        epsilon_raw = stats.norm.rvs(loc=0.0,scale = 0.05,size=5000)
        gammas_raw = stats.norm.rvs(loc=0.0,scale=0.1,size=5000)
        coef0s = stats.norm.rvs(loc=0.0,scale=5,size=5000)

        # Make sure hyperparameters are in the correct range
        Cs = [i for i in Cs_raw if i > 0]
        tols = [i for i in tols_raw if i > 0]
        epsilons = [i for i in epsilon_raw if i > 0]
        gammas = [i for i in gammas_raw if i > 0]
        kernels = ["poly","rbf"]
        degrees = [1,2,3,4,5]

        # Combine parameters
        parameters = {'C':Cs,'tol':tols,'degree':degrees,'kernel':kernels,'gamma':gammas,'coef0':coef0s}

        # Define model based on classification or regression
        if(choice==0):
            model = SVC(probability=True)
        else:
            model = SVR()
            parameters['epsilon'] = epsilons

    elif modelName=="gradBoost":

        # Define hyperparameter sampling distributions
        learning_rate_raw = stats.norm.rvs(loc=0.0,scale=0.1,size=5000)
        n_estimators_raw = stats.norm.rvs(loc=500,scale=300,size=5000)
        max_features_raw = stats.norm.rvs(loc=0.25*(numFeats),scale=0.25*(numFeats),size=5000)
        subsamples_raw = stats.norm.rvs(loc=1.0,scale=0.25,size=5000)
        max_depths_raw = stats.norm.rvs(loc=0.01*(numFeats),scale=0.1*(numFeats),size=5000)
    
        # Make sure hyperparameters are in the correct range
        learning_rates = [i for i in learning_rate_raw if i > 0]
        subsamples = [i for i in subsamples_raw if (i <= 1 and i > 0)]
        n_estimators = [int(round(i)) for i in n_estimators_raw if (int(round(i)) < numFeats and int(round(i)) > 0)]
        max_features = [int(round(i)) for i in max_features_raw if (int(round(i)) < numFeats and int(round(i)) > 0)]
        max_depths = [int(round(i)) for i in max_depths_raw if (int(round(i)) < numFeats and int(round(i)) > 0)]
        losses_binary = ["deviance","exponential"]
        losses_multi = ["deviance"]

        # Combine parameters
        parameters = {'learning_rate':learning_rates,'n_estimators':n_estimators,'max_features':max_features, \
            'subsample':subsamples,'max_depth':max_depths}

        # Define model based on classification or regression
        if(choice==0):
            model = GradientBoostingClassifier(criterion="mae")
        else:
            model = GradientBoostingRegressor(loss="ls",criterion="mae")

    return {'model':model,'parameters':parameters}

# --------------------

def getRandParams(params):

    outDict = {}
    for key in params:
        randInt = int(np.random.randint(0,len(params[key]),1))
        outDict[key] = params[key][randInt]
    return outDict

# --------------------

def errorDiagnostics(superLearner,preds,true):

    # Get prediction errors
    errors = np.subtract(preds,true)
    currMAE = getScore(true,preds)

    # Get Spearman Rank Correlation for this optimized model
    r, r_p = stats.pearsonr(true,preds)

    # Compile stats for returning
    return {'superLearner':superLearner,'bestR':r,'bestRSig':r_p,'bestMAE':currMAE,'bestErrors':errors}

# --------------------

def buildSuperLearner(models,iters,numSim_MB,trainDF,evalDF,labelCol,startOfFeats,kCV,numCores):

    # Hyperparameter Turning
    # Returns a dictionary of optimized models (not yet fit) - model names (from 'models') are the keys
    print("")
    print("   BUILDING & OPTIMIZING MODELS FOR THE SUPER LEARNER")
    print("   ---------------------------------------------------------")
    paramsSet = optimizeModelParams(models,iters,numSim_MB,trainDF,labelCol,\
        startOfFeats,kCV,numCores)
    print("   COMPLETE!")

    # Tuning algorithm coefficients and returning the most generalizable super learner
    print("")
    print("   TUNING SUPER LEARNER & EVALUATING GENERALIZABILITY")
    print("   ---------------------------------------------------------")
    superLearnerStats = optimizeSuperLearner(paramsSet,trainDF,evalDF,labelCol,\
        startOfFeats,numSim_MB,iters,kCV,numCores)
    print("   COMPLETE!")

    return superLearnerStats
    
# --------------------

def optimizeModelParams(models,iters,numSim_MB,trainDF,labelCol,startOfFeats,kCV,numCores):

    optModels ={}

    # Print progress bar for each model and group
    printProgressBar(0,len(models),prefix='   Progress:',suffix='',length=50)
    for i in range(0,len(models)):

        # Get parameter sampling distributions
        modelParams = getModelParams(1,models[i],trainDF.shape[1]-2,trainDF.shape[0])
        model = modelParams['model']
        params = modelParams['parameters']
        bestParams = 0.0
        bestScore = sys.float_info.max

        # Use multithreading to do cross-validated randomized parameter search
        randSearchPartial = partial(randParamSearchCV,model=model,params=params,numSim_MB=numSim_MB,trainDF=trainDF,\
            labelCol=labelCol,startOfFeats=startOfFeats,kCV=kCV)
        output_from_pool = multithreadProcess(randSearchPartial,0,iters,numCores)

        # Find the optimized model with the best score
        for sim in range(0,len(output_from_pool)):
            if output_from_pool[sim]['score'] < bestScore:
                bestScore = output_from_pool[sim]['score']
                bestParams = output_from_pool[sim]['paramSamp']

        # Save optimized model to dict
        optModels[models[i]]=model.set_params(**bestParams)

        # Update Progress Bar
        sleep(0.1)
        printProgressBar(i+1,len(models),prefix='   Progress:',suffix='',length=50)

    # Return model with tuned hyperparameters
    return optModels
    
# --------------------

def randParamSearchCV(currIter,model,params,numSim_MB,trainDF,labelCol,startOfFeats,kCV):

    # Get random sample of possible params
    paramSamp = getRandParams(params)
    model.set_params(**paramSamp)
    cvScores = [0.0]*numSim_MB

    for x in range(0,numSim_MB):

        # Vectorize the new DF
        newTrainVec = vectorizeDataFrame(trainDF,labelCol,startOfFeats)

        # Generate k folds
        kf = KFold(n_splits=kCV,shuffle=True)

        # Get cross validation score
        currScore = np.mean(cross_val_score(estimator=model,X=newTrainVec['x'],y=newTrainVec['y'], \
            scoring="neg_mean_absolute_error",cv=kf))
        if currScore < 0:
            cvScores[x] = -1*currScore
        else:
            cvScores[x] = currScore
        
    # Average CV scores across sim_MB's and save with params
    aveCV = np.mean(cvScores)

    # Return the score for the current set of random parameters
    return {'score':aveCV,'paramSamp':paramSamp}

# --------------------

def optimizeSuperLearner(paramsSet,trainDF,evalDF,labelCol,startOfFeats,numSim_MB,iters,kCV,numCores):

    # Vectorize evaluation dataset
    evalDF_vec = vectorizeDataFrame(evalDF,labelCol,startOfFeats)
        
    # Containers for stats to test
    bestRSig = sys.float_info.max
    bestMAE = sys.float_info.max
    bestSL = 0.0
    bestR = 0.0
    bestErrors = 0.0

    # Test the model with the evaluation set
    printProgressBar(0,numSim_MB,prefix='   Progress:',suffix='',length=50)
    for i in range(0,numSim_MB):

        # Tune the super learner coefficients
        out = tuneSLCoefs(iters,paramsSet,trainDF,evalVec,numSim_MB,labelCol,startOfFeats,kCV)

        currResult = errorDiagnostics(out['superLearner'],out['preds'],out['true'])
        if currResult['bestMAE'] < bestMAE and currResult['bestRSig'] < bestRSig:
            bestMAE = currResult['bestMAE']
            bestRSig = currResult['bestRSig']
            bestSL = currResult['superLearner']
            bestR = currResult['bestR']
            bestErrors = currResult['bestErrors']

        # Update Progress Bar
        sleep(0.1)
        printProgressBar(i+1,numSim_MB,prefix='   Progress:',suffix='',length=50)

    # Return best performing super learner and associated stats
    return {'superLearner':bestSL,'MAE':bestMAE,'R':bestR,'R_p':bestRSig,'errors':bestErrors}

# --------------------

def tuneSLCoefs(iters,optModels,trainDF,evalVec,numSim_MB,labelCol,startOfFeats,kCV):

    # Vectorize the new DF
    newTrainVec = vectorizeDataFrame(trainDF,labelCol,startOfFeats)
            
    # Generate k folds
    kf = KFold(n_splits=kCV,shuffle=True)

    # Train each optimized model and save the hold-out predictions
    holdOutPreds = {'label':newTrainVec['y']}
    for key in optModels:
        holdOutPreds[key] = cross_val_predict(optModels[key],X=newTrainVec['x'],y=newTrainVec['y'],cv=kf)

    # Compile new dataset with out-of-fold predictions and vectorize
    newDF = pd.DataFrame.from_dict(holdOutPreds)
    newDF_vec = vectorizeDataFrame(newDF,0,1)

    # ----------------------------------------
    # Optimize Super Learner coefficients:
    # ----------------------------------------
    # Uses same linear model framework (trained with stochastic gradient descent) as
    # the GLM model within the Super Learner - ridge penalty
    sdgCoef = getModelParams(1,"glm",len(optModels.keys()),trainDF.shape[0])
    model = sdgCoef['model'].set_params(**{'fit_intercept':False})
    params = sdgCoef['parameters']
    bestParams = 0.0
    bestScore = sys.float_info.max

    # Use multithreading to do cross-validated randomized parameter search
    randSearchPartial = partial(randParamSearchCV,model=model,params=params,numSim_MB=numSim_MB,\
        trainDF=newDF,labelCol=0,startOfFeats=1,kCV=kCV)
    output_from_pool = multithreadProcess(randSearchPartial,0,iters,numCores)

    # Save the best parameters to train coefficients
    for i in range(0,len(output_from_pool)):
        if output_from_pool[i]['score'] < bestScore:
            bestScore = output_from_pool[i]['score']
            bestParams = output_from_pool[i]['paramSamp']

    # Save optimized model to dict
    coefModel = model.set_params(**bestParams)
    coefModel_fit = coefModel.fit(newDF_vec['x'],newDF_vec['y'])
    # ----------------------------------------

    # Get coefficients
    coefs = list(coefModel_fit.coef_)
    
    # Train each model on the full dataset
    modelsOut = {}
    for key in optModels:
        modelsOut[key]=optModels[key].fit(newTrainVec['x'],newTrainVec['y'])

    # Create the new Super Learner, make predictions, and return
    currSuperLearner = {'coefs':coefs,'models':modelsOut}
    preds = predictWithSuperLearner(currSuperLearner,evalVec)
    return {'superLearner':currSuperLearner,'preds':preds,'true':evalVec['y']}

# --------------------

def predictWithSuperLearner(superLearner,evalVec):

    preds = [0.0]*len(evalVec['y'])

    for i, key in zip(range(0,len(superLearner['models'].keys())),superLearner['models']):
        currPreds = list(superLearner['models'][key].predict(evalVec['x']))
        updatedPreds = [j*superLearner['coefs'][i] for j in currPreds]
        preds = np.add(preds,updatedPreds)

    return preds

# -----------------------------------------------------------------------------------------------------------------------
# MAIN:
# -----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # NECESSARY FOR UNIX SERVERS:
    mp.set_start_method('forkserver')

    print("")
    print("-----------------------------------------------------------------------------")
    print("    NORMATIVE MODELING and FEATURE INFLUENCE ANALYSIS using SUPER LEARNER    ")
    print("-----------------------------------------------------------------------------")

    # Read in all the args and data
    readIn = check_args_read_data(sys.argv)
    trainDF = readIn['trainDF'] # Training dataset as a Pandas dataframe
    evalDF = readIn['evalDF'] # Evaluation dataset as a Pandas dataframe
    predDF = readIn['predDF'] # Prediction dataset as a Pandas dataframe
    model = readIn['model'] # Model(s) to use
    kCV_paramSearch = readIn['kCV_paramSearch'] # Folds for randomized parameter search cv
    numIter_paramSearch = readIn['numIter_paramSearch'] # Number of iterations for randomized parameter search cv
    numSims_MB = readIn['numSims_MB'] # Number of simulations of model building
    numSims_NPS = readIn['numSims_NPS'] # Number of simulations to use for NPS analysis
    pThresh = readIn['pThresh'] # Threshold for determining statistical significance
    numCores = readIn['numCores'] # Number of cores to use for multi-core processing
    if numCores == -1:
        numCores = mp.cpu_count() # Flag for option to use all available cores
    prefix = readIn['prefix'] # Output file prefix
    labelCol = 1 # Column index for the label variable in train+eval data
    startOfFeats = 2 # Column index for the start of features in train+eval data

    # Ensure this is a regression model
    if(trainDF.dtypes[labelCol]=="int64"):
        trainDF[trainDF.columns[labelCol]] = trainDF[trainDF.columns[labelCol]]*1.0
    print("   Using REGRESSION to PREDICT:",str(list(trainDF.columns.values)[labelCol]))

    # Normalize the data to the training set
    normalizeEval = normalizeDFs(trainDF,evalDF,labelCol,startOfFeats)
    std_trainDF = normalizeEval['trainDF']
    std_evalDF = normalizeEval['testDF']

    # Vectorize the training, evaluation, and prediction data
    trainVec = vectorizeDataFrame(std_trainDF,labelCol,startOfFeats)
    evalVec = vectorizeDataFrame(std_evalDF,labelCol,startOfFeats)

    # Define the models to run
    if model=="all":
        models=["randForest","svm","glm","gradBoost","mlp"]
    else:
        models = [model]

    # Run Super Learner algorithm to generate a normative model, save to file
    fileNameSL = str(prefix+"_superLearner_"+trainDF.columns[labelCol]+"_cv"+str(kCV_paramSearch))
    stats_SL = buildSuperLearner(models,numIter_paramSearch,numSims_MB,std_trainDF,\
        std_evalDF,labelCol,startOfFeats,kCV_paramSearch,numCores)
    joblib.dump(value=stats_SL['superLearner'],filename=fileNameSL,compress=False)

    # Evaluate the Super Learner and its submodels - save performance stats to file
    stats_output = modelEvaluation(stats_SL,std_evalDF,evalVec,labelCol,startOfFeats)
    modName = str(prefix+"_modelPerformances_"+trainDF.columns[labelCol]+"_cv"+str(kCV_paramSearch)+\
        "_simsNPS"+str(numSims_NPS)+".csv")
    saveDFtoFile(pd.DataFrame.from_dict(stats_output),modName)

    # Subdivide the prediction sample by group
    # Create a dictionary of dataframes, one for each group label
    # Group dataframes take-on the same format as normative train + eval sets
    groups = list(predDF[predDF.columns[2]].unique())
    groupDFs = {}
    for group in groups:
        allGroup = predDF.loc[predDF[predDF.columns[2]]==group]
        groupDFs[group] = allGroup.loc[:,allGroup.columns!=allGroup.columns[2]]

    # Generate errors for the eval sample and each group in the prediction sample
    allErrors = compileGroupErrs(stats_SL['superLearner'],groupDFs,evalDF,trainDF,\
        stats_SL['errors'],labelCol,startOfFeats)
    errsName = str(prefix+"_errors_"+trainDF.columns[labelCol]+"_cv"+str(kCV_paramSearch)+\
        "_simsNPS"+str(numSims_NPS)+".csv")
    saveDFtoFile(pd.DataFrame.from_dict(allErrors),errsName)

    # Run feature influence analysis using the NPS algorithm
    statsNPS = runFeatureInfluence(stats_SL['superLearner'],groupDFs,trainVec,std_trainDF,std_evalDF,\
        numSims_NPS,numCores,labelCol,startOfFeats)
    statsName = str(prefix+"_featInfNPS_"+trainDF.columns[labelCol]+ "_cv"+str(kCV_paramSearch)+\
        "_simsNPS"+str(numSims_NPS)+".csv")
    saveDFtoFile(pd.DataFrame.from_dict(statsNPS),statsName)

    print("-----------------------------------------------------------------------------")
    print("                            ALL ANALYSES COMPLETE!                           ")
    print("               Total Execution Time: "+str(datetime.now() - startTime)+"          ")
    print("-----------------------------------------------------------------------------")
    print("")

# -----------------------------------------------------------------------------------------------------------------------
# END MAIN
# -----------------------------------------------------------------------------------------------------------------------
