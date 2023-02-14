"""
Copyright c 2021 by Northwestern University. All Rights Reserved.

@author: Can Aygen
"""

############### Function Definitions ###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
import pdb
import matplotlib as mpl
import os,sys
import numdifftools as nd
import warnings
import six
import time
import pathlib

# from parfor import parfor
from pathlib import Path
from datetime import date
from scipy.optimize import dual_annealing
from scipy import optimize
from scipy import integrate
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import table
from PIL import Image
from PIL import PngImagePlugin
from numpy import linalg as LA
from mpl_toolkits import mplot3d
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.family'] = 'sans-serif'

label_style = {'fontsize': 'medium','fontfamily':'sans-serif'}#, 'usetex': True} 
default_text_style = {'fontsize': 'medium', }

warnings.filterwarnings('ignore')
#%%
class ImportResults:
    def __init__(self, x, fun):
        # self.ret = ret
        self.x = x
        self.fun = fun
        
#%%
def add_watermark(ax,version):
    ax.text(ax.get_xlim()[0], #+ 0.1, 
            ax.get_ylim()[0], #+ 0.1, 
            # "<Company Name>",
            str(version),
            size = 5,
            alpha=0.5)

    # fig, ax = plt.subplots(1, 1)
    # x = np.fromiter((np.random.uniform() for _ in range(100)), dtype=np.float32)
    # y = np.fromiter((np.random.uniform() for _ in range(100)), dtype=np.float32)
    # ax.scatter(x, y)
    # add_watermark(ax)
#%%
def CreateAResultsFolder(filepath):
    path = Path(filepath)
    # print(path.parent.parent.parent)
    ResultsPath = os.path.join(path.parent.parent.parent, 'Results')
    print(ResultsPath)
    if not os.path.isdir(Path(ResultsPath)):
        os.mkdir(ResultsPath)
        
#%%
def PlotFile(file):
    fig,axs = plt.subplots(1, 1, figsize=(5.2, 6))
    file =  file
    csv  = np.genfromtxt(file, delimiter=",")
    DataX = csv[:,0]
    DataY = csv[:,1]
    
    
def PlotAllRaw(folder):
    listOfData = [s for s in os.listdir(path = '../Literature/'+folder) if s.endswith('RAW.csv')]
    
    for datafile in listOfData:
        fig, axs = plt.subplots(1, 1, figsize=(5.2, 3))
        file = '../Literature/{0}/{1}'.format(folder,datafile)
        csv  = np.genfromtxt(file, delimiter=",")
        DataX = csv[:,0]
        DataY = csv[:,1]
        axs.plot(DataX/1e-6, DataY, color = 'b', lw = 0.2 ,label='Data')
        axs.set_yscale('log')
        axs.grid(color='k', linestyle='-', linewidth=0.05 , which ='both')
        axs.set_title(datafile[0:-4])
        add_watermark(ax=axs,version=version)
        
#%%Pico Stuff
def PicoGen(A,T,Back,Data_t,plot=True,**kwargs):
    yi = np.zeros(Data_t.shape)
    df = nd.Derivative(FHTx, n=1)
    if plot:
        fig, axs = plt.subplots(1, 1, figsize=(5.2, 3))
    for Ai,Ti in np.array([A,T]).T:
        y = Ai*np.exp(-Data_t/Ti)
        if plot:
            # axs.plot(Data_t/1e-9,y, '--',color = 'lightblue',lw=2)#no label
            axs.plot(Data_t,y, '--',lw=2, label = '$\\tau = {}\ n s$'.format(Ti))#convert to ns
        yi += y

    # print(DataY)
    # axs.plot(DataX/1e-6, df(np.log(DataX),u,B,df0,m),'green')
    # axs.plot(DataX/1e-6, dFHT(DataX), color = 'r')

    # axs.plot(DataX/1e-6, DataY, color = 'b', lw = 0.5 ,label='Data')
    
    # axs.plot(np.ones(DataX.shape)*np.exp(ret.x[0])/1e-6,np.linspace(1e-1,2e4,DataX.shape[0]),'--',color = 'r',lw=1, label='$tau_{HT}$')
    
    if plot:            
        axs.plot(Data_t, yi   , color = 'gray', lw = 1)
        axs.plot(Data_t, yi +np.ones(Data_t.shape)*Back, color = 'k', lw = 2,label = 'PicoQuant')
        axs.plot(Data_t,     np.ones(Data_t.shape)*Back, color = 'gray', lw = 1)
        
    # if  'ret' in kwargs:
    #     ret    = kwargs.get('ret')
    #     # offset = kwargs.get('offset')
        
    #     # dfHt = lambda t: -(B   *df0   *1e-2*(1-m     )**2*(t/T             )**(B     -1)*np.exp(-(1-m     )*(t/T             )**B     ))/T
    #     dfHt = lambda t: -(ret[1]/yscale*ret[2]*(1-ret[3])**2*(t/np.exp(ret[0]))**(ret[1]-1)*np.exp(-(1-ret[3])*(t/np.exp(ret[0]))**ret[1]))/np.exp(ret[0])
    #     # htDer = dfHt(DataX)+np.ones(DataX.shape)*offset/yscale*10
    #     htDer = dfHt(DataX)+np.ones(DataX.shape)
    #     axs.plot(DataX/1e-6, htDer, color = 'r', lw = 2   ,label='d/dt HT')
        
        # axs.set_ylim(7e-1,1e5)
        if 'ylim' in kwargs:
            axs.set_ylim(ylim)
        # else:
            # axs.set_ylim(5e-1,2e4)
            # axs.set_ylim(0,250)
            
            
        # axs.set_yscale('log')
        # axs.set_xlabel('us')
        axs.set_xlabel('ns')
        axs.grid(color='k', linestyle='-', linewidth=0.05 , which ='both')
        axs.legend(loc='best')
        
        if 'filename' in kwargs:
            filename = kwargs.get('filename')
            axs.set_title(filename)
    return yi + Back


def PicoIntegrate(Data_t, DataY):
    return integrate.cumtrapz(DataY ,Data_t, initial =0)

def PicoCompare(Data_t, 
                PicoY, DataY, dHTy, 
                pdfB, excelOffset,
                filename):
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 3))
    # pdb.set_trace()
    # ylim = (7e-1,1e4)
    # ylim = (7e-1,2e2)
    # ylim = (5e-3,3e2)
    
    axs.plot(Data_t,DataY            ,color='b'         ,label='Raw Data  '   , lw=0.2)
    # axs.plot(Data_t*1e6,DataY-pdfB       ,color='gray'      ,label='Data - PicoAv', lw=0.5)
    # axs.plot(Data_t*1e6,DataY-excelOffset,color='mistyrose',label='Data - EyeAv' , lw=0.5)
    
    axs.plot(Data_t,PicoY,color='k' ,label='PicoQuant', lw=2)       ####
    axs.plot(Data_t, dHTy,color='darkred',label='HT Fit'   , lw=2)     ####
    
    
    axs.grid(color='k', linestyle='-', linewidth=0.05 , which ='both')
    # axs.legend(loc='best')
    axs.set_title(filename)
    axs.set_yscale('log')
    # axs.set_xlabel('t [ns]')
    axs.set_xlabel('t [us]')
    # axs.set_ylim(ylim)
             
    # #-----
    # fig, aks = plt.subplots(1, 1, figsize=(5.2, 3))
    
    # # aks.plot(Data_t*1e6,DataY            ,color='b'         ,label='Raw Data  '   , lw=0.5)
    # aks.plot(Data_t*1e6,DataY-pdfB       ,color='lightgray'      ,label='Data - PicoAv', lw=0.5)
    # # aks.plot(Data_t*1e6,DataY-excelOffset,color='mistyrose',label='Data - EyeAv' , lw=0.5)
    
    # aks.plot(Data_t*1e6, PicoY-pdfB       ,color='gray' ,label='PicoQuant', lw=1)
    # # aks.plot(Data_t*1e6, dHTy-excelOffset ,color='darkred',label='HT Fit'   , lw=1)
    

    # aks.grid(color='k', linestyle='-', linewidth=0.05 , which ='both')
    # aks.legend(loc='best')
    # aks.set_title(filename)
    # aks.set_yscale('log')
    # aks.set_ylim(ylim)
    
    # #-----
    # fig, aks = plt.subplots(1, 1, figsize=(5.2, 3))
    
    # # aks.plot(Data_t*1e6,DataY            ,color='b'         ,label='Raw Data  '   , lw=0.5)
    # # aks.plot(Data_t*1e6,DataY-pdfB       ,color='lightgray'      ,label='Data - PicoAv', lw=0.5)
    # aks.plot(Data_t*1e6, DataY-excelOffset,color='lightcoral',label='Data - EyeAv' , lw=0.5)
    
    # # aks.plot(Data_t*1e6, PicoY-pdfB       ,color='black' ,label='PicoQuant', lw=1)
    # aks.plot(Data_t*1e6, dHTy-excelOffset ,color='darkred',label='HT Fit'   , lw=1)
    

    # aks.grid(color='k', linestyle='-', linewidth=0.05 , which ='both')
    # aks.legend(loc='best')
    # aks.set_title(filename)
    # aks.set_yscale('log')
    # aks.set_ylim(ylim)
    
    return axs

def PicoIntCompare(Data_t, 
                   IntPicoFitBack, IntDataPicoBack, IntDataExcelBack, 
                   ret,m, offset, HTscale, filename):
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 3))
    
    # HTy = (FHTx(np.log(Data_t),*[*ret,m])+offset)/HTscale #3D
    HTy = (FHTx(np.log(Data_t*1e3),*[*ret])+offset)/HTscale #4D
    
    axs.plot(   Data_t, IntDataExcelBack,color='b',linestyle = '-' , label='$\int$ Data - B$_{HT}$' ,lw = 5)
    # axs.plot(   Data_t, IntDataPicoBack ,color='lightgray' ,linestyle = '--', label='$\int$ Data - B$_{PQ}$',lw = 3)
    
    axs.plot(   Data_t, HTy             ,color='red'   ,linestyle = '-' , label='  HT Fit'     ,lw = 2)
    axs.plot(   Data_t, IntPicoFitBack  ,color='k'      ,linestyle = '-' , label='  PQ Fit'     ,lw = 2)
    
    # axs.scatter(Data_t*1e6,DataY,color='b',marker = '+',label='Data')
    axs.grid(color='k', linestyle='-', linewidth=0.05 , which ='both')
    # axs.legend(loc='best')
    # axs.legend([dataline,picoline,htline],['Data','PicoQuant','HT Fit'])
    # axs.set_xlabel('t [ns]')
    # axs.set_xlabel('t [us]')
    axs.set_xlabel('t [s]')
    
    axs.set_title(filename)
    
    # axs.set_xlim(-2,20)
    return axs
    
# PlotPicoComparisons(Datat, DataOg, PicoY ,IntPicoFitBack, IntDataPicoBack, IntDataExcelBack, offset, yscale, excelOffset, pdfB, filename, version)
def PlotPicoComparisons(Data_t, DataOg, PicoY, 
                        IntPicoFitBack, IntDataPicoBack, IntDataExcelBack,  
                        ret, m, offset, yscale,
                        pdfB, excelOffset, 
                        filename, version,
                        savepath,
                        n_init, n_fin):
    
    u,B,df0,m = ret.x; T = np.exp(u); #4D
    # u,B,df0 = ret.x; T = np.exp(u); #3D
    
    # dfHt = lambda t: -(B*df0*(1-m)**2*(t/T)**(B-1)*np.exp(-(1-m)*(t/T)**B))/T
    # dfHt = lambda t: (B*df0*(1-m)**2*(t/T)**B*np.exp((m-1)*(t/T)**B))/t 
    # dfHt = lambda t: -B*df0*((1-m)**2*(t/T)**B*np.exp(-(1-m)*(t/T)**B))/(t*(m*np.exp(-(1-m)*(t/T)**B-1)**2))
    # dfHt = lambda t: -B*df0*((1-m)**2*(t/T)**B*np.exp((1-m)*(t/T)**B))/(t*(np.exp((1-m)*(t/T)**B-m)**2))
    dfHt = lambda t: -B*df0*(1-m)**2*(t**(B-1)/(T**B))*(np.exp((1-m)*(t/T)**B))/(np.exp((1-m)*(t/T)**B)-m)**2
    dHTy = np.sqrt((dfHt(Data_t))**2+(excelOffset)**2)
    # dHTy = dfHt(Data_t)
    
    # HTt = lambda t: fHTt(t,np.exp(ret[0]),*ret[1:]  
    # HTd = HTt(Data_t)
    # HTd = fHTt(Data_t,np.exp(u),B,df0,m)
    
    # df = nd.Derivative(fHTt(x,T,B,df0,m), n=1)
    # dHTy = np.diff(np.array([HTt(Data_t)]))
    # dHTy = np.diff(HTd)*1e6+excelOffset
    # dHTy = df(Data_t)
    # dfHt(Data_t)
    
    # print(dHTy)
    axs = PicoIntCompare(Data_t*1e-3, 
                         IntPicoFitBack, IntDataPicoBack, IntDataExcelBack,
                         ret.x,m,offset,yscale,filename)
    add_watermark(ax=axs,version=version)
    
    axs.figure.savefig(savepath+'Pico1.png',dpi=300)
    
    # axs = PicoCompare(Data_t,PicoY,DataOg,np.append(dHTy,dHTy[-1]),filename)
    axs = PicoCompare(Data_t*1e-3,
                      PicoY,DataOg,dHTy,
                      pdfB,excelOffset,
                      filename)
    add_watermark(ax=axs,version=version)
    axs.figure.savefig(savepath+'Pico2.png',dpi=300)
    
    N = len(Data_t[n_init:n_fin]) # number of datapoints
    # P = 4          # number of floating parameters
    
    Xi_PQ  = PQ_chiSquared(Data_t, PicoY, DataOg, n_init, n_fin)
    # XiR_PQ = Xi_PQ/(N-P)
    print( 'N : ' + str(N) + '; Xn_init: ' + str(Data_t[n_init]) + ' Xn_fin: ' + str(Data_t[n_fin]))# + '; P :' + str(P))
    
    print( 'Xi^2_R (PQ) = ' + str(Xi_PQ) )
    # print( 'Xi^2_R (PQ) = ' + str(XiR_PQ) )
    
    Xi_HT = PQ_chiSquared(Data_t, dHTy, DataOg, n_init, n_fin)
    # XiR_HT = Xi_HT/(N-P)
    
    print( 'Xi^2_R (HT) = ' + str(Xi_HT) )
    # print( 'Xi^2_R (HT) = ' + str(XiR_HT) )
    
    
#%% ChiSquared
def PQ_chiSquared(dataX, dataFit, dataOG, n_init, n_fin,**kwargs):
    # difDiv = (dataOG-dataFit)**2 / dataOG
    
    # numerator   = (SENoise-SEDemo)**2
    # denominator = SEDemo
    # summation   = np.sum(numerator/denominator)/N


    dataX   = dataX[n_init:n_fin]
    dataFit = dataFit[n_init:n_fin]
    dataOG  = dataOG[n_init:n_fin]

    numerator     = (dataOG - dataFit)**2
    denominator   = dataFit
    summation     = np.sum(numerator/denominator)/len(dataX)


    # print(summation)
    
    
    
    # chi = np.sum(difDiv[~np.isinf(difDiv)])
    return summation
#%% ****IN PROGRESS of Modular Integration****

# def plotOffsetXi2or3D(r,PlotM):

#     VecSize = Eig[0].size
#     Xi_i   = np.empty((VecSize,VecSize))*np.nan
#     NewSigma_i = np.empty((VecSize+PlotM,1))*np.nan
#     NewSigma2pl_i = np.copy(NewSigma_i)
#     NewSigma2mn_i = np.copy(NewSigma_i)
    
#     for ii in np.arange(0,VecSize):
#         Xi_i[:,ii] = Xi(sigmaMinSq,Eig[0][ii],Eig[1][:,ii],r)
#         NewSigma_i[ii] = newSigmaByVector(sigmaMinSq,Xi_i[:,ii],Eig[0][ii],CurvTensor)
# #         print("Sigma^2 from Axis {}:".format(ii),+NewSigma_i[ii])


#         NewSigma2pl_i[ii] = Modsigma(DataX,DataY,PlotFun,(PlotParams+Xi_i[:,ii]))**2
#         NewSigma2mn_i[ii] = Modsigma(DataX,DataY,PlotFun,(PlotParams-Xi_i[:,ii]))**2
# #         print("Check Sigma^2 from Axis {} +:".format(ii),+NewSigma2pl_i[ii])
# #         print("Check Sigma^2 from Axis {} -:".format(ii),+NewSigma2mn_i[ii])

#     ############### Error Offset Plotting ###############
# #     fig, axs = plt.subplots(VecSize+PlotM, 1, figsize=(3.8, 10/3*(VecSize+PlotM)))
#     fig, axs = plt.subplots(1,VecSize+PlotM, figsize=(3.8*(VecSize+PlotM), 3.33))
#     xx_lin = np.linspace(DataX[0]-8,DataX[-1]+7,10000) #f


#     colorList = ['blue','red']

#     for panelNum in np.arange(0,VecSize):

# #         ii = panelNum

#         #Original Dataset
#         axs[panelNum].scatter(np.exp(DataX), DataY, color = 'gray' ,marker = 'o',s = 5)  #data is black
# #         axs[panelNum].plot(np.exp(DataX), DataOrigin, color = 'k', lw = 1)
        
#         #Best fit
#         axs[panelNum].plot(np.exp(xx_lin), (PlotFun(xx_lin,*(PlotParams))+calcOffset(DataX,DataY,PlotFun,PlotParams)), color = 'g', lw = 1)


#         #Offsets from the best fit
#         axs[panelNum].plot(np.exp(xx_lin),
#                      (PlotFun(xx_lin,*(PlotParams+Xi_i[:,panelNum])))+calcOffset(DataX,DataY,PlotFun,PlotParams+Xi_i[:,panelNum]) ,
#                      '-' ,color = colorList[0],label = r'$+\xi$',lw=1)
#         axs[panelNum].plot(np.exp(xx_lin), 
#                      (PlotFun(xx_lin,*(PlotParams-Xi_i[:,panelNum])))+calcOffset(DataX,DataY,PlotFun,PlotParams-Xi_i[:,panelNum]) ,
#                      '-' ,color = colorList[1],label = r'$-\xi$',lw=1)

#         axs[panelNum].set_xlabel(r"$t\ [s]$", **label_style)
#         axs[panelNum].set_ylabel(r"Quantitiy", **label_style)

#         add_panel_label(axs[panelNum],r'({})'.format(chr(97+panelNum)),0.05,0.92,color='k')

#         # add_panel_label(axs[ii][jj],r'Input Curve ${2}$:{0:.2f} ${3}$:{1:.2f}'.format(*params,*paraName) ,1.05,0.90,color='k')
#         # add_panel_label(axs[ii][jj],r'Fit Curve     ${2}$:{0:.2f} ${3}$:{1:.2f}'.format(*ret.x,*paraName),1.05,0.80,color='g')

#         add_panel_label(axs[panelNum],r"$\vec{\xi_"+"{}".format(panelNum+1)+"}"+"=$",
#                         0.63,0.72, color='k')
# #                         0.01,0.62, color='k')
#         for kk in np.arange(0,VecSize):
#             add_panel_label(axs[panelNum],r"${}^*$".format(plotParaName[kk])+r"[{0:.02g}]".format(Xi_i[:,panelNum][kk]),
#                         0.63,0.66-kk*0.07, color='k')
# #                          0.01,0.52-kk*0.07, color='k')


#         add_panel_label(axs[panelNum],r"$ \sigma(\vec{x_{fit}}+\vec{\xi_"+"{}".format(panelNum+1)+"}"+")^2/\sigma_0^2 = {0:.2g}$".format(NewSigma2pl_i[panelNum][0]/sigmaMinSq),
#                         0.43,0.92, color=colorList[0])
# #                         0.01,0.82, color=colorList[0])
#         add_panel_label(axs[panelNum],r"$ \sigma(\vec{x_{fit}}-\vec{\xi_"+"{}".format(panelNum+1)+"}"+")^2/\sigma_0^2 = {0:.2g}$".format(NewSigma2mn_i[panelNum][0]/sigmaMinSq),
#                         0.43,0.82, color=colorList[1])
# #                         0.01,0.72, color=colorList[1])
#         add_panel_label(axs[panelNum],r"$ \lambda_"+"{}".format(panelNum+1)+"}$"+" = {0:.2f}".format(Eig[0][panelNum]),
#                         0.05,0.13, color='k')
        
# #         add_panel_label(axs[panelNum],r"$ \sigma_{\vec{x}}^2(\vec{\xi_"+"{}".format(panelNum+1)+"})/\sigma_0^2"+" = {0:.2f}$".format(NewSigma_i[panelNum][0]/sigmaMinSq),
# #                         0.05,0.03, color='k')

#         axs[panelNum].grid(color='k', linestyle='-', linewidth=.05 , which ='both')
#         axs[panelNum].set_xscale('log')
# #         axs[panelNum].set_ylim((-2.1,14.5))
        
        
#     if PlotM == 1:
#         m_pl, m_mn = deltaM(DataX, DataY, function, ret.x, r*sigmaMinSq)
#         NewSigma2pl_i[-1] = Modsigma(DataX,DataY,function,np.append(ret.x[0:3],m_pl))**2
#         NewSigma2mn_i[-1] = Modsigma(DataX,DataY,function,np.append(ret.x[0:3],m_mn))**2
#         #Original Dataset
#         axs[-1].scatter(np.exp(DataX), DataY, color = 'gray' ,marker = 'o',s = 5)  #data is black
# #         axs[-1].plot(np.exp(DataX), DataOrigin, color = 'k', lw = 1)
        
#         #Best fit
#         axs[-1].plot(np.exp(xx_lin), (PlotFun(xx_lin,*(PlotParams))+calcOffset(DataX,DataY,PlotFun,PlotParams)), color = 'g', lw = 1)


#         #Offsets from the best fit
#         axs[-1].plot(np.exp(xx_lin),
#                      (function(xx_lin,*(np.append(ret.x[0:3],m_pl))))+calcOffset(DataX,DataY,function,np.append(PlotParams,m_pl)) ,
#                      '-' ,color = colorList[0],label = r'$+m$',lw=1)
#         axs[-1].plot(np.exp(xx_lin), 
#                      (function(xx_lin,*(np.append(ret.x[0:3],m_mn))))+calcOffset(DataX,DataY,function,np.append(PlotParams,m_mn)) ,
#                      '-' ,color = colorList[1],label = r'$-m$',lw=1)
        
#         axs[-1].set_xlabel(r"$t\ [s]$", **label_style)
#         axs[-1].set_ylabel(r"Quantitiy", **label_style)

#         add_panel_label(axs[-1],r'({})'.format(chr(97+panelNum+1)),0.05,0.92,color='k')

#         add_panel_label(axs[-1],r"$ \sigma(\vec{x_{fit}}+\vec{\Delta m"+"}"+")^2/\sigma_0^2 = {0:.2f}$".format(NewSigma2pl_i[-1][0]/sigmaMinSq),
#                         0.43,0.92, color=colorList[0])
# #                         0.01,0.82, color=colorList[0])
#         add_panel_label(axs[-1],r"$ \sigma(\vec{x_{fit}}-\vec{\Delta m"+"}"+")^2/\sigma_0^2 = {0:.2f}$".format(NewSigma2mn_i[-1][0]/sigmaMinSq),
#                         0.43,0.82, color=colorList[1])
# #                         0.01,0.72, color=colorList[1])
#         add_panel_label(axs[-1],r"$\vec{\Delta m^+} ="+"{0:.2f}".format(m_pl-ret.x[-1])+"$",
#                         0.63,0.72, color='k')
# #                         0.01,0.62, color='k')
#         add_panel_label(axs[-1],r"$\vec{\Delta m^-} ="+"{0:.2f}".format(-m_mn+ret.x[-1])+"$",
#                         0.63,0.72-0.07, color='k')
# #                         0.01,0.62-0.07, color='k')
        
#         axs[-1].grid(color='k', linestyle='-', linewidth=.05 , which ='both')
#         axs[-1].set_xscale('log')
# #         axs[-1].set_ylim((-20.1,14.5))
    
#     axs[0].set_title(r"$r={0:.2f}\ [\sigma^2 = r\sigma_0^2]$   ".format(r))
#     axs[1].set_title(r'[Input] ${4}$:{0:.2f} ${5}$:{1:.2f} ${6}$:{2:.2f} ${7}$:{3:.2f}'.format(*params,*paraName),c='gray')
#     axs[2].set_title(r'[Fit]   ${4}$:{0:.2f} ${5}$:{1:.2f} ${6}$:{2:.2f} ${7}$:{3:.2f}'.format(*ret.x ,*paraName),c='g')
#     axs[3].set_title(r"$r={0:.2f}\ = 1 + (2.576)^2 / N  $ [N = {1}]    ".format(r,numberOfpoints))
# #     axs[0][1].set_title(r'$\theta_{input} = $'+'{0:.2f}'.format(1/params[1])+
# #                         r' $\theta_{fit} = $'+'{0:.2f}'.format(Theta))
#     plt.tight_layout()
#     return Xi_i
#%%
def createArtificialData(function, numberOfPoints=100, uBounds = (-10,10), params=(2.0,0.6,10,0.5), offset=0, seed=33333, noiseAmplitude=1):
    np.random.seed(seed)
    DataX      = np.linspace(*uBounds,numberOfPoints)
    noise      = np.random.normal(0,1,numberOfPoints)*noiseAmplitude
    DataOrigin = function(DataX,*params)
    DataY      = function(DataX,*params)+noise
    return DataX, DataY

#%%
#FHTx (Heavy Tail function as a function of x): this is the function that characterizes the
#curve in (natural) log-time ultimate goal is to optimize the parameters of this function
#to fit the curves 
def FHTx(x,u,B,df0,m):
#     if m == 0: #if if it at the SE limit
#         return df0*np.exp(-np.exp(B*(x-u)))
#     if m == 1: #if HT is at the AD limit
#         return ADx(x,u,B,df0)
    return df0*(1-m)/(np.exp((1-m)*np.exp(B*(x-u)))-m)

def FHTxTheta(x,u,Theta,df0,m):
    return df0*(1-m)/(np.exp((1-m)*np.exp(1/Theta*(x-u)))-m)
#%%
def fHTt(t,T,B,df0,m):
    return df0*(1-m)/(np.exp((1-m)*(t/T)**B)-m)
#%%
#ADX (Algebraic Decay function): At the limit nu->0 the floating point numbers start to 
#get very  small, which causes some computational errors and ultimately results in 
#lower calculated sigma values for waorse fit results. This function is the limit form
#of the generalized heavy tail function FHTx (above) and it helps correct for computational
#due to very small floation point math
def ADx(x,u,B,df0): # m=1
    return df0/(1+np.exp(B*(x-u)))

def FSEx(x,u,B,df0): # m=0
    return df0*np.exp(-np.exp(B*(x-u)))

def FSENorm(x,u,B):
    return 10*np.exp(-np.exp(B*(x-u)))

#----------------------------------------------------
#Lower Dimensional Heavy Tail Functions
def F_uBx(x,p1,p2): #1
    df0 = 10
    m = 0
    return FHTx(x,u=p1,B=p2,df0=df0,m=m)

def F_uDf0x(x,p1,p2): #2
    B = 0.6
    m = 0
    return FHTx(x,u=p1,B=B,df0=p2,m=m)

def F_uMx(x,p1,p2): #3
    B = 0.6
    df0 = 10
    return FHTx(x,u=p1,B=B,df0=df0,m=p2)

def F_BDf0x(x,p1,p2): #4
    u = 2.0
    m = 0
    return FHTx(x,u=u,B=p1,df0=p2,m=m)

def F_BMx(x,p1,p2): #5
    u = 2.0
    df0 = 10
    return FHTx(x,u=u,B=p1,df0=df0,m=p2)

def F_Df0Mx(x,p1,p2,): #6
    u = 2.0
    B = 0.6
    return FHTx(x,u=u,B=B,df0=p1,m=p2)

def F_uThetDf0x(x,p1,p2,p3):
    m = 1-1e-9
    # m=0.8
    # print('m is fixed at {}'.format(m))
    return FHTx(x,u=p1,B=p2,df0=p3,m=m)

def F_uThetx(x,p1,p2,df0,m):
    df0 = 10
    m = 0
    return FHTxTheta(x,u=p1,Theta=p2,df0=df0,m=m)

#----------------------------------------------------

def boundSelector(argument, uBounds, bBounds, fdeltBounds, mBounds): 
    switcher = { 
        F_uBx  :[uBounds, bBounds  ], 
        F_uDf0x:[uBounds, fdeltBounds], 
        F_uMx  :[uBounds, mBounds  ], 
        F_BDf0x:[bBounds, fdeltBounds],
        F_BMx  :[bBounds, mBounds  ],
        F_Df0Mx:[fdeltBounds, mBounds],
        F_uThetDf0x:[uBounds,bBounds,fdeltBounds],
        FHTx   :[uBounds, bBounds, fdeltBounds, mBounds]
    } 
    return switcher.get(argument, "nothing")

def constraintSelector(param, *argv):
    switcher = {
        'u'   : (np.amin(argv),np.amax(argv)),
        'b'   : (0,1),
        'fdelt': (-np.amax(argv)*10 , np.amax(argv)*10),
        'm'   : (0,1),
        }
    return switcher.get(param, "no such parameter")

def constraintBuilder(param, bound, constraint):
    try:
        idx   = constraint[0].index(param)
        if constraint[1][idx] == '>':
            bound = (constraint[2][idx],bound[1])
        elif constraint[1][idx] == '<':
            bound = (bound[0],constraint[2][idx])
        elif constraint[1][idx] == '=':
            bound    = (constraint[2][idx]-1e-9, constraint[2][idx]+1e-9)
        else:
            print('wrong constraint format!')
            sys.exit(1)
    except: 
        print('no constraint for the parameter ' + param)
    
    return bound

#----------------------------------------------------


    


#%%
#ModWeightedAverage (Modified Weighted Average Function):
#This function distrubutes the importnce of each datapoint. Since the data collectrender_mpl_tableion
#usually takes place in linear time with equally spaced times analysis in log-time has
#drastically higher denstiy of points in later times. Using the spacing of each data-point
#in log-time, this function makes sure that the weight is distributed based on the density
#of the data. Furthermore, x_w's first two values are the first value of the x array
#and similarly the last two values are the last two values of the x array. This 
#allows first and last datapoints to have half the weight of the previous and the
#last interval respectively
def ModWeightedAverage(x,F,N):
    ii=1
    summ = 0
    while(ii <= N):
        wi = weight(x,ii,N) #calculate wi:  eqn.37
        summ += wi*F[ii]    #calculate sum: eqn.38
        ii   +=1
    return summ

#%%#
def OffsetAverage(dataX,dataY):
    
    N = dataY.size
#    
#    #x_0 = x_1 and x_N+1 = x_N
#    x = np.array(dataX[0])
#    x = np.append(x,dataX)
#    x = np.append(x,dataX[-1])
    
    x = xPad(dataX)
    
    #pad y with a nan to have consistent notation as the paper (correcting for prog. lang.)
    y = np.append(np.array([np.nan]),dataY)
    
    average  = ModWeightedAverage(x,y,N)
    
    return average

def calcOffsetData(dataX,dataY,fitX,fitY,**kwargs):
    FdataAv    = OffsetAverage(dataX,dataY)
    FfitAv     = OffsetAverage(fitX,fitY)
    offset     = FdataAv-FfitAv
    return offset

def calcOffset(dataX,dataY,fun,params,**kwargs):
    FdataAv    = OffsetAverage(dataX,dataY)
    FfitAv     = OffsetAverage(dataX,fun(dataX,*params,**kwargs))
    offset     = FdataAv-FfitAv
    return offset

#%%#
def weight(x,ii,N):
    # RightBehaveIndex = 307363
    # LeftBehaveIndex  = 307364
    # if ii == RightBehaveIndex:
    #     wi = ((x[ii]-x[ii-1])/2)/(x[N]-x[1])
    # elif ii == LeftBehaveIndex:
    #     wi = ((x[ii+1]-x[ii])/2)/(x[N]-x[1])
    # else:
    #     wi = ((x[ii+1]-x[ii-1])/2)/(x[N]-x[1]) #calculate wi:  eqn.37
    # return wi
    wi = ((x[ii+1]-x[ii-1])/2)/(x[N]-x[1]) #calculate wi:  eqn.37
    return wi
def xPad(x):
    x_pad = np.concatenate((np.array([x[0]]),x,np.array([x[-1]])),axis = 0)
    return x_pad
#%%
def render_mpl_table(data, col_width=1.0, row_height=0.625, font_size=14,
                      header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                      bbox=[0, 0, 1, 1], header_columns=0,
                      ax=None, **kwargs):
        
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
        
    # pdb.set_trace()
    
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns ,cellLoc='right', **kwargs)    
    # mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, colLoc='center', cellLoc='right', **kwargs)
    # mpl_table = ax.table(cellText=data.values, colLabels=data.columns, loc='center', cellLoc='center')
    mpl_table.auto_set_column_width(col=list(range(len(data.columns)+2)))
    # pdb.set_trace()
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        # cell.set_edgecolor(edge_color)
        cell.set_edgecolor(row_colors[k[0]%len(row_colors)])
        if k[0] == 0:
            cell.set_text_props(color='w')
            cell.set_facecolor(header_color)
            cell.set_edgecolor(header_color)
            # pass
            # cell.set_facecolor(row_colors[k[0]%len(row_colors)])
            cell.cellLoc = 'center'
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
            if k[1] >= header_columns:
                cell.cellLoc = 'left'
            # cell.cellLoc = 'left'
        # cell.set_facecolor(row_colors[k[0]%len(row_colors)])
        
        
    # if  'CurvTensor' in kwargs:
    #     CurvTensor = kwargs.get('CurvTensor')
    #     # print("import Curvature")
    # else:
    #     CurvTensor = tensorByNumdiff(DataX,DataY,PlotFun,PlotParams,m = ret.x[-1])
    #     # print("calculate Curvature")
    
    return ax

#%%
#Standard standard deviation of fit from experimental values
def MeetingStdDev(dataX,dataY,fitU,fitB,fitdf0,fitM):
    N = dataX.size
    
#    #x_0 = x_1 and x_N+1 = x_N
#    x = np.array(dataX[0])
#    x = np.append(x,dataX)
#    x = np.append(x,dataX[-1])
    x = xPad(dataX)
    
    #pad y with a nan to have consistent notation as the paper (correcting for prog. lang.)
    y = np.append(np.array([np.nan]),dataY)
    
    #calculate the fit curve's y values based of predicted fit variables: u,B,df0 and m
    Ffit = FHTx(dataX,fitU,fitB,fitdf0,fitM)
    #pad yfit with a nan to have consistent notation as the paper (correcting for prog. lang.)
    yfit = np.append(np.array([np.nan]),Ffit)
    
    summ  = 0
    numer = 0
    denom = 1
    ii = 1
    while(ii<N):
        wi = weight(x,ii,N) #calculate wi:  eqn.37
        summ += (y[ii]-yfit[ii])**2*wi
        ii += 1
    
    stdDev = np.sqrt(summ/N)
    return stdDev

#%% verify curvature
def sigVerify(sigMin,curv,delta):
    return sigMin + 1/2 * delta**2 * curv

# def sigGradient(sigX,sigY):
#     return np.gradient(sigY,sigX)


    
#%%
def sigMultiParamSweep(DataX ,DataY, 
                       fitU, fitB, fitdf0, fitM,
                       uBounds   = [-5,10], bBounds = [0,1],
                       df0Bounds = [0 , 2], mBounds = [0,1], fun = FHTx):
    
    uBounds  =[fitU    -abs(fitU)  *2 ,fitU  +abs(fitU)  *2 ]
    df0Bounds=[fitdf0-abs(fitdf0)*0.5,fitdf0+abs(fitdf0)*0.5]
    
    B_x  = np.arange(0, 1.01, 0.01)
    M_x  = np.arange(0, 1.01, 0.03)
    U_x  = np.arange(uBounds[0],uBounds[1],0.01)
    D_x  = np.arange(df0Bounds[0],df0Bounds[1], 0.01)
    
    # color = plt.cm.Spectral
    color = plt.cm.CMRmap
    fig = plt.figure(figsize=plt.figaspect(50))
    fitParam = [fitU,fitB,fitdf0,fitM]
    zup = Modsigma(DataX,DataY,fun,fitParam)**2*2
    
#=============================================
# u vs. B ------------------------------------
    UB_z = np.zeros(shape=(B_x.size,U_x.size))   
    i_x = 0
    for u in U_x:
        i_y = 0
        for b in B_x:
            UB_z[i_y][i_x] = Modsigma(DataX,DataY,fun,[u,b,fitdf0,fitM])**2
            i_y += 1
        i_x += 1
    
    # Plot the surface.
    X,Y = np.meshgrid(U_x, B_x)
    ax = fig.add_subplot(6, 1, 1, projection='3d')
    ax.plot_surface(X, Y, UB_z, rstride=1, cstride=1,
                cmap=color, edgecolor='none')
    ax.set_xlabel(r"u", **label_style)
    ax.set_ylabel(r"$\beta$", **label_style)
    ax.set_zlabel(r"$\sigma^2$", **label_style)
    
# u vs. df0 ----------------------------------
    UD_z = np.zeros(shape=(D_x.size,U_x.size))
    i_x = 0
    for u in U_x:
        i_y = 0
        for d in D_x:
            UD_z[i_y][i_x] = Modsigma(DataX,DataY,fun,[u,fitB,d,fitM])**2
            i_y += 1
        i_x += 1
    
    # Plot the surface.
    X,Y = np.meshgrid(U_x, D_x)
    ax = fig.add_subplot(6, 1, 2, projection='3d')
    ax.plot_surface(X, Y, UD_z, rstride=1, cstride=1,
                cmap=color, edgecolor='none')
    ax.set_xlabel(r"u", **label_style)
    ax.set_ylabel(r"$\Delta f_0$", **label_style)
    ax.set_zlabel(r"$\sigma^2$", **label_style)
    
# u vs. m ------------------------------------
    UM_z = np.zeros(shape=(M_x.size,U_x.size))
    i_x = 0
    for u in U_x:
        i_y = 0
        for m in M_x:
            UM_z[i_y][i_x] = Modsigma(DataX,DataY,fun,[u,fitB,fitdf0,m])**2
            i_y += 1
        i_x += 1
    
    # Plot the surface.
    X,Y = np.meshgrid(U_x, M_x)
    ax = fig.add_subplot(6, 1, 3, projection='3d')
    ax.plot_surface(X, Y, UM_z, rstride=1, cstride=1,
                cmap=color, edgecolor='none')
    ax.set_xlabel(r"u", **label_style)
    ax.set_ylabel(r"m", **label_style)
    ax.set_zlabel(r"$\sigma^2$", **label_style)
    
# B vs. df0 ----------------------------------
    BD_z = np.zeros(shape=(D_x.size,B_x.size))
    i_x = 0
    for b in B_x:
        i_y = 0
        for d in D_x:
            BD_z[i_y][i_x] = Modsigma(DataX,DataY,fun,[fitU,b,d,fitM])**2
            i_y += 1
        i_x += 1
        
    # Plot the surface.
    X,Y = np.meshgrid(B_x, D_x)
    ax = fig.add_subplot(6, 1, 4, projection='3d')
    ax.plot_surface(X, Y, BD_z, rstride=1, cstride=1,
                cmap=color, edgecolor='none')
    ax.set_xlabel(r"$\beta$", **label_style)
    ax.set_ylabel(r"$\Delta f_0$", **label_style)
    ax.set_zlabel(r"$\sigma^2$", **label_style)
    
# B vs. m ------------------------------------
    BM_z = np.zeros(shape=(M_x.size,B_x.size))
    i_x = 0
    for b in B_x:
        i_y = 0
        for m in M_x:
            BM_z[i_y][i_x] = Modsigma(DataX,DataY,fun,[fitU,b,fitdf0,m])**2/1000
            i_y += 1
        i_x += 1
    
    # Plot the surface.
    X,Y = np.meshgrid(B_x, M_x)
    ax = fig.add_subplot(6, 1, 5, projection='3d')
    ax.plot_surface(X, Y, BM_z, rstride=1, cstride=1,
                cmap=color, edgecolor='none')
    
    ax.set_zlim(0,zup)
    
    ax.set_xlabel(r"$\beta$", **label_style)
    ax.set_ylabel(r"m", **label_style)
    ax.set_zlabel(r"$\sigma^2$", **label_style)
    
# df0 vs. m ----------------------------------
    DM_z = np.zeros(shape=(M_x.size,D_x.size))
    i_x = 0
    for d in D_x:
        i_y = 0
        for m in M_x:
            DM_z[i_y][i_x] = Modsigma(DataX,DataY,fun,[fitU,fitB,d,m])**2
            i_y += 1
        i_x += 1
    
    # Plot the surface.
    X,Y = np.meshgrid(D_x, M_x)
    ax = fig.add_subplot(6, 1, 6, projection='3d')
    ax.plot_surface(X, Y, DM_z, rstride=1, cstride=1,
                cmap=color, edgecolor='none')
    ax.set_xlabel(r"$\Delta f_0$", **label_style)
    ax.set_ylabel(r"m", **label_style)
    ax.set_zlabel(r"$\sigma^2$", **label_style)
    
    plt.tight_layout()
    plt.show()
    
#%%
def deltaM(dataX, dataY, fun, ret, r):
    resolution = 0.001
    param = ret.x
    rSigminSq = r*ret.fun**2
    m_test = param[3]  
    while (Modsigma(dataX, dataY, fun, np.append(param[0:3], m_test))**2 <= rSigminSq) and (m_test<= 1-1e-6):
        m_test += resolution
    m_hi = m_test-resolution
    
    m_test = param[3]
    while (Modsigma(dataX, dataY, fun, np.append(param[0:3], m_test))**2 <= rSigminSq) and (m_test>= 0):
        m_test -= resolution
    m_lo = m_test+resolution
    
    return (m_hi, m_lo)

def deltaPar3d(Hessian,r,sigmaMinSq):
    # pdb.set_trace()
    dim = Hessian.shape[0]
    Diag = np.diagonal(Hessian,offset=0)
    OffD = np.diagonal(np.hstack((Hessian,Hessian)),offset=1)
    delta = np.empty(dim)
    
#     if dim == 2:
#         print("dim 2")
#     elif dim == 3:
#         print("dim 3")
    if dim == 3:
#         print("dim 3")
        
        
#         for ii in np.arange(0,3):
#             delta[ii] = np.sqrt(1/
#              (Diag[ii]+
#              (2*np.prod(OffD)-
#              Diag[np.mod(ii+1,dim)]*OffD[np.mod(ii+2,dim)]**2-
#              Diag[np.mod(ii+2,dim)]*OffD[np.mod(ii+1,dim)]**2)/
#             (Diag[np.mod(ii+1,dim)]*Diag[np.mod(ii+2,dim)]-OffD[np.mod(ii+1,dim)])))
        a,b,c = Diag
        d,e,f = OffD
        delta[0] = np.sqrt(2*(r-1)*sigmaMinSq/
                           (a+(2*d*e*f-b*f**2-c*d**2)/(b*c-e**2)))
        delta[1] = np.sqrt(2*(r-1)*sigmaMinSq/
                           (b+(2*d*e*f-c*d**2-a*e**2)/(c*a-f**2)))
        delta[2] = np.sqrt(2*(r-1)*sigmaMinSq/
                           (c+(2*d*e*f-a*e**2-b*f**2)/(a*b-d**2)))
    return delta
        
    
def tensorByNumdiff(dataX,dataY,fun,fitParam,**kwargs):
    f = lambda x: Modsigma(dataX,dataY,fun,x,**kwargs)**2
    H = nd.Hessian(f)((fitParam[:]))
    return H

def round_to_n(x,n):
   return round(x, -int(np.floor(np.log10(abs(x))))+n-1)
#%%
def MakeResultsTable(DataX,DataY,function,ret,r,y_unit,title, **kwargs):
    # function = FHTx
    retTheta = np.copy(ret.x)
    Theta = 1/retTheta[1]
    retTheta[1] = Theta
    PlotParams = retTheta[0:3]
    PlotFun = function
    
    # pdb.set_trace()
    
    if  'CurvTensor' in kwargs:
        CurvTensor = kwargs.get('CurvTensor')
        # print("import Curvature")
    else:
        CurvTensor = tensorByNumdiff(DataX,DataY,PlotFun,PlotParams,m = ret.x[-1])
        # print("calculate Curvature")
    
    if 'm' in kwargs:
        m = kwargs.get('m')
        
    Eig = LA.eig(CurvTensor)
    sigmaMinSq = ret.fun**2
    
    # pdb.set_trace()
    
    # delta = deltaPar3d(CurvTensor,r,ret.fun**2)
    
    #--------------------deltaPar3d explicit!
    Hessian = CurvTensor
    dim = Hessian.shape[0]
    Diag = np.diagonal(Hessian,offset=0)
    OffD = np.diagonal(np.hstack((Hessian,Hessian)),offset=1)
    delta = np.empty(dim)
    
    if dim == 3:
        a,b,c = Diag
        d,e,f = OffD
        delta[0] = np.sqrt(2*(r-1)*sigmaMinSq/
                           (a+(2*d*e*f-b*f**2-c*d**2)/(b*c-e**2)))
        delta[1] = np.sqrt(2*(r-1)*sigmaMinSq/
                           (b+(2*d*e*f-c*d**2-a*e**2)/(c*a-f**2)))
        delta[2] = np.sqrt(2*(r-1)*sigmaMinSq/
                           (c+(2*d*e*f-a*e**2-b*f**2)/(a*b-d**2)))
        
    #--------------------
    offset = calcOffset(DataX,DataY,function,ret.x)

    # yscale = 1
    #u    +/-
    u_hi, u_lo = ret.x[0]+delta[0], ret.x[0]-delta[0]
    #tau  +/-
    # tScale = 1
    tScale = 1
    tUnit  = 's'
    t_hi, t_lo = np.exp(u_hi), np.exp(u_lo)
    if (t_hi < 1e-7) or (t_lo<1e-7):
        tScale = 1e9
        tUnit  = r'$ns$'
    elif (t_hi < 1e-3) or (t_lo<1e-3):
        tScale = 1e6
        tUnit  = r'$\mu s$'
    elif (t_hi > 1e6) or (t_lo>1e6):
        tScale = 1e-5
        tUnit  = r'$10^5\ s$'
    elif (t_hi > 1e4) or (t_lo>1e4):
        tScale = 1e-3
        tUnit  = r'$10^3\ s$'

    sigfigs = [2,2,2,3]
    
    
    #Theta +/-
    T_hi, T_lo = retTheta[1]+delta[1], retTheta[1]-delta[1]

    #beta +/-
    b_lo, b_hi = 1/T_hi, 1/T_lo

    #fdelt +/-
    f_hi, f_lo = ret.x[2]+delta[2], ret.x[2]-delta[2]

    #offset +/-
    o_hi, o_lo = offset+delta[2], offset-delta[2]

    #f_0 +/-
    f0_hi, f0_lo = f_hi+offset, f_lo+offset

    #m    +/-
    if function == FHTx:
        m_hi, m_lo = deltaM(DataX, DataY, function, ret, r)
        mVal = '{0:.{1}f}'.format(ret.x[3]       ,sigfigs[3])
        m = ret.x[3]
    else:
        m_hi, m_lo = 9,9
        mVal = '{0:.{1}f}'.format(9       ,sigfigs[3])
        
    if abs(m_hi)>1: m_hi = 1
    if abs(m_lo)<0: m_lo = 0
    
    allDelt = np.append(delta,m_hi)
    sigfigs = (np.ceil(abs(np.log10(allDelt)))).astype(int)
               
    sigfig_t = (np.ceil(abs(np.log10((t_hi-t_lo)/2*tScale)))-abs(np.log10(tScale))+1).astype(int)
    # pdb.set_trace()
    # sigfig_t = 3
    
    sigfigs[0] = 2
    sigfigs[1] = 2
    sigfigs[3] = 3
    
    sigfigs = [2,2,6,2]
    
    sigfigs = [1,1,1,1]
    #Formatted Values
    uVal = '{0:.{1}f}'.format(ret.x[0]       ,sigfigs[0])
    tVal = '{0}'.format(round_to_n(np.exp(ret.x[0])*tScale,2))

    TVal = '{0:.{1}f}'.format(1/ret.x[1]     ,sigfigs[1])
    bVal = '{0:.{1}f}'.format(ret.x[1]       ,sigfigs[1])

    fVal = '{0:.{1}f}'.format(ret.x[2]       ,sigfigs[2])
    oVal = '{0:.{1}f}'.format(offset         ,sigfigs[2])
    f0Val= '{0:.{1}f}'.format(ret.x[2]+offset,sigfigs[2])
    # f0Val = ret.x[2]+offset
    
    # fVal = "%1.{}e".format(3)%ret.x[2]
    # oVal = "%1.{}e".format(3)%offset
    # f0Val= "%1.{}e".format(3)%f0Val
    
    


    #Formatted Hi-Lo
    uHiLo = '{0:.{1}f}'.format(u_hi,sigfigs[0]) ,'{0:.{1}f}'.format(u_lo,sigfigs[0])
    tHiLo = '{0}'.format(round_to_n(t_hi*tScale,2)),'{0}'.format(round_to_n(t_lo*tScale,2))

    THiLo = '{0:.{1}f}'.format(T_hi,sigfigs[1]) ,'{0:.{1}f}'.format(T_lo,sigfigs[1])
    bHiLo = '{0:.{1}f}'.format(b_hi,sigfigs[1]) ,'{0:.{1}f}'.format(b_lo,sigfigs[1])

    fHiLo = '{0:.{1}f}'.format(f_hi,sigfigs[2]) ,'{0:.{1}f}'.format(f_lo,sigfigs[2])
    oHiLo = '{0:.{1}f}'.format(o_hi,sigfigs[2]) ,'{0:.{1}f}'.format(o_lo,sigfigs[2])
    f0HiLo= '{0:.{1}f}'.format(f0_hi,sigfigs[2]),'{0:.{1}f}'.format(f0_lo,sigfigs[2])

    mHiLo = '{0:.{1}f}'.format(m_hi,sigfigs[3]) ,'{0:.{1}f}'.format(m_lo,sigfigs[3])

    #Formatted Physical Parameter Table
    reslts = pd.DataFrame()
    reslts[' ']         = [
        r'$\tau$ ='   ,
        r'$\beta$ ='  ,
        r'$m$ ='      ,       
        r'$f_0$ ='    ,
        r'$f_{\Delta}$ =',
        r'$f_\infty$ =']
    
    reslts['  ']     = [
        ' {} '.format(tVal),
        ' {} '.format(bVal),
        ' {} '.format(mVal),
        ' {} '.format(f0Val),
        ' {} '.format(fVal),
        ' {} '.format(oVal)]
        
        
    reslts[title]   = [
        ' [{} , {}] '.format( tHiLo[1] , tHiLo[0] ),
        ' [{} , {}] '.format( bHiLo[1] , bHiLo[0] ),
        ' [{} , {}] '.format( mHiLo[1] , mHiLo[0] ),
        ' [{} , {}] '.format( f0HiLo[1], f0HiLo[0]),
        ' [{} , {}] '.format( fHiLo[1] , fHiLo[0] ), 
        ' [{} , {}] '.format( oHiLo[1] , oHiLo[0] ) ]
    
    reslts['    '] =[
        tUnit,
        '',
        '',
        y_unit,
        y_unit,
        y_unit]#unit
    
    #Formatted Symmetric Parameter Table
    debug = pd.DataFrame()
    debug['r = {0:.2f}'.format(r)]         = [
        '$u$' ,
        '$T$' ,
        r'$f_{\Delta}$',
        '$\sigma_0^2$']
    #     r'$f_\infty$']
    debug['Value']         = [
        uVal, 
        TVal,
        fVal,
        "{0:.3g}".format(sigmaMinSq)
    ]
    #     oVal]
    debug[' $\pm \Delta$'] = [
        "{0:.3f}".format(delta[0]), 
        "{0:.3f}".format(delta[1]),
        "{0:.3f}".format(delta[2]),
        " "]
    #     oDel]

#     fig, axs = plt.subplots(1,3, figsize=(3*3.3, 2.5))
    # fig, axs = plt.subplots(1,1, figsize=(3.5, 2.5))
    # for ii in axs:
    #     # no axes
    #     ii.xaxis.set_visible(False)
    #     ii.yaxis.set_visible(False)
    #     # no frame
    #     ii.set_frame_on(False)
    
    # # no axes
    # axs.xaxis.set_visible(False)
    # axs.yaxis.set_visible(False)
    # # no frame
    # axs.set_frame_on(False)


    # #Create The Visual table
    # render_mpl_table(reslts, header_columns=0, col_width=1.4, row_height=0.425, ax=None)    
    # render_mpl_table(debug , header_columns=0, col_width=1.4, row_height=0.425, ax=None)
    # 
    
    #Console Print Format
    # uConf     ='u    = {0:0.4f} [{1:.4f}, {2:.4f}] '.format(       ret.x[0]        ,u_lo,u_hi)
    # tConf     ='tau  = {0:0.4g} [{1:.4g}, {2:.4g}] seconds'.format(np.exp(ret.x[0]),t_lo,t_hi)
    # betaConf  ='beta = {0:0.4g} [{1:.4g}, {2:.4g}] '.format(       ret.x[1]        ,b_lo ,b_hi)
    # fdeltConf ='fdelt = {0:0.4g} [{1:.4g}, {2:.4g}] '.format(      ret.x[2]        ,f_lo ,f_hi)  + y_unit
    # f0Conf    ='f0   = {0:0.4g} [{1:.4g}, {2:.4g}] '.format(       ret.x[2]+offset ,f0_lo,f0_hi) + y_unit
    # oConf     ='f_inf= {0:0.4g} [{1:.4g}, {2:.4g}] '.format(       offset          ,o_lo ,o_hi)  + y_unit
    # mConf     ='m    = {0:0.5g} [{1:.5g}, {2:.5g}] '.format(       m               ,m_lo ,m_hi)
    # sigmaPrint='sigma_0^2 = {}'.format(sigmaMinSq)
    
    uConf     ='u    = {0:0.4f} '.format(       ret.x[0]         ,u_lo,u_hi)
    tConf     ='tau  = {0:0.4g}  seconds'.format(np.exp(ret.x[0]),t_lo,t_hi)
    betaConf  ='beta = {0:0.4g} '.format(       ret.x[1]        ,b_lo ,b_hi)
    fdeltConf ='fdelt= {0:0.4g} '.format(      ret.x[2]         ,f_lo ,f_hi)  + y_unit
    f0Conf    ='f0   = {0:0.4g} '.format(       ret.x[2]+offset ,f0_lo,f0_hi) + y_unit
    oConf     ='f_inf= {0:0.4g} '.format(       offset          ,o_lo ,o_hi)  + y_unit
    mConf     ='m    = {0:0.5g} '.format(       m               ,m_lo ,m_hi)
    sigmaPrint='sigma_0^2 = {}'.format(sigmaMinSq)

    #Print Results in the console
    print('=========================================')
    print(uConf)
    print(tConf)
    print(betaConf)
    print(f0Conf)
    print(fdeltConf)
    print(oConf)
    print(mConf)
    print('-----------------------------------------')
    print(sigmaPrint)
    print('=========================================')
    
    plt.tight_layout()
    
    b = np.asarray([['Parameter','Values'                  ],#,'lo','hi'],
                   ['u'        ,'{0:.{1}g}'.format(ret.x[0],5)],#,u_lo,u_hi],
                   ['beta'     ,'{0:.{1}g}'.format(ret.x[1],5)],#,b_lo,b_hi],
                   ['fdelt'    ,'{0:.{1}g}'.format(ret.x[2],5)],#,f_lo,f_hi],
                   ['m'        ,'{0:.{1}g}'.format(m+1     ,5)],#,m_lo,m_hi],
                   ['sigma'    ,'{0:.2g}'.format(ret.fun)     ],# ,'',''],
                   ['offset'   ,'{0:.5f}'.format(offset)      ]]# ,'','']]
                   ,dtype = 'object')
    
#     CompleteResult = pd.DataFrame()
    
    return b, delta, reslts, debug

    
#%%
#Modified Sigma (standard deviation): 
#classical standard deviation calculation with weights coming from the ModWeightedAverage function

def Modsigma(dataX,dataY,fun,fitParam,**kwargs):
    
    N = dataY.size
    
    #x_0 = x_1 and x_N+1 = x_N
    x = np.array(dataX[0])
    x = np.append(x,dataX)
    x = np.append(x,dataX[-1])
    
    #pad y with a nan to have consistent notation as the paper (correcting for prog. lang.)
    y = np.append(np.array([np.nan]),dataY)
    
    #calculate the fit curve's y values based of predicted fit : u,B,df0 and m
    offset = calcOffset(dataX,dataY,fun,fitParam,**kwargs)
    Ffit = fun(dataX,*fitParam,**kwargs)+offset
    #pad yfit with a nan to have consistent notation as the paper (correcting for prog. lang.)

    yfit = np.append(np.array([np.nan]),Ffit)

    #calculate Fav(xi)
    FdataAv  = ModWeightedAverage(x,y,N)
    #calculate Fav_fit(xi)
    FfitAv   = ModWeightedAverage(x,yfit,N)
    
    summ = 0
    ii = 1
    while(ii <= N):         #calculate the sum in eq.39
        wi = weight(x,ii,N) #calculate wi:  eqn.37
        summ += wi*((y[ii]-FdataAv)-(yfit[ii]-FfitAv))**2 
        ii   += 1
    sigma = np.sqrt(summ)
    return sigma

def ModsigmaData(dataX,dataY,fitX,fitY,**kwargs):
    plt.plot(dataX,dataY,fitX,fitY,'--')
    
    N = dataY.size
    
    f = interp1d(fitX, fitY, kind= 'cubic')
    xnew = np.linspace(fitX[0], fitX[-1], num = N, endpoint=True)
    fitX,fitY = xnew,f(xnew);
    
    #x_0 = x_1 and x_N+1 = x_N
    x = np.array(dataX[0])
    x = np.append(x,dataX)
    x = np.append(x,dataX[-1])
    
    #pad y with a nan to have consistent notation as the paper (correcting for prog. lang.)
    y = np.append(np.array([np.nan]),dataY)
    
    #calculate the fit curve's y values based of predicted fit : u,B,df0 and m        
    offset = calcOffsetData(dataX,dataY,fitX,fitY,**kwargs)
    
    #pad yfit with a nan to have consistent notation as the paper (correcting for prog. lang.)
    fitY = np.append(np.array([np.nan]),fitY)

    #calculate Fav(xi)
    FdataAv  = ModWeightedAverage(x,y,N)
    #calculate Fav_fit(xi)
    FfitAv   = ModWeightedAverage(x,fitY,N)
    
    summ = 0
    ii = 1
    while(ii <= N):         #calculate the sum in eq.39
        wi = weight(x,ii,N) #calculate wi:  eqn.37
        summ += wi*((y[ii]-FdataAv)-(fitY[ii]-FfitAv))**2 
        ii   += 1
    sigma = np.sqrt(summ)
    return sigma
#%%
# def EigenAnalysis(EigenTuple, sigmaMin):
#     EigenValuesSq  = EigenTuple[0]**2
#     EigenVectorsSq = EigenTuple[1]**2
    
#     xVect = sigmaMin*np.sqrt(np.matmul(2/EigenValuesSq,EigenVectorsSq))
#     return xVect
    
#%%
def EigenAnalysisVerify(EigenTuple,sigmaMin):
    EigenValuesSq  = EigenTuple[0]**2
    EigenVectorsSq = EigenTuple[1]**2
    
    xVect = np.zeros((1,4))[0]
    for jj in np.arange(0,4):
        for ii in np.arange(0,4):
            xVect[jj] += 2/EigenValuesSq[ii] * EigenVectorsSq[ii,jj]
             
    xVect = sigmaMin * np.sqrt(xVect)
    return xVect

def newSigmaByVector(sigmaMinSq,Vector,Value,CurvTensor):
    return sigmaMinSq+1/2*np.matmul(Vector.T,np.matmul(CurvTensor,Vector))

def Xi(sigmaMinSq,EigValue,EigVector,r=2):
    return np.sqrt(2*(r-1)*sigmaMinSq/EigValue)*EigVector

#%%
#ModSigmaAD:
#Same as the Modsigma above, but it uses the ADx function instead of FHTx function
#this also means there is one less optimization parameter, which is nu = 0 (m=1)
#def ModsigmaAD(dataX,dataY,fitU,fitB,fitdf0):
#    
#    N = dataX.size
#    b = np.array(dataX[0])
#    b = np.append(b,dataX)
#    b = np.append(b,dataX[-1])
#    
#    x = b
#    dataY = np.append(np.array([np.nan]),dataY)
#    
#    Ffit = ADx(dataX,fitU,fitB,fitdf0)
#    FdataAv  = ModWeightedAverage(dataX,dataY)
#    FfitAv   = ModWeightedAverage(dataX,Ffit)
#    
#    summ = 0
#    numer = 0
#    denom = 1
#    ii = 1
#    while(ii < N):
#        wi = (x[ii+1]-x[ii-1])/(2*(x[N]-x[1]))
#        summ += wi*((dataY[ii]-FdataAv)-(Ffit[ii]-FfitAv))**2
#        ii   += 1
#    numer = summ
#    denom = np.abs(dataX[N]-dataX[0])
#    return np.sqrt(numer/denom)


#%%
#Residual Modified Sigma:
#Serves the save function of residual calculation,but it does not calculate the residual 
#difference from the fit. Instead it leaves it up to Modsigma function and effectively 
# acts as a carrier function for passing parameters into the optimization algorithm
def residualModSig(p, x, y, fun):
    return Modsigma(x,y,fun,p)

# #%%
# #Residual Modified Sigma for ADx:
# #same function as the residualModSig above, it optimizes ADx instead of FHTx
# def residualModSigAD(p, x, y):
#     return ModsigmaAD(x,y,*p)
#%%
#EXPERIMENTAL!: Determines the confidence level in the quality of the fit
def Confidence(sigmaNu,sigmaAD,sigmaSE):
    return 1-(sigmaNu)/np.sqrt((sigmaAD**2+sigmaSE**2)/2)
#%%
#Easy funcion to add text into the figure
def add_panel_label(ax, s, x=0.05, y=0.08, color='k'):
    ax.text(x, y, s, transform=ax.transAxes, **default_text_style, color = color)

#%%
#The Actual Fit Function


callback_num = 0
def callbackF(x, f, context):
    global callback_num
    print('\n num:', callback_num)
    print('x:', x)
    print('f:', f)
    print('context:', context)
    callback_num += 1
    
    
def heavyTailFit(x, y, fun, pBounds, maxiterations = 2500, initialTemp = 1000):
    #x: Array-like
    #y: Array-like corresponding to and must have the same length as x
    #fun: function the fit is being applied to
    #pBounds: parameter search Boundaries format -tuple array...[(p0_min,p0_max),(p1_min,p1_max),etc...]
    
    # if 'mfix' in kwargs:
    #     mval = kwargs.get('mfix')
    # print('# iter = {}'.format(maxiterations))
    # print('bounds = {}'.format(pBounds))
    NumParams = np.shape(pBounds)[0]
    p_0 = np.empty([1,NumParams])[0]
    
    #Create a random initial condition within boundaries
    for ii in np.arange(1,NumParams):
        Bound = np.array(pBounds[ii])
        TenPower = 2 - np.int(np.floor(np.log10(Bound[1])))
        Bound = np.round(Bound *10**TenPower)
        p_0[ii] = float(random.randrange(Bound[0], Bound[1], 1)/10**TenPower)
        
    p_0[0] = (pBounds[0][0]+pBounds[0][1])/2
    #Optimization Settings
    accept = -5e4 #probability of acceptance: The lower the smaller the probability of acceptance. Default value is -5.0 with a range (-1e4, -5].

    #ACTUAL FIT IS HAPPENING HERE#
    ret = dual_annealing(residualModSig, x0 = p_0,
                           initial_temp = initialTemp,
                           args = (x, y, fun),
                           bounds = (pBounds),
                           maxiter=maxiterations,
                           accept = accept)#,
                           # callback=callbackF)
    
    return ret