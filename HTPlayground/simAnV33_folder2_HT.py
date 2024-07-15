"""
Copyright c 2021 by Northwestern University. All Rights Reserved.

@author: Can Aygen
"""
#%%#Import the Functions
#Import the Functions
#********************#
from FunDefV39_2 import *
#********************#
SE = 0;
AD = 1-1e-9;

#%% 
# pdb.set_trace()
#%%
#File Import
#Initialization Constants
version     = '33P39_f '
save        =  True
save        =  False
interpolateData = True
interpolateData = False
savelabel = str(date.today())+'11d2_'

RawOnly = True
RawOnly = False

PlotRaw     = True#False
PlotRaw = False
PlotFitFast = False

LinearPlot = False #True
ForceRecalculation = False #if the previous saved if is not satifactory set this to True to re-do the fit
ForceRecalculation = True
yscale = 1; xscale = 1
quantity = 'f'
constraint = np.nan
dataOffset = 0;
topLabel = ' '; bottomLabel = ' ';
# cansFolder = "~/Google Drive/GraysonLab/AmorhousOxideCharacterization/Manuscript w: Jiajun"
HTDataColor = 'lightcoral'; HTFitColor = 'darkred'
try:
    del yrange
    del xrange
except:
    pass
# %%

############### ############### ############### ############### ###############
#Example of File Import

#Muenster Data
# PQ_Mode = True
# folder = 'Muenster/StefanBuss/Grayson_data/csvPtCl'
# filename ='PtCl_77K_LT'; bottomLabel = 'Sbu'; quantity = ' counts '


topLabel = '{\ }'; unit = ''; timeunit = 's'
# folder =''
# folder ='../../Non-Int-Python-NB'
# folder = 'Muenster/StefanBuss/Grayson_data/csvPtCl@Zif8NoShift'

# filename = 'PtCl_Zif8_air1'  ;  bottomLabel = ' Zif8air   '; quantity = ' counts '; pdfA = [180.13,845.9,1093.1]; pdfT = [22.648e-6,6.146e-6,1.641e-6 ]; pdfB = 1.299
# filename = 'PtCl_Zif8_degas1';  bottomLabel = ' Zif8degas '; quantity = ' counts '; pdfA = [628.0,2009.8,2407  ]; pdfT = [17.286e-6,6.262e-6,1.5153e-6]; pdfB = 0.871; logFlatB = pdfB;

# filename = 'dataAllZSII'; 
# filename = 'DarkdataAllZSII';
# filename = 'BGPDS-DTDA_160C.csv'
# filename = 'out.csv'

filename = 'Sil_t_DCMMeOH_496_50_0.csv'
# filename = 'JL01-JL02_DCMMeOH_487_5000_0.csv'

# filename = 'Sil_t_DCMMeOH_496_50_0_FULL_RAW.dat.csv'
# filename = '../Sil_t_DCMMeOH_496_5000_0.csv'


# filename = '../JL01-JL02_DCMMeOH_487_5000_0.csv'

# filename = '../JL01-JL02_DCMMeOH_487_50_0.csv'



# bottomLabel = ' Pt-tBu-AmPt 280K 510 P  ' 
# filename = 'Pt-tBu-AmP_t_280K_510_P'
# filename = '20211127_row=5_col=2_relaxation_5hr_c1p3_truncate_60s'
unit = ' cts '                ### Axis label UNITS for table
# savelabe =  'm0'
# yscale = -1/170230;
yscale = 1;
xscale = 1;

#%%File Import
import pandas as pd
import copy
# file = '{0}/{1}'.format(folder,filename)
file = filename
# file = '../Literature/{0}/{1}'.format(folder,filename)  ### Replace "Literature" with subfolder name, if desired
# dataOffset = 117
# dataOffset = 1
dataOffset = 0

#%%File Import
# file = '../Literature/{0}/{1}.csv'.format(folder,filename)
# file = '../Literature/{0}/{1}'.format(folder,filename)
# dataOffset = 117
# dataOffset = 3
# csv  = np.genfromtxt(file, delimiter=",")
# csv  = np.genfromtxt(file, delimiter=",", skip_header = 2 + dataOffset)

# csv  = np.genfromtxt(file, delimiter="\t", skip_header = 2 + dataOffset)
# csv  = np.genfromtxt(file, delimiter="\t", skip_header = 1)

csv = pd.read_csv(file,header = None)
# csv = pd.read_csv(file, delimiter="\t")
# csv = csv.tail(-1)
# csv = csv[~np.isnan(csv).any(axis=1)] # removes rows containing NaN from the data
print('Analyzing ' + file)

# %%Data To Variable

DataX = np.log(np.array((csv.iloc[:,0])*xscale)) #Logtime
DataX =        np.array((csv.iloc[:,0])*xscale) #lintime
# DataX = np.array(np.log(csv.timeAll))
DataY =          np.array(csv.iloc[:,1]*yscale)
# DataY = np.array(csv.Vaa_x)/2e-6*1.4301
    
DataX,DataY = DataX[np.logical_not(np.isnan(DataX))], DataY[np.logical_not(np.isnan(DataX))] # removes rows containing NaN from the data
DataX,DataY = DataX[np.logical_not(np.isinf(DataX))], DataY[np.logical_not(np.isinf(DataX))] # removes rows containing inf from the data
# DataX = csv[:,0][37:]
# DataY = csv[:,1][37:]
# pdb.set_trace()
print('data imported')

#%%Interpolation
if interpolateData:
    startInterpAt = 1000
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 3.0))
    f = interp1d(DataX[startInterpAt:], DataY[startInterpAt:], kind='linear')
    xnew = np.linspace(DataX[startInterpAt], DataX[-1], num=1000, endpoint=True)
    
    plt.plot(DataX, DataY, 'o', np.append(DataX[:startInterpAt],xnew), np.append(DataY[:startInterpAt],f(xnew)), 'x')
    
    DataX,DataY = np.append(DataX[:startInterpAt],xnew),np.append(DataY[:startInterpAt],f(xnew));
    
    # DataX,DataY = DataX[~np.isnan(DataX).any(axis=0)], DataY[~np.isnan(DataX).any(axis=0)] # removes rows containing NaN from the data
    # plt.legend(['data', 'cubic'], loc='best')
    plt.show()

#%%Yscaling
# pdb.set_trace()
# yscale = 10**-np.floor(np.log10(np.amax(DataY)))
# DataY = DataY * yscale

# if yscale != 1:
#     unit = '$10^{' + '{}'.format(int(-np.log10(yscale)))+ '}$ ' + unit
#     try:
#         comParams[2] *= yscale
#         savelabel = savelabel + 'Comp'
#         compare = True
#     except:
#         pass
#%% Import the fit parameters if there is a previous successful fit for this data with this version of the code
# path = pathlib.Path(file[:-4]+'ResultV{}{}.csv'.format(version,savelabel))
if save:
    ResultsFolderPath = pathlib.Path(file[:-4]+'Result{}.csv'.format(savelabel)[:-4])
    ResultFolderExist = ResultsFolderPath.is_dir()
    if not ResultFolderExist:
        ResultsFolderPath.mkdir()
    
    path = pathlib.Path(str(ResultsFolderPath)+'/'+filename[:-4]+'Result{}.csv'.format(savelabel))
    try:
        # path
        FitExists = path.is_file()
    except:
        FitExists = path.is_file()
    
#
############### ############### ############### ############### ###############

# #Artificial Data Generator
# topLabel = ' {Artificial}'; timeunit = ' '; 
# bottomLabel = '10\% noise'
# unit =' '
# DataX, DataY  = createArtificialData(FHTx, noiseAmplitude = 1)

#%% Plt Raw Data pre-fit
#-----------------------------------------------------------------------------
#Plot the Raw Data
if PlotRaw: 
    fig, axs = plt.subplots(2, 1, figsize=(3.2, 6.3))
    axs[0].scatter(DataX[1:], DataY[1:], color = 'black' ,label = ' ' ,marker = 'o',s = 5)
    axs[1].scatter(np.exp(DataX), DataY, color = 'black' ,label = ' ' ,marker = 'o',s = 5)
    axs[0].set_xlabel(r"$x$", **label_style)
    axs[1].set_xlabel(r"$t$", **label_style)
    axs[0].set_ylabel(r"Quantitiy", **label_style)
    axs[1].set_ylabel(r"Quantitiy", **label_style)
    axs[0].grid(color='k', linestyle='-', linewidth=0.03 , which ='both')
    axs[1].grid(color='k', linestyle='-', linewidth=0.03 , which ='both')
    axs[0].set_title('Raw ln(t)')
    axs[1].set_title('Raw t')
    axs[1].set_xscale('log')
    # xlimpow = np.array([-7,10])
    # axs.set_xticks(10**xtickpow,minor = False)
    # axs.set_xlim(np.exp(DataX.min()), np.exp(DataX.max()))
    # axs.set_xlim(np.exp(DataX.min()), np.exp(DataX.max()))
    plt.tight_layout()
    print('N = {}'.format(DataX.shape[0]))
    
    
############### ############### ############### ############### ###############

#%%
if RawOnly:
    sys.exit(0)

# pdb.set_trace()
#%%############ ############### ############### ############### ###############
############### Fitting ###############

paraName=['u','\\beta','\\bar{f}','m']; plotParaName =['u','T','\\bar{f}','m']
# function = derFHTt; print('function: FHTx (4D Solver)')
function = FHTx;
# function = BiexpFun;
# function = ADx; print('function: FHTx (4D Solver)')
# function = F_uThetDf0x; print('function: F_uThetDf0x (3D Solver)')
#boundaries parameter space

# constraint = (['u','fdelt'],['>','<'],[5,3])
# constraint = (['m','fdelt'],['>','>'],[1-9e-3,-50]); savelabel = savelabel + 'm_one2';
# constraint = (['m'],['<'],[1e-3])  ; savelabel = savelabel + 'm_zero'
# constraint = (['u','u'],['>','<'],[3,11])
# constraint = (['fdelt','m'],['>','<'],[-50,1e-3])
# constraint = (['fdelt'],['>'],[-50])

# print(function)
# savelabel = savelabel + 'm_free'



if  ForceRecalculation or not FitExists:
    # uBounds    = constraintBuilder('u'   , constraintSelector('u'  , DataX), constraint)
    # # uBounds = (-3,8)
    # bBounds    = constraintBuilder('b'   , constraintSelector('b'  ,np.nan), constraint) 
    # fdeltBounds= constraintBuilder('fdelt', constraintSelector('fdelt',DataY), constraint)
    # # fdeltBounds= (-2,1)
    # # fdeltBounds= (-5000,5000)
    # # fdeltBounds = (-0.1, 0.1)
    # mBounds    = constraintBuilder('m'   , constraintSelector('m'  ,np.nan), constraint) 
    # # mBounds    = (0,1e-3)
    # # mBounds    = (1-1e-2 ,1)
    # # funBounds  = boundSelector(function, uBounds, bBounds, fdeltBounds, mBounds)
    # # funBounds  = boundSelector[uBounds, bBounds, df0Bounds, mBounds]
    # funBounds = [uBounds,bBounds,(-5e12,10),mBounds]
    TBounds = (DataX[0] ,  DataX[-1]*10)
    ABounds = (min(DataY), max(DataY))
    BBounds = (0, 1e3)
    funBounds = [TBounds, ABounds, TBounds, ABounds, BBounds]#biexp
    
    
    BBounds = (0, 1)
    df0bounds = (-2e4,2e4)
    mBounds = (0,1)
    bBounds = (0,1e3)
    funBounds = [TBounds, BBounds, df0bounds, mBounds, bBounds]#HT
    
    
    
    
    
    
    #Simulated Annealing Settings
    maxiterations = 1000
    # accept = -5
    # initialTemp = 5
    cost = 20e10
    jj = 0
    
    # #Sil 50
    # ti = 1.89e3
    # t1 = 13.6e3
    # A1 = 0.15 * np.exp(ti/t1)
    # t2 = 6e3
    # A2 = 0.85 * np.exp(ti/t2)
    
    #Michael Sil50 (pdf)
    Mti = 1890
    Mt1 = 13620
    MA1 = 1.5395e+03
    # MA1 = 1.34e3
    Mt2 = 6050
    MA2 = 1.0524e+04
    # MA2 = 7.7e3
    MB  = 2.4
    
    # #Sil 5000
    # ti = 125.6e3
    # t1 = 14.3e3
    # A1 = .23 * np.exp(ti/t1)
    # t2 = 6.28e3
    # A2 = .77 * np.exp(ti/t2)
    
    # #JL12 50
    # ti = 2.52e3
    # t1 = 16.1e3
    # A1 = .46* np.exp(ti/t1)
    # t2 = 10.3e3
    # A2 = .54* np.exp(ti/t2)
    
    # #JL12 5000
    # ti = 125.9e3
    # t1 = 16.6e3
    # A1 = .48 * np.exp(ti/t1)
    # t2 = 10.7e3
    # A2 = .51 * np.exp(ti/t2)
    
    
    
    # comp = [t1, A1, t2, A2]#amplitude weighted
    Mcomp = [Mt1, MA1, Mt2, MA2, MB]
    fullFitBack = 2.26428
    BBounds = (0, 10)
    
    # JL12_50 = [16.1e3, ]
    CalcBackground = 1.8862
    
    # comp = [13.6e3, .28, 6e3, .72]#intensity weighted
    # def biexp(x, comp):
    #     PQ_fit = comp[1]*np.exp(-x/comp[0])+comp[3]*np.exp(-x/comp[2])
    #     return PQ_fit*1e4
    #****************************************************************************#
    #ACTUAL FIT IS HAPPENING HERE#
    ret = heavyTailFit(x = DataX, y = DataY, fun = function, pBounds = funBounds, maxiterations = maxiterations)
    while abs(cost) >= 3.5:
        print("jj: "+str(jj)+"     cost:" + str(cost))
        jj+=1
        ret = heavyTailFit(x = DataX, y = (DataY)/1e4, fun = function, pBounds = funBounds, maxiterations = maxiterations,jj=jj,p0 = ret.x)
        cost = ret.fun*1e4
        # fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))
        # axs.scatter(DataX,DataY, color = 'black', label = 'data', marker = 'o', s = 5)
        # axs.plot(DataX, function(DataX,*ret.x))
    #****************************************************************************#
    print("Data fit done!")
    print("cost: " + str(cost))
    PQ_cost = ChiSquaredData(DataX, DataY, DataX, BiexpFun(DataX,*Mcomp))
    real_cost = ChiSquaredData(DataX, DataY, DataX, function(DataX,*ret.x)+MB)
    print("PQ cost: " + str(PQ_cost))
else:
    try:
        print("Fit Results Importing")
        importResultFile = np.genfromtxt(path, delimiter=",")[1:,1]
        ret = ImportResults(importResultFile[0:4],importResultFile[4])
        print("Results Import Successful")
    except:
        print("Import Failed")
    
    
#%%
FdataAv    = OffsetAverage(DataX,DataY)
FfitAv     = OffsetAverage(DataX,function(DataX,*ret.x))
offset     = (FdataAv-FfitAv)
#%%
############### Fast Plotting Fit with Data (no special labels) ###############
PlotFitFast = True
if PlotFitFast:
    fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))
    xx_lin = np.linspace(DataX[0]-800,DataX[-1]+70000,10000)
    
    
    # axs.scatter(np.exp(DataX ), DataY, color = 'black' ,label = 'data' ,marker = 'o',s = 5)  #data is black
    # axs.plot(np.exp(xx_lin), (function(xx_lin,*ret.x))+offset ,'-' ,color = 'blue',label = 'HT fit',lw=1)#our fit is blue
    
    # axs.scatter(np.exp(DataX3), DataY3, color = 'green' ,label = 'data' ,marker = 'o',s = 5)  #data is black
    # axs.plot(np.exp(xx_lin), (function(xx_lin,*ret3.x))+offset3, '--', color = 'red', label = 'June fit', lw = 1)
    
    axs.scatter(DataX,(DataY), color = 'black', label = 'data', marker = 'o', s = 5)
    # axs.plot(xx_lin, function(xx_lin,*ret.x)*1e4+MB,color='green')
    axs.plot(xx_lin, function(xx_lin,*ret.x)*1e4,color = 'red')
    
    retBiFit = ([1.1462e+04, 3.023e+03, 5.470e+03, 9.09500000e+03,
                 3.328e+00])
    
    
    # axs.plot(xx_lin,BiexpFun(xx_lin,*comp))
    axs.plot(xx_lin,BiexpFun(xx_lin,*Mcomp),color='blue')
    axs.plot(xx_lin,BiexpFun(xx_lin,*retBiFit), color = 'green')
    
    # ddtMihaelHT = np.array([6855, 0.8833, -693100, 0])
    
    # axs.plot(xx_lin,derFHTt(xx_lin,*ddtMihaelHT)*400/np.pi)
   

    
    axs.set_xlabel(r"$t\ [ns]$", **label_style)
    axs.set_ylabel(r"Quantitiy", **label_style)
    
    axs.grid(color='k', linestyle='-', linewidth=.05 , which ='both')

    # axs.set_xscale('log')
    axs.set_yscale('log')
    # axs.set_xlim(left = 0)
    # axs.set_ylim(bottom = 1e0, top = 2e4)
    # axs.set_xlim(left=0,right = DataX[-1]*1.05)
    axs.set_title(filename)
#%% Residual Comparison
# fig, axs = plt.subplots(1, 1, figsize=(3.2, 2.0))
fig, axs = plt.subplots(1, 1, figsize=(9.2, 2.0))
axs.plot(DataX/1e3,(DataY-function(DataX,*ret.x)*1e4)/np.sqrt(DataY),linewidth=0.5,color='red')
# axs.plot(DataX/1e3,(DataY-function(DataX,*ret.x)*1e4+MB)/np.sqrt(DataY),linewidth=0.5,color='green')
# axs.plot(DataX,DataY-function(DataX,*comp)*1e4)
axs.plot(DataX/1e3,(DataY-BiexpFun(DataX,*Mcomp))/np.sqrt(DataY),linewidth=0.5,color='blue')
axs.plot(DataX/1e3,(DataY-BiexpFun(DataX,*retBiFit))/np.sqrt(DataY),linewidth=0.5,color='green')

# axs.set_xlim((0,100))
axs.set_xlim((0,3))
axs.set_xticks(np.linspace(0,320,9))
axs.set_ylim((-5,5))
axs.grid()

PQ_cost      = ChiSquaredCost(DataX, DataY, BiexpFun, Mcomp)
ddHT_cost    = ChiSquaredCost(DataX, DataY/1e4, function, ret.x)*1e4
# ddHT_MB_cost = ChiSquaredCost(DataX, (DataY-MB)/1e4, function, ret.x)*1e4
SABixp_cost  = ChiSquaredCost(DataX, DataY, BiexpFun, retBiFit)


# axs.plot(DataX,DataY - derFHTt(DataX,*ddtMihaelHT)*400/np.pi)
# fig, axs = plt.subplots(1, 1, figsize=(3, 2.0))
# axs.scatter(DataX,(DataY), color = 'black', label = 'data', marker = 'o', s = 5)
# axs.plot(xx_lin, function(xx_lin, *ret.x), color='blue')
#%% Biexp Breakdown
# fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))
# axs.scatter(DataX ,DataY , color='k')

# axs.set_yscale('log')

# axs.plot(DataX, Mcomp[1]*np.exp(-DataX/Mcomp[0]))
# axs.plot(DataX, Mcomp[3]*np.exp(-DataX/Mcomp[2]))
# axs.plot(DataX, Mcomp[4]+(DataX*0))
# axs.plot(DataX, Mcomp[1]*np.exp(-DataX/Mcomp[0]) + 
#                 Mcomp[3]*np.exp(-DataX/Mcomp[2]) +
#                 Mcomp[4]+(DataX*0))


# # axs.plot(DataX, ret.x[1]*np.exp(-DataX/ret.x[0]))
# # axs.plot(DataX, ret.x[3]*np.exp(-DataX/ret.x[2]))
# # axs.plot(DataX, ret.x[4]+(DataX*0))
# # axs.plot(DataX, ret.x[1]*np.exp(-DataX/ret.x[0]) + 
# #                 ret.x[3]*np.exp(-DataX/ret.x[2]) +
# #                 ret.x[4]+(DataX*0))


# axs.set_ylim(bottom = .1)
# axs.set_title('PicoQuant Biexp Breakdown')
# axs.set_xlabel('t [ns]')
# axs.set_ylabel(' A [cts]')
# axs.set_xlim(left = 0)
# axs.grid()

# #%% SA Breakdown
# fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))
# axs.scatter(DataX ,DataY , color='k')

# axs.set_yscale('log')

# # axs.plot(DataX, Mcomp[1]*np.exp(-DataX/Mcomp[0]))
# # axs.plot(DataX, Mcomp[3]*np.exp(-DataX/Mcomp[2]))
# # axs.plot(DataX, Mcomp[4]+(DataX*0))
# # axs.plot(DataX, Mcomp[1]*np.exp(-DataX/Mcomp[0]) + 
# #                 Mcomp[3]*np.exp(-DataX/Mcomp[2]) +
# #                 Mcomp[4]+(DataX*0))


# axs.plot(DataX, ret.x[1]*np.exp(-DataX/ret.x[0]))
# axs.plot(DataX, ret.x[3]*np.exp(-DataX/ret.x[2]))
# axs.plot(DataX, ret.x[4]+(DataX*0))
# axs.plot(DataX, ret.x[1]*np.exp(-DataX/ret.x[0]) + 
#                 ret.x[3]*np.exp(-DataX/ret.x[2]) +
#                 ret.x[4]+(DataX*0))


# axs.set_ylim(bottom = .1)
# axs.set_title('Simulated Annealing Biexp Breakdown')
# axs.set_xlabel('t [ns]')
# axs.set_ylabel(' A [cts]')
# axs.set_xlim(left = 0)
# axs.grid()