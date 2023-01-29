"""
Copyright c 2021 by Northwestern University. All Rights Reserved.

@author: Can Aygen
"""
#%%#Import the Functions
#Import the Functions
#********************#
from FunDefV38demoDec import *
#********************#
SE = 0;
AD = 1-1e-9;

#%%
# pdb.set_trace()
#%%
#File Import
#Initialization Constants
version     = '32P38MGf_Dec '
save        =  True
save        =  False
interpolateData = True
# interpolateData = False
# today = date.today().strftime("%d-%m-%y")
# savelabel = '-ExcelAv'
# savelabel = 'beforeT'
# m = 0.0
# m=AD
savelabel = str(date.today())
# savelabel   = ' m={},1000.B2'.format(m)
# savelabel   = ' m=AD,1000.B2'
# savelabel = '4D'
print(savelabel)
# savelabel =''

RawOnly = True
RawOnly = False

PlotRaw     = True#False
PlotFitFast = False
PQ_Mode = False

compare = False
# compare = True
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
PQ_Mode = False#True
# folder = 'Muenster/StefanBuss/Grayson_data/csvPtCl'
# filename ='PtCl_77K_LT'; bottomLabel = 'Sbu'; quantity = ' counts '


topLabel = '{\ S.Buss}'; unit = ''; timeunit = 's'
folder = '.'
# folder = 'Muenster/StefanBuss/Grayson_data/csvPtCl@Zif8NoShift'

# filename = 'PtCl_Zif8_air1.csv'  ;  bottomLabel = ' Zif8air   '; quantity = ' counts '; pdfA = [180.13,845.9,1093.1]; pdfT = [22.648e-6,6.146e-6,1.641e-6 ]; pdfB = 1.299; logFlatB = pdfB-0.9;
filename = 'PtCl_Zif8_degas1.csv';  bottomLabel = ' Zif8degas '     ; quantity = ' counts '; pdfA = [628.0,2009.8,2407  ]; pdfT = [17.286e-6,6.262e-6,1.5153e-6]; pdfB = 0.871; logFlatB = pdfB;
# filename = 'PtCl_Zif8_degas.dat'
# filename = 'PtCl_Zif8_air.dat'
# filename = 'CF3_N DCM 77K t0.csv';  bottomLabel = 'CF3_N DCM 77K t0'; quantity = ' counts '; pdfA = [839.8, 977.9];          pdfT= [11.909e-6,4.755e-6];            pdfB = 5.598; logFlatB = pdfB+2.5;
# xscale = 1e-6
xscale = 1


filename = 'MG103_t_RT_Ar_520_2b_0.csv'      ; bottomLabel = ' RT_Ar_520_2b_0'     ; quantity = ''
# filename = 'MG103_t_RT_Ar_520_16384_0.csv'   ; bottomLabel = ' RT_Ar_520_16384_0'  ; qunatity = ''

# bottomLabel = ' Pt-tBu-AmPt 280K 510 P  '
# filename = 'Pt-tBu-AmP_t_280K_510_P'
# filename = '20211127_row=5_col=2_relaxation_5hr_c1p3_truncate_60s'
unit = ' cts '                ### Axis label UNITS for table

filename = 'RelTimeCroppedAndMerged.csv'     ; bottomLabel = ' Relax';  quantitry = ' rho_{xxyy} '; unit = 'ohms'
filename = 'RelTimeCroppedAndMerged_MG_Crop.csv'
filename = 'data_AllZQV_n1.csv';

filename = 'Sil_t_DCMMeOH_530_5_INT.csv'
filename = 'Sil_t_DCMMeOH_496_1_INT.csv'


#-----------------------
folder = 'csv_files'
# filename = 'JL01_t_RT_Ar_525_6_INT.csv'
# filename = 'JL01_t_RT_Ar_525_1500_0_INT.csv'
# filename = 'MG103_77K_506_t_5_INT.csv'
# filename = 'MG103_77K_506_t_8500_INT.csv'
# filename = 'MG103_77K_530_t_5_INT.csv'
# filename = 'MG103_t_DCMMeOH_520_5_INT.csv'
# filename = 'MG103_t_DCMMeOH_540_5b_INT.csv'

# filename = 'Sil_t_DCMMeOH_530_3000_INT.csv'


# filename = 'Sil_t_DCMMeOH_496_1_INT.csv'


# folder = 'rezoomfortomorrow'
# filename = 'Sil_77K_530_t_3000.csv'; logFlatB = 0.0

# filename = 'Sil_77K_496_t_3000.csv'

# filename = 'Sil_77K_530_t_1'

folder = 'Jan27'
filename = 'Sil_t_DCMMeOH_496_1_INT_adj.csv'
# PQ_Mode = True

# yscale = 1e-6
# xscale = 1e-9

# RawOnly = True
unit = 'cts'
# filename = 'csv'
# RawOnly = True
# save = True
# savelabe =  'm0'
# yscale = -1

#%%File Import
import pandas as pd
import copy
# file = '{0}/{1}.csv'.format(folder,filename)
file = '{0}/{1}'.format(folder,filename)
# file = '../Literature/{0}/{1}'.format(folder,filename)  ### Replace "Literature" with subfolder name, if desired
# dataOffset = 117
# dataOffset = 1
dataOffset = 0

#%%File Import
# file = '../Literature/{0}/{1}.csv'.format(folder,filename)
# file = '../Literature/{0}/{1}'.format(folder,filename)
# dataOffset = 117
# dataOffset = 3
csv  = np.genfromtxt(file, delimiter=",")
# csv  = np.genfromtxt(file, delimiter=','   , skip_header = 1 + dataOffset)
# csv  = np.genfromtxt(file, delimiter='\t'   , skip_header = 1 + dataOffset)

# csv  = np.genfromtxt(file, delimiter="\t", skip_header = 2 + dataOffset)
# csv  = np.genfromtxt(file, delimiter="\t", skip_header = 1)

csv = csv[~np.isnan(csv).any(axis=1)] # removes rows containing NaN from the data
fitMode = input('Would you like to fit to a \'Streched-Exponential\' (0) or a \'Power-Law\' decay equation? (1) \nPlease enter 0 or 1: ')
print('Analyzing' + file)

# %%Data Integration
if PQ_Mode:
    Datat = (csv[:,0]-dataOffset*2.0)*xscale
    # Datat  = csv[:,0]*xscale
    DataOg = csv[:,1]
    PicoDataColor = 'lightgray'; PicoFitColor = 'gray'

    # xx_lin = np.linspace(0,Datat[-1],1000)
    xx_lin = np.linspace(0,Datat[-1],Datat.shape[0])
    #First Plot
    PicoY = PicoGen(A=pdfA,T=pdfT,Back=pdfB, Data_t = Datat, filename=filename, plot=True)

    IntPicoFitBack   = PicoIntegrate(Datat,PicoY)
    IntDataPicoBack  = PicoIntegrate(Datat,DataOg-pdfB)
    IntDataExcelBack = PicoIntegrate(Datat,DataOg-logFlatB)

    #Second Plot
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 3.0))
    axs.plot(Datat ,DataOg, color ='b', lw = 0.2, label = 'Data')
    axs.plot(xx_lin, PicoY, color = 'k', lw = 2, label = 'PQ')
    axs.plot(xx_lin, PicoY - pdfB, color = 'gray', label = 'PQ - B$_{PQ}$')
    axs.plot(xx_lin, PicoY - logFlatB, label = 'PQ - B$_{HT}$')
    axs.set_yscale('log')
    axs.grid(color='k', linestyle='-', linewidth=0.03 , which ='both')
    axs.set_title(filename)
    add_watermark(axs, version)
    axs.legend()

    #Third Plot
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 3.0))
    axs.plot(Datat,IntPicoFitBack  ,'-' ,color = PicoFitColor , lw = 1, label = 'PQ reconstructed')
    axs.plot(Datat,IntDataPicoBack ,'-' ,color = PicoDataColor, lw = 2, label = 'Data - B$_{PQ}$')
    axs.plot(Datat,IntDataExcelBack,'--',color = HTDataColor  , lw = 2, label = 'Data - B$_{HT}$')

    axs.legend()
    # axs.set_yscale('log')
    axs.set_xscale('log')
    axs.set_title(filename)
    axs.grid(color='k', linestyle='-', linewidth=0.03 , which ='both')
    add_watermark(axs, version)
#%X and Y transfer
    DataX = np.log(csv[:,0]*xscale)
    DataY = IntDataExcelBack
else:
    DataX = np.log(csv[:,0]*xscale)
    DataY = csv[:,1]
# DataX = csv[:,0][37:]
# DataY = csv[:,1][37:]
# pdb.set_trace()
#%%fz
# def printDuplicates(arr):
#     dict = {}

#     for ele in arr:
#         try:
#             dict[ele] += 1
#         except:
#             dict[ele] = 1

#     for item in dict:

#          # if frequency is more than 1
#          # print the element
#         if(dict[item] > 1):
#             print(item, end=" ")

#     print("\n")
# pdb.set_trace()
# printDuplicates(DataX)

#%%Interpolation
if interpolateData:
    startInterpAt = 1000
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 3.0))
    f = interp1d(DataX[startInterpAt:], DataY[startInterpAt:], kind='linear')
    xnew = np.linspace(DataX[startInterpAt], DataX[-1], num=2000, endpoint=True)

    plt.plot(DataX, DataY, 'o', np.append(DataX[:startInterpAt],xnew), np.append(DataY[:startInterpAt],f(xnew)), 'x')

    DataX,DataY = np.append(DataX[:startInterpAt],xnew),np.append(DataY[:startInterpAt],f(xnew));
    # plt.legend(['data', 'cubic'], loc='best')
    plt.show()

#%%Yscaling
# pdb.set_trace()
# yscale = 10**-np.floor(np.log10(np.amax(DataY)))
DataY = DataY * yscale

if yscale != 1:
    unit = '$10^{' + '{}'.format(int(-np.log10(yscale)))+ '}$ ' + unit
    # try:
    #     comParams[2] *= yscale
    #     savelabel = savelabel + 'Comp'
    #     compare = True
    # except:
    #     pass
#%% Import the fit parameters if there is a previous successful fit for this data with this version of the code
# path = pathlib.Path(file[:-4]+'ResultV{}{}.csv'.format(version,savelabel))
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

#%% Temporary Plots


fig, axs = plt.subplots(1, 1, figsize=(5.2, 3.0))
firstpoints = 700


# plt.plot(Datat[:firstpoints], DataOg[:firstpoints], '-', label = 'OG Data [AFT]')
# plt.plot(Datat[:firstpoints]-32, DataOg[:firstpoints], '-', label = '-31 ns [MID]')
# plt.plot(Datat[:firstpoints]-64, DataOg[:firstpoints], '-', label = '-64 ns [BEF]')

# plt.plot(Datat[:firstpoints]-30, DataOg[:firstpoints], '-', label = '-30 ns [AFT]')
# plt.plot(Datat[:firstpoints]-208, DataOg[:firstpoints], '-', label = '-208 ns [MID]')
# plt.plot(Datat[:firstpoints]-386, DataOg[:firstpoints], '-', label = '-386 ns [BEF]')

axs.grid(color='k', linestyle='-', linewidth=0.03 , which ='both')
titleText = 'RT Burst linlin 500 points'
# titleText = '77 K Burst linlog all points ymax=8e3'
axs.set_title(titleText)
# axs.legend()
# axs.set_xscale('log')
# axs.set_yscale('log')

# axs.set_ylim([0, 8e3])
axs.set_ylim([0, 1.5e4])
axs.set_xlabel('t [ns]')
axs.set_ylabel(' counts')
add_watermark(ax=axs,version=version)
# fig.savefig(str(ResultsFolderPath)+'/'+filename[:-4]+'{}.png'.format(titleText),dpi=fig.dpi)



#%%
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

    #%%
#Demo Plot Extra
# def jumpZone():


# fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.2))
# # axs.scatter(DataX[1:], DataY[1:], color = 'black' ,label = ' ' ,marker = 'o',s = 5)
# axs.scatter(np.exp(DataX), DataY/1e6, color = 'black' ,label = ' ' ,marker = 'o',s = 5)
# axs.set_xlabel(r"$t\ [sec]$", **label_style)
# # axs.set_ylabel(r"$f^{\ Münster\ UP}_{E. Coli\ 25ºC} $ x $10^6$ [count]", **label_style)
# axs.set_ylabel(r"$Rxx^{\ InOH}_{A7/ IIb}$  [ohms]", **label_style)
# axs.grid(color='k', linestyle='-', linewidth=0.03 , which ='both')
# # axs.set_title('Raw ln(t)')

# axs.grid(b = True, linestyle='-', linewidth=0.05 , which ='major')
# axs.set_xscale('log')
# # xlimpow = np.array([-2,64])
# # xtickpow = np.linspace(xlimpow[0],xlimpow[1]+1,(int(abs(xlimpow[0]-xlimpow[1]))+2))
# # xtickpow = np.linspace(-7,-1,3)
# # axs.set_xticks(10**xtickpow,minor = False)
# # axs.set_yticks(np.linspace(0,2,5))
# # axs.set_xlim([1e-10,1e1])
# axs.set_ylim([0,7e-5])



# plt.tight_layout()





############### ############### ############### ############### ###############

#%%
if RawOnly:
    sys.exit(0)

# pdb.set_trace()
#%%############ ############### ############### ############### ###############
############### Fitting ###############

paraName=['u','\\beta','\\bar{f}','m']; plotParaName =['u','T','\\bar{f}','m']
function = FHTx; print('function: FHTx (4D Solver)')
# function = ADx; print('function: FHTx (4D Solver)')
# function = F_uThetDf0x; print('function: F_uThetDf0x (3D Solver)')
#boundaries parameter space

# constraint = (['u','fdelt'],['>','<'],[5,3])
# constraint = (['m'],['>'],[1-9e-3])
# constraint = (['m','b','fdelt'],['<','>','<'],[0.01,0.99,100]) # m=0, B=1.0
# constraint = (['m','b','fdelt'],['<','<','<'],[0.01,0.50,300]) # m=0, B=0.5
# constraint = (['m','b','fdelt'],['>','>','<'],[0.50,0.99,100]) # m=0.5, B=1.0
# constraint = (['m','b','fdelt'],['>','<','<'],[0.50,0.50,100]) # m=0.5, B=0.5
# constraint = (['m','b','fdelt'],['>','>','<'],[0.99,0.99,100]) # m=1, B=1
# constraint = (['m','b','fdelt'],['>','<','<'],[0.99,0.5,1e3]) # m=1, B=0.5

# constraint = (['b','fdelt'],['>','<'],[0.99,300]) # m=free, B=1
# constraint = (['b','fdelt'],['<','<'],[0.5,460]) # m=free, B=0.5
# constraint = (['b','fdelt'],['<','<'],[0.1,2e3]) # m=free, B=0.1

# constraint = (['u','u'],['>','<'],[3,11])
# constraint = (['fdelt'],['>'],[-4e6])

if fitMode:
    constraint = (['m'],['<'],[6e-4])
    # constraint = (['m'],['>'],[1-9e-3])
else:
    # constraint = (['m'],['<'],[6e-4])
    constraint = (['m'],['>'],[1-9e-3])
    
# 
# print(function)

if not FitExists or ForceRecalculation:
    uBounds    = constraintBuilder('u'   , constraintSelector('u'  , DataX), constraint)
    # uBounds = (-3,21)
    bBounds    = constraintBuilder('b'   , constraintSelector('b'  ,np.nan), constraint)
    fdeltBounds= constraintBuilder('fdelt', constraintSelector('fdelt',DataY), constraint)
    # fdeltBounds= (-2,1)
    # fdeltBounds= (-5000,5000)
    # fdeltBounds = (-0.1, 0.1)
    mBounds    = constraintBuilder('m'   , constraintSelector('m'  ,np.nan), constraint)
    # mBounds    = (1-1e-9,1)
    funBounds  = boundSelector(function, uBounds, bBounds, fdeltBounds, mBounds)
    # funBounds  = boundSelector[uBounds, bBounds, df0Bounds, mBounds]

    #Simulated Annealing Settings
    maxiterations = 1000
    # accept = -5
    # initialTemp = 5

    #****************************************************************************#
    #ACTUAL FIT IS HAPPENING HERE#
    ret = heavyTailFit(x = DataX, y = DataY, fun = function, pBounds = funBounds, maxiterations = maxiterations)
    #****************************************************************************#
    print("Data fit done!")
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
# roundSigFig = 2
funBounds[-1] = (0,1e-9)
if compare == 2:
    ret3 = heavyTailFit(x = DataX3, y = DataY3, fun = function,  pBounds = funBounds, maxiterations=maxiterations)
#%%
    compareSig2= Modsigma(DataX,DataY,function,ret3.x)**2
    sigmaRatio = r' x{:.2f}'.format((compareSig2/ret.fun**2))
    FdataAv3   = OffsetAverage(DataX3,DataY3)
    FfitAv3    = OffsetAverage(DataX3,FHTx(DataX3,*ret3.x))
    offset3    = (FdataAv3-FfitAv3)

    #m2 = ret2.x[3]
    #offset2 = (1-m2)/m2*ret2.x[2]

#%%
############### Fast Plotting Fit with Data (no special labels) ###############
if PlotFitFast:
    fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))
    xx_lin = np.linspace(DataX[0]-8,DataX[-1]+7,10000)

    axs.scatter(np.exp(DataX), DataY, color = 'black' ,label = 'data' ,marker = 'o',s = 5)  #data is black
    axs.plot(np.exp(xx_lin), (function(xx_lin,*ret.x))+offset ,'-' ,color = 'blue',label = 'HT fit',lw=1)#our fit is blue

    axs.scatter(np.exp(DataX3), DataY3, color = 'green' ,label = 'data' ,marker = 'o',s = 5)  #data is black
    axs.plot(np.exp(xx_lin), (function(xx_lin,*ret3.x))+offset3, '--', color = 'red', label = 'June fit', lw = 1)

    axs.set_xlabel(r"$t\ [s]$", **label_style)
    axs.set_ylabel(r"Quantitiy", **label_style)

    axs.grid(color='k', linestyle='-', linewidth=.05 , which ='both')
    axs.set_xscale('log')
############### ############### ############### ############### ###############
#%%############ ############### ############### ############### ###############
############### 'Nice' Plotting ###############

fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))
xx_lin = np.linspace(DataX[0]-2,DataX[-1]+2,10000)

tempRet = [-11.1,1,-11.2e-3,0.0]
# tempRet = ret.x
axs.scatter(np.exp(DataX), DataY, color = HTDataColor ,label = 'data' ,marker = 'o',s = 5)  #data is black
axs.plot(np.exp(xx_lin), (function(xx_lin,*ret.x))+offset ,'-' ,color = HTFitColor,label = 'HT fit',lw=0.5)#our fit is blue
# axs.plot(np.exp(xx_lin), (FHTx(xx_lin,*tempRet))-tempRet[2]-0.0005 ,'-' ,color = 'yellow',label = 'HT fit',lw=1)#our fit is blue

# axs.plot(np.exp(xx_lin), (function(xx_lin,*ret3.x))+offset3, '--', color = 'gray', label = 'June fit', lw = 1)
try:
    # CfitAv     = OffsetAverage(DataX,FHTx(DataX,*comParams))
    CfitAv     = OffsetAverage(DataX,FHTx(DataX,*comParams))
    C_offset   = FdataAv-CfitAv
    axs.plot(np.exp(xx_lin), (FHTx(xx_lin,*comParams))+C_offset ,'-' ,color = 'gray',label = 'Comp fit',lw=1)#our fit is blue
except:
    pass

# dfHt = lambda t: (B*df0*(1-m)**2*(t/T)**B*np.exp((m-1)*(t/T)**B))/t
# dfHt = lambda t: -B*df0*((1-m)**2*(t/T)**B*np.exp(-(1-m)*(t/T)**B))/(t*(m*np.exp(-(1-m)*(t/T)**B-1)**2))
# diffHt = PicoIntegrate(Datat[:-1],np.diff((FHTx(np.log(Datat),*ret.x))))

# dHTy = np.sqrt((dfHt(Datat))**2+(logFlatB)**2)
# axs.plot(Datat,dHTy)
# diffcheckO = PicoIntegrate(Datat,dHTy*1.3)
# diffcheckG = PicoIntegrate(Datat,dHTy*1)
# axs.plot(Datat,diffcheckO,'orange')
# axs.plot(Datat,diffcheckG,'g')
# axs.plot(Datat[:-1],diffHt,'r')


axs.set_xlabel(r"$t$ [{}]".format(timeunit), **label_style)
# axs.set_ylabel(r"${}".format(quantity)+"_{" + '{}'.format(bottomLabel) + "}^"+"{}$ {}".format(topLabel,unit), **label_style)
# axs.set_ylabel(r"${}$".format(quantity) + r"  {}".format(unit), **label_style)

# axs.grid(b=True, which='minor', color='k', linestyle='-')
# axs.grid(b=True, which='minor', color='r', linestyle='--')

axs.grid(b = True, linestyle='-', linewidth=0.05 , which ='major')
# axs.grid(b = True, which='minor', color='r', linestyle='--', axis ='x')

axs.set_xscale('log')

xlimpow = np.floor(np.log10(axs.get_xlim()))
xtickpow = np.linspace(xlimpow[0],xlimpow[1]+1,(int(abs(xlimpow[0]-xlimpow[1]))+2))


#Optional Axes Formatiing Settings:
# axs.set_xlim(1e-3,1e7)
# yrange = (0, 2e5)
try:
    axs.set_ylim(yrange)
except:
    pass
# xrange = (1e-2,1e9)
try:
    axs.set_xlim(xrange)
except:
    pass
axs.set_xticks(10**xtickpow,minor = False)
# axs.set_xticks([1e0,1e3,1e6,1e9],minor = False)
# axs.set_xticks([1e-2,1e-0,1e2,1e4,1e6,1e8],minor = False)

plt.tight_layout()
# add_watermark(ax=axs,version=version+savelabel+sigmaRatio)

if save:
    # plt.savefig(file[:-4]+'V{}{}.png'.format(version,savelabel),dpi=200)
    # imPath = file[:-4]+'V{}{}.png'.format(version,savelabel)
    # plt.savefig(file[:-4]+'/'+filename[:-4]+'{}.png'.format(savelabel),dpi=300)

    # imPath = file[:-4]+'{}.png'.format(savelabel)
    imPath = str(ResultsFolderPath)+'/'+filename[:-4]+'.png'
    plt.savefig(imPath,dpi=300)



#%% Add metadata to the saved plot
if save:
    im = Image.open(imPath)
    METADATA = {"Code Version":"Spyder{}{}".format(savelabel,version),
                "Optimized Parameters" :"u,B,fdelt,m = {0},sigma={1:.6f}, offset={2:.4f}".format(ret.x, ret.fun, offset),
                # "Version":version,
                # "Plus/Minus error" : "u,B,fdelt,m = {0}".format(deltaParam),
                "fdelt and offset scaling factor":"yscale = {0}".format(yscale)}
    meta = PngImagePlugin.PngInfo()

    for x in METADATA:
        meta.add_text(x, METADATA[x])

    im.save(imPath, "png", pnginfo=meta)

#%%Linear time Plot
if LinearPlot:
    fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))
    xx_lin = np.linspace(DataX[0]-8,DataX[-1]+3,10000)

    axs.scatter(np.exp(DataX), DataY, color = 'black' ,label = 'data' ,marker = 'o',s = 5)  #data is black
    axs.plot(np.exp(xx_lin), (FHTx(xx_lin,*ret.x))+offset ,'-' ,color = 'blue',label = 'HT fit',lw=1)#our fit is blue

    # try:
    #     CfitAv     = OffsetAverage(DataX,FHTx(DataX,*comParams))
    #     C_offset   = FdataAv-CfitAv
    #     axs.plot(np.exp(xx_lin), (FHTx(xx_lin,*comParams))+C_offset ,'-' ,color = 'red',label = 'Comp fit',lw=1)#our fit is blue
    # except:
    #     pass

    axs.set_xlabel(r"$t$ [{}]".format(timeunit), **label_style)
    # axs.set_ylabel(r"${}".format(quantity)+"_{" + '{}'.format(bottomLabel) + "}^"+"{}$ {}".format(topLabel,unit), **label_style)
    axs.set_ylabel(r"${}$".format(quantity) + r"  {}".format(unit), **label_style)

    axs.grid(color='k', linestyle='-', linewidth=0.05 , which ='both')
    # axs.set_xscale('log')
    xrange = (-1e7,1e8)
    try:
        axs.set_ylim(yrange)
    except:
        pass

    try:
        axs.set_xlim(xrange)
    except:
        pass
    axs.set_xticks([0,.2e8,.4e8,.6e8,.8e8,1e8],minor = False)

#%% Error Estimation and Tabulation

# CurvTensor = tensorByNumdiff(DataX,DataY,F_uThetDf0x,ret.x[0:3],m=ret.x[3])
# Eig = LA.eig(CurvTensor);

r = 1 + (2.576)**2/len(DataX)
# pdb.set_trace()
CurvTensor = tensorByNumdiff(DataX,DataY,function,ret.x)
b,delta,reslts,debug = MakeResultsTable(DataX,DataY,function,ret,r = r, y_unit = unit, title = bottomLabel[1:-1], CurvTensor=CurvTensor)# m=mHiLo)
try:
    print('sigma_c^2 = {0:3g}'.format(Modsigma(DataX, DataY, FHTx, comParams)**2))
except:
    pass
print("Confidence Intervals Calculated!")
#%%
#Create The Visual table(s)
fig, axs = plt.subplots()
axs.xaxis.set_visible(False)
axs.yaxis.set_visible(False)
axs.set_frame_on(False)
render_mpl_table(reslts, header_columns=0, col_width=1.4, row_height=0.425, ax=axs)
add_watermark(ax=axs,version=version+savelabel)
if save == True:
    fig.savefig(str(ResultsFolderPath)+'/'+filename[:-4]+'{}Results.png'.format(savelabel),dpi=fig.dpi)
    # fig.savefig(file[:-4]+'{}Results.png'.format(savelabel),dpi=fig.dpi)
fig, axs = plt.subplots()
axs.xaxis.set_visible(False)
axs.yaxis.set_visible(False)
axs.set_frame_on(False)
render_mpl_table(debug , header_columns=0, col_width=1.4, row_height=0.425, ax=axs)
add_watermark(ax=axs,version=version+savelabel)
# fig.savefig('test2.png',dpi=fig.dpi)
if save == True:
    # fig.savefig(file[:-4]+'{}Delta.png'.format(savelabel),dpi=fig.dpi)
    fig.savefig(str(ResultsFolderPath)+'/'+filename[:-4]+'{}Delta.png'.format(savelabel),dpi=fig.dpi)
# delta = deltaPar3d(CurvTensor,r,ret.fun**2)

#Saved Table
if save:
    np.savetxt(str(ResultsFolderPath)+'/'+filename[:-4]+'Result{}.csv'.format(savelabel), b, delimiter=",",fmt='%s')
    # np.savetxt(file[:-4]+'Result{}.csv'.format(savelabel), b, delimiter=",",fmt='%s')
#%%Pico
# m=ret.x[-1]
savepath = str(ResultsFolderPath)+'/'+filename[:-4]
# sys.exit(0)
# pdb.set_trace()

if PQ_Mode:
    PlotPicoComparisons(Datat, DataOg, PicoY,
                        IntPicoFitBack, IntDataPicoBack, IntDataExcelBack,
                        ret, ret.x[-1], offset, yscale, #m
                        # ret, offset, yscale, #m
                        pdfB, logFlatB,
                        bottomLabel, version + savelabel, savepath, n_init = 0, n_fin = 501)
