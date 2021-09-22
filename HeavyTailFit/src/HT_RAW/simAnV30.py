"""
Copyright c 2021 by Northwestern University. All Rights Reserved.

@author: Can Aygen
"""
#%%#Import the Functions
#Import the Functions
#********************#
from FunDefV36 import *
#********************#
SE = 0;
AD = 1-1e-9;

#%% 
# pdb.set_trace()
#%%
#File Import
#Initialization Constants
version     = '29P35'
# save        =  True
save        =  False
interpolateData = True
# interpolateData = False
# today = date.today().strftime("%d-%m-%y")
# savelabel = '-ExcelAv'
# savelabel = 'beforeT'
# m = 0.0
# m=AD
savelabel = ''
# savelabel   = ' m={},1000.B2'.format(m)
# savelabel   = ' m=AD,1000.B2'
# savelabel = '4D'
print(savelabel)
# savelabel =''

# RawOnly = True
RawOnly = False
PlotRaw     = True#False
PlotFitFast = False
PQ_Mode = False

compare = False
compare = True
LinearPlot = False #True
ForceRecalculation = False #if the previous saved if is not satifactory set this to True to re-do the fit
# ForceRecalculation = True
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
# folder = 'Muenster/Pentyl-Pt-mPy_attenuator'
# timeunit = 's'; quantity = "counts"
# topLabel = '{\ Pentyl-Pt-mPy}';
# yrange = (0,9); #optional

# filename = '77K01percent'   ; bottomLabel = '{0.1 \%}'  ;yscale=1e4; unit = 'x $ 10^{4}$'
# filename = '77K1percent'    ; bottomLabel = '{1.0 \%}'  ;yscale=1e4; unit = 'x $ 10^{4}$'
# filename = '77K10percent'   ; bottomLabel = '{10.0 \%}' ;yscale=1e4; unit = 'x $ 10^{4}$'
# filename = '77K100percent'  ; bottomLabel = '{100.0 \%}';yscale=1e4; unit = 'x $ 10^{4}$'
#-----------------------------------------------------------------------------
# folder   =  'Jiajun/5.2b!'
# timeunit =  's'; quantity = 'S'
# topLabel = "'16"; bottomLabel = 'IGZO'

# filename = '5.2bReduced'

# folder   = 'June(Bio)'
# timeunit = 's'; quantity = 'a.u.'
# topLabel = "June '13"; bottomLabel = 'Cartilage'

# filename = 'June4d'
# filename = 'June4c'
# unit ='='


# folder   = 'Klein(earthquake)'
# filename = 'eathquake'

# folder   = "Muenster/E coli lifetime measurements Oct2" 
# filename = "Coli 25C Obs 650"

folder    = "Mystery"
filename  = "RxxClean3"
timeunit  = 'sec'
unit      = 'ohms' 
#-----------------------------------------------------------------------------

# folder = 'Muenster/Published/Reported_Complexes_data/HT_csv'

# timeunit = 's'; quantity = 'counts'
# topLabel = '{\ Reported}'; yscale = 1 ; unit = ' '

# yrange = (-0.5,6.5); xrange = (1e-8,1e-2)
# filename = 'tBu_Sb 77K' ; bottomLabel = ' tBuSb ' ; quantity = ' BuSb\ counts  '; #Pico = True
# filename = 'tBu_P 77K'  ; bottomLabel = ' tBuP  ' ; quantity = ' BuP\ counts   '
# filename = 'tBu_As 77K' ; bottomLabel = ' tBuAs  '; quantity = ' BuAs\ counts  '

# yrange = (-0.5,2); xrange = (1e-8,1e-2)
# filename = 'CF3_Sb 77K' ; bottomLabel = ' CF3 Sb'; quantity = ' CF3 Sb counts'
# filename = 'CF3_As 77K' ; bottomLabel = ' CF3 As'; quantity = ' CF3 As counts '
# filename = 'CF3_P 77K'  ; bottomLabel = ' CF3 P' ; quantity = ' CF3 P counts '

# folder = 'Muenster/Published/Reported_Complexes_data/RAW/CSV'
# filename = 'tBu_As 77KRAW' ; bottomLabel = ' $\\rm ~{}^tBu/As\ 77\ K$     ' ; quantity = ' ^tBu/As\ counts '; pdfA = [2304.8,2145.3]; pdfT = [7.427e-6,15.352e-6]; pdfB = 4.166; logFlatB = 4.841574652
# filename = 'tBu_P 77KRAW'  ; bottomLabel = ' $\\rm ~{}^tBu/P\ 77\ K$      ' ; quantity = ' ^tBu/P\ counts  '; pdfA = [931.1        ]; pdfT = [18.482e-6         ]; pdfB = 1.338; logFlatB = 1.821909033
# filename = 'tBu_SbRAW'     ; bottomLabel = ' $\\rm ~{}^tBu/Sb\ 77\ K$     ' ; quantity = ' ^tBu/Sb\ counts '; pdfA = [216.7,330.0  ]; pdfT = [7.889e-6,16.540e-6]; pdfB = 3.006; logFlatB = 3.66168942

# filename = 'CF3_As 77KRAW' ; bottomLabel = ' $\\rm CF_3/As\ 77\ K$ ' ; quantity = ' CF_3As\ counts '; pdfA = [928.8          ]; pdfT = [13.244e-6         ]; pdfB = 2.320; logFlatB = 3.443021767
# filename = 'CF3_PRAW'      ; bottomLabel = ' $\\rm CF_3/P\ 77\ K$  ' ; quantity = ' CF_3P\  counts '; pdfA = [713.9          ]; pdfT = [14.129e-6         ]; pdfB = 4.200; logFlatB = 5.30112
# filename = 'CF3_Sb 77KRAW' ; bottomLabel = ' $\\rm CF_3/Sb\ 77\ K$ ' ; quantity = ' CF_3Sb\ counts '; pdfA = [2733.5,1934.4  ]; pdfT = [12.5095e-6,5.657e-6];pdfB = 1.519; logFlatB = 2.205128205;#logFlatB = 2.590792839

#-------

# folder = 'Muenster/StefanBuss/Grayson_data/csvPtCl'
# filename ='PtCl_77K_LT'; bottomLabel = 'Sbu'; quantity = ' counts '


# topLabel = '{\ S.Buss}'; unit = ''

# folder = 'Muenster/StefanBuss/Grayson_data/csvPtClNoShift'
# filename ='PtCl_77K_LT1RAW'; bottomLabel = ' 77K '; quantity = ' counts '; pdfA = [3524.5,5881.5]; pdfT = [39.306e-6,26.274e-6]; pdfB = 1.161

# folder = 'Muenster/StefanBuss/Grayson_data/csvPtCl@MOF5NoShift'
# filename ='PtCl_MOF5_degas1'; bottomLabel = ' air '; quantity = ' counts '; pdfA = [250.88,1258.9,2731]; pdfT = [0.05776e-3,0.017354e-3,0.003927e-3]; pdfB = 1.310

# folder = 'Muenster/StefanBuss/Grayson_data/csvPtCl@Zif8NoShift'
# filename = 'PtCl_Zif8_air1'  ;  bottomLabel = ' Zif8air   '; quantity = ' counts '; pdfA = [180.13,845.9,1093.1]; pdfT = [22.648e-6,6.146e-6,1.641e-6 ]; pdfB = 1.299
# filename = 'PtCl_Zif8_degas1';  bottomLabel = ' Zif8degas '; quantity = ' counts '; pdfA = [628.0,2009.8,2407  ]; pdfT = [17.286e-6,6.262e-6,1.5153e-6]; pdfB = 0.871

#%%

# g = plt.plot(np.exp(x), y-y.min())
# u = -11.24; B = 1; df0 = 1.35e-2; m =0.1
# PicoFun()
#%%

# PlotAllRaw(folder)
#%%
#-----------------------------------------------------------------------------
# folder = 'Muenster/Ir_ppy_3_77K_attenuator'
# timeunit = 's'; yscale = 1
# topLabel = '{\ Ir-ppy-3}';

# # filename = '77K0.1percent'    ; bottomLabel = '{0.1 \%}';yscale=1e4; unit = 'x $ 10^{4}$'
# # filename = '77K1percent'  ; bottomLabel = '{1.0 \%}';yscale=1e4 ; unit = 'x $ 10^{4}$'
# filename = '77K10percent' ; bottomLabel = '{10.0 \%}';yscale=1e4; unit = 'x $ 10^{4}$'

# def MunsterFit(Ti,Ai,x):
#     summ = np.zeros((x.size,))
#     integ = summ
#     for ii in np.arange(0,Ti.size-1,1):
#         summ+=Ai[ii]*np.exp(-x/Ti[ii])
        
#     for ii in np.arange(1,x.size,1):
#         integ[ii]+=summ[ii]
#     return integ
#-----------------------------------------------------------------------------
# folder = 'Muenster/satish'
# timeunit = 's'
# topLabel = '{\ Ir-ppy-3}';

# filename = 'Ir(ppy)3, I=9, Attenuator 10%, BW=32 ns'; bottomLabel = '{10 \%} I=9 BW = 32ns'; unit = 'counts'
#%%
# folder = 'Muenster/Published/Decay_files_DCM_and_solid_samples/DCM/CSV'
# timeunit = 's';
# filename = 'CF3_As DCM-RT-Ar'; bottomLabel = 'CF3_As DCM-RT-Ar'; unit = 'counts'; pdfA = [74.45,158.3]; pdfT = [11.843e-9,2.075e-9]; pdfB = 13.807; logFlatB = pdfB

#%%
# def JumpHere():
    # return np.nan

# PQ_Mode = True
# RawOnly = True
# RawOnly = False
# ForceRecalculation = True

# folder = 'Muenster\Published\Ada_P_complex\Ada_P_complex_decays\RAW\CSV'
# timeunit = 's'; unit = 'counts'
# filename = 'Ada_P DCM 77K'; quantity = ' Ada_P 77 K counts '; pdfA = [80.42          ]; pdfT = [18.793e-6         ]; pdfB = 1.558; logFlatB = 2.2#1.717
# filename = 'Ada_P DCM RT O2'; quantity = ' Ada_P O2 counts '; pdfA = [250.12, 735.0  ]; pdfT = [0.26673e-6, 0.006026]; pdfB = 0.371; #logFlatB = 3.443021767
# 
# folder = 'Muenster/Published/CF3_N_complex_decays/CSV'
# folder = 'Muenster/Published/testFolder/CSV'

# timeunit = 'ns'; unit = 'counts'; xscale = 1e-9
# filename = 'CF3_N DCM 77K t0'  ; pdfA = [839.8, 977.9];        pdfT = [11.909e-6  , 4.755e-6  ]; pdfB = 5.598; logFlatB = pdfB;
# filename = 'CF3_N DCM 77K'  ; pdfA = [839.8, 977.9];        pdfT = [11.909e-6  , 4.755e-6  ]; pdfB = 5.598; logFlatB = pdfB;
# filename = 'CF3_N DCM RT Ar'; pdfA = [3042, 4537.8, 134.8]; pdfT = [0.001527e-6, 0.20181e-6, 0.3713e-6]; pdfB = 5.093; logFlatB = pdfB + 0.05;
# filename = 'CF3_N DCM RT O2'; pdfA = [160.79]; pdfT = [0.17163e-6]; pdfB = 0.499; logFlatB = pdfB +0.6;
# filename = 'CF3_N DCM RT O2 full'; pdfA = [160.79]; pdfT = [0.17163e-6]; pdfB = 0.499; logFlatB = pdfB +0.6;

#####
#tBu_P_decay_data_17.03.2021/
#
# folder = 'Muenster/tBu_P_decay_data_17.03.2021/DAT_NoHead'

##filename= name of the file including the extension;
##pdfA = PQ amplitude fit info; 
##pdfT = PQ time fit info (in terms of data time units); 
##pdfB = background value (value at t=+inf)

# xscale = 1
#1
# filename = 'tBu_P 77 K Ar burst mode.dat'                 ; pdfA = [2423.2, 576.9]; pdfT = [18.724e-6 *1e9, 5.638e-6 *1e9 ]; pdfB = 4.141; logFlatB = pdfB   ; dataOffset = 0; #dataOffset = 2
#2
# filename = 'tBu_P 77 K Ar without burst mode I = 8.5.dat' ; pdfA = [10.43, 57.94 ]; pdfT = [4.38e-6  *1e9, 17.319e-6 *1e9]; pdfB = 0.066; logFlatB = pdfB+.1; dataOffset = 7
# #3
# filename = 'tBu_P 77 K Ar without burst mode I = 10.dat'  ; pdfA = [1488.5, 345.8]; pdfT = [18.725e-6*1e9, 5.735e-6  *1e9]; pdfB = 2.312; logFlatB = pdfB+1; dataOffset = 2
# #4
# filename = 'tBu_P RT Ar burst mode.dat'                   ; pdfA = [3265.0, 849.7]; pdfT = [1.16869e-6*1e9  , 0.8166e-6*1e9  ]; pdfB = 1.394; logFlatB = pdfB-1.0; dataOffset =0 #dataOffset = 250
# #5
# filename = 'tBu_P RT Ar without burst mode I = 8.5.dat'   ; pdfA = [952.9 , 219.2]; pdfT = [1.0960e-6*1e9   , 0.1728e-6*1e9  ]; pdfB = 0.405; logFlatB = pdfB+0.3; dataOffset = 20
# #6
# filename = 'tBu_P RT Ar without burst mode I = 10.dat'    ; pdfA = [1141.1 , -6606.7, 6997.0]; pdfT = [1.10029e-6*1e9, 0.17924e-6*1e9, 0.17918e-6*1e9  ]; pdfB = 0.165; logFlatB = pdfB+0.2;
# #7
# filename = 'tBu_P RT O2 I = 10.dat'                       ; pdfA = [709.4, 140.8]; pdfT = [0.18581e-6*1e9  , 0.0610e-6*1e9  ]; pdfB = -0.131; logFlatB = pdfB+0.25;
#8
# filename = 'tBu_P RT O2 I=8.5.dat'                        ; pdfA = [875.9, 322.6]; pdfT = [0.17832e-6*1e9  , 0.02459e-6*1e9  ]; pdfB = 0.961; logFlatB = pdfB-0.5;

# timeunit = 'ns'; unit=''; savelabel = ''
# quantity = filename + 'counts'
#%%
# folder   = 'ASU_Collab_IHO/DesktopMeasurements/Data'
# timeunit = 'time'

# filename = 'Rxx' ; unit = 'ohms'

# file = '../{0}/{1}'.format(folder,filename)

#-----------------------------------------------------------------------------
# Lintao
# PQ_Mode = False
# folder = 'Lintao/fig4a'; quantity = '\sigma'
# topLabel = '{\ Peng\ BP}'; unit = ' [S]'; timeunit = 'sec';
# # yrange = (0,9); xrange = (1e-4, 1e8); SE = 0;

# filename =  '288K200'       ; bottomLabel = r'{288K}'; timeunit='sec' ; comParams = np.array([12.24 ,0.434 ,66.65e-6   ,SE])
# filename =  '299K200'       ; bottomLabel = r'{299K}'; timeunit='sec' ; comParams = np.array([10.215,0.443 ,58.47e-6   ,SE])
# filename =  '314K200'       ; bottomLabel = r'{314K}'; timeunit='sec' ; comParams = np.array([ 7.05 ,0.549 ,48.50e-6   ,SE])
# filename =  '322K200'       ; bottomLabel = r'{322K}'; timeunit='sec' ; comParams = np.array([ 6.80 ,0.561 ,49.43e-6   ,SE])

# PQ_Mode = False
# folder = 'Lintao/fig2a'; quantity = '\sigma'
# topLabel = '{\ Peng\ BP}'; unit = ' [S]'; #yscale=1e5
# xrange = (1e0, 5e8); yrange = (7e-4, 14e-4); AD = 1-1e-9;
# filename =  '269K'       ; bottomLabel = r'{269K}'; timeunit='sec' ;comParams = np.array([11.491,0.568,4.41E-4,AD])
# filename =  '279K'       ; bottomLabel = r'{279K}'; timeunit='sec' ;comParams = np.array([9.365,0.589,4.41E-4,AD])
# filename =  '289K'       ; bottomLabel = r'{289K}'; timeunit='sec' ;comParams = np.array([7.364,0.616,4.36E-4,AD])
# filename =  '299K'       ; bottomLabel = r'{299K}'; timeunit='sec' ;comParams = np.array([6.686,0.615,4.1E-4,AD])
# filename =  '305K'       ; bottomLabel = r'{305K}'; timeunit='sec' ;comParams = np.array([5.628,0.611,4.07E-4,AD])
# filename =  '314K'       ; bottomLabel = r'{314K}'; timeunit='sec'; del comParams #data relevant for this file cannot be found!
# filename =  '317K'       ; bottomLabel = r'{317K}'; timeunit='sec' ;comParams = np.array([3.332,0.655,3.88E-4,AD])



#%%File Import
file = '../Literature/{0}/{1}.csv'.format(folder,filename)
# file = '../Literature/{0}/{1}'.format(folder,filename)
dataOffset = 117
dataOffset = 1
# csv  = np.genfromtxt(file, delimiter=",")
csv  = np.genfromtxt(file, delimiter=",", skip_header = 2 + dataOffset)

# csv  = np.genfromtxt(file, delimiter="\t", skip_header = 2 + dataOffset)
# csv  = np.genfromtxt(file, delimiter="\t", skip_header = 1)

csv = csv[~np.isnan(csv).any(axis=1)] # removes rows containing NaN from the data
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
#%%
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
    xnew = np.linspace(DataX[startInterpAt], DataX[-1], num=1000, endpoint=True)
    
    plt.plot(DataX, DataY, 'o', np.append(DataX[:startInterpAt],xnew), np.append(DataY[:startInterpAt],f(xnew)), 'x')

    DataX,DataY = np.append(DataX[:startInterpAt],xnew),np.append(DataY[:startInterpAt],f(xnew));
    # plt.legend(['data', 'cubic'], loc='best')
    plt.show()

#%%Yscaling
# pdb.set_trace()
# yscale = 10**-np.floor(np.log10(np.amax(DataY)))
# DataY = DataY * yscale

if yscale != 1:
    unit = '$10^{' + '{}'.format(int(-np.log10(yscale)))+ '}$ ' + unit
    try:
        comParams[2] *= yscale
        savelabel = savelabel + 'Comp'
        compare = True
    except:
        pass
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
    
    
fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.2))
# axs.scatter(DataX[1:], DataY[1:], color = 'black' ,label = ' ' ,marker = 'o',s = 5)
axs.scatter(np.exp(DataX), DataY/1e6, color = 'black' ,label = ' ' ,marker = 'o',s = 5)
axs.set_xlabel(r"$t\ [sec]$", **label_style)
# axs.set_ylabel(r"$f^{\ Münster\ UP}_{E. Coli\ 25ºC} $ x $10^6$ [count]", **label_style)
axs.set_ylabel(r"$Rxx^{\ InOH}_{A7/ IIb}$  [Ω]", **label_style)
axs.grid(color='k', linestyle='-', linewidth=0.03 , which ='both')
# axs.set_title('Raw ln(t)')

axs.grid(b = True, linestyle='-', linewidth=0.05 , which ='major')
axs.set_xscale('log')
# xlimpow = np.array([-2,64])
# xtickpow = np.linspace(xlimpow[0],xlimpow[1]+1,(int(abs(xlimpow[0]-xlimpow[1]))+2))
# xtickpow = np.linspace(-7,-1,3)
# axs.set_xticks(10**xtickpow,minor = False)
# axs.set_yticks(np.linspace(0,2,5))
# axs.set_xlim([1e-10,1e1])
axs.set_ylim([0,7e-5])



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
constraint = (['m'],['<'],[1e-9])
# print(function)

if not FitExists or ForceRecalculation:
    uBounds    = constraintBuilder('u'   , constraintSelector('u'  , DataX), constraint)
    # uBounds = (-14,-8)
    bBounds    = constraintBuilder('b'   , constraintSelector('b'  ,np.nan), constraint) 
    fdeltBounds= constraintBuilder('fdelt', constraintSelector('fdelt',DataY), constraint)
    # fdeltBounds= (-5000,5000)
    # fdeltBounds = (-0.1, 0.1)
    mBounds    = constraintBuilder('m'   , constraintSelector('m'  ,np.nan), constraint) 
    
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
#%%
############### Fast Plotting Fit with Data (no special labels) ###############
if PlotFitFast:
    fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))
    xx_lin = np.linspace(DataX[0]-8,DataX[-1]+7,10000)
    
    axs.scatter(np.exp(DataX), DataY, color = 'black' ,label = 'data' ,marker = 'o',s = 5)  #data is black
    axs.plot(np.exp(xx_lin), (function(xx_lin,*ret.x))+offset ,'-' ,color = 'blue',label = 'HT fit',lw=1)#our fit is blue
    
    axs.set_xlabel(r"$t\ [s]$", **label_style)
    axs.set_ylabel(r"Quantitiy", **label_style)
    
    axs.grid(color='k', linestyle='-', linewidth=.05 , which ='both')
    axs.set_xscale('log')
############### ############### ############### ############### ###############
#%%############ ############### ############### ############### ###############
############### 'Nice' Plotting ###############
     
fig, axs = plt.subplots(1, 1, figsize=(3.2, 3.0))
xx_lin = np.linspace(DataX[0]-4,DataX[-1]+1,10000)

tempRet = [-11.1,1,-11.2e-3,0.0]
# tempRet = ret.x
axs.scatter(np.exp(DataX), DataY, color = HTDataColor ,label = 'data' ,marker = 'o',s = 5)  #data is black
axs.plot(np.exp(xx_lin), (function(xx_lin,*ret.x))+offset ,'-' ,color = HTFitColor,label = 'HT fit',lw=0.5)#our fit is blue
# axs.plot(np.exp(xx_lin), (FHTx(xx_lin,*tempRet))-tempRet[2]-0.0005 ,'-' ,color = 'yellow',label = 'HT fit',lw=1)#our fit is blue

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
# yrange = (0.0008, 0.0015)
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
# axs.set_xticks([1e-1,1e0,1e2],minor = False)

plt.tight_layout()
add_watermark(ax=axs,version=version+savelabel)

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
                        bottomLabel, version + savelabel, savepath)
    