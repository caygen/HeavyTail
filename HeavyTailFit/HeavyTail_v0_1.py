"""
Copyright c 2021 by Northwestern University. All Rights Reserved.

@author: Can Aygen
"""
#%%#Import the Functions
#Import the Functions
#********************#
from Functions_v0_1 import *
#********************#
SE = 0;
AD = 1-1e-9;

#%%
# pdb.set_trace()
#%%
#File Import
#Initialization Constants
version     = 'Gitv0_1'
save        =  True
interpolateData = False
savelabel = str(date.today())
print(savelabel)
RawOnly = False
PlotRaw     = False
PlotFitFast = False
PQ_Mode = False
compare = False
LinearPlot = False
ForceRecalculation = True
yscale = 1; xscale = 1
quantity = 'f'
constraint = np.nan
dataOffset = 0;
topLabel = ' '; bottomLabel = ' ';
HTDataColor = 'lightcoral'; HTFitColor = 'darkred'
folder = '.'; timeunit = ' '; unit = ' '

filename = input('please enter the data-file ({data-file}.csv): ')
file = '{0}/{1}'.format(folder,filename)
df = pd.read_csv(file)
HeaderExist = input('do you have a header row in the data? (y/n) ')
if HeaderExist == 'y':
    NameAxes = True
    AxesLabels = df.columns
if HeaderExist == 'n':
    NameAxes = False
    AxesLabels = ('','')
csv = df.to_numpy('float')
# filename = 'June4b.csv'; 
try:
    del yrange
    del xrange
except:
    pass

#%%File ImportName
# import pandas as pd
# import copy
# file = '{0}/{1}'.format(folder,filename)
# dataOffset = 0

#%%File Import
# csv_reader = csv.reader(file, delimiter = ',')
# csv = pd.read_csv(file)
# Hea
# csv  = np.genfromtxt(file, delimiter=",")
# csv  = np.genfromtxt(file, delimiter=','   , skip_header = 1 + dataOffset)
# csv  = np.genfromtxt(file, delimiter='\t'   , skip_header = 1 + dataOffset)
csv = csv[~np.isnan(csv).any(axis=1)] # removes rows containing NaN from the data
fitMode = -1+float(input('Would you like to fit to a (1) \'Streched-Exponential\' or a (2) \'Power-Law\' decay equation?  \nPlease enter 1 or 2: '))

print('Analyzing' + file)

DataX = np.log(csv[:,0]*xscale)
DataY = csv[:,1]
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
ResultsFolderPath = pathlib.Path(file[:-4]+'Result{}.csv'.format(savelabel)[:-4])
ResultFolderExist = ResultsFolderPath.is_dir()
if not ResultFolderExist:
    ResultsFolderPath.mkdir()
path = pathlib.Path(str(ResultsFolderPath)+'/'+filename[:-4]+'Result{}.csv'.format(savelabel))
try:
    FitExists = path.is_file()
except:
    FitExists = path.is_file()

#%%############ ############### ############### ############### ###############
############### Fitting ###############
paraName=['u','\\beta','\\bar{f}','m']; plotParaName =['u','T','\\bar{f}','m']
function = FHTx; #print('function: FHTx (4D Solver)')
# function = ADx; print('function: FHTx (4D Solver)')
# function = F_uThetDf0x; print('function: F_uThetDf0x (3D Solver)')


if fitMode == 0:
    # constraint = (['m'],['<'],[0.01])
    # constraint = (['m'],['>'],[1-9e-3])
    mBounds = (0,0.0001)
elif fitMode == 1:
    # constraint = (['m'],['<'],[6e-4])
    # constraint = (['m'],['>'],[0.99])
    mBounds = (0.995,1)
    
# 
# print(function)

if not FitExists or ForceRecalculation:
    uBounds    = constraintBuilder('u'   , constraintSelector('u'  , DataX), constraint)
    bBounds    = constraintBuilder('b'   , constraintSelector('b'  ,np.nan), constraint)
    fdeltBounds= constraintBuilder('fdelt', constraintSelector('fdelt',DataY), constraint)
    # mBounds    = constraintBuilder('m'   , constraintSelector('m'  ,np.nan), constraint)
    
    funBounds  = boundSelector(function, uBounds, bBounds, fdeltBounds, mBounds)

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
#%%############ ############### ############### ############### ###############
############### 'Nice' Plotting ###############

fig, axs = plt.subplots(1, 1, figsize=(3.4, 3.0))
xx_lin = np.linspace(DataX[0]-1,DataX[-1]+2,10000)

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

# axs.set_xlabel(r"$t$ [{}]".format(timeunit), **label_style)

axs.grid(b = True, linestyle='-', linewidth=0.05 , which ='major')
# axs.grid(b = True, which='minor', color='r', linestyle='--', axis ='x')

axs.set_xscale('log')

xlimpow = np.floor(np.log10(axs.get_xlim()))
xtickpow = np.linspace(xlimpow[0],xlimpow[1]+1,(int(abs(xlimpow[0]-xlimpow[1]))+2))


#Optional Axes Formatiing Settings:
# axs.set_xlim(1e-3,1e7)
# yrange = (0, 2e5)

if NameAxes:
    axs.set_xlabel(AxesLabels[0])
    axs.set_ylabel(AxesLabels[1])
    
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
# print("Confidence Intervals Calculated!")
#%%
# #Create The Visual table(s)
# fig, axs = plt.subplots()
# axs.xaxis.set_visible(False)
# axs.yaxis.set_visible(False)
# axs.set_frame_on(False)
# render_mpl_table(reslts, header_columns=0, col_width=1.4, row_height=0.425, ax=axs)
# add_watermark(ax=axs,version=version+savelabel)
# if save == True:
#     fig.savefig(str(ResultsFolderPath)+'/'+filename[:-4]+'{}Results.png'.format(savelabel),dpi=fig.dpi)
#     # fig.savefig(file[:-4]+'{}Results.png'.format(savelabel),dpi=fig.dpi)
# fig, axs = plt.subplots()
# axs.xaxis.set_visible(False)
# axs.yaxis.set_visible(False)
# axs.set_frame_on(False)
# render_mpl_table(debug , header_columns=0, col_width=1.4, row_height=0.425, ax=axs)
# add_watermark(ax=axs,version=version+savelabel)
# # fig.savefig('test2.png',dpi=fig.dpi)
# if save == True:
#     # fig.savefig(file[:-4]+'{}Delta.png'.format(savelabel),dpi=fig.dpi)
#     fig.savefig(str(ResultsFolderPath)+'/'+filename[:-4]+'{}Delta.png'.format(savelabel),dpi=fig.dpi)
# # delta = deltaPar3d(CurvTensor,r,ret.fun**2)

# #Saved Table
if save:
    np.savetxt(str(ResultsFolderPath)+'/'+filename[:-4]+'Result{}.csv'.format(savelabel), b, delimiter=",",fmt='%s')
    # np.savetxt(file[:-4]+'Result{}.csv'.format(savelabel), b, delimiter=",",fmt='%s')
#%%Pico
# m=ret.x[-1]
savepath = str(ResultsFolderPath)+'/'+filename[:-4]
# sys.exit(0)
# pdb.set_trace()
