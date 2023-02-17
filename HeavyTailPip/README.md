# HeavyTail Fitting Code Packaged Version
  use the package folloiwing the example
```
  from HeavyTailPip import HeavyTail
```

On your terminal page execute to download and setup the pakage:
```
pip install HeavyTail
```

Afterwards in your python analysis code all you need to do the following 
import HeavyTail as ht #run this at the beginning of your code
…
#data collection instrument control etc. or anything else
…
## Analysis
only need to choose one of the below options
```
ht.Fit(filename)     #queries if you want the m=1 (SE) or m=2 (AD) fit
ht.FitSE(filename)   #fits the input file to the m = 1 (stretched exponential relaxation) - unimolecular relaxation
ht.FitAD(filename)   #fits the inpit file to the m = 2 (algebraic decay relaxation)       - bimolecular relaxation
Ht.FitFree(filename)
```
