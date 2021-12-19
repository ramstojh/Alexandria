from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
from PyAstronomy import pyasl
import time
import sys
from tqdm import trange
import pandas as pd
from kapteyn import kmpfit
from PyPDF2 import PdfFileReader, PdfFileWriter
from numpy.random import normal
from pylab import *
from scipy.stats import *
from scipy.stats import t
path = os.getcwd()
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore', category=UserWarning, append=True)

#data path
this_dir, this_filename = os.path.split(__file__)

#########################################################################################################
####################################### For making plots ################################################
#########################################################################################################
#set global settings for plotting
def plotting():
  #  plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 17                                     #set the size of the font numbers       
    plt.rcParams['font.family'] = 'fantasy'                            #choose the style of the numbers
    plt.rcParams['axes.labelsize'] = 20                                #set the size of the word axes
    #plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    #plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']        #set the size of the label x
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']        #set the size of the label y
    plt.rcParams['xtick.major.size'] = 7                               #set the length of the x tick
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.major.width'] = 2                              #set the width of the x tick 
    #plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 7                               #set the length of the y tick
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['ytick.major.width'] = 2                              #set the width of the y tick
    #plt.rcParams['ytick.minor.width'] = 1.2
    #plt.rcParams['legend.frameon'] = True
    #plt.rcParams['legend.loc'] = 'upper left'
    plt.rcParams['axes.linewidth'] = 3

 #   plt.gca().spines['right'].set_color('none')
 #   plt.gca().spines['top'].set_color('none')
 #   plt.gca().xaxis.set_ticks_position('bottom')
 #   plt.gca().yaxis.set_ticks_position('left')


#########################################################################################################
####################################### Stellar Parametes ###############################################
#########################################################################################################
#logg from parallaxes
#Bolometric corrections from Melendez et. al 2016 (http://adsabs.harvard.edu/abs/2006ApJ...641L.133M)
#M_{bol, \odot} = 4.736 (Bessel et al. 2018)
def logg_from_plx(teff, err_teff, mass, err_mass, vmag, err_vmag, plx, err_plx):
  #Sun's parameters
  teff_sun = 5777
  mass_sun = 1.0
  logg_sun = 4.44
  err_BC   = 0.02
  Mbol_sun = 4.74 #Bessel 1998

  theta          = 5040./teff
  BC             = -1.6240 + 4.5066*theta - 3.12936*theta**2  #(\sigma = 0.02 mag)
  X              = np.log10(mass/mass_sun) + 4*np.log10(teff/teff_sun) + 0.4*vmag + 0.4*BC +2*np.log10(plx/1000) + 0.104
  logg_plx       = logg_sun + X

  sigma_logg_plx = np.sqrt((0.434*err_mass/mass)**2 + (0.434*err_teff/teff)**2 + err_vmag**2 + err_BC**2 + (0.434*err_plx/plx)**2) 

  return logg_plx, sigma_logg_plx

#########################################################################################################
####################################### Vmacro ##########################################################
#########################################################################################################
#vmacro for a star following Tucci Maia 2015 and do Santos 2016.
def vmacro_sf(vmacro_sun, teff, logg):
    #errors for vm_leo = 0.4 km/s
    k1       = -1.81 #+/-0.26
    k2       = -0.05 #+/-0.03
    
    vm_tucci = vmacro_sun + (teff-5777)/486.
    vm_leo   = vmacro_sun - 0.00707*teff + 9.2422*10**(-7)*teff**2 + 10. + k1*(logg-4.44) + k2
    
    return vm_leo, vm_tucci    



#########################################################################################################
####################################### Age from Li #####################################################
#########################################################################################################
#vmacro for a star following Carlos et al. 2019.
def Liage(Li, err_Li):
    
    Age     = (2.44 - Li)/0.2
    err_sum = np.sqrt(err_Li**2 + 0.1**2)
    err_age = Age*np.sqrt((err_sum/(2.44 - Li))**2 + (0.02/0.2)**2)
    
    return Age, err_age



#########################################################################################################
################################################# GCE ###################################################
#########################################################################################################
#the gce is performed with the slope
#b is the slope in spina+ 2016
#m is the slope in bedell+ 2018
def CGE_2018(specie, ab_specie, err_ab_specie, age, err_age):
    Age_Sun = 4.6
    PATH_GCE = os.path.join(this_dir, 'data', 'GCE_Megan_2018.csv')
    table    = pd.read_csv(PATH_GCE, index_col ='Species')
    #table   = pd.read_csv('~/Dropbox/PyJao/tables/GCE_Megan_2018.csv', index_col ='Species')
    rows     = table.loc[specie]

    if Age_Sun > age:
        diff_age  = Age_Sun - age
        gce       = ab_specie + rows['m']*diff_age
        if rows['m'] == 0.0:
            err_gce = err_ab_specie
        else:
            err_age_b = rows['m']*(diff_age)*np.sqrt((rows['err_m']/rows['m'])**2 + (err_age/diff_age)**2)
            err_gce   = np.sqrt(err_ab_specie**2 + (err_age_b)**2)

    else:
        diff_age  = age - Age_Sun
        gce       = ab_specie + rows['m']*diff_age
        if rows['m'] == 0.0:
            err_gce = err_ab_specie
        else:
            err_age_b = rows['m']*(diff_age)*np.sqrt((rows['err_m']/rows['m'])**2 + (err_age/diff_age)**2)
            err_gce   = np.sqrt(err_ab_specie**2 + (err_age_b)**2)

    return gce, err_gce

#########################################################################################################
#generating tables after GCE
#input is a table
def tab_GCE(input, age, err_age):
  data           = pd.read_csv(input)
  gce_result     = []
  err_gce_result = []
  for i in range(len(data['element'])):
      gce, err_gce    = CGE_2018(data['element'].values[i], data['[X/H]'].values[i], data['err_[X/H]'].values[i], age, err_age)
      gce_result.append(gce)
      err_gce_result.append(err_gce)

  data['gce']     = gce_result
  data['err_gce'] = err_gce_result
  #data['err_gce'] = data['err_gce'].fillna(0)

  #media ponderada Sc
  ScI                  = data.loc[data['element'] == 'ScI']
  ScII                 = data.loc[data['element'] == 'ScII']
  XSc, err_XSc         = med_pond(ScI['[X/H]'].values, ScI['err_[X/H]'].values, ScII['[X/H]'].values, ScII['err_[X/H]'].values)
  XSc_gce, err_XSc_gce = med_pond(ScI['gce'].values, ScI['err_gce'].values, ScII['gce'].values, ScII['err_gce'].values)
  dat_pond_Sc          = {'Z': 21, 'element': 'Sc', 'Tcond': 1659, '[X/H]': XSc, 'err_[X/H]': err_XSc, 'gce': XSc_gce, 'err_gce': err_XSc_gce}
  df_Sc                = pd.DataFrame(dat_pond_Sc)

  #media ponderada Ti
  TiI                  = data.loc[data['element'] == 'TiI']
  TiII                 = data.loc[data['element'] == 'TiII']
  XTi, err_XTi         = med_pond(TiI['[X/H]'].values, TiI['err_[X/H]'].values, TiII['[X/H]'].values, TiII['err_[X/H]'].values)
  XTi_gce, err_XTi_gce = med_pond(TiI['gce'].values, TiI['err_gce'].values, TiII['gce'].values, TiII['err_gce'].values)
  dat_pond_Ti          = {'Z': 22, 'element': 'Ti', 'Tcond': 1582, '[X/H]': XTi, 'err_[X/H]': err_XTi, 'gce': XTi_gce, 'err_gce': err_XTi_gce}
  df_Ti                = pd.DataFrame(dat_pond_Ti)

  #media ponderada Cr
  CrI                  = data.loc[data['element'] == 'CrI']
  CrII                 = data.loc[data['element'] == 'CrII']
  XCr, err_XCr         = med_pond(CrI['[X/H]'].values, CrI['err_[X/H]'].values, CrII['[X/H]'].values, CrII['err_[X/H]'].values)
  XCr_gce, err_XCr_gce = med_pond(CrI['gce'].values, CrI['err_gce'].values, CrII['gce'].values, CrII['err_gce'].values)
  dat_pond_Cr          = {'Z': 24, 'element': 'Cr', 'Tcond': 1296, '[X/H]': XCr, 'err_[X/H]': err_XCr, 'gce': XCr_gce, 'err_gce': err_XCr_gce}
  df_Cr                = pd.DataFrame(dat_pond_Cr)

  #removing rows
  data = data[data.element != 'ScI'] 
  data = data[data.element != 'ScII']
  data = data[data.element != 'TiI']
  data = data[data.element != 'TiII']
  data = data[data.element != 'CrI']
  data = data[data.element != 'CrII']

  #concatenating rows
  frames = [data, df_Sc, df_Ti, df_Cr]
  result = pd.concat(frames, ignore_index=True)
  result = result.sort_values(by=['Z'])
  result.to_csv('GCE'+input, index=False)
  
  return result

#########################################################################################################
#transformig from [X/H] and [Y/H] to [X/Y]
def ab_XY(X, err_X, Y, err_Y):
    XY     = X - Y
    err_XY = np.sqrt(err_X**2 + err_Y**2)
    return XY, err_XY

#estimating ages from [Y/MG]-age (Spina 2018) 
def chemical_clock_spina(Y, err_Y, Mg, err_Mg, Al, err_Al):
  #for YMG-age relation (equation 4 therein)
  YMg, err_YMg = ab_XY(Y, err_Y, Mg, err_Mg)
  age_YMg      = (0.204 - YMg)/0.046
  diff_YMg     = np.sqrt(0.014**2 + err_YMg**2)
  err_age_YMg  = age_YMg*np.sqrt((diff_YMg/(0.204-YMg))**2 + (0.002/0.046)**2)

  #for YAl-age relation (equation 5 therein)
  YAl, err_YAl = ab_XY(Y, err_Y, Al, err_Al)
  age_YAl      = (0.231 - YAl)/0.051
  diff_YAl     = np.sqrt(0.014**2 + err_YAl**2)
  err_age_YAl  = age_YAl*np.sqrt((diff_YAl/(0.231-YAl))**2 + (0.002/0.051)**2)

  return age_YMg, err_age_YMg, age_YAl, err_age_YAl

#estimating ages from [Y/MG]-age (Nissen 2018)
def chemical_clock_nissen(Y, err_Y, Mg, err_Mg, Al, err_Al):
  #for YMG-age relation (equation 5 therein)
  YMg, err_YMg = ab_XY(Y, err_Y, Mg, err_Mg)
  age_YMg      = (0.170 - YMg)/0.0371
  diff_YMg     = np.sqrt(0.009**2 + err_YMg**2)
  err_age_YMg  = age_YMg*np.sqrt((diff_YMg/(0.170-YMg))**2 + (0.0013/0.0371)**2)

  #for YAl-age relation (equation 6 therein)
  YAl, err_YAl = ab_XY(Y, err_Y, Al, err_Al)
  age_YAl      = (0.196 - YAl)/0.0427
  diff_YAl     = np.sqrt(0.009**2 + err_YAl**2)
  err_age_YAl  = age_YAl*np.sqrt((diff_YAl/(0.196-YAl))**2 + (0.0014/0.0427)**2)

  return age_YMg, err_age_YMg, age_YAl, err_age_YAl

#estimating age from activity-age correlation (Diego 2018)
def activity_age_diego(logRhk, err_logRhk):
  #equation 11 therein if your solar twin is in Diego 2018
  log_Age   = 0.0534 - 1.92*logRhk
  Age       = 10**(log_Age)*10**-9

  return Age

#########################################################################################################
#getting stellar parameters, ages, and abundances of solar twins
def sp_st_2018():
    PATH_st_ab = os.path.join(this_dir, 'data', 'st_ab_Megan_Spina_2018.csv')
    data = pd.read_csv(PATH_st_ab)
    #data = pd.read_csv('~/Dropbox/PyJao/tables/st_ab_Megan_Spina_2018.csv')
    return data

#########################################################################################################
#getting stellar parameters, ages, and abundances kepler legacy stars Nissen+ 2017
def kepler_legacy_2017():
    PATH_kepler = os.path.join(this_dir, 'data', 'st_ab_Megan_Spina_2018.csv')
    data = pd.read_csv(PATH_kepler)
    #data = pd.read_csv('~/Dropbox/PyJao/tables/kepler_legacy_nissen_2017.csv')
    return data


#########################################################################################################
######################################## Stellar Activity ###############################################
#########################################################################################################
#Calculate logRhk(teff) according to Lorenzo-Oliveira + 2018
def R_teff(S,eS,teff):
  def LO(S,eS,teff):
	  #### MC input
	  #single S index measured
	  #single sigma S index measured
	  #teff 
	  #### MC output
	  # 10000 logRhk(teff) values following S +- eS measurements
	  S = normal(loc=S, scale=eS, size=int(1e4))
	  Rphot = (10**(-4.78845 -3.70700/((1. + (teff/4598.92)**(17.5272)))))
	  logCcf= -1.70*(10**(-7))*teff**2 + 2.25*(10**(-3))*teff - 7.31
	  RHK = 1.34*(10**(-4))*(10**logCcf)*S
	  xx = log10((RHK - Rphot))
	  return xx

  #calculate logRhk(teff) distribution using single Sindex values
  #xx = LO(S=0.170,eS=0.003,teff=5777)
  xx = LO(S, eS, teff)
  xx = xx[~np.isnan(xx)]

  #estimate median and std
  lrlhk,slrlhk=np.median(xx),np.std(xx)

  #estimate cycle variations using LO18 eq.
  slrlhk_cycle = 0.62+0.119*lrlhk

  #propagate cycle dispersion with single error measurements
  slrlhk=sqrt(slrlhk**2+slrlhk_cycle**2)
  
  #estimate chromospheric ages
  tt=10**(-9+ 0.0534 -1.92*lrlhk)
  tt1=10**(-9+ 0.0534 -1.92*(lrlhk+slrlhk) )
  tt2=10**(-9+ 0.0534 -1.92*(lrlhk-slrlhk))
  

  return lrlhk, slrlhk, tt,tt1,tt2


#########################################################################################################
################################### Reading FITS (ascii) Table ##########################################
#########################################################################################################
#reading FITS (ascii) from vizier
from astropy.table import Table
def read_fits_ascii(input, output):
  t = Table.read(input)
  t.write(output, delimiter=',')

#########################################################################################################
####################################### making histograms ##############################################
#########################################################################################################
#histograms
def simple_histogram(x, xlabel, bins, output):
  plt.hist(x=x, bins=bins, color='#3498db')
  plt.xlabel(xlabel, fontsize=17)
  plt.ylabel('N', fontsize=17)
  #plt.tick_params(axis='both', labelsize=15)
  plt.savefig(output)

#########################################################################################################
###################################### Fitting with kmpfit ##############################################
#########################################################################################################
#simple fit
#important to give beta = [a, b] as first guess
def kmpfit_both_errs(x, y, errx, erry, beta):
  # Model is staight line: y = a + b*x
  def model(p, x):
    a, b = p
    return a + b*x
  
  # Residuals function for effective variance
  def residuals(p, data):
    a, b = p
    x, y, ex, ey = data
    w = ey*ey + ex*ex*b*b
    wi = np.sqrt(np.where(w==0.0, 0.0, 1.0/(w)))
    d = wi*(y-model(p,x))
    return d
  
  #Prepare fit routine
  #beta   = [a, b]
  n       = len(y) 
  # Prepare fit routine
  fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errx, erry))
  try:
    fitobj.fit(params0=beta)
  except Exception as mes:
   print("Something wrong with fit: ", mes)
   raise SystemExit

  y_modkmp = fitobj.params[1]*x + fitobj.params[0]     #y_model       
  rmse_kmp = np.linalg.norm(y_modkmp - y) / np.sqrt(n)  #rmse

  print ("\n\n======== Results kmpfit: w1 = ey*ey + b*b*ex*ex =========")
  print ("\nFit status kmpfit:", fitobj.message)
  print ("Params:                 ", fitobj.params)
  print ("Covariance errors:      ", fitobj.xerror)
  print ("Standard errors         ", fitobj.stderr)
  print ("Chi^2 min:              ", fitobj.chi2_min)
  print ("Reduced Chi^2:          ", fitobj.rchi2_min)
  print ("RMSE                    ", rmse_kmp)

  return fitobj.params[0], fitobj.params[1]


#simple linear fit with kmpfit
def kmpfit_simple_fit(x, y, beta):
  # The model
  def model(p, x):
    a,b = p
    y = a + b*x
    return y

  # The residual function
  def residuals(p, data):
    x, y = data
    return y - model(p,x)
  
  # Prepare a 'Fitter' object'
  #beta  = [0, 0])
  n       = len(y)
  fitobj = kmpfit.Fitter(residuals=residuals, data=(x,y))

  try:
    fitobj.fit(params0=beta)
  except Exception as mes:
    print("Something wrong with fit: ", mes)
    raise SystemExit
  
  y_modkmp = fitobj.params[1]*x + fitobj.params[0]     #y_model
  rmse_kmp = np.linalg.norm(y_modkmp - y) / np.sqrt(n)  #rmse

  print("Fit status: ", fitobj.message)
  print("Best-fit parameters:      ", fitobj.params)
  print("Covariance errors:        ", fitobj.xerror)
  print("Standard errors           ", fitobj.stderr)
  print("Chi^2 min:                ", fitobj.chi2_min)
  print("Reduced Chi^2:            ", fitobj.rchi2_min)
  print ("RMSE                     ", rmse_kmp)
  #print("Iterations:               ", fitobj.niter)
  #print("Number of function calls: ", fitobj.nfev)
  #print("Number of free pars.:     ", fitobj.nfree)
  #print("Degrees of freedom:       ", fitobj.dof)
  #print("Number of pegged pars.:   ", fitobj.npegged)
  #print("Covariance matrix:\n", fitobj.covar)

  return fitobj.params[0], fitobj.params[1]


#linear fit with errors in y only
def kmpfit_fit_erry(x, y, erry, beta):
    #the model
    def model(p, x):
        a, b = p
        return a + b*x
    
    #the residual function
    def residuals2(p, data):
        # Residuals function for data with errors in y only
        a, b = p
        x, y, ey = data
        wi = np.where(ey==0.0, 0.0, 1.0/ey)
        d = wi*(y-model(p,x))
        return d
    
    # Prepare a 'Fitter' object'
    #beta  = [0, 0])
    n       = len(y)
    fitobj2 = kmpfit.Fitter(residuals=residuals2, data=(x, y, erry))
    fitobj2.fit(params0=beta)
    
    y_modkmp = fitobj2.params[1]*x + fitobj2.params[0]     #y_model       
    rmse_kmp = np.linalg.norm(y_modkmp - y) / np.sqrt(n)  #rmse
    
    print("\n\n======== Results kmpfit errors in Y only =========")
    print("Fitted parameters:      ", fitobj2.params)
    print("Covariance errors:      ", fitobj2.xerror)
    print("Standard errors         ", fitobj2.stderr)
    print("Chi^2 min:              ", fitobj2.chi2_min)
    print("Reduced Chi^2:          ", fitobj2.rchi2_min)
    print ("RMSE                   ", rmse_kmp)
    
    return fitobj2.params[0], fitobj2.params[1]
  
  
 
########### confidence band ##########
def linear_confidence_band(x, y, errx, erry, beta):
    def model(p, x):
        # Model is staight line: y = a + b*x
        a, b = p
        return a + b*x
    
    def residuals(p, data):
        # Residuals function for effective variance
        # Residuals function for data with errors in both coordinates
        a, b = p
        x, y, ex, ey = data
        w = ey*ey + b*b*ex*ex
        wi = np.sqrt(np.where(w==0.0, 0.0, 1.0/(w)))
        d = wi*(y-model(p,x))
        return d
    
    def confidence_band(x, dfdp, confprob, fitobj, f, abswei=False):
        # Given the confidence probability confprob = 100(1-alpha)
        # we derive for alpha: alpha = 1 - confprob/100 
        alpha = 1 - confprob/100.0
        prb = 1.0 - alpha/2
        tval = t.ppf(prb, fitobj.dof)
   
        C = fitobj.covar
        n = len(fitobj.params)              # Number of parameters from covariance matrix
        p = fitobj.params
        N = len(x)
        if abswei:
            covscale = 1.0
        else:
            covscale = fitobj.rchi2_min
        df2 = np.zeros(N)
        for j in range(n):
            for k in range(n):
                df2 += dfdp[j]*dfdp[k]*C[j,k]
        df = np.sqrt(fitobj.rchi2_min*df2)
        y = f(p, x)
        delta = tval * df   
        upperband = y + delta
        lowerband = y - delta 
        return y, upperband, lowerband
    
    #Prepare fit routine
    #beta   = [a, b]
    n       = len(y) 
    # Prepare fit routine
    fitobj = kmpfit.Fitter(residuals=residuals, data=(x, y, errx, erry))
    fitobj.fit(params0=beta)
    y_modkmp = fitobj.params[1]*x + fitobj.params[0]     #y_model       
    rmse_kmp = np.linalg.norm(y_modkmp - y) / np.sqrt(n)  #rmse
    
    #calling confidence band at 95%
    x_range = np.linspace(11, -1, 100)
    df = [1, x_range]
    confprob = 95.0
    ymmm, upperband, lowerband = confidence_band(x_range, df, confprob, fitobj, model)
    verts = list(zip(x_range, lowerband)) + list(zip(x_range[::-1], upperband[::-1]))
    
    #model to plot
    y_mod = fitobj.params[0] + fitobj.params[1]*x_range
    
    return x_range, y_mod, verts#, upperband, lowerband UNCOMMENT THIS LINE FOR OTHER PLOTS


########### simple linear fit with statsmod ##########
def linear_fit_statsmod(x, y):
    x = sm.add_constant(x)     #Add the column of ones to the inputs to calculate the intercept b0. It doesn’t takes b0 into account by default
    model = sm.OLS(y, x)       #Regression model based on ordinary least squares
    results = model.fit()
    
    return results.summary()


#########################################################################################################
####################################### Jao Python Modules ##############################################
#########################################################################################################
#media ponderada de dos numeros y sus errores
#https://www.uv.es/zuniga/07_Valores_medios_ponderados.pdf
def med_pond(x, err_x, y, err_y):
  wx = 1/err_x**2
  wy = 1/err_y**2

  X     = (wx*x + wy*y)/(wx + wy)
  sigma = 1/np.sqrt(wx + wy)

  return X, sigma




    






