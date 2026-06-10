import numpy as np
from astropy.io import fits
from astropy.modeling import functional_models, models, fitting
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.aperture import ApertureStats
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.wcs import WCS
import glob
import os
import scipy
from lmfit import Model, Parameters
import lib

cfg = lib.get_config_file("./config_focusY.yaml")

def gauss2D_circ(x, y, A, x0, y0, sigma):
    #2d circular gaussian
    sigma_x = sigma
    sigma_y = sigma
    
    theta = 0 #np.radians(theta)
    sigx2 = sigma_x**2; 
    sigy2 = sigma_y**2
    
    a = np.cos(theta)**2/(2*sigx2) + np.sin(theta)**2/(2*sigy2)
    b = np.sin(theta)**2/(2*sigx2) + np.cos(theta)**2/(2*sigy2)
    c = np.sin(2*theta)/(4*sigx2) - np.sin(2*theta)/(4*sigy2)
    expo = -a*(x-x0)**2 - b*(y-y0)**2 - 2*c*(x-x0)*(y-y0)
            
    return A*np.exp(expo)

def DoubleGauss2D_circ(x,y,x0,y0,A1,A2,sigma1,sigma2):
    #double circular gaussian (primary beam + halo)
    
    return gauss2D_circ(x,y, A1, x0, y0, sigma1) + gauss2D_circ(x,y,A2,x0,y0,sigma2)

def gauss2D_rot(x, y, A, x0, y0, sigma_x, sigma_y, theta):
    theta = np.radians(theta)
    sigx2 = sigma_x**2; sigy2 = sigma_y**2

    a = np.cos(theta)**2/(2*sigx2) + np.sin(theta)**2/(2*sigy2)
    b = np.sin(theta)**2/(2*sigx2) + np.cos(theta)**2/(2*sigy2)
    c = np.sin(2*theta)/(4*sigx2) - np.sin(2*theta)/(4*sigy2)
    expo = -a*(x-x0)**2 - b*(y-y0)**2 - 2*c*(x-x0)*(y-y0)

    f = A*np.exp(expo)

    return f

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)
gauss_model = Model(gaussian)

def gaussian_sigma(x, amp, cen, wid):
    return - amp * np.exp(-(x-cen)**2 / wid)
gauss_sigma_model = Model(gaussian_sigma)

def parabola(x, a, b, c):
    return a * x**2 + b * x + c
par_model = Model(parabola)

def parabola_sigma(x, a, b, c):
    return -a * x**2 + b * x + c
par_sigma_model = Model(parabola_sigma)

def fwhm_arcsec(sigma):
    return cfg['pixel_size']*2.355*np.array(sigma) 

def plot_zfocus_circ(srp_tz, A_focus, A_err_focus, sigma1_focus, sigma1_err_focus):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].scatter(srp_tz, A_focus, label='data')
    #ax[0].errorbar(srp_tz, A_focus, A_err_focus)
    #result_gauss = par_model.fit(A_focus[0:-1], x=srp_tz[0:-1])#, a=np.max(A_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(A_err_focus[0:-1]))
    fit_params_A, pcov_A = scipy.optimize.curve_fit(parabola, srp_tz, A_focus, sigma=A_err_focus)
    y_fit_A = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_A)
    ax[0].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), y_fit_A, '-', label='best parabolic fit', c='orange')
    #plt.plot(srp_tz, result_par.best_fit, '-', label='best parabolic fit')
    #ax[0].scatter(srp_tz, A_focus_conv, label='Fit on convolved image')
    max_A = np.nanargmin(y_fit_A)
    ax[0].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[max_A], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[max_A]:.2f}',linestyle='--',  c='black', alpha=0.3)
    ax[0].set_title('Amplitude')
    ax[0].legend()
    ax[0].set_xlabel('z')
    ax[0].set_ylabel('A')
    ax[1].scatter(srp_tz, fwhm_arcsec(sigma1_focus), label='data')
    #ax[1].errorbar(srp_tz, fwhm_arcsec(sigma1_focus), sigma1_err_focus)
    #result_gauss1 = par_sigma_model.fit(sigma1_focus[0:-1], x=srp_tz[0:-1])#, a=np.min(sigma1_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(sigma1_err_focus[0:-1]))
    fit_params_sigma, pcov_sigma = scipy.optimize.curve_fit(parabola, srp_tz, sigma1_focus, sigma=sigma1_err_focus)
    y_fit_sigma = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_sigma)
    ax[1].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), fwhm_arcsec(y_fit_sigma), '-', label='best parabolic fit', c='orange')
    min_sigma = np.nanargmin(y_fit_sigma)
    ax[1].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma]:.2f}',linestyle='--',  c='black', alpha=0.3)
    #ax[1].scatter(srp_tz, sigma_x_focus_conv, label='Fit on convolved image')
    ax[1].legend()
    ax[1].set_title('FWHM')
    ax[1].set_xlabel('z')
    ax[1].set_ylabel('FWHM [arcsec]')
    plt.show()

def plot_zfocus_with_error_circ(srp_tz, A_focus, A_err_focus, sigma1_focus, sigma1_err_focus):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].scatter(srp_tz, A_focus, label='data')
    ax[0].errorbar(srp_tz, A_focus, A_err_focus)
    #result_gauss = par_model.fit(A_focus[0:-1], x=srp_tz[0:-1])#, a=np.max(A_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(A_err_focus[0:-1]))
    fit_params_A, pcov_A = scipy.optimize.curve_fit(parabola, srp_tz, A_focus, sigma=A_err_focus)
    y_fit_A = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_A)
    ax[0].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), y_fit_A, '-', label='best parabolic fit', c='orange')
    #plt.plot(srp_tz, result_par.best_fit, '-', label='best parabolic fit')
    #ax[0].scatter(srp_tz, A_focus_conv, label='Fit on convolved image')
    max_A = np.nanargmin(y_fit_A)
    ax[0].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[max_A], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[max_A]:.2f}',linestyle='--',  c='black', alpha=0.3)
    ax[0].set_title('Amplitude')
    ax[0].legend()
    ax[0].set_xlabel('z')
    ax[0].set_ylabel('A')
    ax[1].scatter(srp_tz, fwhm_arcsec(sigma1_focus), label='data')
    ax[1].errorbar(srp_tz, fwhm_arcsec(sigma1_focus), sigma1_err_focus)
    #result_gauss1 = par_sigma_model.fit(sigma1_focus[0:-1], x=srp_tz[0:-1])#, a=np.min(sigma1_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(sigma1_err_focus[0:-1]))
    fit_params_sigma, pcov_sigma = scipy.optimize.curve_fit(parabola, srp_tz, sigma1_focus, sigma=sigma1_err_focus)
    y_fit_sigma = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_sigma)
    ax[1].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), fwhm_arcsec(y_fit_sigma), '-', label='best parabolic fit', c='orange')
    min_sigma = np.nanargmin(y_fit_sigma)
    ax[1].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma]:.2f}',linestyle='--',  c='black', alpha=0.3)
    #ax[1].scatter(srp_tz, sigma_x_focus_conv, label='Fit on convolved image')
    ax[1].legend()
    ax[1].set_title('FWHM')
    ax[1].set_xlabel('z')
    ax[1].set_ylabel('FWHM [arcsec]')
    plt.show()

def plot_zfocus_ell(srp_tz, A_focus, A_err_focus, sigma1_focus, sigma1_err_focus, sigma2_focus, sigma2_err_focus, A_mappa):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].scatter(srp_tz, A_focus, label='data')
    ax[0].scatter(srp_tz, A_mappa, label='map min')
    #ax[0].errorbar(srp_tz, A_focus, A_err_focus)
    #result_gauss = par_model.fit(A_focus[0:-1], x=srp_tz[0:-1])#, a=np.max(A_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(A_err_focus[0:-1]))
    fit_params_A, pcov_A = scipy.optimize.curve_fit(parabola, srp_tz, A_focus, sigma=A_err_focus)
    y_fit_A = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_A)
    ax[0].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), y_fit_A, '-', label='best parabolic fit', c='orange')
    #plt.plot(srp_tz, result_par.best_fit, '-', label='best parabolic fit')
    #ax[0].scatter(srp_tz, A_focus_conv, label='Fit on convolved image')
    max_A = np.nanargmin(y_fit_A)
    ax[0].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[max_A], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[max_A]:.2f}',linestyle='--',  c='black', alpha=0.3)
    ax[0].set_title('Amplitude')
    ax[0].legend()
    ax[0].set_xlabel('z')
    ax[0].set_ylabel('A')
    ax[1].scatter(srp_tz, fwhm_arcsec(sigma1_focus), label='data')
    #ax[1].errorbar(srp_tz, fwhm_arcsec(sigma1_focus), sigma1_err_focus)
    #result_gauss1 = par_sigma_model.fit(sigma1_focus[0:-1], x=srp_tz[0:-1])#, a=np.min(sigma1_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(sigma1_err_focus[0:-1]))
    fit_params_sigma, pcov_sigma = scipy.optimize.curve_fit(parabola, srp_tz, sigma1_focus, sigma=sigma1_err_focus)
    y_fit_sigma = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_sigma)
    ax[1].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), fwhm_arcsec(y_fit_sigma), '-', label='best parabolic fit', c='orange')
    min_sigma = np.nanargmin(y_fit_sigma)
    ax[1].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma]:.2f}',linestyle='--',  c='black', alpha=0.3)
    #ax[1].scatter(srp_tz, sigma_x_focus_conv, label='Fit on convolved image')
    ax[1].legend()
    ax[1].set_title('FWHM_x')
    ax[1].set_xlabel('z')
    ax[1].set_ylabel('FWHM [arcsec]')
    ax[2].scatter(srp_tz, fwhm_arcsec(sigma2_focus), label='data')
    #ax[2].errorbar(srp_tz, fwhm_arcsec(sigma2_focus), sigma2_err_focus)
    #result_gauss1 = par_sigma_model.fit(sigma1_focus[0:-1], x=srp_tz[0:-1])#, a=np.min(sigma1_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(sigma1_err_focus[0:-1]))
    fit_params_sigma, pcov_sigma = scipy.optimize.curve_fit(parabola, srp_tz, sigma2_focus, sigma=sigma2_err_focus)
    y_fit_sigma = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_sigma)
    ax[2].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), fwhm_arcsec(y_fit_sigma), '-', label='best parabolic fit', c='orange')
    min_sigma = np.nanargmin(y_fit_sigma)
    ax[2].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma]:.2f}',linestyle='--',  c='black', alpha=0.3)
    #ax[1].scatter(srp_tz, sigma_x_focus_conv, label='Fit on convolved image')
    ax[2].legend()
    ax[2].set_title('FWHM_y')
    ax[2].set_xlabel('z')
    ax[2].set_ylabel('FWHM [arcsec]')
    plt.show()
    
def plot_zfocus_with_error_ell(srp_tz, A_focus, A_err_focus, sigma1_focus, sigma1_err_focus, sigma2_focus, sigma2_err_focus):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].scatter(srp_tz, A_focus, label='data')
    ax[0].errorbar(srp_tz, A_focus, A_err_focus)
    #result_gauss = par_model.fit(A_focus[0:-1], x=srp_tz[0:-1])#, a=np.max(A_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(A_err_focus[0:-1]))
    fit_params_A, pcov_A = scipy.optimize.curve_fit(parabola, srp_tz, A_focus, sigma=A_err_focus)
    y_fit_A = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_A)
    ax[0].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), y_fit_A, '-', label='best parabolic fit', c='orange')
    #plt.plot(srp_tz, result_par.best_fit, '-', label='best parabolic fit')
    #ax[0].scatter(srp_tz, A_focus_conv, label='Fit on convolved image')
    max_A = np.nanargmin(y_fit_A)
    ax[0].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[max_A], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[max_A]:.2f}',linestyle='--',  c='black', alpha=0.3)
    ax[0].set_title('Amplitude')
    ax[0].legend()
    ax[0].set_xlabel('z')
    ax[0].set_ylabel('A')
    ax[1].scatter(srp_tz, fwhm_arcsec(sigma1_focus), label='data')
    ax[1].errorbar(srp_tz, fwhm_arcsec(sigma1_focus), sigma1_err_focus)
    #result_gauss1 = par_sigma_model.fit(sigma1_focus[0:-1], x=srp_tz[0:-1])#, a=np.min(sigma1_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(sigma1_err_focus[0:-1]))
    fit_params_sigma, pcov_sigma = scipy.optimize.curve_fit(parabola, srp_tz, sigma1_focus, sigma=sigma1_err_focus)
    y_fit_sigma = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_sigma)
    ax[1].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), fwhm_arcsec(y_fit_sigma), '-', label='best parabolic fit', c='orange')
    min_sigma = np.nanargmin(y_fit_sigma)
    ax[1].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma]:.2f}',linestyle='--',  c='black', alpha=0.3)
    #ax[1].scatter(srp_tz, sigma_x_focus_conv, label='Fit on convolved image')
    ax[1].legend()
    ax[1].set_title('FWHM_x')
    ax[1].set_xlabel('z')
    ax[1].set_ylabel('FWHM [arcsec]')
    ax[2].scatter(srp_tz, fwhm_arcsec(sigma2_focus), label='data')
    ax[2].errorbar(srp_tz, fwhm_arcsec(sigma2_focus), sigma2_err_focus)
    #result_gauss1 = par_sigma_model.fit(sigma1_focus[0:-1], x=srp_tz[0:-1])#, a=np.min(sigma1_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(sigma1_err_focus[0:-1]))
    fit_params_sigma, pcov_sigma = scipy.optimize.curve_fit(parabola, srp_tz, sigma2_focus, sigma=sigma2_err_focus)
    y_fit_sigma = parabola(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), *fit_params_sigma)
    ax[2].plot(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1), fwhm_arcsec(y_fit_sigma), '-', label='best parabolic fit', c='orange')
    min_sigma = np.nanargmin(y_fit_sigma)
    ax[2].axvline(np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma], label=f'min at {np.arange(np.min(srp_tz), np.max(srp_tz), 0.1)[min_sigma]:.2f}',linestyle='--',  c='black', alpha=0.3)
    #ax[1].scatter(srp_tz, sigma_x_focus_conv, label='Fit on convolved image')
    ax[2].legend()
    ax[2].set_title('FWHM_y')
    ax[2].set_xlabel('z')
    ax[2].set_ylabel('FWHM [arcsec]')
    plt.show()

def plot_yfocus(srp_ty, A_focus, A_err_focus, sigma1_focus, sigma1_err_focus):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].scatter(srp_ty, A_focus, label='data')
    ax[0].errorbar(srp_ty, A_focus, A_err_focus)
    #result_gauss = par_model.fit(A_focus[0:-1], x=srp_tz[0:-1])#, a=np.max(A_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(A_err_focus[0:-1]))
    fit_params_A, pcov_A = scipy.optimize.curve_fit(parabola, srp_ty, A_focus, sigma=A_err_focus)
    y_fit_A = parabola(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1), *fit_params_A)
    ax[0].plot(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1), y_fit_A, '-', label='best parabolic fit', c='orange')
    #plt.plot(srp_tz, result_par.best_fit, '-', label='best parabolic fit')
    #ax[0].scatter(srp_tz, A_focus_conv, label='Fit on convolved image')
    max_A = np.nanargmax(y_fit_A)
    ax[0].axvline(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1)[max_A], label=f'max at {np.arange(np.min(srp_ty), np.max(srp_ty), 0.1)[max_A]:.2f}',linestyle='--',  c='black', alpha=0.3)
    ax[0].set_title('Amplitude')
    ax[0].legend()
    ax[0].set_xlabel('y')
    ax[0].set_ylabel('A')
    ax[1].scatter(srp_ty, fwhm_arcsec(sigma1_focus), label='data')
    ax[1].errorbar(srp_ty, fwhm_arcsec(sigma1_focus), sigma1_err_focus)
    #result_gauss1 = par_sigma_model.fit(sigma1_focus[0:-1], x=srp_tz[0:-1])#, a=np.min(sigma1_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(sigma1_err_focus[0:-1]))
    fit_params_sigma, pcov_sigma = scipy.optimize.curve_fit(parabola, srp_ty, sigma1_focus, sigma=sigma1_err_focus)
    y_fit_sigma = parabola(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1), *fit_params_sigma)
    ax[1].plot(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1), fwhm_arcsec(y_fit_sigma), '-', label='best parabolic fit', c='orange')
    min_sigma = np.nanargmin(y_fit_sigma)
    ax[1].axvline(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1)[min_sigma], label=f'min at {np.arange(np.min(srp_ty), np.max(srp_ty), 0.1)[min_sigma]:.2f}',linestyle='--',  c='black', alpha=0.3)
    #ax[1].scatter(srp_tz, sigma_x_focus_conv, label='Fit on convolved image')
    ax[1].legend()
    ax[1].set_title('FWHM_x')
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('FWHM [arcsec]')
    plt.show()
    
def plot_yfocus_with_error(srp_ty, A_focus, A_err_focus, sigma1_focus, sigma1_err_focus):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].scatter(srp_ty, A_focus, label='data')
    ax[0].errorbar(srp_ty, A_focus, A_err_focus)
    #result_gauss = par_model.fit(A_focus[0:-1], x=srp_tz[0:-1])#, a=np.max(A_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(A_err_focus[0:-1]))
    fit_params_A, pcov_A = scipy.optimize.curve_fit(parabola, srp_ty, A_focus, sigma=A_err_focus)
    y_fit_A = parabola(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1), *fit_params_A)
    ax[0].plot(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1), y_fit_A, '-', label='best parabolic fit', c='orange')
    #plt.plot(srp_tz, result_par.best_fit, '-', label='best parabolic fit')
    #ax[0].scatter(srp_tz, A_focus_conv, label='Fit on convolved image')
    max_A = np.nanargmax(y_fit_A)
    ax[0].axvline(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1)[max_A], label=f'max at {np.arange(np.min(srp_ty), np.max(srp_ty), 0.1)[max_A]:.2f}',linestyle='--',  c='black', alpha=0.3)
    ax[0].set_title('Amplitude')
    ax[0].legend()
    ax[0].set_xlabel('y')
    ax[0].set_ylabel('A')
    ax[1].scatter(srp_ty, fwhm_arcsec(sigma1_focus), label='data')
    ax[1].errorbar(srp_ty, fwhm_arcsec(sigma1_focus), sigma1_err_focus)
    #result_gauss1 = par_sigma_model.fit(sigma1_focus[0:-1], x=srp_tz[0:-1])#, a=np.min(sigma1_focus[0:-1]))#, cen=-7, wid=0.5, weights=1./np.array(sigma1_err_focus[0:-1]))
    fit_params_sigma, pcov_sigma = scipy.optimize.curve_fit(parabola, srp_ty, sigma1_focus, sigma=sigma1_err_focus)
    y_fit_sigma = parabola(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1), *fit_params_sigma)
    ax[1].plot(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1), fwhm_arcsec(y_fit_sigma), '-', label='best parabolic fit', c='orange')
    min_sigma = np.nanargmin(y_fit_sigma)
    ax[1].axvline(np.arange(np.min(srp_ty), np.max(srp_ty), 0.1)[min_sigma], label=f'min at {np.arange(np.min(srp_ty), np.max(srp_ty), 0.1)[min_sigma]:.2f}',linestyle='--',  c='black', alpha=0.3)
    #ax[1].scatter(srp_tz, sigma_x_focus_conv, label='Fit on convolved image')
    ax[1].legend()
    ax[1].set_title('FWHM_x')
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('FWHM [arcsec]')
    plt.show()




