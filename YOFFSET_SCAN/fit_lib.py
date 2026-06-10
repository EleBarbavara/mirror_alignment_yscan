import numpy as np
import lmfit
from astropy.convolution import convolve, Gaussian2DKernel
from lmfit import Model, Parameters
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
import matplotlib
from astropy.modeling import models, fitting

def bkg_plane(x,y, x0,y0,offset,a,b):
    return a*(x-x0) + b*(y-y0) + offset

def gauss2D_circ(x,y,A,x0,y0,sigma):
    offset=0
    plane_a=0
    plane_b=0

    sigma_y = sigma
    sigma_x = sigma
    theta = 0
    sigx2 = sigma_x**2; sigy2 = sigma_y**2

    a = np.cos(theta)**2/(2*sigx2) + np.sin(theta)**2/(2*sigy2)
    b = np.sin(theta)**2/(2*sigx2) + np.cos(theta)**2/(2*sigy2)
    c = np.sin(2*theta)/(4*sigx2) - np.sin(2*theta)/(4*sigy2)
    expo = -a*(x-x0)**2 - b*(y-y0)**2 - 2*c*(x-x0)*(y-y0)

    plane = plane_a*(x-x0) + plane_b*(y-y0) 

    bkg = bkg_plane(x,y,x0,y0,offset, plane_a, plane_b)

    return A*np.exp(expo)

def DoubleGauss2D_circ(x,y,x0,y0,A1,A2,sigma1,sigma2,bkg):
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

def gauss2D_rot_bkg(x, y, A, x0, y0, sigma_x, sigma_y, theta,offset, plane_a, plane_b):
    theta = np.radians(theta)
    sigx2 = sigma_x**2; sigy2 = sigma_y**2

    a = np.cos(theta)**2/(2*sigx2) + np.sin(theta)**2/(2*sigy2)
    b = np.sin(theta)**2/(2*sigx2) + np.cos(theta)**2/(2*sigy2)
    c = np.sin(2*theta)/(4*sigx2) - np.sin(2*theta)/(4*sigy2)
    expo = -a*(x-x0)**2 - b*(y-y0)**2 - 2*c*(x-x0)*(y-y0)

    bkg = bkg_plane(x,y,x0,y0,offset, plane_a, plane_b)

    f = A*np.exp(expo) + bkg

    return f

def fit_source_least_squaresss(mapp, background=False, gauss_circ=False, gauss_ell=False, airy=False, twogauss_circ=False):
    if gauss_circ==False and gauss_ell==False and airy==False and twogauss_circ==False:
        print('Select model!!')

    x, y = np.meshgrid(np.arange(0,np.shape(mapp)[0]), np.arange(0,np.shape(mapp)[1]))
    mapmax = np.nanargmin(mapp)
    x0 = x.ravel()[mapmax] #x position of the brightest pixel
    y0 = y.ravel()[mapmax] #y position of the brightest pizel
    delta=20
    A =np.nanmin(mapp)

    pars = Parameters()

    if background == True and gauss_ell == True:
        pars.add("A", value=A, min=0.00001, max=1)
        pars.add("x0", value=x0, min=x0-delta, max=x0+delta)
        pars.add("y0", value=y0, min=y0-delta, max=y0+delta)
        pars.add("sigma_x", value=1, min=0.5, max=+3)
        pars.add("sigma_y", value=1, min=0.5, max=+3)
        pars.add("theta", value=0, min=-180, max=+180)
        pars.add("offset", value=0)
        pars.add("plane_a", value=0)
        pars.add("plane_b", value=0)

        model = Model(gauss2D_rot_bkg, independent_vars=('x','y'),nan_policy="omit")#Model(gauss2D_rot, independent_vars=('x','y'),nan_policy="omit")
        result = model.fit(mapp, x=x, y=y, params=pars, method="leastsq")
        fit = model.func(x, y, **result.best_values)
        residuals = mapp-fit
        bkg = bkg_plane(x, y, result.params["x0"], result.params["y0"], 0, 0, 0)
        
        import sys
        file = open('output.txt', 'a')
        sys.stdout = file

        lmfit.report_fit(result)
        
        file.close()

    elif background == False and gauss_ell==True:
        pars.add("A", value=A, min=0.00001, max=1)
        pars.add("x0", value=x0, min=x0-delta, max=x0+delta)
        pars.add("y0", value=y0, min=y0-delta, max=y0+delta)
        pars.add("sigma_x", value=1, min=0.5, max=+3)
        pars.add("sigma_y", value=1, min=0.5, max=+3)
        pars.add("theta", value=0, min=-180, max=+180)

        model = Model(gauss2D_rot, independent_vars=('x','y'),nan_policy="omit")#Model(gauss2D_rot, independent_vars=('x','y'),nan_policy="omit")
        result = model.fit(mapp, x=x, y=y, params=pars, method="leastsq")
        fit = model.func(x, y, **result.best_values)
        residuals = mapp-fit
        bkg = bkg_plane(x, y, result.params["x0"], result.params["y0"], 0, 0, 0)
        
        import sys
        file = open('output.txt', 'a')
        sys.stdout = file
        lmfit.report_fit(result)
        file.close()
        
        lmfit.report_fit(result)

    elif gauss_circ==True:
        pars.add("A", value=A, min=0.00001, max=1)
        pars.add("x0", value=x0, min=x0-delta, max=x0+delta)
        pars.add("y0", value=y0, min=y0-delta, max=y0+delta)
        pars.add("sigma", value=10, min=1, max=+20)
        #print(A, x0, y0)
        
        model = Model(gauss2D_circ, independent_vars=('x','y'),nan_policy="omit")#Model(gauss2D_rot, independent_vars=('x','y'),nan_policy="omit")
        result = model.fit(mapp, x=x, y=y, params=pars, method="leastsq")
        fit = model.func(x, y, **result.best_values)
        residuals = mapp-fit
        bkg = bkg_plane(x, y, result.params["x0"], result.params["y0"], 0, 0, 0)
        
        import sys
        file = open('output.txt', 'a')
        sys.stdout = file

        lmfit.report_fit(result)
        
        file.close()
    
    elif twogauss_circ == True:
        pars.add("x0", value=x0, min=x0-delta, max=x0+delta)
        pars.add("y0", value=y0, min=y0-delta, max=y0+delta)
        pars.add("A1", value=A, min=0.00001, max=1)
        pars.add("A2", value=A, min=0.00001, max=1)
        pars.add("sigma1", value=1, min=0.5, max=3)
        pars.add("sigma2", value=5, min=1, max=20)
        pars.add("bkg", value=0)

        model = Model(DoubleGauss2D_circ, independent_vars=('x','y'),nan_policy="omit")
        result = model.fit(mapp, x=x, y=y, params=pars, method="leastsq")
        fit = model.func(x, y, **result.best_values)
        residuals = mapp-fit
        bkg = bkg_plane(x, y, result.params["x0"], result.params["y0"], 0, 0, 0)
        
        import sys
        file = open('output.txt', 'a')
        sys.stdout = file

        lmfit.report_fit(result)
        
        file.close()
        
        
    elif airy==True:
        model = models.AiryDisk2D(amplitude=A, x_0=x0, y_0=y0, radius=delta)
        fittare = fitting.LevMarLSQFitter()
        fit = fittare(model, x, y, mapp)
        residuals = mapp-fit(x, y)
        bkg = bkg_plane(x, y, fit.x_0.value, fit.y_0.value, 0, 0, 0)
        
        result = [fit.amplitude.value, fit.x_0.value, fit.y_0.value, fit.radius.value]
    
        print('[[Fit Statistics]]')
        print('\t # fitting method   = leastsq')
        print('\t # chi-square  = //')
        print('\t # reduced chi-square  = //')
        print('\t # Akaike info crit  = //')
        print('\t # Bayesian info crit  = //')
        print('\t # R-squared  = //')
        print('[[Variables]]')
        print('\t # A  = ', fit.amplitude.value )
        print('\t # x0  = ', fit.x_0.value)
        print('\t # y0  = ', fit.y_0.value)
        print('\t # radius  = ', fit.radius.value)
        print('[[Correlations]] (unreported correlations are < 0.100)')
        print('\t # C(A, sigma_y)  = //')
        print('\t # C(A, sigma_x)  = //')
        print('\t # C(sigma_x, theta)  = //')
        print('\t # C(sigma_y, theta)  = //')
        

    return fit, residuals, bkg, result

        
def show_fit(img, fit, residuals, result, savefig=False):
    #title = "fwhm_x =" + str(result.params["sigma_x"]*2.633)
    
    norm = simple_norm(data=img, stretch="linear")
    
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(18,5))
    
    #beam = Gaussian2DKernel(x_stddev=4.5/2.633)
    #img_conv = convolve(img, beam)
    #residuals_conv = convolve(residuals, beam)
    
    im = ax1.imshow(img, cmap='viridis')#, vmin=np.nanmedian(img))
    ax2.imshow(fit, cmap='viridis')#, vmin=np.nanmedian(fit))
    ax3.imshow(residuals, cmap='viridis')#, vmin=np.nanmedian(img))
    
    c = matplotlib.patches.Ellipse((5,5), width=4.5, height=4.5, facecolor="none", edgecolor="white", lw=1) #label="M2 beam = 9''=4.5px")
    #ax2.add_patch(c)
    #ax2.legend()
    
    ax1.set_title("Map")
    ax2.set_title("Model")
    ax3.set_title("Residuals")
        
    #fig.suptitle("source="+cut.split("/")[-1].split(".fits")[0])
    
    cbar = plt.colorbar(im, ax=ax1, fraction=0.05, pad=0.25, label="Phase [rad]")
    fig.subplots_adjust(wspace=0.03)
    plt.show()
    
    if savefig==True:
        plt.savefig('plot/plot_fit.png')
    
    return ax1, ax2, ax3

def grand_fit_gauss_circ(x_edge, result, residuals):
    pix = x_edge[1]-x_edge[0] #grandezza di un pixel
    print('Pixel size =', pix)
    
    sigma = result.params['sigma'] #FWHM in pixel
    FWHM_deg = 2.355*sigma*pix #2.633
    FWHM_arcsec = FWHM_deg*3600
    print('FWHM in arcsec =', FWHM_arcsec)
    
    #inc_beam_deg = 2.355*0.002*pix
    #inc_beam_arcsec = inc_beam_deg*3600
    #print('Inc beam in arcsec =', inc_beam_arcsec)

    noise = np.nanstd(residuals[40:80,40:80])
    print('Noise =', noise)

def grand_fit_gauss_ell(x_edge, result, residuals):
    pix = x_edge[1]-x_edge[0] #grandezza di un pixel
    print('Pixel size =', pix)
    
    sigma_x = result.params['sigma_x'] #FWHM asse x in pixel
    FWHM_x_deg = 2.355*sigma_x*pix #2.633
    FWHM_x_arcsec = FWHM_x_deg*3600
    print('FWHM asse x in arcsec =', FWHM_x_arcsec)
    
    sigma_y = result.params['sigma_y'] #FWHM asse y in pixel
    FWHM_y_deg = 2.355*sigma_y*pix #2.633
    FWHM_y_arcsec = FWHM_y_deg*3600
    print('FWHM asse y in arcsec =', FWHM_y_arcsec)
    
    #inc_beam_y_deg = 2.355*0.002*pix
    #inc_beam_arcsec = inc_beam_deg*3600
    #print('Inc beam in arcsec =', inc_beam_arcsec)

    noise = np.nanstd(residuals[40:80,40:80])
    print('Noise =', noise)


