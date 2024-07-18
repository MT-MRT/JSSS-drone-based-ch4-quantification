"""
@author: Lars Kistner
Affiliation: Department of Measurement and Control
Mechanical Engineering Faculty - Universty of Kassel
email: lars.kistner@mrt.uni-kassel.de
"""

import locale

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from numpy import nan

def use_locals(use: bool, lang_en: bool=True, use_serif: bool=False):
    """activate or deactivate localization support in python and matplotlib
    
    supported languages are en_US if lang_en=True or de_DE if False
    also font can be set to sans-serif or serif
    """
    if use:
        if lang_en:
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except locale.Error:
                locale.setlocale(locale.LC_ALL, '')
                print("WARNING: locale 'en_US.UTF-8' not supported on this system use default!")
        else:
            try:
                locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
            except locale.Error:
                locale.setlocale(locale.LC_ALL, '')
                print("WARNING: locale 'de_DE.UTF-8' not supported on this system use default!")
        matplotlib.rcParams['axes.formatter.use_locale'] = True
    else:
        locale.setlocale(locale.LC_ALL, '')
        matplotlib.rcParams['axes.formatter.use_locale'] = False
    
    if use_serif:
        matplotlib.rcParams["font.family"] = "STIXGeneral"
        matplotlib.rcParams['mathtext.fontset'] = "stix"
    else:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams['mathtext.fontset'] = "dejavusans"

def locals(number: float):
    """helber function return the number according to current localization"""
    return f"{number:n}"

def _Celsius_to_Kelvin(T: float) -> float:
    """ temperture from celsius to kelvin """
    return T + 273.15
Celsius_to_Kelvin = np.vectorize(_Celsius_to_Kelvin)

def _A(distance, alpha):
    """ calculate the measuring cone area based on distance and alpha """
    return 2*np.power(distance,2)*np.tan(alpha/2)/2
A = np.vectorize(_A)

def _Qn(p, T, A, C_avg, v_avg, Tn=273.15, pn=101325):
    return (p*Tn)/(pn*T) * A * C_avg * v_avg
Qn = np.vectorize(_Qn)

def _C_avg(C_int, distance):
    return C_int / distance
C_avg = np.vectorize(_C_avg)

def quantify_points(pointcloud, alpha, pu=101325, Tu=Celsius_to_Kelvin(0), wind_channel="anemometer"):
    data = pointcloud

    pressure = np.ones(data.shape)*pu
    temp = np.ones(data.shape)*Tu
    distance = data["distance"]
    ppmm = data["ppmm"]
    area = A(distance, alpha)
    wind = data[wind_channel]

    ch4_avg = C_avg(ppmm, distance) / 1_000_000 # ppmm -> anteil
    quantification = Qn(pressure, temp, area, ch4_avg, wind) # m^3/s
    quantification = quantification * 60 * 1_000 # m^3/s -> l/min
    return quantification

def moving_average(x, w, force=False):
    """
    Moving average of window length w over dimension d for samples N

    Supported shapes of x are (N,) (N, 1) or (N, d) where N is number of samples.
    """
    if (len(x.shape) > 2) and not force:
        raise ValueError("Only 2 dimensional arrays supported")
    elif len(x.shape) == 1:
        x = np.atleast_2d(x).T
    elif (x.shape[0] < x.shape[1]) and not force:
        raise ValueError("Dimensions > Samples. Is this correct?")
    
    new_x = np.zeros(x.shape)
    for i in range(x.shape[1]):
        tmp = np.hstack([np.repeat(x[0, i], w), x[:, i], np.repeat(x[-1, i], w)])
        new_x[:, i] = (np.convolve(tmp, np.ones(w), 'same') / w)[w:-w]
    if new_x.shape[1] == 1:
        new_x = new_x.reshape([-1,])
    return new_x


def fehlermass(y_true, y_pred) -> dict:
    rmse = np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
    sqr = np.sum(np.power((y_pred - y_true), 2))
    sqt = np.sum(np.power((y_true - np.mean(y_true)), 2))
    r2 = 1 - (sqr / sqt)
    nrmse = np.sqrt(sqr) / np.sqrt(sqt)
    bfr = 1 - nrmse
    return {"BFR": bfr, "RMSE": rmse, "R2": r2, "NRMSE": nrmse}

def plot_quantify_hist(quantification, x_lim = None, y_lim = False, expected_value=False, bins=None, log=True, quanti_resolution=0.25, en=True):
    if bins == None and x_lim != None:
        bins = int((max(x_lim)-min(x_lim))/quanti_resolution)
    plt.hist(quantification, bins=bins, range=x_lim)
    if en:
        plt.xlabel("$\\hat{q}_\\mathrm{v}$ in $\\mathrm{L_n \\; min^{-1}}$")
    else:
        plt.xlabel("$\\hat{q}_\\mathrm{v}$ in $\\mathrm{L_n \\; min^{-1}}$")
    if en:
        plt.ylabel("count in -")
    else:
        plt.ylabel("Anzahl in -")
    if x_lim:
        plt.xticks(range(0, int(x_lim[1])+1))
        plt.xlim(*x_lim)
    if y_lim:
        plt.ylim(*y_lim)
    if expected_value:
        # values of 0.0 wont be ploted if range is set to [0, ...]
        if expected_value < 0.001:
            expected_value = 0.01
        #plt.vlines(expected_value, y_lim[0], y_lim[1], colors="r")
        plt.axvline(x=expected_value, linestyle="--", color="r")
    if log:
        plt.yscale("log")
    plt.grid()

def plot_pointcloud_2d(x, y, colored_point_channel, arrow_direction_channel=False, arrow_length_channel=False, size_channel=32, x_lim=False, y_lim=False, alpha=1.0, norm=None, en=True):
    if arrow_direction_channel != False or arrow_length_channel != False:
        qu = arrow_length_channel.flatten() * np.cos(np.radians(arrow_direction_channel.flatten()))
        qv = arrow_length_channel.flatten() * np.sin(np.radians(arrow_direction_channel.flatten()))
        plt.quiver(x, y, qu, qv, pivot='middle')
    
    if type(size_channel) == int:
        plt.scatter(x, y, s=size_channel, c=colored_point_channel, alpha=alpha, norm=norm)
    else:
        cmap = matplotlib.cm.get_cmap("viridis")
        for i in range(len(x)):
            scaled_ = (colored_point_channel[i] - norm.vmin) * 255 / norm.vmax
            color = cmap(scaled_)
            plt.gca().add_patch(plt.Circle((x[i], y[i]), size_channel[i], alpha=alpha, facecolor=color, edgecolor=(0,0,0,0)))
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.xlabel("x in m")
    plt.ylabel("z in m")

def plot_quantify_pc_hist_box(pc_low, pc_high, quantification, pc_x_lim=[None, None], pc_y_lim=[None, None], ppmm_range=[None, None], q_expected=nan, q_x_lim=[None, None], q_y_lim=[None, None], q_resolution=0.25, q_num_of_points=11, en=True):
    plt.figure(figsize=[6,2.5], dpi=150)

    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 1.5, 1.5]) 
    
    plt.subplot(gs[0])
    if en:
        plt.title(f"point cloud")
    else:
        plt.title("Punktwolke der Messung")
    plot_pointcloud_2d(pc_low["x"], pc_low["y"], pc_low["ppmm"], x_lim=pc_x_lim, y_lim=pc_y_lim, alpha=0.03, norm=ppmm_range, en=en)
    plot_pointcloud_2d(pc_high["x"], pc_high["y"], pc_high["ppmm"], x_lim=pc_x_lim, y_lim=pc_y_lim, alpha=1, norm=ppmm_range, en=en)
    if en:
        plt.colorbar().set_label("$C_\\mathrm{ppmm}$ in ppmm")
    else:
        plt.colorbar().set_label("$C_\\mathrm{ppmm}$ in ppmm")
    plt.grid()

    plt.subplot(gs[1])
    #plt.title(f"$\dot{{q}}_\mathrm{{V}} = {locals(q_expected)} \quad \mathrm{{l_n/min}}$ CH4")
    if en:
        plt.title(f"histogram")
    else:
        plt.title("Häufigkeitsverteilung der Quantifizierung")
    plot_quantify_hist(quantification, expected_value=q_expected, x_lim=q_x_lim, y_lim=q_y_lim, quanti_resolution=q_resolution, en=en)

    plt.subplot(gs[2])
    if en:
        plt.xlabel("$q_\\mathrm{v}$ in $\\mathrm{L_n \\; min^{-1}}$")
        plt.ylabel("$\\hat{q}_\\mathrm{v}$ in $\\mathrm{L_n \\; min^{-1}}$")
    else:
        plt.ylabel("Normvolumenstrom in $\\mathrm{L_n \\; min^{-1}}$")
    plt.boxplot(np.sort(quantification)[-q_num_of_points:-1])
    plt.xticks([1], [locals(q_expected)])
    plt.grid()
    plt.tight_layout()

    q = np.sort(quantification)[-q_num_of_points:-1]
    print(f"mean: {np.mean(q)}")
    print(f"median: {np.median(q)}")
    print(f"std: {np.std(q)}")