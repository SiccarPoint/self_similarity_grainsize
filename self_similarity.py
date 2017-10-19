import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit, minimize
from scipy.integrate import quad


# filename = './B_50/01_3.csv'

mu_list = []
stdev_list = []
xmin = 0.
xmax = 200.
folders = ('B', 'T')
self_sim = True
data_dict = {'B': {}, 'T': {}}
fit_equ = 'log'

histvals = {'B': [], 'T': []}
#### DO NOT CHANGE THESE BINVALS; THEY ARE HARDWIRED BELOW
binvals = np.array([x/10. for x in range(-20, 50, 5)])
binmeans = (binvals[:-1] + binvals[1:])/2.

goodpts_T = np.arange(7, dtype=int)
goodpts_B = np.arange(9, dtype=int)
goodhists = {'B': [], 'T': []}
goodstds = {'B': [], 'T': []}
goodmeans = {'B': [], 'T': []}

for folder_prefix in folders:
    folder = folder_prefix + '_long'
    plt.figure('GS_dist_' + folder_prefix)
    if folder_prefix == 'T':
        goodpts = goodpts_T
    else:
        goodpts = goodpts_B

    for filename in os.listdir('./' + folder):
        if not filename.startswith('.'):
            reader = csv.reader(open(folder + '/' + filename, 'rU'),
                                dialect='excel')

            x = [int(i[0]) for i in reader]

            data = np.array(x)

            # now save the data for the future:
            run = int(filename[:2])
            rep = int(filename[3])
            if run not in data_dict[folder_prefix].keys():
                data_dict[folder_prefix][run] = {}
            data_dict[folder_prefix][run][rep] = data

            mu, std = norm.fit(data)
            mu_list.append(mu)
            stdev_list.append(std)

            if self_sim:
                xmin = -2.
                xmax = 5.
                n, _, _ = plt.hist((data - mu)/std, bins=binvals, normed=True,
                                   histtype='step', alpha=0.3)
                histvals[folder_prefix].append(n)
                if int(filename[:2]) in goodpts:
                    goodhists[folder_prefix].append(n)
                    goodstds[folder_prefix].append(std)
                    goodmeans[folder_prefix].append(mu)
                title = "Normalized grainsize distributions"
            else:
                plt.hist(data, bins=20, normed=True, alpha=0.6, color='g')
                title = "All grainsize distributions"

            # xmin_new, xmax_new = plt.xlim()
            # xmin = min(xmin, xmin_new)
            # xmax = max(xmax, xmax_new)
            # x = np.linspace(xmin, xmax, 20)
            # p = norm.pdf(x, mu, std)
            # plt.plot(x, p, linewidth=2)

            plt.xlim((xmin, xmax))
            if self_sim:
                plt.figure('all_GS_dists')
                plt.hist((data - mu)/std, bins=binvals, normed=True,
                         histtype='step', alpha=0.3)
                plt.figure('GS_dist_' + folder_prefix)
    histvals[folder_prefix] = np.array(histvals[folder_prefix])
    goodhists[folder_prefix] = np.array(goodhists[folder_prefix])
    goodstds[folder_prefix] = np.array(goodstds[folder_prefix])
    goodmeans[folder_prefix] = np.array(goodmeans[folder_prefix])
    if self_sim:
        plt.plot(binmeans, histvals[folder_prefix].mean(axis=0))
        plt.figure('all_GS_dists')
        plt.plot(binmeans, histvals[folder_prefix].mean(axis=0))

plt.title(title)

# set the good pts, i.e., just good topsets
locs_dict = {}
locs_dict['T'] = np.array([4416.808695, 4042.029765, 3454.451352,
                           2540.089843, 2350.310208, 2042.634689,
                           1762.745577, 1975.541615, 1494.362944])
locs_dict['B'] = np.array([4176.222804, 4109.126285, 3963.423321, 3745.857280,
                           3169.788791, 3169.788791, 2935.932485, 2963.725783,
                           2813.249370, 2706.849629, np.nan, 2623.469736])
# ^NB: loc B11 is missed out
if fit_equ == 'exp':
    equ_form = (lambda t,a,b: a*np.exp(b*t))
elif fit_equ == 'log':
    equ_form = (lambda t,a,b,c: c + b*np.log(t + a))
else:
    raise NameError
params_dict = {'T': {}, 'B': {}}

plt.figure('mean_GS_dists_of_topsets')
for folder in folders:
    if folder == 'T':
        goodpts = goodpts_T
    else:
        goodpts = goodpts_B
    locs = locs_dict[folder]
    goodlocs = locs[goodpts]
    goodD50s = np.empty_like(goodlocs, dtype=float)
    goodD84s = np.empty_like(goodD50s)

    for loc in goodpts:
        D50s = []
        D84s = []
        stds = []
        for sample in data_dict[folder][loc+1].values():
            D50s.append(np.percentile(sample, 50.))
            D84s.append(np.percentile(sample, 84.))
            stds.append(sample.std())
        sample_D50 = np.mean(D50s)
        sample_D84 = np.mean(D84s)
        sample_std = np.mean(stds)
        goodD50s[loc] = sample_D50
        goodD84s[loc] = sample_D84
    # plt.figure('mean_GS_' + folder + '_dstr_topsets')
    plt.plot(goodlocs, goodD50s)
    plt.plot(goodlocs, goodD84s)
    plt.ylim(ymin=0.)
    # do some curve fitting:
    datarange = np.arange(goodlocs[-1], goodlocs[0], 100.)
    if fit_equ == 'exp':
        (a, b), _ = curve_fit(equ_form, goodlocs, goodD50s, p0=(1., 1.))
        sim_data = equ_form(datarange, a, b)
        params_dict[folder]['D50'] = (a, b)
    else:
        (a, b, c), _ = curve_fit(equ_form, goodlocs, goodD50s,
                                 p0=(1., -100., 1.))
        sim_data = equ_form(datarange, a, b, c)
        params_dict[folder]['D50'] = (a, b, c)
    plt.plot(datarange, sim_data)
    if fit_equ == 'exp':
        try:
            (a, b), _ = curve_fit(equ_form, goodlocs, goodD84s, p0=(1., 1.))
            sim_data = equ_form(datarange, a, b)
            params_dict[folder]['D84'] = (a, b)
        except RuntimeError:
            p = np.polyfit(goodlocs, goodD84s, 1)
            sim_data = np.poly1d(p)(datarange)
            params_dict[folder]['D84'] = p
    else:
        try:
            (a, b, c), _ = curve_fit(equ_form, goodlocs, goodD84s,
                                     p0=(1., -100., 1.))
            sim_data = equ_form(datarange, a, b, c)
            params_dict[folder]['D84'] = (a, b, c)
        except RuntimeError:
            p = np.polyfit(goodlocs, goodD84s, 1)
            sim_data = np.poly1d(p)(datarange)
            params_dict[folder]['D84'] = p
    plt.plot(datarange, sim_data)

# now, down here, let's do some GS normalization work!
xi = binmeans
xi = np.array([x/4. for x in range(-40, 400)])
# these are Mitch's best fit params:
c_g = 0.15  # per F&P07
C_v = 0.72
C2overC1 = 1./C_v  # assumed by self-sim for now, tho in principle could fit it
porosity = 0.3  # following Mitch's assertion

# free params:
a_g = 0.15
b_g = 2.2
C1 = 0.7

# where's the first xi > 0?
pos_xi = np.argmax(xi > 0)
spacing = xi[1] - xi[0]
J = a_g * np.exp(-b_g * xi) + c_g
Jprime = -a_g * b_g * np.exp(-b_g * xi)
lilphi = 1./(C1 * (1. + C2overC1 * xi)) * (1. - 1./J) - Jprime/J
phi = np.zeros_like(xi)

assert pos_xi != 0, ("you must set xi such that it contains within it xi=0, " +
                     "and not as first or last entry")

how_far_between = xi[pos_xi]/(xi[pos_xi] - xi[pos_xi - 1])  # back from pos_xi
lilphi_at_zero = (how_far_between*lilphi[pos_xi-1] +
                  (how_far_between-1.)*lilphi[pos_xi])
for i in range(len(xi)):
    # This is how Mitch does it, which surely can't be right. The solution
    # becomes a function of the choice of initial xi this way.
    # phi[i] = -np.trapz(y=lilphi[:(i+1)], x=xi[:(i+1)])
    # Instead, we integrate about 0, which is surely what F&P meant. This is
    # fairly cumbersome, but I think works:
    if i == 0:
        phi[i] = -(np.trapz(y=lilphi[pos_xi::-1], x=xi[pos_xi::-1]) -
                   spacing*how_far_between*(lilphi[pos_xi]+lilphi_at_zero)/2.)
    elif i < pos_xi:  # -ve xi. Remove the overshot partial trapezoid
        phi[i] = -(np.trapz(y=lilphi[pos_xi:(i-1):-1], x=xi[pos_xi:(i-1):-1]) -
                   spacing*how_far_between*(lilphi[pos_xi]+lilphi_at_zero)/2.)
    else:  # same again
        phi[i] = -(np.trapz(y=lilphi[(pos_xi-1):(i+1)], x=xi[(pos_xi-1):(i+1)]) -
                   spacing*(1.-how_far_between)*(lilphi[pos_xi-1] +
                                                 lilphi_at_zero)/2.)
f_notnorm = np.exp(-phi)
f = f_notnorm/np.trapz(y=f_notnorm, x=xi)


def calc_curve(xi, a_g, b_g, C1):# alternatively x for xi, remove def below, and final redef of f
    # where's the first xi > 0?
    # xi = np.arange(-2, 6, 0.0654)  # make sure you choose step to NOT overlap with the desired vals_out
    pos_xi = np.argmax(xi > 0)
    spacing = xi[1] - xi[0]
    J = a_g * np.exp(-b_g * xi) + c_g
    Jprime = -a_g * b_g * np.exp(-b_g * xi)
    lilphi = 1./(C1 * (1. + C2overC1 * xi)) * (1. - 1./J) - Jprime/J
    phi = np.zeros_like(xi)

    assert pos_xi != 0, ("you must set xi such that it contains within it xi=0, " +
                         "and not as first or last entry")

    how_far_between = xi[pos_xi]/(xi[pos_xi] - xi[pos_xi - 1])  # back from pos_xi
    lilphi_at_zero = (how_far_between*lilphi[pos_xi-1] +
                      (how_far_between-1.)*lilphi[pos_xi])
    for i in range(len(xi)):
        # This is how Mitch does it, which surely can't be right. The solution
        # becomes a function of the choice of initial xi this way.
        # phi[i] = -np.trapz(y=lilphi[:(i+1)], x=xi[:(i+1)])
        # Instead, we integrate about 0, which is surely what F&P meant. This is
        # fairly cumbersome, but I think works:
        if i == 0:
            phi[i] = -(np.trapz(y=lilphi[pos_xi::-1], x=xi[pos_xi::-1]) -
                       spacing*how_far_between*(lilphi[pos_xi]+lilphi_at_zero)/2.)
        elif i < pos_xi:  # -ve xi. Remove the overshot partial trapezoid
            phi[i] = -(np.trapz(y=lilphi[pos_xi:(i-1):-1], x=xi[pos_xi:(i-1):-1]) -
                       spacing*how_far_between*(lilphi[pos_xi]+lilphi_at_zero)/2.)
        else:  # same again
            phi[i] = -(np.trapz(y=lilphi[(pos_xi-1):(i+1)], x=xi[(pos_xi-1):(i+1)]) -
                       spacing*(1.-how_far_between)*(lilphi[pos_xi-1] +
                                                     lilphi_at_zero)/2.)
    f_notnorm = np.exp(-phi)
    f = f_notnorm/np.trapz(y=f_notnorm, x=xi)

    # f = np.interp(x, xi, f)  # this interp selects the values to return, to compare with the data
    return f

# x, y are:
x_in = binmeans
topmeans = goodhists['T'].mean(axis=0)
bottommeans = goodhists['B'].mean(axis=0)
numcounts_top = goodhists['T'].shape[0]
numcounts_bottom = goodhists['B'].shape[0]
# so a corrected total weighted avg is
y_in = ((topmeans*numcounts_top + bottommeans*numcounts_bottom) /
        (numcounts_top + numcounts_bottom))

init_vals = [0.15, 2.2, 0.7]  # following Mitch

best_vals, covar = curve_fit(calc_curve, x_in, y_in, p0=init_vals)
# NOTE: This is the best fit for all distns at once

# record the new values for the variables:
(a_g, b_g, C1) = best_vals

# now check Cv = std/mu
Cv_T = goodstds['T']/goodmeans['T']
Cv_B = goodstds['B']/goodmeans['B']
plt.figure('Cv')
plt.plot(Cv_T)
plt.plot(Cv_B)
plt.ylim([0., 1.2])
Cv_mean = (Cv_T.sum() + Cv_B.sum())/(Cv_T.size + Cv_B.size)

# Now we can proceed with the fit, again following Mitch
# retain Mitch's (8) & (10) (geometric fan spread is logarithmic); but (9) for
# a delta probably should be r*(x*) = r0, i.e., a const, & not a fn(x*).
# For (10), we measure W, W0 ~ 1000m, Wapex = 1800m. Need a bunch more transect
# measurements to do this properly, but for now just assume W = 1500m.
# Gobo indicates L_T ~ 2450m, L_B ~ 1650m.
# These two simplifications make everything super easy, and (8) becomes:
# Qs(x*) = Qs0 - (1-porosity)*r0*W0* x*
# And (7) is
# R*(x*) = (1-porosity)*L*r0*W0/Qs(x*)
# R*(x*) = ((1-porosity)*L*r0*W0 /
#           (Qs0 - (1-porosity)*r0*W0* x*))
# i.e.
# R*(x*) ~ K1 / (1 - K2 * x*)
# And: K1 = (1-porosity)*L*r0*W0*Qs0;
#      K2 = (1-porosity)*r0*W0/Qs0 = K1/(L*Qs0**2)
# NOTE: by including L here, I think there's an assumption about sed exhaustion
# so our equ's may need tweaking. Think this will be OK, since it remains
# geometrically controlled. ...NO, if equ's are set right, exhaustion or export
# happen correctly.

# Our expression to fit GS comes from (4) & (5). (4) is easy:
# y*(x*) = integral(R*(x*) dx*), 0 to x*. This has a nice analytic solution for
# our R*,
# y*(x*) = -K1*log(1-K2* x*) /K2

# So using our best guess values (r0 ~ 0.001 m/y?):
#     K1_T ~ 2450 * Qs0
#     K2_T ~ 1. / Qs0
#     K1_T/K2_T = 2450. * Qs0**2
# Now D depends in large part on the value of exp(y*), which we can simplify:
# exp(y*) = (1 - k2* x*)**(-K1/K2), which unfortunately is explosive at large K


plt.figure('all_GS_dists')
# plt.plot(xi, f)
plt.plot(binmeans, calc_curve(binmeans, best_vals[0], best_vals[1], best_vals[2]))
plt.xlim((xmin, xmax))

plt.show()
# Note that (by definition?) the D50s at each level converge at infinite
# distance. The D84s very much do not!
