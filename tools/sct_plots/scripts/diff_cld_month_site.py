import os
import xarray as xr
from glob import glob
import numpy as np
import string
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl

# Plot kwargs
fsize = 11
font = {'size':fsize}
mpl.rc('font', **font)
mpl.rc('lines', linewidth=1.5)
mpl.rc('figure', figsize=[4.,4*4/3.])
mpl.rc('figure', facecolor='w')

fig_dir = '/groups/esm/ilopezgo/CalibrateEDMF.jl/tools/sct_plots/figures/'
tc_root = '/groups/esm/ilopezgo/CalibrateEDMF.jl/tools/results/'
# Stretched nz=55, inv, mb20, best
# tc_root = tc_root + 'best_val_amip4k_Inversion_dt_3.0_p16_e33_i100_mb_LES_2022-07-11_22-50_oDa/'
# Stretched nz=55, inv, mb20, best nn mean
tc_root = tc_root +'best_nn_val_amip4k_Inversion_dt_3.0_p16_e33_i100_mb_LES_2022-07-11_22-50_oDa/'

# eval with constant grid, dz = 50m; last nn mean from stretched nz=55, inv, mb20
# tc_root = tc_root +'last_nn_constant_dz_Inversion_dt_3.0_p16_e33_i100_mb_LES_2022-07-11_22-50_oDa/'
# eval with constant grid, dz = 50m; best nn mean from stretched nz=55, inv, mb20
# tc_root = tc_root +'best_nn_val_constant_dz_Inversion_dt_3.0_p16_e33_i100_mb_LES_2022-07-11_22-50_oDa/'

#### Trained with Stretched nz=55, inv, mb20, trained on q_l, LWP
# tc_root = tc_root +'last_nn_val_stretched_ql_lwp_Inv_dt_3.0_p16_e33_i100_mb_LES_2022-07-14_18-59_szG/'

# Sites and dictionaries
hadgem_2_sites_dict = {
    '01':['2','3','4','5','6','7','8','9','10','11','12','13','14',
          '21','22','23'],
    '04':['2','3','4','5','6','7','8','9','10','11','12','13','14',
          '19','20','21','22','23'],
    '07':['2','3','4','5','6','7','8','9','10','11','12','13','14','15',
          '17','18','19','20','21','22','23'],
    '10':['2','3','4','5','6','7','8','9','10','11','12','13','14','15',
          '17','18','19','20','21','22','23']}

# Dict: pycles => label
var_name_label_dict = {'cloud_fraction':r'$\Delta$ Cloud cover','lwp':r'$\Delta$ LWP (g m$^{-2}$)','cloud_base':r'$\Delta$ Cloud base (km)','cloud_top':r'$\Delta$ Cloud top (km)'}
# Dict: pycles => GCM
var_names_dict_gcm = {'lwp':'clwvi'}
# Dict: pycles => TC.jl
var_names_dict_scm = {"z_half":"zc",
                      "z":"zf",
                      "lwp":"lwp_mean",
                      "cloud_base":"cloud_base_mean",
                      "cloud_top":"cloud_top_mean",
                      "cloud_fraction":"cloud_cover_mean"}

# lowest allowable cloud top diagnostic (TC)
domain_bottom = 100.0
# highest allowable cloud base diagnostic (TC)
domain_top = 4000.0 - 100.0

# Available months
months = ['01','04','07','10']
month_labels = ['Jan','Apr','Jul','Oct']
nmonth = np.size(months)

fs = []
# LES data directory
exp_root = '/groups/esm/zhaoyi/GCMForcedLES/cfsite/'

# Check what the times for these indices are (LES)
t_mask = slice(144*5+1,144*6+1)
# times, last 3 hours
t_mask_scm = slice(3600.0*3, 3600.0*6)

# LES var_names to plot
var_names = ['cloud_fraction','lwp','cloud_base','cloud_top']
nvar = np.size(var_names)

# ymins = [-0.05,-2,0.3,0.3]
# ymaxs = [1.2,75,1.05,3.1]
scs = [1.,1000.,1./1000.,1./1000.]

# Model and experiment
models = ['HadGEM2-A'] 
exps = ['amip', 'amip4K']

# Sites (This should be fetched from dictionary)
all_sites = ['2','3','4','5','6','7','8','9','10','11','12','13','14','15',
         '17','18','19','20','21','22','23']
all_sites_num = [int(site) for site in all_sites]
nsite = np.size(all_sites)
nse = 14
ct = 0
subtitles = [x for x in string.ascii_lowercase]
cmap = plt.cm.viridis
cmap_values = [0.05,0.35,0.65,0.95]

# Plot SCM - LES bias
for model in models:
    for exp in exps:

        fig,axes = plt.subplots(ncols=2, nrows=nvar, figsize=[6,9.5/4*nvar], gridspec_kw={'width_ratios': [2,1]})
        fig2,axes2 = plt.subplots(ncols=2, nrows=nvar, figsize=[6,9.5/4*nvar], gridspec_kw={'width_ratios': [2,1]})
        figname = fig_dir+'diff_scm_les_cld_sitex_'+model+'_'+exp+'_'+'.pdf'
        figname2 = fig_dir+'avg_diff_scm_les_cld_sitex_'+model+'_'+exp+'_'+'.pdf'

        fig_list = [fig, fig2]
        axes_list = [axes, axes2]
        figname_list = [figname, figname2]
        
        save = True

        # Fetch LES
        les_data = np.zeros((nvar,nmonth,nsite))
        top = np.zeros((nmonth,nsite))
        for vari in range(nvar):
            var_name = var_names[vari]
            for monthi in range(nmonth):
                month = months[monthi]
                for sitei in range(nsite):
                    site = all_sites[sitei]
                    f_root = os.path.join(exp_root,month,model,exp,'Output.cfsite'+site+'_'+model+'_'+exp+'_2004-2008.'+month+'.4x','stats/')
                    fs = sorted(glob(f_root+'Stats*.nc'))
                    if fs:
                        f = fs[-1]
                        with xr.open_dataset(f,group='timeseries') as ds:
                            da = ds.data_vars[var_name].isel(t=t_mask).mean('t')
                        les_data[vari,monthi,sitei] = da
                        if np.abs(da)>10000.:
                            les_data[vari,monthi,sitei] = np.nan
                        if vari==0:
                            da = ds.data_vars['cloud_top'].isel(t=t_mask).mean('t')
                            top[monthi,sitei] = da

                    else:
                        les_data[vari,monthi,sitei] = np.nan

        # Fetch SCM
        scm_data = np.zeros((nvar,nmonth,nsite))
        for vari in range(nvar):
            # Get SCM variable
            var_name_les = var_names[vari]
            var_name = var_names_dict_scm[var_name_les]

            for monthi in range(nmonth):
                month = months[monthi]
                # Fetch states for current month
                sites = hadgem_2_sites_dict[month]
                
                # All sites, even not included
                for sitei in range(nsite):
                    if all_sites[sitei] in sites:
                        site = all_sites[sitei]
                        dirname = 'Output.LES_driven_SCM.cfsite'+site+'_'+model+'_'+exp+'_2004-2008.'+month
                        dir_path = glob(tc_root+dirname+'*')[0]
                        f_root = os.path.join(dir_path,'stats/')
                        fs = sorted(glob(f_root+'Stats*.nc'))
                        if fs:
                            f = fs[-1]
                            with xr.open_dataset(f,group='timeseries') as ds:
                                da = ds.data_vars[var_name].sel(t=t_mask_scm).mean('t')
                            scm_data[vari,monthi,sitei] = da
                            if np.abs(da)>10000.:
                                scm_data[vari,monthi,sitei] = np.nan
                            elif "cloud_base" in var_name and scm_data[vari,monthi,sitei] > domain_top:
                                scm_data[vari,monthi,sitei] = np.nan
                            elif "cloud_top" in var_name and scm_data[vari,monthi,sitei] < domain_bottom:
                                scm_data[vari,monthi,sitei] = np.nan

                        else:
                            scm_data[vari,monthi,sitei] = np.nan
                    else:
                        scm_data[vari,monthi,sitei] = np.nan

        for vari in range(nvar):
            var_name = var_names[vari]
            les_data[vari,np.where(top>3000)[0],np.where(top>3000)[1]] = np.nan

            for i in range(2):
                ax = axes[vari,i]
                for monthi in range(nmonth):
                    ax.scatter(range(nsite), np.subtract(scm_data, les_data)[vari,monthi,:] * scs[vari],
                            s=12,color=cmap(cmap_values[monthi]),label=month_labels[monthi])
            
                ax.set_xticks(range(nsite))
                ax.set_xticklabels(all_sites,rotation=60)
                # ax.set_ylim(ymins[vari],ymaxs[vari]) # Need to set this

                # avg figure
                ax2 = axes2[vari,i]
                time_mean_diff = np.nanmean(np.subtract(scm_data, les_data), axis=1) # [var, month, site]
                ax2.scatter(range(nsite), time_mean_diff[vari,:] * scs[vari],
                        s=12,color=cmap(cmap_values[0]),label='Avg. diff')

                ax2.set_xticks(range(nsite))
                ax2.set_xticklabels(all_sites,rotation=60)

            for fig_axes in axes_list:
                fig_axes[vari,0].set_xlim(-0.5,nse-0.5)
                fig_axes[vari,1].set_xlim(nse-0.5,nsite-0.5)

            if var_name in var_name_label_dict.keys():
                ylabel = var_name_label_dict[var_name]
            else:
                ylabel = var_name

            for fig_axes in axes_list:
                ax0 = fig_axes[vari,0]
                ax0.set_ylabel(ylabel)
                ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                ax0.set_title(subtitles[vari],loc='left',fontweight='bold',fontsize=fsize)
                ax1 = fig_axes[vari,1]
                ax1.set_yticklabels([])

        for fig_axes, figure, name in zip(axes_list, fig_list, figname_list):     
            fig_axes[-1,0].set_xlabel('SE Pacific')
            fig_axes[-1,1].set_xlabel('NE Pacific')
            fig_axes[0,0].legend(loc='upper right',ncol=2,columnspacing=0.2,handletextpad=0.1,bbox_to_anchor=[1.0,1.0])
            figure.tight_layout()
            if save:
                figure.savefig(name)

