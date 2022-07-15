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


#### Trained with Stretched nz=55, inv, mb20

# best val
# tc_root = tc_root + 'best_val_amip4k_Inversion_dt_3.0_p16_e33_i100_mb_LES_2022-07-11_22-50_oDa/'
# best nn mean
tc_root = tc_root +'best_nn_val_amip4k_Inversion_dt_3.0_p16_e33_i100_mb_LES_2022-07-11_22-50_oDa/'

# eval with constant grid, dz = 50m; last nn mean
#tc_root = tc_root +'last_nn_constant_dz_Inversion_dt_3.0_p16_e33_i100_mb_LES_2022-07-11_22-50_oDa/'
# eval with constant grid, dz = 50m; best nn mean
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
var_name_label_dict = {'cloud_fraction':'Cloud cover','lwp':'LWP (g m$^{-2}$)','cloud_base':'Cloud base (km)','cloud_top':'Cloud top (km)'}
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

ymins = [-0.05,-2,0.3,0.3]
ymaxs = [1.2,75,1.05,3.1]
scs = [1.,1000.,1./1000.,1./1000.]

# Model and experiment
models = ['HadGEM2-A'] # 'CNRM-CM6-1', 'CNRM-CM5', 
exps = ['amip', 'amip4K']

all_sites = ['2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                 '17','18','19','20','21','22','23']
all_sites_num = [int(site) for site in all_sites]

nsite = np.size(all_sites)
nse = 14
ct = 0
subtitles = [x for x in string.ascii_lowercase]
cmap = plt.cm.viridis
cmap_values = [0.05,0.35,0.65,0.95]
save = True

# Plot LES
for model in models:
    for exp in exps:

        fig,axes = plt.subplots(ncols=2, nrows=nvar, figsize=[6,9.5/4*nvar], gridspec_kw={'width_ratios': [2,1]})
        figname = fig_dir+'cld_les_sitex_'+model+'_'+exp+'_'+'.pdf'
        
        data_les = np.zeros((nvar,nmonth,nsite))
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
                        data_les[vari,monthi,sitei] = da
                        if np.abs(da)>10000.:
                            data_les[vari,monthi,sitei] = np.nan
                        if vari==0:
                            da = ds.data_vars['cloud_top'].isel(t=t_mask).mean('t')
                            top[monthi,sitei] = da

                    else:
                        data_les[vari,monthi,sitei] = np.nan

        for vari in range(nvar):
            var_name = var_names[vari]
            data_les[vari,np.where(top>3000)[0],np.where(top>3000)[1]] = np.nan
            
            for monthi in range(nmonth):
                for i in range(2):
                    ax = axes[vari,i]
                    ax.scatter(range(nsite),data_les[vari,monthi,:]*scs[vari],s=12,color=cmap(cmap_values[monthi]),label=month_labels[monthi])
                    ax.set_xticks(range(nsite))
                    ax.set_xticklabels(all_sites,rotation=60)
                    # ax.set_ylim(ymins[vari],ymaxs[vari])
                axes[vari,0].set_xlim(-0.5,nse-0.5)
                axes[vari,1].set_xlim(nse-0.5,nsite-0.5)
            if var_name in var_name_label_dict.keys():
                ylabel = var_name_label_dict[var_name]
            else:
                ylabel = var_name
            ax = axes[vari,0]
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax.set_title(subtitles[vari],loc='left',fontweight='bold',fontsize=fsize)
            ax = axes[vari,1]
            ax.set_yticklabels([])
        axes[vari,0].set_xlabel('SE Pacific')
        axes[vari,1].set_xlabel('NE Pacific')

        axes[0,0].legend(loc='upper right',ncol=2,columnspacing=0.2,handletextpad=0.1,bbox_to_anchor=[1.0,1.0])

        plt.tight_layout()

        if save:
            plt.savefig(figname)


# Plot TurbulenceConvection

for model in models:
    for exp in exps:

        fig2,axes2 = plt.subplots(ncols=2, nrows=nvar, figsize=[6,9.5/4*nvar], gridspec_kw={'width_ratios': [2,1]})
        fig3,axes3 = plt.subplots(ncols=2, nrows=nvar, figsize=[6,9.5/4*nvar], gridspec_kw={'width_ratios': [2,1]})

        figname2 = fig_dir+'/cld_scm_sitex_'+model+'_'+exp+'_'+'.pdf'
        figname3 = fig_dir+'/avg_cld_scm_les_sitex_'+model+'_'+exp+'_'+'.pdf'

        data_scm = np.zeros((nvar,nmonth,nsite))

        # Store all data in numpy array
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
                            data_scm[vari,monthi,sitei] = da
                            # Filtering
                            if np.abs(da)>10000.:
                                data_scm[vari,monthi,sitei] = np.nan
                            elif "cloud_base" in var_name and data_scm[vari,monthi,sitei] > domain_top:
                                data_scm[vari,monthi,sitei] = np.nan
                            elif "cloud_top" in var_name and data_scm[vari,monthi,sitei] < domain_bottom:
                                data_scm[vari,monthi,sitei] = np.nan
                        else:
                            data_scm[vari,monthi,sitei] = np.nan
                    else:
                        data_scm[vari,monthi,sitei] = np.nan

        for vari in range(nvar):
            var_name_les = var_names[vari]
            var_name = var_names_dict_scm[var_name_les]

            for i in range(2):
                ax = axes2[vari,i]
                # Monthly plot
                for monthi in range(nmonth):
                    month = months[monthi]
                    ax.scatter(range(nsite),data_scm[vari,monthi,:]*scs[vari],s=12,color=cmap(cmap_values[monthi]),label=month_labels[monthi])
                    ax.set_xticks(range(nsite))
                    ax.set_xticklabels(all_sites,rotation=60)
                    # ax.set_ylim(ymins[vari],ymaxs[vari])

                ax_avg = axes3[vari, i]
                data_scm_avg = np.nanmean(data_scm, axis=1)
                ax_avg.scatter(range(nsite),data_scm_avg[vari,:]*scs[vari],s=12,color=cmap(cmap_values[0]),label='SCM')
                data_les_avg = np.nanmean(data_les, axis=1)
                ax_avg.scatter(range(nsite),data_les_avg[vari,:]*scs[vari],s=12,color=cmap(cmap_values[2]),label='LES')
                ax_avg.set_xticks(range(nsite))
                ax_avg.set_xticklabels(all_sites,rotation=60)

            if var_name_les in var_name_label_dict.keys():
                ylabel = var_name_label_dict[var_name_les]
            else:
                ylabel = var_name

            for fig_axes in [axes2, axes3]:
                fig_axes[vari,0].set_xlim(-0.5,nse-0.5)
                fig_axes[vari,1].set_xlim(nse-0.5,nsite-0.5)
                ax = fig_axes[vari,0]
                ax.set_ylabel(ylabel)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                ax.set_title(subtitles[vari],loc='left',fontweight='bold',fontsize=fsize)
                ax = fig_axes[vari,1]
                ax.set_yticklabels([])

        for fig_axes in [axes2, axes3]:
            fig_axes[-1,0].set_xlabel('SE Pacific')
            fig_axes[-1,1].set_xlabel('NE Pacific')
            fig_axes[0,0].legend(loc='upper right',ncol=2,columnspacing=0.2,handletextpad=0.1,bbox_to_anchor=[1.0,1.0])

        for figure, name in zip([fig2, fig3], [figname2, figname3]):
            figure.tight_layout()
            if save:
                figure.savefig(name)
