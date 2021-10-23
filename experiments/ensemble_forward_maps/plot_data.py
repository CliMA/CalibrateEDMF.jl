import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from netCDF4 import Dataset
import os

def read_scm_data(scm_data_path):
    """
    Read data from netcdf file into a dictionary that can be used for plots
    Input:
    scm_data_path  - path to scampy netcdf dataset with simulation results
    """
    scm_data = Dataset(scm_data_path, 'r')
    
    variables = ["temperature_mean", "thetal_mean", "qt_mean", "ql_mean", "qr_mean",\
                 "buoyancy_mean", "b_mix","u_mean", "v_mean", "tke_mean",\
                 "updraft_buoyancy", "updraft_area", "env_qt", "updraft_qt", "env_ql", "updraft_ql", "updraft_thetal",\
                 "env_qr", "updraft_qr", "env_RH", "updraft_RH", "updraft_w", "env_w", "env_thetal",\
                 "massflux_h", "diffusive_flux_h", "total_flux_h", "diffusive_flux_u", "diffusive_flux_v",\
                 "massflux_qt","diffusive_flux_qt","total_flux_qt","turbulent_entrainment",\
                 "eddy_viscosity", "eddy_diffusivity", "mixing_length", "mixing_length_ratio",\
                 "entrainment_sc", "detrainment_sc", "massflux", "nh_pressure", "nh_pressure_b", "nh_pressure_adv", "nh_pressure_drag", "eddy_diffusivity",\
                 "Hvar_mean", "QTvar_mean", "HQTcov_mean", "env_Hvar", "env_QTvar", "env_HQTcov",\
                 "Hvar_dissipation", "QTvar_dissipation", "HQTcov_dissipation",\
                 "Hvar_entr_gain", "QTvar_entr_gain", "HQTcov_entr_gain",\
                 "Hvar_detr_loss", "QTvar_detr_loss", "HQTcov_detr_loss",\
                 "Hvar_shear", "QTvar_shear", "HQTcov_shear", "H_third_m", "QT_third_m", "W_third_m",\
                 "Hvar_rain", "QTvar_rain", "HQTcov_rain","tke_entr_gain","tke_detr_loss",\
                 "tke_buoy","tke_dissipation","tke_pressure","tke_shear"\
                ]  # "tke_advection" "tke_transport"

    data = {"z_half" : np.divide(np.array(scm_data["profiles/z_half"][:]),1000.0),\
            "t" : np.divide(np.array(scm_data["profiles/t"][:]),3600.0),\
            "rho_half": np.array(scm_data["reference/rho0_half"][:])}

    for var in variables:
        data[var] = []
        if var=="QT_third_m":
            data[var] = np.transpose(np.array(scm_data[f"profiles/{var}"][:, :]))*1e9  #g^3/kg^3
        elif ("qt" in var or "ql" in var or "qr" in var):
            try:
                data[var] = np.transpose(np.array(scm_data[f"profiles/{var}"][:, :])) * 1000  #g/kg
            except:
                data[var] = np.transpose(np.array(scm_data["profiles/w_mean" ][:, :])) * 0  #g/kg
        else:
            data[var] = np.transpose(np.array(scm_data[f"profiles/{var}"][:, :]))
            
    return data

def read_les_data(les_data):
    """
    Read data from netcdf file into a dictionary that can be used for plots
    Input:
    les_data - pycles netcdf dataset with specific fileds taken from LES stats file
    """
    variables = ["temperature_mean", "thetali_mean", "qt_mean", "ql_mean", "buoyancy_mean",\
                 "u_mean", "v_mean", "tke_mean","v_translational_mean", "u_translational_mean",\
                 "updraft_buoyancy", "updraft_fraction", "env_thetali", "updraft_thetali",\
                 "env_qt", "updraft_qt","env_RH", "updraft_RH", "env_ql", "updraft_ql",\
                 "diffusive_flux_u", "diffusive_flux_v","massflux","massflux_u", "massflux_v","total_flux_u", "total_flux_v",\
                 "qr_mean", "env_qr", "updraft_qr", "updraft_w", "env_w",  "env_buoyancy", "updraft_ddz_p_alpha",\
                 "thetali_mean2", "qt_mean2", "env_thetali2", "env_qt2", "env_qt_thetali",\
                 "tke_prod_A" ,"tke_prod_B" ,"tke_prod_D" ,"tke_prod_P" ,"tke_prod_T" ,"tke_prod_S",\
                 "Hvar_mean" ,"QTvar_mean" ,"env_Hvar" ,"env_QTvar" ,"env_HQTcov", "H_third_m", "QT_third_m", "W_third_m",\
                 "massflux_h" ,"massflux_qt" ,"total_flux_h" ,"total_flux_qt" ,"diffusive_flux_h" ,"diffusive_flux_qt"]

    data = {"z_half" : np.divide(np.array(les_data["z_half"][:]),1000.0),\
            "t" : np.divide(np.array(les_data["t"][:]),3600.0),\
            "rho": np.array(les_data["profiles/rho"][:]),\
            "p0": np.divide(np.array(les_data["profiles/p0"][:]),100.0)}

    for var in variables:
        data[var] = []
        if ("QT_third_m" in var ):
            data[var] = np.transpose(np.array(les_data["profiles/"  + var][:, :]))*1e9  #g^3/kg^3
        elif ("qt" in var or "ql" in var or "qr" in var):
            try:
                data[var] = np.transpose(np.array(les_data["profiles/"  + var][:, :])) * 1000  #g/kg
            except:
                data[var] = np.transpose(np.array(les_data["profiles/w_mean" ][:, :])) * 0  #g/kg
        else:
            data[var] = np.transpose(np.array(les_data["profiles/"  + var][:, :]))


    return data

def get_LES_data(case, les_dir=Path.cwd()):
    les_data_path = les_dir / f"{case}.nc"
    if not les_data_path.is_file():
        les_dir.mkdir(parents=True, exist_ok=True)
        url_ = LES_data_url(case) 
        os.system(f"curl -sLo {les_data_path} '{url_}'")
    les_data = Dataset(les_data_path, 'r')
    return read_les_data(les_data)

def LES_data_url(case):
	urls = {
		"ARM_SGP": r"https://caltech.box.com/shared/static/4osqp0jpt4cny8fq2ukimgfnyi787vsy.nc",
		"Bomex": r"https://caltech.box.com/shared/static/jci8l11qetlioab4cxf5myr1r492prk6.nc",
		"DYCOMS_RF01": r"https://caltech.box.com/shared/static/toyvhbwmow3nz5bfa145m5fmcb2qbfuz.nc",
		"Gabls": r"https://caltech.box.com/shared/static/zraeiftuzlgmykzhppqwrym2upqsiwyb.nc",
		"Nieuwstadt": r"https://caltech.box.com/shared/static/7upt639siyc2umon8gs6qsjiqavof5cq.nc",
		"Rico": r"https://caltech.box.com/shared/static/johlutwhohvr66wn38cdo7a6rluvz708.nc",
		"Soares": r"https://caltech.box.com/shared/static/pzuu6ii99by2s356ij69v5cb615200jq.nc",
		"life_cycle_Tan2018": r"https://caltech.box.com/shared/static/jci8l11qetlioab4cxf5myr1r492prk6.nc",  # same as Bomex
		"TRMM_LBA": r"https://caltech.box.com/shared/static/ivo4751camlph6u3k68ftmb1dl4z7uox.nc"
	}
	return urls[case]

def time_bounds(data, tmin, tmax):
    """Get index of time bounds for scm/les data """
    t0_ind = int(np.where(np.array(data["t"]) > tmin)[0][0])
    t1_ind = int(np.where(np.array(tmax<= data["t"]))[0][0])
    return t0_ind, t1_ind

def initialize_plot(nrows, ncols, labs, zmin, zmax):
    """ Initialize a new plotting frame """
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    for ax, lab in zip(axs.flatten(), labs):
        # ax.grid(True)
        ax.set_xlabel(lab, fontsize = 20)
        ax.set_ylim([zmin, zmax])
        ax.tick_params(labelsize=16)
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")
    
    # Set ylabel on leftmost plots
    for i in range(nrows):
        ax = axs[i,0] if (nrows > 1) else axs[0]
        ax.set_ylabel("z [km]", fontsize = 20)
    
    return fig, axs

def finalize_plot(fig, axs, folder, title):
    """ Finalize and save plot """
    axs.flatten()[-1].legend()
    fig.tight_layout()
    Path(folder).mkdir(exist_ok=True, parents=True)
    fig.savefig(Path(folder) / title)
    fig.clf()

def plot_scm(
    axs,
    scm_data_files, 
    tmin, tmax, 
    scm_vars,
    scm_label,
    color_mean="lightblue", color_realizations="darkblue",
):
    """ Plot profiles from stochastic SCAMPy simulations
    
    Parameters:
    -----------
    scm_data_files                  :: scm stats files
    les_data                        :: les stats file
    tmin, tmax                      :: lower and upper bound for time mean
    scm_vars, les_vars              :: lists of variable identifiers for scm and les data
    scm_label                       :: label name for scm mean line
    color_mean, color_realizations  :: line color for mean line and individual realizations
    """

    # Process and plot SCM data
    scm_z_half = scm_data_files[0]["z_half"]
    scm_processed = {}
    for scm_data in scm_data_files:
        t0_scm, t1_scm = time_bounds(scm_data, tmin, tmax)

        for ax, var in zip(axs.flatten(), scm_vars):
            mean_var = np.nanmean(scm_data[var][:, t0_scm:t1_scm], axis=1)
            # Stack every realization of a variable to common array (for mean & std later) 
            if var in scm_processed:
                scm_processed[var] = np.vstack([scm_processed[var], mean_var])
            else:
                scm_processed[var] = mean_var
        
            # Add each realization to the plot
            ax.plot(
                mean_var, scm_z_half, "-", 
                color=color_realizations, lw=1, alpha=0.2,
            )

    # Plot mean and std
    for ax, var in zip(axs.flatten(), scm_vars):
        scm_var = scm_processed[var]
        mean_var = np.nanmean(scm_var, axis=0)
        std_var = np.std(scm_var, axis=0)
        ax.plot(  # plot mean value
            mean_var, scm_z_half, "-", 
            color=color_mean, label=scm_label, lw=3,
        )
        ax.fill_betweenx(  # plot standard deviation
            scm_z_half, mean_var - std_var, mean_var + std_var,
            color=color_mean, alpha=0.4, 
        )

def plot_les(axs, data, vars, tmin, tmax):
    # plot LES data
    t0_les, t1_les = time_bounds(data, tmin, tmax)
    
    # Plot LES data
    for ax, var in zip(axs.flatten(), vars):
        ax.plot(
            np.nanmean(data[var][:, t0_les:t1_les], axis=1),
            data["z_half"], '-', color='gray', label='les', lw=3,
        )

class PlottingArgs:
    def __init__(self, plotdir):
        self.nrows=1
        self.ncols=1
        self.scm_vars = []
        self.les_vars = []
        self.labs = []
        self.title = ""
        self.plotdir = plotdir

        self.tmin = 4
        self.tmax = 6
        self.zmin = 0.0
        self.zmax = 2.3

        self.fig = None
        self.axs = None

class MeansFluxes(PlottingArgs):
    def __init__(self, plotdir, title_prefix="output", title_suffix=""):
        super().__init__(plotdir)
        self.nrows = 2
        self.ncols = 2
        scm_vars = les_vars = [
            "qt_mean",          "ql_mean",          # qt_mean??
            "total_flux_h",     "total_flux_qt",
        ]
        self.scm_vars = scm_vars
        self.les_vars = les_vars
        self.labs = [
            "mean qt [g/kg]",                                                   "mean ql [g/kg]",
            r'$ \langle w^* \theta_l^* \rangle  \; [\mathrm{kg K /m^2s}]$',     r'$ \langle w^* q_t^* \rangle  \; [\mathrm{g /m^2s}]$',
        ]
        self.title = f"{title_prefix}_means_fluxes{title_suffix}.pdf"

class VarsCovars(PlottingArgs):
    def __init__(self, plotdir, title_prefix="output", title_suffix=""):
        super().__init__(plotdir)
        self.nrows = 2
        self.ncols = 2
        self.scm_vars = [
            "tke_mean",     "HQTcov_mean",
            "Hvar_mean",    "QTvar_mean", 
        ]
        self.les_vars = [
            "tke_mean",     "env_HQTcov",
            "env_Hvar",     "env_QTvar", 
        ]
        self.labs = [
            r'$TKE [\mathrm{m^2/s^2}]$',    "HQTcov",
            "Hvar",                         "QTvar",
        ]
        self.title = f"{title_prefix}_var_covar{title_suffix}.pdf"

class ThirdOrder(PlottingArgs):
    # Third-order moments
    def __init__(self, plotdir, title_prefix="output", title_suffix=""):
        super().__init__(plotdir)
        self.nrows = 1
        self.ncols = 2
        scm_vars = les_vars = [
            "H_third_m",    "QT_third_m",
        ]
        self.scm_vars = scm_vars
        self.les_vars = les_vars
        self.labs = [
            r'$ \langle \theta_l^*\theta_l^*\theta_l^* \rangle [K^3] $',    r'$ \langle q_t^*q_t^*q_t^* \rangle [g^3/kg^3] $',
        ]
        self.title = f"{title_prefix}_third{title_suffix}.pdf"

# Functions using PlottingArgs as args:
def initialize_plot2(cls: PlottingArgs) -> None: 
    fig, axs = initialize_plot(cls.nrows, cls.ncols, cls.labs, cls.zmin, cls.zmax)
    cls.fig = fig
    cls.axs = axs
def finalize_plot2(cls: PlottingArgs) -> None: finalize_plot(cls.fig, cls.axs, cls.plotdir, cls.title)
def plot_scm2(cls: PlottingArgs, data, scm_label, color) -> None: plot_scm(cls.axs, data, cls.tmin, cls.tmax, cls.scm_vars, scm_label, color, color)
def plot_les2(cls: PlottingArgs, data) -> None: plot_les(cls.axs, data, cls.les_vars, cls.tmin, cls.tmax)

def load_scm_data(scm_folders):
    data = {}
    for folder in scm_folders:
        files = Path(folder).glob("Stats.*.nc")
        data[folder.name] = [read_scm_data(file) for file in files]
    return data

# Folders
# root_folder = Path("/Users/haakon/Documents/CliMA/SEDMF/output/fix_noise2")
case = "Bomex"  # used to fetch correct LES data
scm_name = f"Stochastic{case}"
_root = Path("/central/groups/esm/hervik/calibration/EnsembleKalmanProcesses.jl/examples/SCM/experiments/ensemble_forward_maps/output")
# root_folder = _root / "results_ens_p4_e40_Bomex_sde_full_vert_corr"  # sde
# root_folder = _root / "results_ens_p2_e40_Bomex_lognormal_med_new"  # lognormal
root_folder = _root / "results_ens_p2_e40_Bomex_lognormal_med_noise"  # lognormal eki
scm_root_folder = root_folder / "scm_data"
les_folder = root_folder / "les_data"
plotsdir = root_folder / "plots"
print(f"Plot creator. Output located within: {root_folder}")

# Load data:
# (Read in parallel: https://stackoverflow.com/a/38378869)
print("Loading data...")
scm_folders = [x for x in scm_root_folder.glob("noise*") if x.is_dir()]
all_scm_data = load_scm_data(scm_folders)

# get LES data
les_data = get_LES_data(case, les_folder)

def all_plots(scm_data, suffix="", labels=None):
    # Initialize PlottingArgs objects
    means_fluxes = MeansFluxes(plotsdir, title_prefix=scm_name, title_suffix=suffix)
    vars_covars = VarsCovars(plotsdir, title_prefix=scm_name, title_suffix=suffix)
    third = ThirdOrder(plotsdir, title_prefix=scm_name, title_suffix=suffix)

    print("Initializing plots...")
    # Initialize plots
    initialize_plot2(means_fluxes)
    initialize_plot2(vars_covars)
    initialize_plot2(third)

	# Plot LES
    plot_les2(means_fluxes, les_data)
    plot_les2(vars_covars, les_data)
    plot_les2(third, les_data)

	# colors:  https://colorbrewer2.org/?type=diverging&scheme=Spectral&n=10
    # colors = ['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2']  # 10 colors
    # colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f']  
    # colors = ["#1b9e77", "#d95f02", "#7570b3"]
    # colors = ['#d53e4f','#fc8d59','#fee08b','#ffffbf','#e6f598','#99d594','#3288bd']  # 7 colors
    colors = ['#d53e4f','#fc8d59','#fee08b','#e6f598','#99d594','#3288bd']  # 6 colors

    print("Plotting SCM...")
	# Plot SCM
    for i, key in enumerate(sorted(scm_data, key=lambda x: float(x[5:]))):
        data = all_scm_data[key]

        if labels is None:
            scm_label = key[5:]
        else:
            scm_label = labels[i]

        plot_scm2(means_fluxes, data, scm_label, colors[i])
        plot_scm2(vars_covars, data, scm_label, colors[i])
        plot_scm2(third, data, scm_label, colors[i])

    finalize_plot2(means_fluxes)
    finalize_plot2(vars_covars)
    finalize_plot2(third)
    print("Done!")

# all
labels = ["determ", "EKI params", "true params"]  # eki
# labels = None  # fixed param plots
all_plots(all_scm_data, "_all", labels=labels)

# # 0.0, 0.4
# keys = ["noise0.0", "noise0.4"]
# _data = {key: all_scm_data[key] for key in keys if key in all_scm_data}
# all_plots(_data, "_0.4") if set(keys).issubset(_data) else None

# # 0.0, 0.8
# keys = ["noise0.0", "noise0.8"]
# _data = {key: all_scm_data[key] for key in keys if key in all_scm_data}
# all_plots(_data, "_0.8") if set(keys).issubset(_data) else None

# # 0.0, 5.0
# keys = ["noise0.0", "noise5.0"]
# _data = {key: all_scm_data[key] for key in keys if key in all_scm_data}
# all_plots(_data, "_5.0") if set(keys).issubset(_data) else None

# # 0.0, 0.4, 0.8, 5.0
# keys = ["noise0.0", "noise0.4", "noise0.8", "noise5.0"]
# _data = {key: all_scm_data[key] for key in keys if key in all_scm_data}
# all_plots(_data, "_all") if set(keys).issubset(_data) else None

# 0.0, 0.4, 0.8
# keys = ["noise0.0", "noise0.4", "noise0.8"]
# _data = {key: all_scm_data[key] for key in keys if key in all_scm_data}
# all_plots(_data, "_all") if set(keys).issubset(_data) else None
