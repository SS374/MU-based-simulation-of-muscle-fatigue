#!/usr/bin/env python3
"""
Converted from: Potvin & Fuglevand 2017 MATLAB data
Preserves equations, parameters, and plotting layout.
Dependencies: numpy, matplotlib, tkinter
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from matplotlib.colors import LinearSegmentedColormap
import csv
import os
import sys
import subprocess
import tkinter as tk

_CLI_FTHSCALE = None
_CLI_FTHTIME = None
_CLI_ANIMATE = False

if '--run' in sys.argv:
    run_i = sys.argv.index('--run')
    if len(sys.argv) <= run_i + 2:
        raise SystemExit("Usage: mvic.py --run <fthscale> <fthtime>")
    _CLI_FTHSCALE = float(sys.argv[run_i + 1])
    _CLI_FTHTIME = float(sys.argv[run_i + 2])
    _CLI_ANIMATE = '--animate' in sys.argv
else:
    # Create main window with modern styling
    root = tk.Tk()
    root.title('MU-based Fatigue Model')
    
    # Set window size and center it on screen
    window_width = 400
    window_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Configure style
    root.configure(bg='#f0f0f0')
    root.resizable(False, False)
    
    # Create main container with padding
    main_frame = tk.Frame(root, bg='#f0f0f0', padx=20, pady=20)
    main_frame.pack(expand=True, fill='both')
    
    # Add title
    title_frame = tk.Frame(main_frame, bg='#f0f0f0')
    title_frame.pack(fill='x', pady=(0, 20))
    tk.Label(title_frame, 
             text='Motor Unit Control Simulation', 
             font=('Helvetica', 14, 'bold'), 
             bg='#f0f0f0').pack()
    
    # Create input fields frame
    input_frame = tk.Frame(main_frame, bg='#f0f0f0')
    input_frame.pack(fill='x', pady=10)
    
    # Configure grid weights
    input_frame.columnconfigure(1, weight=1)
    
    # Input fields
    fthscale_var = tk.StringVar(value='80')
    fthtime_var = tk.StringVar(value='20')
    error_var = tk.StringVar(value='')
    
    # Configure consistent column widths
    input_frame.columnconfigure(0, minsize=200)  # Label column
    input_frame.columnconfigure(1, minsize=150)  # Entry column
    
    # Common entry style
    entry_style = {
        'width': 5,  # Standardized to 5 characters
        'font': ('Helvetica', 10),
        'relief': 'solid',
        'borderwidth': 1
    }
    
    # Percentage of MVIC
    tk.Label(input_frame, 
             text='Percentage of MVIC (0-100%):', 
             bg='#f0f0f0',
             font=('Helvetica', 10)).grid(row=0, column=0, sticky='w', pady=5)
    percent_entry = tk.Entry(input_frame, 
                           textvariable=fthscale_var, 
                           **entry_style)
    percent_entry.grid(row=0, column=1, sticky='w', pady=5, padx=5)
    
    # Time in seconds
    tk.Label(input_frame, 
             text='Duration (seconds):', 
             bg='#f0f0f0',
             font=('Helvetica', 10)).grid(row=1, column=0, sticky='w', pady=5)
    time_entry = tk.Entry(input_frame, 
                         textvariable=fthtime_var, 
                         **entry_style)
    time_entry.grid(row=1, column=1, sticky='w', pady=5, padx=5)

    animate_var = tk.BooleanVar(value=True)
    animate_check = tk.Checkbutton(
        input_frame,
        text='Realtime MU animation',
        variable=animate_var,
        bg='#f0f0f0',
        font=('Helvetica', 10)
    )
    animate_check.grid(row=2, column=0, columnspan=2, sticky='w', pady=(10, 0))
    
    # Error message
    error_label = tk.Label(main_frame, 
                          textvariable=error_var, 
                          fg='red', 
                          bg='#f0f0f0',
                          wraplength=350,
                          justify='left')
    error_label.pack(pady=(10, 0))
    
    def _on_graph():
        try:
            s = float(fthscale_var.get())/100
            t = float(fthtime_var.get())
            if t <= 0:
                raise ValueError('Duration must be greater than 0')
            if s <= 0 or s > 1:
                raise ValueError('MVIC percentage must be between 0 and 100')
            error_var.set('')
            args = [sys.executable, __file__, '--run', str(s), str(t)]
            if animate_var.get():
                args.append('--animate')
            subprocess.Popen(args)
        except ValueError as e:
            error_var.set(f'Error: {str(e)}')
        except Exception as e:
            error_var.set(f'Unexpected error: {str(e)}')
    
    # Button with improved styling
    button_frame = tk.Frame(main_frame, bg='#f0f0f0')
    button_frame.pack(pady=(20, 0))
    
    graph_button = tk.Button(button_frame, 
                           text='Run Simulation', 
                           command=_on_graph, 
                           width=20,
                           bg='#4CAF50',
                           fg='white',
                           font=('Helvetica', 10, 'bold'),
                           relief='flat',
                           padx=10,
                           pady=5)
    graph_button.pack()
    
    # Add hover effects
    def on_enter(e):
        graph_button['bg'] = '#45a049'
    
    def on_leave(e):
        graph_button['bg'] = '#4CAF50'
    
    graph_button.bind('<Enter>', on_enter)
    graph_button.bind('<Leave>', on_leave)

    root.mainloop()
    raise SystemExit(0)

# -------------------------
# Model input parameters
# -------------------------
nu = 120  # number of neurons (motor units) - optimized for 120
samprate = 10  # sample rate (Hz)
res = 100  # resolution of activations (100 -> 0.01 activation resolution)
hop = 20  # hopping factor for excitation search
r = 50  # range of recruitment thresholds
fat = 180  # range of fatigue rates across MUs
FatFac = 0.0225  # fatigue factor (FF/S) percent of peak force of MU per second

tau = 22  # adaptation time constant
adaptSF = 0.67  # adaptation scale factor
ctSF = 0.379  # contraction time slowing scale factor

mthr = 1  # minimum recruitment threshold
a = 1  # recruitment gain parameter
minfr = 8  # minimum firing rate (imp/s)
pfr1 = 35  # peak firing rate of first recruited MU
pfrL = 25  # peak firing rate of last recruited MU
mir = 1  # slope of firing rate increase vs excitation
rp = 100  # range of twitch tensions
rt = 3  # range of contraction times (3-fold)
tL = 90  # longest contraction time (ms in original; code uses same units)

# -------------------------
# Trial / task parameters
# -------------------------
fthscale = 1  # sets %MVC level for the trial duration (100% MVC is 1.00)
con = '0.50'  # for output file names
fthtime = 100  # duration to run trial (seconds)

if _CLI_FTHSCALE is not None:
    fthscale = float(_CLI_FTHSCALE)
if _CLI_FTHTIME is not None:
    fthtime = float(_CLI_FTHTIME)

# -------------------------
# Derived sizes and arrays
# -------------------------
fthsamp = int(fthtime * samprate)
fth = np.full(fthsamp, fthscale)

# -------------------------
# Recruitment Threshold Excitation (thr)
# -------------------------
n_idx = np.arange(1, nu + 1)  # 1..nu to match MATLAB indexing in formulas
b = np.log(r + (1 - mthr)) / (nu - 1)
thr = a * np.exp((n_idx - 1) * b) - (1 - mthr)  # thr[0] corresponds to MU 1

# -------------------------
# Peak Firing Rate (frp)
# -------------------------
frdiff = pfr1 - pfrL
frp = pfr1 - frdiff * ((thr - thr[0]) / (r - thr[0]))
maxex = thr[-1] + (pfrL - minfr) / mir  # maximum excitation
maxact = int(round(maxex * res))
ulr = 100 * thr[-1] / maxex  # recruitment threshold (%) of last MU

# -------------------------
# Firing Rates for each MU with increased excitation (mufr)
# mufr shape: (nu, maxact)
# -------------------------
acti = (np.arange(1, maxact + 1) / res)[None, :]
thr_col = thr[:, None]
frp_col = frp[:, None]
mufr_raw = mir * (acti - thr_col) + minfr
mufr = np.where(acti >= thr_col, np.minimum(mufr_raw, frp_col), 0.0)

mufr_max_all = mufr[:, -1]

k = np.arange(1, maxact + 1)

# -------------------------
# Twitch peak force P
# -------------------------
b_p = np.log(rp) / (nu - 1)
P = np.exp(b_p * (n_idx - 1))  # P[0] = 1, P[-1] = rp (approx)

# -------------------------
# Twitch contraction time ct
# -------------------------
c = np.log(rp) / np.log(rt)
ct = tL * (1.0 / P) ** (1.0 / c)

# -------------------------
# Normalized MU firing rates nmufr (CT * FR / 1000)
# -------------------------
nmufr = ct[:, None] * (mufr / 1000.0)

# -------------------------
# Force-frequency mapping (Pr) based on Fuglevand style
# -------------------------
sPr = 1 - np.exp(-2 * (0.4 ** 3))
Pr = np.where(
    nmufr <= 0.4,
    (nmufr / 0.4) * sPr,
    1.0 - np.exp(-2.0 * (nmufr ** 3))
)

# -------------------------
# MU force muP with increased excitation
# -------------------------
muP = Pr * P[:, None]
totalP = muP.sum(axis=0)
maxP = totalP[maxact - 1]

# -------------------------
# Initialize Pnow (current MU peak forces over time)
# Pnow shape: (nu, fthsamp+1) to allow Pnow[:, i+1] updates
# -------------------------
Pnow = np.zeros((nu, fthsamp + 1))
Pnow[:, 0] = P.copy()

# -------------------------
# Fatigue parameters
# -------------------------
# Note: MATLAB code computes mufatrate from 'fat' range
b2 = np.log(fat) / (nu - 1)
mufatrate = np.exp(b2 * (n_idx - 1))

# The MATLAB code references 'rec' and computes murecrate but 'rec' is not defined.
# We'll set recovery to zero as in the MATLAB snippet and skip murecrate.
recovery = np.zeros(nu)

fatigue = mufatrate * (FatFac / fat) * P

# -------------------------
# Establish rested excitation required for each target load level
# startact for forces 1..100 (MATLAB 1..100)
# -------------------------
startact = np.zeros(100, dtype=int)
force_percent = (totalP / maxP) * 100.0
forces = np.arange(1, 101)[:, None]
counts = np.sum(force_percent[None, :] < forces, axis=1)
startact = np.maximum(counts - 1, 0).astype(int)

# -------------------------
# Pchangecurves for display (not used in dynamics)
# -------------------------
Pchangecurves = (fatigue * P)[:, None] * Pr

# -------------------------
# Prepare arrays for main loop
# -------------------------
TmuPinstant = np.zeros((nu, fthsamp))
mufrFAT = np.zeros((nu, fthsamp))
ctFAT = np.zeros((nu, fthsamp))
ctREL = np.zeros((nu, fthsamp))
nmufrFAT = np.zeros((nu, fthsamp))
PrFAT = np.zeros((nu, fthsamp))
muPt = np.zeros((nu, fthsamp))
TPt = np.zeros(fthsamp)
Ptarget = np.zeros(fthsamp)
Tact = np.zeros(fthsamp, dtype=int)
Pchange = np.zeros((nu, fthsamp))
TPtMAX = np.zeros(fthsamp)
muPtMAX = np.zeros((nu, fthsamp))
muON = np.zeros(nu, dtype=int)  # time index when MU turned on (0 means not on yet)
adaptFR = np.zeros((nu, fthsamp))
Rdur = np.zeros(nu)
acttemp = np.zeros((fthsamp, maxact), dtype=int)
muPna = np.zeros((nu, fthsamp))
muForceCapacityRel = np.zeros((nu, fthsamp))

# Some variables referenced in MATLAB but not defined in snippet:
# recminfr: threshold for recovery vs fatigue. We'll set equal to minfr (reasonable default).
recminfr = minfr
# mujump used later for output loop; set to 1 to include all MUs
mujump = 1

thr_scale = (thr - 1.0) / (thr[-1] - 1.0)

timer = 0

# -------------------------
# Main loop: move through force time-history and determine excitation required
# -------------------------
for i in range(fthsamp):
    force_idx = int(round(fth[i] * 100.0)) + 1
    if force_idx > 100:
        force_idx = 100

    s = startact[force_idx - 1] - (5 * res)
    if s < 1:
        s = 1

    acthop = int(round(maxact / hop))
    act = int(s)

    recruited_mask = muON > 0
    Rdur[:] = 0.0
    Rdur[recruited_mask] = (i - (muON[recruited_mask] - 1)) / samprate
    adapt_time_term = (1.0 - np.exp(-1.0 * Rdur / tau))

    pnow_col = Pnow[:, i]
    ctFAT_col = ct * (1.0 + ctSF * (1.0 - pnow_col / P))
    ctFAT[:, i] = ctFAT_col
    ctREL[:, i] = ctFAT_col / ct

    # search for excitation (act) that meets target force at this time
    for a_search in range(1, maxact + 1):
        acttemp[i, a_search - 1] = act

        act_idx = act - 1
        mufr_val = mufr[:, act_idx]
        adaptFR_col = thr_scale * adaptSF * (mufr_val - minfr + 2.0) * adapt_time_term
        adaptFR_col = np.clip(adaptFR_col, 0.0, None)
        adaptFR[:, i] = adaptFR_col

        mufrFAT_col = np.clip(mufr_val - adaptFR_col, 0.0, None)
        mufrFAT[:, i] = mufrFAT_col

        mufrMAX = mufr_max_all - adaptFR_col

        nmufrFAT_col = ctFAT_col * (mufrFAT_col / 1000.0)
        nmufrFAT[:, i] = nmufrFAT_col

        PrFAT_col = np.where(
            nmufrFAT_col <= 0.4,
            (nmufrFAT_col / 0.4) * sPr,
            1.0 - np.exp(-2.0 * (nmufrFAT_col ** 3))
        )
        PrFAT[:, i] = PrFAT_col

        muPt_col = PrFAT_col * pnow_col
        muPt[:, i] = muPt_col

        nmufrMAX = ctFAT_col * (mufrMAX / 1000.0)
        PrMAX_col = np.where(
            nmufrMAX <= 0.4,
            (nmufrMAX / 0.4) * sPr,
            1.0 - np.exp(-2.0 * (nmufrMAX ** 3))
        )
        muPtMAX[:, i] = PrMAX_col * pnow_col

        TPt[i] = np.sum(muPt_col) / maxP
        TPtMAX[i] = np.sum(muPtMAX[:, i]) / maxP

        # speed-up search logic (same as MATLAB)
        if TPt[i] < fth[i] and act == maxact:
            break
        if TPt[i] < fth[i]:
            act = act + acthop
            if act > maxact:
                act = maxact
        if TPt[i] >= fth[i] and acthop == 1:
            break
        if TPt[i] >= fth[i] and acthop > 1:
            act = act - (acthop - 1)
            if act < 1:
                act = 1
            acthop = 1

    # after search, record recruitment times for MUs that turned on
    newly_on = (muON == 0) & ((act / res) >= thr)
    muON[newly_on] = i + 1

    Ptarget[i] = TPt[i]
    Tact[i] = act

    fat_mask = mufrFAT[:, i] >= recminfr
    Pchange_col = np.where(
        fat_mask,
        -1.0 * (fatigue / samprate) * PrFAT[:, i],
        recovery / samprate
    )
    Pchange[:, i] = Pchange_col

    Pnow_next = Pnow[:, i] + Pchange_col
    Pnow_next = np.minimum(Pnow_next, P)
    Pnow_next = np.maximum(Pnow_next, 0.0)
    Pnow[:, i + 1] = Pnow_next

# -------------------------
# Compute Tstrength (non-adapted total strength) across time
# -------------------------
muPna[:, :] = Pnow[:, :fthsamp] * (muP[:, -1] / P)[:, None]
Tstrength = np.sum(muPna, axis=0) / maxP

# -------------------------
# Determine endurance time (first time TPtMAX < fth)
# -------------------------
endurtime = None
for i in range(fthsamp):
    if TPtMAX[i] < fth[i]:
        endurtime = (i + 1) / samprate
        break

if endurtime is None:
    endurtime = fthtime  # if never fell below target within simulation
    print("Endurance time (s):", endurtime, "(simulation completed without falling below target)")
else:
    print("Endurance time (s):", endurtime, "(time when force first dropped below target)")

# -------------------------
# Prepare muForceCapacityRel for outputs (percentage)
# -------------------------
muForceCapacityRel[::mujump, :] = Pnow[::mujump, :fthsamp] * 100.0 / P[::mujump, None]

# -------------------------
# Calculate relative MU force capacity (used for plotting)
# -------------------------
# combo = [ns(:)/samprate, fth(:), Tact(:)/res/maxex * 100,Tstrength(:) * 100 ,Ptarget(:) * 100,TPtMAX(:)* 100];
ns = np.arange(1, fthsamp + 1)
combo = np.column_stack((
    ns / samprate,
    fth,
    Tact / (res * maxex) * 100.0,
    Tstrength * 100.0,
    Ptarget * 100.0,
    TPtMAX * 100.0
))
# -------------------------
# Plotting: 4 vertically stacked panels sharing x-axis
# Panel A: Excitation (Tact converted to excitation units)
# Panel B: Firing rate of each MU over time (thin lines), highlight every 20th MU
# Panel C: Force contribution of each MU over time (thin lines), highlight every 20th MU
# Panel D: MU capacity at endurance time vs MU index
# -------------------------
time = ns / samprate

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axA = axes[0, 0]
axB = axes[0, 1]
axC = axes[1, 0]
axD = axes[1, 1]

# Set common x-axis limits for time-based plots (A, B, C)
for ax in [axA, axB, axC]:
    ax.set_xlim(0, fthtime)

# Panel A
# Calculate muscle force as percentage of max
total_force_capacity = (np.sum(Pnow[:, :fthsamp], axis=0) / maxP) * 100.0

# Find the index corresponding to endurance time
endur_idx = int(endurtime * samprate) if endurtime is not None else fthsamp - 1
endur_idx = min(endur_idx, fthsamp - 1)  # Ensure we don't go out of bounds

# Get the target force at endurance time
target_force_percent = fth[0] * 100.0  # Target force as percentage of max

# Calculate the current force values at start and endurance time
initial_force = total_force_capacity[0]
endurance_force = total_force_capacity[endur_idx] if endur_idx < len(total_force_capacity) else total_force_capacity[-1]

# Create a scaling factor to adjust the entire curve
if initial_force > 0 and endurance_force > 0:
    # Calculate the desired scaling factor at each point
    # We want to map [initial_force, endurance_force] to [100, target_force_percent]
    if initial_force != endurance_force:  # Avoid division by zero
        # Linear scaling: y = mx + b
        # At t=0: 100 = m*initial_force + b
        # At t=endur_idx: target_force_percent = m*endurance_force + b
        m = (100 - target_force_percent) / (initial_force - endurance_force)
        b = 100 - m * initial_force
        
        # Apply the scaling to the entire curve
        total_force_capacity = m * total_force_capacity + b
    else:
        # If initial_force equals endurance_force, just scale to match the target
        scale_factor = target_force_percent / initial_force
        total_force_capacity = total_force_capacity * scale_factor

# Get excitation and normalize it so that value at endurance time = 100%
excitation = Tact / res
if endurtime is not None:
    # Find the index corresponding to endurance time
    endur_idx = int(endurtime * samprate) - 1
    if 0 <= endur_idx < len(excitation):
        max_excitation = excitation[endur_idx]
        if max_excitation > 0:
            excitation = (excitation / max_excitation) * 100.0

# Panel A: excitation
if endurtime is not None:
    axA.axvline(endurtime, color='k', linestyle=':', label=f'Endurance time ({endurtime:.1f}s)')
axA.plot(time, np.clip(excitation, 0, 100), color='green', label='Excitation (% of max at endurance)', linewidth=2)
axA.set_ylim(0, 100)  # Fixed 0-100% range for Panel A
axA.set_ylabel('Percentage of maximum')
axA.set_title('Panel A: Excitation')

# Panel B: firing rates
# Only plot MUs with non-zero firing rates
for mu in range(nu):
    mask = mufrFAT[mu, :] > 0
    if np.any(mask):  # Only plot if MU has non-zero firing rate
        # Create masked arrays to only show non-zero values
        masked_time = np.ma.masked_where(~mask, time)
        masked_fr = np.ma.masked_where(~mask, mufrFAT[mu, :])
        axB.plot(masked_time, masked_fr, color='lightblue', linewidth=0.5)

# highlight specific MUs: 1, 20, 40, 60, 80, 100, 120
# but only if they have non-zero firing rates
highlight_mus = [mu for mu in [0, 19, 39, 59, 79, 99, 119] if mu < nu and np.any(mufrFAT[mu, :] > 0)]
highlight_colors = {
    0: 'darkred',
    19: 'red',
    39: 'yellow',
    59: 'green',
    79: 'lightblue',
    99: 'blue',
    119: 'darkviolet'
}
for mu in highlight_mus:
    mask = mufrFAT[mu, :] > 0
    if np.any(mask):
        masked_time = np.ma.masked_where(~mask, time)
        masked_fr = np.ma.masked_where(~mask, mufrFAT[mu, :])
        axB.plot(masked_time, masked_fr, linewidth=1.5, color=highlight_colors.get(mu, None), label=f'MU {mu+1}')
axB.set_ylabel('Firing rate (imp/s)')
axB.set_title('Panel B: MU firing rates over time')
if endurtime is not None:
    axB.axvline(endurtime, color='k', linestyle=':', label=f'Endurance time ({endurtime:.1f}s)')

# Panel C: MU force contributions
# Only plot MUs with non-zero force contributions
for mu in range(nu):
    mask = muPt[mu, :] > 0
    if np.any(mask):  # Only plot if MU has non-zero force contribution
        # Create masked arrays to only show non-zero values
        masked_time = np.ma.masked_where(~mask, time)
        masked_force = np.ma.masked_where(~mask, muPt[mu, :])
        axC.plot(masked_time, masked_force, color='lightgray', linewidth=0.5)

# highlight the same specific MUs as in Panel B, but only if they have non-zero force
for mu in highlight_mus:
    mask = muPt[mu, :] > 0
    if np.any(mask):  # Only plot if MU has non-zero force contribution
        masked_time = np.ma.masked_where(~mask, time)
        masked_force = np.ma.masked_where(~mask, muPt[mu, :])
        axC.plot(masked_time, masked_force, linewidth=1.5, color=highlight_colors.get(mu, None), label=f'MU {mu+1}')
axC.set_ylabel('MU force contribution')
axC.set_title('Panel C: MU force contributions over time')
if endurtime is not None:
    axC.axvline(endurtime, color='k', linestyle=':', label=f'Endurance time ({endurtime:.1f}s)')

# Panel D: MU capacity at endurance time vs MU index
if endurtime is None:
    end_idx = fthsamp - 1
else:
    end_idx = int(min(int(endurtime * samprate) - 1, fthsamp - 1))
fc_at_end = (Pnow[:, end_idx] / P) * 100.0
mu_indices = np.arange(1, nu + 1)
axD.plot(mu_indices, fc_at_end, marker='o', markersize=3, linestyle='-', color='blue', label='FC (%) at endurance')
# mark exhausted MUs (FC <= 5%)
exhausted_mask = fc_at_end <= 5.0
axD.plot(mu_indices[exhausted_mask], fc_at_end[exhausted_mask], 'ro', markersize=3, label='Exhausted (<=5%)')
axD.set_ylim(0, 100)  # Set y-axis to 0-100%
axD.set_xlabel('MU index')
axD.set_ylabel('Force capacity at endurance time (% max)')
axD.set_title('Panel D: MU force capacity at endurance time')
axD.set_xlim(0, nu)  # Set x-axis limit to show all motor units

# Track all MUs we want to check (1, 20, 40, 60, 80, 100, 120)
target_mus = [1, 20, 40, 60, 80, 100, 120]
mu_activation_times = []
non_activated_mus = []

for mu in [m-1 for m in target_mus]:  # Convert to 0-based index
    if mu < nu:
        # Find first index where force > 0
        force_nonzero = np.where(muPt[mu, :] > 0)[0]
        if len(force_nonzero) > 0:
            first_activation = time[force_nonzero[0]]
            # Set activation time to 0 if less than 1 second
            if first_activation < 1.0:
                first_activation = 0.0
            mu_activation_times.append((mu + 1, first_activation))
        else:
            non_activated_mus.append(mu + 1)  # Store 1-based index

# Calculate additional metrics
# 1. Average force output as % of target
if len(total_force_capacity) > 0:
    avg_force_pct = np.mean(total_force_capacity)
    target_force_pct = fth[0] * 100  # Convert to percentage
    force_error_pct = ((avg_force_pct - target_force_pct) / target_force_pct) * 100
    force_accuracy = f'{avg_force_pct:.1f}% (Target: {target_force_pct:.1f}%)'

# 2. Time to 80% MU recruitment
sorted_activation_times = sorted([t[1] for t in mu_activation_times])
if sorted_activation_times:
    idx_80pct = min(int(len(sorted_activation_times) * 0.8), len(sorted_activation_times) - 1)
    time_to_80pct = sorted_activation_times[idx_80pct] if idx_80pct >= 0 else 0
else:
    time_to_80pct = np.nan

# Create statistics text
stats_text = []
if endurtime is not None:
    stats_text.append(f'Endurance Time: {endurtime:.2f} s')

# Add performance metrics
stats_text.append(f'Avg Force Output: {force_accuracy}')
stats_text.append(f'Time to 80% MU Recruited: {time_to_80pct:.2f} s')

# Calculate recruitment percentage
activated_count = len(mu_activation_times)
total_checked = len([m for m in target_mus if m <= nu])
recruitment_pct = (activated_count / total_checked) * 100 if total_checked > 0 else 0

# Add recruitment percentage and rating
stats_text.append(f'Motor Unit Recruitment: {recruitment_pct:.1f}%')

# Determine hypertrophy rating
if recruitment_pct < 70:
    rating = 'Bad'
elif recruitment_pct < 80:
    rating = 'Okay'
elif recruitment_pct < 90:
    rating = 'Good'
else:
    rating = 'Great'

stats_text.append(f'Hypertrophy Potential: {rating}\n')

# Add activated MUs
if mu_activation_times:
    stats_text.append('Activated MUs:')
    for mu_num, act_time in sorted(mu_activation_times):
        stats_text.append(f'  MU{mu_num} @{act_time:.2f} s')

# Add non-activated MUs
if non_activated_mus:
    stats_text.append('\nNot Activated MUs:')
    # Group consecutive MUs for cleaner display
    non_activated_mus.sort()
    ranges = []
    if non_activated_mus:
        start = non_activated_mus[0]
        prev = start
        for mu in non_activated_mus[1:]:
            if mu != prev + 1:
                if start == prev:
                    ranges.append(str(start))
                else:
                    ranges.append(f'{start}-{prev}')
                start = mu
            prev = mu
        if start == prev:
            ranges.append(str(start))
        else:
            ranges.append(f'{start}-{prev}')
    stats_text.append('  ' + ', '.join(ranges))

# Add text box with statistics
stats_str = '\n'.join(stats_text)
fig.text(0.98, 0.02, stats_str, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
         fontsize=9, family='monospace')

# Adjust layout to make room for the text box
fig = plt.gcf()
fig.tight_layout(rect=(0.18, 0.0, 0.98, 1.0), h_pad=2.0, w_pad=3.0)
fig.subplots_adjust(hspace=0.45, wspace=0.55)

# Move legends outside each quadrant without overlapping neighboring panels
axA.legend(loc='center right', bbox_to_anchor=(-0.22, 0.5), fontsize=7)
axC.legend(loc='center right', bbox_to_anchor=(-0.22, 0.5), fontsize=7)
axB.legend(loc='center left', bbox_to_anchor=(1.22, 0.5), fontsize=7)
axD.legend(loc='center left', bbox_to_anchor=(1.22, 0.5), fontsize=7)

if _CLI_ANIMATE:
    animate_mus = [0, 19, 39, 59, 79, 99, 119]
    animate_idx = np.array([mu for mu in animate_mus if mu < nu], dtype=int)
    n_active = int(animate_idx.size)

    if n_active > 0:
        fig_anim, ax_anim = plt.subplots(figsize=(6, 6))

        # Arrange the 7 MUs like a small bundle of muscle fibers (cross-section)
        # Layout: center + hex-like ring positions
        base_positions = np.array(
            [
                [0.0, 0.0],
                [1.2, 0.2],
                [0.7, 1.1],
                [-0.6, 1.0],
                [-1.2, 0.0],
                [-0.6, -1.0],
                [0.7, -1.1],
            ],
            dtype=float,
        )
        pos = base_positions[:n_active]
        xs = pos[:, 0]
        ys = pos[:, 1]

        mu_numbers = animate_idx + 1

        # Size circles by MU index (MU1 smallest ... MU120 largest)
        # matplotlib scatter sizes are in points^2
        min_s = 450.0
        max_s = 3400.0
        size_scale = (mu_numbers.astype(float) - 1.0) / max(float(nu - 1), 1.0)
        sizes = min_s + (max_s - min_s) * size_scale

        mupt_active = muPt[animate_idx, :]
        max_val = float(np.max(mupt_active)) if np.max(mupt_active) > 0 else 1.0

        force_cmap = LinearSegmentedColormap.from_list('force_red', ['#ffd6d6', '#8b0000'])

        vals0 = mupt_active[:, 0] / max_val
        colors0 = force_cmap(vals0)
        colors0[:, 3] = np.where(mupt_active[:, 0] > 0, 1.0, 0.0)  # transparent if force == 0

        sc = ax_anim.scatter(
            xs,
            ys,
            s=sizes,
            facecolors=colors0,
            edgecolors='black',
            linewidths=0.8,
        )

        # Labels for each MU
        label_texts = []
        for i, mu_num in enumerate(mu_numbers):
            label_texts.append(
                ax_anim.text(
                    xs[i],
                    ys[i],
                    f'MU{mu_num}',
                    ha='center',
                    va='center',
                    fontsize=9,
                    color='black',
                )
            )

        ax_anim.set_aspect('equal', adjustable='box')
        ax_anim.set_xlim(xs.min() - 2.0, xs.max() + 2.0)
        ax_anim.set_ylim(ys.min() - 2.0, ys.max() + 2.0)
        ax_anim.set_xticks([])
        ax_anim.set_yticks([])
        ax_anim.set_title('Realtime Motor Unit Contractions')
        time_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes, ha='left', va='top')

        # Colorbar for normalized force
        mappable = plt.cm.ScalarMappable(cmap=force_cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        mappable.set_array([])
        plt.colorbar(mappable, ax=ax_anim, fraction=0.046, pad=0.04, label='Normalized force')

        interval_ms = int(round(1000.0 / samprate)) if samprate > 0 else 100

        def _update(frame_i):
            vals = mupt_active[:, frame_i] / max_val
            rgba = force_cmap(vals)
            rgba[:, 3] = np.where(mupt_active[:, frame_i] > 0, 1.0, 0.0)  # transparent if force == 0
            sc.set_facecolor(rgba)
            time_text.set_text(f't = {time[frame_i]:.2f} s')
            return (sc, time_text, *label_texts)

        mu_anim = mpl_animation.FuncAnimation(
            fig_anim,
            _update,
            frames=fthsamp,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
        fig_anim._mu_anim = mu_anim
plt.show()

# End of script