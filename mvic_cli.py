#!/usr/bin/env python3
"""
Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as mpl_animation
from matplotlib.colors import LinearSegmentedColormap
import argparse

# -------------------------
# Parse command line arguments
# -------------------------
parser = argparse.ArgumentParser(description='Run MVIC model with customizable parameters')
parser.add_argument('--nu', type=int, default=200,
                    help='Number of motor units (default: 200)')
parser.add_argument('--fthscale', type=float, default=0.80,
                    help='Fraction of MVIC (0.50 = 50%% MVC) (default: 0.80)')
parser.add_argument('--fthtime', type=float, default=20.0,
                    help='Duration of the task in seconds (default: 20.0)')
parser.add_argument('--save-csv', action='store_true',
                    help='Save output data to CSV files (default: False)')
parser.add_argument('--animate', action='store_true',
                    help='Enable animation of motor unit contraction (default: False)')
args = parser.parse_args()

import numpy as np

# -------------------------
# Model input parameters
# -------------------------
nu = args.nu
samprate = 10
res = 100
hop = 20
r = 50
fat = 180
FatFac = 0.0225

tau = 22
adaptSF = 0.67
ctSF = 0.379

mthr = 1
a = 1
minfr = 8
pfr1 = 35
pfrL = 25
mir = 1
rp = 100
rt = 3
tL = 90

# Task parameters (change MVIC/target here)
fthscale = args.fthscale  # fraction of MVIC (0.50 = 50% MVC)
con = str(fthscale)
fthtime = args.fthtime  # seconds

# -------------------------
# Derived sizes and arrays
# -------------------------
fthsamp = int(fthtime * samprate)
fth = np.full(fthsamp, fthscale)

n_idx = np.arange(1, nu + 1)

# Recruitment thresholds
b_thr = np.log(r + (1 - mthr)) / (nu - 1)
thr = a * np.exp((n_idx - 1) * b_thr) - (1 - mthr)

# Peak firing rates and excitation grid
frdiff = pfr1 - pfrL
frp = pfr1 - frdiff * ((thr - thr[0]) / (r - thr[0]))
maxex = thr[-1] + (pfrL - minfr) / mir
maxact = int(round(maxex * res))

# Precompute mufr table (rested firing rates vs excitation)
mufr = np.zeros((nu, maxact), dtype=np.float64)
acts = (np.arange(1, maxact + 1) / res)
for mu in range(nu):
    valid = acts >= thr[mu]
    mufr_vals = mir * (acts - thr[mu]) + minfr
    mufr_vals[~valid] = 0.0
    mufr_vals = np.minimum(mufr_vals, frp[mu])
    mufr[mu, :] = mufr_vals

# Twitch peak force P
b_p = np.log(rp) / (nu - 1)
P = np.exp(b_p * (n_idx - 1))

# Contraction times
c = np.log(rp) / np.log(rt)
ct = tL * (1.0 / P) ** (1.0 / c)

# Normalized firing rates table (rested)
nmufr = ct[:, None] * (mufr / 1000.0)

# Force-frequency mapping (rested)
sPr = 1 - np.exp(-2 * (0.4 ** 3))
Pr = np.where(nmufr <= 0.4, (nmufr / 0.4) * sPr, 1 - np.exp(-2 * (nmufr ** 3)))

# MU force at each excitation (rested)
muP = Pr * P[:, None]
totalP = muP.sum(axis=0)
maxP = totalP[-1]

# Initialize dynamic arrays
Pnow = np.zeros((nu, fthsamp + 1), dtype=np.float64)
Pnow[:, 0] = P.copy()

# Fatigue parameters
b2 = np.log(fat) / (nu - 1)
mufatrate = np.exp(b2 * (n_idx - 1))
fatigue = mufatrate * (FatFac / fat) * P

# Preallocate time-history arrays (preserve all)
mufrFAT = np.zeros((nu, fthsamp), dtype=np.float64)
ctFAT = np.zeros((nu, fthsamp), dtype=np.float64)
ctREL = np.zeros((nu, fthsamp), dtype=np.float64)
nmufrFAT = np.zeros((nu, fthsamp), dtype=np.float64)
PrFAT = np.zeros((nu, fthsamp), dtype=np.float64)
muPt = np.zeros((nu, fthsamp), dtype=np.float64)
muPtMAX = np.zeros((nu, fthsamp), dtype=np.float64)
TPt = np.zeros(fthsamp, dtype=np.float64)
TPtMAX = np.zeros(fthsamp, dtype=np.float64)
Tact = np.zeros(fthsamp, dtype=int)
Pchange = np.zeros((nu, fthsamp), dtype=np.float64)
muON = np.zeros(nu, dtype=int)
adaptFR = np.zeros((nu, fthsamp), dtype=np.float64)
muPna = np.zeros((nu, fthsamp), dtype=np.float64)
muForceCapacityRel = np.zeros((nu, fthsamp + 1), dtype=np.float64)

recminfr = minfr
recovery = np.zeros(nu, dtype=np.float64)

startact = np.zeros(100, dtype=int)
for force in range(1, 101):
    startact[force - 1] = 0
    for act in range(1, maxact + 1):
        if (totalP[act - 1] / maxP * 100.0) < force:
            startact[force - 1] = act - 1

# Rdur vector for durations since recruitment
Rdur = np.zeros(nu, dtype=np.float64)

# Cache frequently used arrays as locals for speed
mufr_lastcol = mufr[:, -1].copy()
P_arr = P  # alias
ct_arr = ct
thr_arr = thr
sPr_val = sPr
minfr_val = minfr
adaptSF_val = adaptSF
tau_val = tau
ctSF_val = ctSF
mir_val = mir

# Precompute some constants for adaptation scaling
thr_scale = (thr_arr - 1.0) / (thr_arr[-1] - 1.0)

# -------------------------
# Main loop (binary-search for act, vectorized MU computations)
# -------------------------
# We'll perform a bounded binary search for act in [1, maxact] that yields TPt >= fth[i]
# The candidate evaluation uses the same vectorized computations as the original code,
# so results remain identical to the original while-loop approach.

for i in range(fthsamp):
    # Determine force index for startact (same as original)
    force_idx = int(round(fth[i] * 100.0))
    if force_idx < 1:
        force_idx = 1
    if force_idx > 100:
        force_idx = 100

    s = startact[force_idx - 1] - (5 * res)
    if s < 1:
        s = 1

    # Binary search bounds (1-based act)
    lo = 1
    hi = maxact

    # Optional: use start bound as a hint to narrow search window
    # Keep it safe: ensure lo <= s <= hi
    hint = int(s)
    if hint < lo:
        hint = lo
    if hint > hi:
        hint = hi

    # Narrow initial window around hint to reduce iterations (keeps identical semantics)
    # We'll set lo..hi to [max(1, hint - window), min(maxact, hint + window)]
    # window chosen as a fraction of maxact but not zero
    window = max(1, int(maxact // 8))
    lo = max(1, hint - window)
    hi = min(maxact, hint + window)

    # Ensure full range if hint window misses solution; binary search will expand if needed
    # We'll allow up to a small number of expansions if solution not found in window
    found_act = None
    max_expand = 3  # expand window up to 3 times if necessary

    for expand in range(max_expand + 1):
        # Standard binary search within [lo, hi]
        while lo <= hi:
            mid = (lo + hi) // 2
            act_idx = min(max(mid - 1, 0), maxact - 1)
            mufr_val = mufr[:, act_idx]

            # update Rdur for recruited MUs (vectorized)
            recruited_mask = muON > 0
            Rdur = np.where(recruited_mask, (i - (muON - 1)) / samprate, 0.0)

            # adaptation (vectorized)
            adaptFR_vec = thr_scale * adaptSF_val * (mufr_val - minfr_val + 2.0) * (1.0 - np.exp(-Rdur / tau_val))
            np.maximum(adaptFR_vec, 0.0, out=adaptFR_vec)

            # mufrFAT
            mufrFAT_vec = mufr_val - adaptFR_vec
            np.maximum(mufrFAT_vec, 0.0, out=mufrFAT_vec)

            # mufrMAX_vec
            mufrMAX_vec = mufr_lastcol - adaptFR_vec

            # contraction time slowing
            Pnow_i = Pnow[:, i]
            ctFAT_vec = ct_arr * (1.0 + ctSF_val * (1.0 - (Pnow_i / P_arr)))

            # normalized firing rate and fusion curve
            nmufrFAT_vec = ctFAT_vec * (mufrFAT_vec / 1000.0)
            # PrFAT
            mask_low = nmufrFAT_vec <= 0.4
            PrFAT_vec = np.empty_like(nmufrFAT_vec)
            PrFAT_vec[mask_low] = (nmufrFAT_vec[mask_low] / 0.4) * sPr_val
            high_idx = ~mask_low
            PrFAT_vec[high_idx] = 1.0 - np.exp(-2.0 * (nmufrFAT_vec[high_idx] ** 3))

            # muPt and muPtMAX
            muPt_candidate = PrFAT_vec * Pnow_i

            nmufrMAX_vec = ctFAT_vec * (mufrMAX_vec / 1000.0)
            mask_low_max = nmufrMAX_vec <= 0.4
            PrMAX_vec = np.empty_like(nmufrMAX_vec)
            PrMAX_vec[mask_low_max] = (nmufrMAX_vec[mask_low_max] / 0.4) * sPr_val
            high_idx_max = ~mask_low_max
            PrMAX_vec[high_idx_max] = 1.0 - np.exp(-2.0 * (nmufrMAX_vec[high_idx_max] ** 3))
            muPtMAX_candidate = PrMAX_vec * Pnow_i

            TPt_candidate = np.sum(muPt_candidate) / maxP

            # Binary search decision: find minimal act such that TPt >= fth[i]
            if TPt_candidate < fth[i]:
                lo = mid + 1
            else:
                # candidate meets or exceeds target; try lower acts to find minimal
                found_act = mid
                hi = mid - 1

        if found_act is not None:
            act = found_act
            break
        else:
            # Expand window and retry
            # Expand by doubling window size around hint
            window = window * 2
            lo = max(1, hint - window)
            hi = min(maxact, hint + window)
            # If we've already covered full range, break and pick maxact
            if lo == 1 and hi == maxact:
                # final fallback: set act to maxact (matches original loop's fallback)
                act = maxact
                break

    # If binary search loop ended without setting act (shouldn't happen), fallback
    if 'act' not in locals():
        act = maxact

    # Now compute and store the final per-time-step arrays using the chosen act
    act_idx = min(max(act - 1, 0), maxact - 1)
    mufr_val = mufr[:, act_idx]

    # update Rdur for recruited MUs (vectorized)
    recruited_mask = muON > 0
    Rdur = np.where(recruited_mask, (i - (muON - 1)) / samprate, 0.0)

    # adaptation (vectorized)
    adaptFR_vec = thr_scale * adaptSF_val * (mufr_val - minfr_val + 2.0) * (1.0 - np.exp(-Rdur / tau_val))
    np.maximum(adaptFR_vec, 0.0, out=adaptFR_vec)
    adaptFR[:, i] = adaptFR_vec

    # mufrFAT
    mufrFAT_vec = mufr_val - adaptFR_vec
    np.maximum(mufrFAT_vec, 0.0, out=mufrFAT_vec)
    mufrFAT[:, i] = mufrFAT_vec

    # mufrMAX_vec
    mufrMAX_vec = mufr_lastcol - adaptFR_vec

    # contraction time slowing
    Pnow_i = Pnow[:, i]
    ctFAT_vec = ct_arr * (1.0 + ctSF_val * (1.0 - (Pnow_i / P_arr)))
    ctFAT[:, i] = ctFAT_vec
    ctREL[:, i] = ctFAT_vec / ct_arr

    # normalized firing rate and fusion curve
    nmufrFAT_vec = ctFAT_vec * (mufrFAT_vec / 1000.0)
    nmufrFAT[:, i] = nmufrFAT_vec

    mask_low = nmufrFAT_vec <= 0.4
    PrFAT_vec = np.empty_like(nmufrFAT_vec)
    PrFAT_vec[mask_low] = (nmufrFAT_vec[mask_low] / 0.4) * sPr_val
    high_idx = ~mask_low
    PrFAT_vec[high_idx] = 1.0 - np.exp(-2.0 * (nmufrFAT_vec[high_idx] ** 3))
    PrFAT[:, i] = PrFAT_vec

    muPt[:, i] = PrFAT_vec * Pnow_i

    nmufrMAX_vec = ctFAT_vec * (mufrMAX_vec / 1000.0)
    mask_low_max = nmufrMAX_vec <= 0.4
    PrMAX_vec = np.empty_like(nmufrMAX_vec)
    PrMAX_vec[mask_low_max] = (nmufrMAX_vec[mask_low_max] / 0.4) * sPr_val
    high_idx_max = ~mask_low_max
    PrMAX_vec[high_idx_max] = 1.0 - np.exp(-2.0 * (nmufrMAX_vec[high_idx_max] ** 3))
    muPtMAX[:, i] = PrMAX_vec * Pnow_i

    TPt[i] = np.sum(muPt[:, i]) / maxP
    TPtMAX[i] = np.sum(muPtMAX[:, i]) / maxP

    # record recruitment times
    newly_recruited = (muON == 0) & ((act / res) >= thr_arr)
    if np.any(newly_recruited):
        muON[newly_recruited] = i + 1

    Tact[i] = act

    # fatigue update
    rec_mask = mufrFAT_vec >= recminfr
    neg_fat_term = -1.0 * (fatigue / samprate) * PrFAT_vec
    Pchange[:, i] = np.where(rec_mask, neg_fat_term, recovery / samprate)

    # update Pnow for next time step
    Pnext = Pnow_i + Pchange[:, i]
    np.clip(Pnext, 0.0, P_arr, out=Pnext)
    Pnow[:, i + 1] = Pnext

# Compute Tstrength (non-adapted total strength)
Tstrength = np.zeros(fthsamp, dtype=np.float64)
for i in range(fthsamp):
    muPna[:, i] = Pnow[:, i] * muP[:, -1] / P_arr
    Tstrength[i] = np.sum(muPna[:, i]) / maxP

# Determine endurance time
endurtime = None
for i in range(fthsamp):
    if TPtMAX[i] < fth[i]:
        endurtime = (i + 1) / samprate
        break
if endurtime is None:
    endurtime = fthtime

print("endurtime (s):", endurtime, flush=True)

# muForceCapacityRel (percentage) with matching columns to Pnow
muForceCapacityRel = (Pnow * 100.0) / P_arr[:, None]

# -------------------------
# Prepare time array for plotting and CSV output
# -------------------------
ns = np.arange(1, fthsamp + 1)
time = ns / samprate

# -------------------------
# Save CSV outputs (same names as MATLAB) if --save-csv flag is set
# -------------------------
if args.save_csv:
    combo = np.column_stack((
        time,
        fth,
        Tact / (res * maxex) * 100.0,
        Tstrength * 100.0,
        TPt * 100.0,
        TPtMAX * 100.0
    ))
    np.savetxt(f"{con} A - Target - Act - Strength (no adapt) - Force - Strength (w adapt).csv", combo, delimiter=',',
               header='time_s,target,act_percent,Strength_no_adapt_percent,Force_percent,Strength_with_adapt_percent', comments='')
    np.savetxt(f"{con} B - Firing Rate.csv", mufrFAT.T, delimiter=',')
    np.savetxt(f"{con} C - Individual MU Force Time-History.csv", muPt.T, delimiter=',')
    np.savetxt(f"{con} D - MU Capacity - relative.csv", muForceCapacityRel.T, delimiter=',')
    print("CSV files saved with prefix:", con)
else:
    print("CSV output disabled. Use --save-csv to enable saving CSV files.")

# -------------------------
# Plotting: 2x2 grid
# Panels B and C start each MU line at first nonzero sample
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
axA = axes[0, 0]  # top-left
axB = axes[0, 1]  # top-right
axC = axes[1, 0]  # bottom-left
axD = axes[1, 1]  # bottom-right

# --- Beautiful Toggle Keys Button (smaller) ---
from matplotlib.widgets import Button

# Smaller footprint under the plots
button_ax = fig.add_axes([0.45, 0.01, 0.10, 0.045])   # <-- reduced size

toggle_button = Button(
    button_ax,
    'Toggle Keys',
    color='#e0e0e0',        # light gray background
    hovercolor='#c0c0c0'    # darker gray on hover
)

# Slightly smaller text
toggle_button.label.set_color('black')
toggle_button.label.set_fontsize(8)      

button_ax.set_facecolor('#f5f5f5')

legend_visible = [True]

def toggle_legends(event):
    legend_visible[0] = not legend_visible[0]
    for ax in [axA, axB, axC, axD]:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_visible(legend_visible[0])
    plt.draw()

toggle_button.on_clicked(toggle_legends)
# -------------------------------------

end_idx = int(min(max(int(endurtime * samprate) - 1, 0), fthsamp))

# Rainbow colormap from dark red → violet for panels B and C
cmap = plt.get_cmap("rainbow_r")  # reversed rainbow
colors = [cmap(i / (nu - 1)) for i in range(nu)]

# Panel A: excitation (act/res), total force capacity, target
# Raw excitation (act/res)
excitation = Tact / res

# Normalize excitation so that excitation at endurance time = 100%
# Only do this if endurance time occurs before the end of the simulation
if endurtime < fthtime:
    # end_idx is the index corresponding to endurance time
    ex_at_end = excitation[end_idx]
    if ex_at_end > 0:
        excitation = excitation / ex_at_end * 100.0
else:
    # If the muscle never failed within the simulation, just express excitation as percent of max
    excitation = excitation / np.max(excitation) * 100.0

# Raw total muscle force capacity (0–1)
total_force_capacity = TPtMAX.copy()

# Normalize so that force capacity equals target force at endurance time
if endurtime < fthtime:
    cap_at_end = total_force_capacity[end_idx]
    target_at_end = fth[end_idx]
    if cap_at_end > 0:
        scale = target_at_end / cap_at_end
        total_force_capacity = total_force_capacity * scale * 100.0
    else:
        total_force_capacity = total_force_capacity * 0.0
else:
    # If no failure occurred, normalize to start at 100%
    total_force_capacity = total_force_capacity / total_force_capacity[0] * 100.0
axA.plot(time, excitation, color='green', label='Excitation (act/res)')
axA.plot(time, total_force_capacity, color='black', label='Total muscle force capacity (% max)')
axA.hlines(fthscale * 100.0, time[0], time[-1], colors='gray', linestyles='dashed', label='Target force (%MVC)')
axA.axvline(endurtime, color='k', linestyle=':', label='Endurance time')
axB.axvline(endurtime, color='k', linestyle=':')
axC.axvline(endurtime, color='k', linestyle=':')
axA.set_ylabel('Excitation / Force (%)')
axA.set_title('Panel A: Excitation and Total Muscle Force Capacity')
axA.legend(loc='upper right', fontsize = "small")
axA.set_ylim(1, 105)  

# Panel B: firing rates, start lines at first nonzero firing
for mu in range(nu):
    fr_trace = mufrFAT[mu, :]
    nz = np.nonzero(fr_trace > 0)[0]
    if nz.size == 0:
        continue
    start = nz[0]
    axB.plot(time[start:], fr_trace[start:], color='lightblue', linewidth=0.6)

# create highlight indices: MU1 (index 0), then MU20 (index 19), MU40 (index 39), ...
highlight_indices = [0] + list(range(19, nu, 20))
# Panel B highlight
for mu in highlight_indices:
    fr_trace = mufrFAT[mu, :]
    nz = np.nonzero(fr_trace > 0)[0]
    if nz.size == 0:
        continue
    start = nz[0]
    axB.plot(time[start:], fr_trace[start:], color=colors[mu], linewidth=2.2, label=f'MU {mu+1}')

axB.set_ylabel('Firing rate (imp/s)')
axB.set_title('Panel B: MU firing rates over time')
axB.legend(loc='upper right', fontsize='small')

# Panel C: MU force contributions, start lines at first nonzero force
for mu in range(nu):
    f_trace = (muPt[mu, :] / maxP) * 100.0
    nz = np.nonzero(f_trace > 0)[0]
    if nz.size == 0:
        continue
    start = nz[0]
    axC.plot(time[start:], f_trace[start:], color='lightgray', linewidth=0.6)
# highlight every 20th MU
# Panel C highlight (replace existing highlight loop)
for mu in highlight_indices:
    f_trace = (muPt[mu, :] / maxP) * 100.0
    nz = np.nonzero(f_trace > 0)[0]
    if nz.size == 0:
        continue
    start = nz[0]
    axC.plot(time[start:], f_trace[start:], color=colors[mu], linewidth=2.2, label=f'MU {mu+1}')
axC.set_ylabel('MU force contribution (% max)')
axC.set_title('Panel C: MU force contributions over time')
axC.legend(loc='upper right', fontsize='small')

# Panel D: MU capacity at endurance time vs MU index
end_idx = int(min(max(int(endurtime * samprate) - 1, 0), fthsamp))
fc_at_end = (Pnow[:, end_idx] / P) * 100.0
mu_indices = np.arange(1, nu + 1)
axD.plot(mu_indices, fc_at_end, marker='o', linestyle='-', color='blue', label='FC (%) at endurance')
exhausted_mask = fc_at_end <= 5.0
axD.plot(mu_indices[exhausted_mask], fc_at_end[exhausted_mask], 'ro', label='Exhausted (<=5%)')
axD.set_xlabel('MU index')
axD.set_ylabel('Force capacity (%)')
axD.set_title('Panel D: MU force capacity at endurance time')
axD.legend(loc='upper right', fontsize="small")
axD.set_ylim(1, 100)

# Add animation if requested
if args.animate:

    # Select MU numbers 1, 10, 20, 30, ... (one-based), then convert to zero-based indices
    one_based = [1] + list(range(10, nu + 1, 10))
    # Remove any values > nu (defensive) and convert to zero-based numpy array
    animate_idx = np.array([x - 1 for x in one_based if x <= nu], dtype=int)
    n_active = animate_idx.size


    if n_active > 0:
        fig_anim, ax_anim = plt.subplots(figsize=(6, 6))

        # Grid layout for the selected MUs
        cols = int(np.ceil(np.sqrt(n_active)))
        rows = int(np.ceil(n_active / cols))
        xs = []
        ys = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= n_active:
                    break
                xs.append(c)
                ys.append(-r)
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        # Center and scale grid to reasonable display coordinates
        if xs.size > 1:
            xs = (xs - xs.mean()) / max(np.ptp(xs), 1) * 3.0
        if ys.size > 1:
            ys = (ys - ys.mean()) / max(np.ptp(ys), 1) * 3.0

        mu_numbers = animate_idx + 1

        # Size circles by MU index (MU1 smallest ... MUN largest)
        min_s = 350
        max_s = 2200
        size_scale = (mu_numbers.astype(float) - 1.0) / max(float(nu - 1), 1.0)
        sizes = min_s + (max_s - min_s) * size_scale

        # Extract force traces for selected MUs
        mupt_active = muPt[animate_idx, :]  # shape (n_active, fthsamp)
        max_val = float(np.max(mupt_active)) if np.max(mupt_active) > 0 else 1.0

        # Colormap from light to dark red
        force_cmap = LinearSegmentedColormap.from_list('force_red', ['#ffd6d6', '#8b0000'])

        # Initial colors (frame 0)
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
        pad = 1.0
        ax_anim.set_xlim(xs.min() - pad, xs.max() + pad)
        ax_anim.set_ylim(ys.min() - pad, ys.max() + pad)
        ax_anim.set_xticks([])
        ax_anim.set_yticks([])
        ax_anim.set_title('Realtime Motor Unit Contractions (every 10th MU)')
        time_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes, ha='left', va='top')

        # Colorbar for normalized force
        mappable = plt.cm.ScalarMappable(cmap=force_cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        mappable.set_array(np.linspace(0, 1, 256))
        plt.colorbar(mappable, ax=ax_anim, fraction=0.046, pad=0.04, label='Normalized force')

        interval_ms = int(round(1000.0 / samprate)) if samprate > 0 else 100

        def _update(frame_i):
            vals = mupt_active[:, frame_i] / max_val
            rgba = force_cmap(vals)
            rgba[:, 3] = np.where(mupt_active[:, frame_i] > 0, 1.0, 0.0)  # transparent if force == 0
            sc.set_facecolor(rgba)
            # subtle pulsing by scaling sizes with normalized force
            size_scale_now = 1.0 + 0.20 * vals  # up to +20%
            sc.set_sizes(sizes * size_scale_now)
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