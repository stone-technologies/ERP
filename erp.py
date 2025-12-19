#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduces the Emissions-Aware Robust Portfolio Optimization (EAPO) results and figures

Inputs:
    - CLEAN_PRICES.csv            : Daily adjusted-close prices with columns [Date, <tickers...>]
    - CLEAN_EMISSIONS.csv         : Annual emissions with columns [YEAR, TICKER, SCOPE_1] (scope-1 absolute)
    - stock_revenues_imputed.csv  : Annual revenues with columns [YEAR, <tickers...>] (USD millions)

Outputs:
    - Figures:
        R2_{variations}.png
    - Tables:
        R2_{variations}.csv
"""
import os, sys, io, json, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__author__ = 'kqureshi'

plt.rcParams['figure.dpi'] = 160

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of v onto the probability simplex."""
    v = np.asarray(v, float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    idx = np.nonzero(u*np.arange(1, v.size+1) > (cssv - 1))[0]
    if idx.size == 0:
        # fallback to uniform weights if projection degenerates
        return np.ones_like(v)/v.size
    rho = idx[-1]
    theta = (cssv[rho]-1)/(rho+1.0)
    return np.maximum(v-theta,0.0)

def portfolio_metrics(gross_series: pd.Series) -> dict:
    """Annualized metrics from daily gross return series."""
    r = gross_series.values - 1.0
    n = len(r)
    if n == 0:
        return dict(Ann_Return=np.nan, Ann_Vol=np.nan, Sharpe=np.nan, MaxDD=np.nan)
    m = np.nanmean(r); s = np.nanstd(r, ddof=1)
    ann_ret = (np.prod(1+r)**(252.0/n)-1.0)*100.0
    ann_vol = (s*np.sqrt(252.0))*100.0
    sharpe = 0.0 if s==0 else (m/s)*np.sqrt(252.0)
    wealth = np.cumprod(1+r); peak = np.maximum.accumulate(wealth)
    mdd = (wealth/peak - 1.0).min()*100.0
    return dict(Ann_Return=ann_ret, Ann_Vol=ann_vol, Sharpe=sharpe, MaxDD=mdd)

def newey_west_se_mean(x: np.ndarray, L: int = 20) -> float:
    """Newey–West (HAC) standard error for the mean of x, with lag L."""
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    n = len(x)
    gamma0 = np.nanmean(x*x)
    var = gamma0
    for ell in range(1, L+1):
        cov = np.nanmean(x[ell:]*x[:-ell])
        w = 1.0 - ell/(L+1.0)
        var += 2*w*cov
    se = np.sqrt(max(var, 0.0)/n)
    return se

def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def build_panels(prices_csv='CLEAN_PRICES.csv',
                 emissions_csv='CLEAN_EMISSIONS.csv',
                 revenues_csv='stock_revenues_imputed.csv',
                 start='2012-01-01'):
    """Load data and construct panels: daily gross returns R, daily intensity matrix L."""
    prices = pd.read_csv(prices_csv)
    emiss  = pd.read_csv(emissions_csv)
    revs   = pd.read_csv(revenues_csv)

    prices['Date'] = pd.to_datetime(prices['Date'], utc=True, errors='coerce')
    prices = prices.set_index('Date').sort_index()

    universe = sorted(list(set(prices.columns) & set(emiss['TICKER'].unique()) & set([c for c in revs.columns if c!='YEAR'])))
    # Daily gross returns
    R = prices[universe] / prices[universe].shift(1)
    R = R.dropna(how='all')
    R = R.loc[start:]
    R.index = pd.to_datetime(R.index, utc=True)

    # yearly scope-1 intensity: SCOPE_1 / REVENUE
    rev_long = revs.melt(id_vars=['YEAR'], var_name='TICKER', value_name='REVENUE')
    e1 = emiss[['YEAR','TICKER','SCOPE_1']].copy()
    em_rev = pd.merge(e1, rev_long, on=['YEAR','TICKER'], how='inner').sort_values(['TICKER','YEAR'])
    em_rev['REVENUE'] = em_rev.groupby('TICKER')['REVENUE'].apply(lambda s: s.ffill().bfill())
    em_rev['INTENSITY_S1'] = em_rev['SCOPE_1'] / em_rev['REVENUE']
    I_year = em_rev.pivot(index='YEAR', columns='TICKER', values='INTENSITY_S1').sort_index().ffill().bfill()

    common = sorted(list(set(R.columns) & set(I_year.columns)))
    R = R[common].copy()

    L = pd.DataFrame(index=R.index, columns=common, dtype=float)
    min_year, max_year = int(I_year.index.min()), int(I_year.index.max())
    for y in np.unique(R.index.year):
        y_clip = min(max(int(y), min_year), max_year)
        L.loc[R.index.year==y, common] = I_year.loc[y_clip, common].values
    L = L.astype(float)
    return R, L, e1, (min_year, max_year)

def penalty_adjusted_returns(R: pd.DataFrame, L: pd.DataFrame, n_penalty=10) -> pd.DataFrame:
    """Apply Definition 2.1 penalty: Re = ((1 - λ/λmax)^n) * R (elementwise)."""
    lam_max = L.max(axis=1).replace(0, np.nan)
    penalty_factor = (1.0 - (L.div(lam_max, axis=0))).clip(lower=0.0)**n_penalty
    Re = penalty_factor * R
    return Re

def eapo_solver(mu: np.ndarray, Sigma: np.ndarray | None,
                Gamma=3.5, theta=0.5, n_iter=80, step=0.1) -> np.ndarray:
    """Projected-gradient ascent on the simplex for the EAPO objective."""
    x = np.ones_like(mu)/mu.size
    for _ in range(n_iter):
        g = mu - Gamma*(x/(np.linalg.norm(x)+1e-12))
        if (Sigma is not None) and (theta>0):
            g = g - 2.0*theta*(Sigma @ x)
        x = project_to_simplex(x + step*g)
    return x

def inv_var_weights(var: np.ndarray) -> np.ndarray:
    """Inverse-variance (diagonal GMV proxy)."""
    iv = 1.0/var
    iv[~np.isfinite(iv)] = 0.0
    s = iv.sum()
    return iv/s if s>0 else np.ones_like(iv)/len(iv)


def run_baseline(R, L, e1, years_range, lookback=252, Gamma=3.5, theta=0.5, cost_rate=0.0002):
    month_ends = R.groupby(pd.Grouper(freq='M')).apply(lambda df: df.index[-1]).values
    ret_series, int_series, weights_ts, turnovers = {}, {}, {}, {}
    prev = {}
    rets = ['Equal Weight', 'GMV (inv-var)', 'EMW (1/Scope1)', f'EAPO-θ={theta} (Γ={Gamma}, n=10)']
    for k in rets:
        prev[k] = pd.Series(0.0, index=R.columns)
        ret_series[k] = []; int_series[k] = []; turnovers[k] = []; weights_ts[k] = []
    min_year, max_year = years_range

    for mi, t in enumerate(month_ends):
        win = R.loc[:t].tail(lookback)
        if win.shape[0] < lookback:
            continue
        valid = win.columns[win.notna().all()].tolist()
        if len(valid) < 50:
            continue

        mu_e = penalty_adjusted_returns(R.loc[win.index, valid], L.loc[win.index, valid]).mean(axis=0).values
        ret_win = win[valid] - 1.0
        var = ret_win.var(axis=0, ddof=1).values
        Sigma = np.cov(ret_win.T, ddof=1)

        # EMW scope-1 inverse
        y_clip = min(max(int(pd.Timestamp(t).year), min_year), max_year)
        g_year = e1[e1['YEAR']==y_clip].set_index('TICKER')['SCOPE_1'].reindex(valid)
        inv_g = 1.0/g_year.replace({0:np.nan})
        if np.isfinite(inv_g).sum()>0:
            w_emw = (inv_g/np.nansum(inv_g)).fillna(0.0).values
        else:
            w_emw = np.ones(len(valid))/len(valid)

        W = {
            'Equal Weight': np.ones(len(valid))/len(valid),
            'GMV (inv-var)': inv_var_weights(var),
            'EMW (1/Scope1)': w_emw,
            f'EAPO-θ={theta} (Γ={Gamma}, n=10)': eapo_solver(mu_e, Sigma, Gamma=Gamma, theta=theta, n_iter=80, step=0.1)
        }

        if mi+1 < len(month_ends):
            hold_idx = R.loc[t:month_ends[mi+1]].index[1:]
        else:
            hold_idx = R.loc[t:].index[1:]

        for s,w in W.items():
            w = pd.Series(w, index=valid)
            wf = pd.Series(0.0, index=R.columns); wf.loc[w.index]=w.values
            tc = cost_rate*(wf - prev[s]).abs().sum()
            port_R = R.loc[hold_idx, w.index].mul(w.values, axis=1).sum(axis=1)
            if len(port_R)>0: port_R.iloc[0]-=tc
            It = L.loc[hold_idx, w.index].mul(w.values, axis=1).sum(axis=1)
            ret_series[s].append(port_R); int_series[s].append(It)
            turnovers[s].append(pd.Series((wf - prev[s]).abs().sum(), index=[t]))
            weights_ts[s].append(wf.rename(t))
            prev[s]=wf.copy()

    ret_series = {k: pd.concat(v) for k,v in ret_series.items()}
    int_series = {k: pd.concat(v) for k,v in int_series.items()}
    weights_ts = {k: pd.DataFrame(v).fillna(0.0) for k,v in weights_ts.items()}
    turnovers = {k: pd.concat(v) for k,v in turnovers.items()}
    summary_rows = []
    for k, ser in ret_series.items():
        M = portfolio_metrics(ser); M['Avg_Intensity'] = float(int_series[k].mean()); M['Avg_Turnover (L1)'] = float(turnovers[k].mean())
        summary_rows.append(pd.Series(M, name=k))
    summary = pd.DataFrame(summary_rows).round(3)
    return ret_series, int_series, weights_ts, turnovers, summary

def pareto_frontier(R, L, mu_grid, lookback=252, cost_rate=0.0002):
    month_ends = R.groupby(pd.Grouper(freq='M')).apply(lambda df: df.index[-1]).values
    pts = []
    for mu in mu_grid:
        prev = pd.Series(0.0, index=R.columns)
        SR, SI = [], []
        for mi, t in enumerate(month_ends):
            win = R.loc[:t].tail(lookback)
            if win.shape[0] < lookback: continue
            valid = win.columns[win.notna().all()].tolist()
            if len(valid)<50: continue
            mu_hat = R.loc[win.index, valid].mean(axis=0).values
            lam_vec = L.loc[win.index[-1], valid].values
            grad = mu_hat - mu*lam_vec
            w = project_to_simplex(grad)
            w = pd.Series(w, index=valid)
            if mi+1 < len(month_ends):
                hold_idx = R.loc[t:month_ends[mi+1]].index[1:]
            else:
                hold_idx = R.loc[t:].index[1:]
            wf = pd.Series(0.0, index=R.columns); wf.loc[w.index]=w.values
            tc = cost_rate*(wf - prev).abs().sum()
            port_R = R.loc[hold_idx, w.index].mul(w.values, axis=1).sum(axis=1)
            if len(port_R)>0: port_R.iloc[0]-=tc
            It = L.loc[hold_idx, w.index].mul(w.values, axis=1).sum(axis=1)
            SR.append(port_R); SI.append(It); prev=wf.copy()
        serR = pd.concat(SR); serI = pd.concat(SI)
        r = serR.values - 1.0; n = len(r); m = r.mean(); sd = r.std(ddof=1)
        ann_ret = (np.prod(1+r)**(252.0/n)-1.0)*100.0; ann_vol = sd*np.sqrt(252.0)*100.0
        sharpe = 0.0 if sd==0 else m/sd*np.sqrt(252.0)
        pts.append({'mu':mu, 'Ann_Return':ann_ret, 'Ann_Vol':ann_vol, 'Sharpe':sharpe, 'Avg_Intensity':float(serI.mean())})
    return pd.DataFrame(pts)

def run_eapo_theta(R, L, theta, lookback=252, Gamma=3.5, tau=None, cost_rate=0.0002):
    month_ends = R.groupby(pd.Grouper(freq='M')).apply(lambda df: df.index[-1]).values
    prev = pd.Series(0.0, index=R.columns)
    SR, SI, TT = [], [], []
    monthly_w = []
    for mi, t in enumerate(month_ends):
        win = R.loc[:t].tail(lookback)
        if win.shape[0] < lookback: continue
        valid = win.columns[win.notna().all()].tolist()
        if len(valid)<50: continue
        Re_win = penalty_adjusted_returns(R.loc[win.index, valid], L.loc[win.index, valid])
        mu_e = Re_win.mean(axis=0).values
        Sigma = np.cov((win[valid]-1.0).T, ddof=1)
        w_un = eapo_solver(mu_e, Sigma, Gamma=Gamma, theta=theta, n_iter=80, step=0.1)
        wf = pd.Series(0.0, index=R.columns); wf.loc[valid]=w_un
        if tau is not None:
            d = np.abs(wf.values - prev.values).sum()
            if d > tau and d>0:
                alpha = min(1.0, tau/d)
                wf = pd.Series(prev.values + alpha*(wf.values - prev.values), index=R.columns)
                wf = wf.clip(lower=0.0); s = wf.sum(); wf = wf/s if s>0 else pd.Series(np.ones(len(wf))/len(wf), index=wf.index)
        turnover = np.abs(wf - prev).sum()
        monthly_w.append(wf.rename(t))
        if mi+1 < len(month_ends):
            hold_idx = R.loc[t:month_ends[mi+1]].index[1:]
        else:
            hold_idx = R.loc[t:].index[1:]
        port_R = R.loc[hold_idx, :].mul(wf, axis=1).sum(axis=1)
        tc = cost_rate*turnover
        if len(port_R)>0: port_R.iloc[0]-=tc
        It = L.loc[hold_idx, :].mul(wf, axis=1).sum(axis=1)
        SR.append(port_R); SI.append(It); TT.append(pd.Series(turnover, index=[t]))
        prev = wf.copy()
    serR = pd.concat(SR); serI = pd.concat(SI); serT = pd.concat(TT); W = pd.DataFrame(monthly_w).fillna(0.0)
    M = portfolio_metrics(serR); M['Avg_Intensity']=float(serI.mean()); M['Avg_Turnover (L1)']=float(serT.mean())
    M['Avg_TC per Rebalance (bps)'] = 10000.0*cost_rate*M['Avg_Turnover (L1)'] if M['Avg_Turnover (L1)']==M['Avg_Turnover (L1)'] else np.nan
    return serR, serI, serT, W, pd.Series(M)


def plot_all(ret_series, int_series, theta_results_no, theta_results_tau, pareto_df,
             weights_ts, turnovers_tau, outdir='.'):
    os.makedirs(outdir, exist_ok=True)
    # Baseline cumulative wealth
    fig = plt.figure()
    cum_wealth = pd.DataFrame({k: np.cumprod(v.values) for k,v in ret_series.items()},
                              index=list(ret_series.values())[0].index)
    cum_wealth.plot(legend=True); plt.title("Cumulative Wealth — Baseline (EW, GMV, EMW, EAPO θ=0.5)"); plt.xlabel("Date"); plt.ylabel("Wealth (start=1)")
    savefig(os.path.join(outdir, "R2_cum_wealth.png"))
    # Baseline monthly intensity
    fig = plt.figure()
    int_df = pd.DataFrame({k: v for k,v in int_series.items()})
    int_df.resample('M').mean().plot(legend=True); plt.title("Portfolio Scope‑1 Intensity (Monthly Avg) — Baseline"); plt.xlabel("Date"); plt.ylabel("tCO2e / USD mm")
    savefig(os.path.join(outdir, "R2_intensity_monthly.png"))
    # Pareto frontier (line)
    fig = plt.figure()
    plt.plot(pareto_df['Avg_Intensity'], pareto_df['Ann_Return'], marker='o')
    plt.xlabel("Average Scope‑1 Intensity (tCO2e / USD mm)"); plt.ylabel("Annualized Return (%)"); plt.title("Pareto Frontier: Return vs Scope‑1 Intensity (μ‑sweep)"); plt.grid(True, alpha=0.3)
    savefig(os.path.join(outdir, "R2_pareto_frontier.png"))
    # Pareto bubble
    fig = plt.figure()
    sizes = 10.0 + pareto_df['Ann_Vol'].values
    plt.scatter(pareto_df['Avg_Intensity'], pareto_df['Ann_Return'], s=sizes)
    for i,row in pareto_df.iterrows():
        if i%3==0:
            plt.annotate(f"μ={row['mu']:.1e}", (row['Avg_Intensity'], row['Ann_Return']))
    plt.xlabel("Average Scope‑1 Intensity"); plt.ylabel("Annualized Return (%)"); plt.title("Pareto Frontier (bubble size ∝ annualized vol)"); plt.grid(True, alpha=0.3)
    savefig(os.path.join(outdir, "R2_pareto_bubble.png"))
    # Theta sweeps (no cap)
    fig = plt.figure()
    for th, d in theta_results_no.items():
        plt.plot(d['serR'].index, np.cumprod(d['serR'].values), label=f"θ={th}")
    plt.legend(); plt.title("EAPO θ‑Sweep (no turnover cap): Cumulative Wealth"); plt.xlabel("Date"); plt.ylabel("Wealth")
    savefig(os.path.join(outdir, "R2_theta_wealth_nocap.png"))
    fig = plt.figure()
    for th, d in theta_results_no.items():
        serI = d['serI']
        plt.plot(serI.resample('M').mean().index, serI.resample('M').mean().values, label=f"θ={th}")
    plt.legend(); plt.title("EAPO θ‑Sweep (no cap): Scope‑1 Intensity (Monthly Avg)"); plt.xlabel("Date"); plt.ylabel("tCO2e / USD mm")
    savefig(os.path.join(outdir, "R2_theta_intensity_nocap.png"))
    fig = plt.figure()
    theta_no = pd.DataFrame({th: d['metrics'] for th,d in theta_results_no.items()}).T.sort_index()
    plt.plot(theta_no['Avg_Intensity'], theta_no['Ann_Return'], marker='o')
    for th in theta_no.index:
        plt.annotate(f"θ={th}", (theta_no.loc[th,'Avg_Intensity'], theta_no.loc[th,'Ann_Return']))
    plt.xlabel("Average Scope‑1 Intensity (tCO2e / USD mm)"); plt.ylabel("Annualized Return (%)"); plt.title("θ‑Sweep Trade‑off (no turnover cap)"); plt.grid(True, alpha=0.3)
    savefig(os.path.join(outdir, "R2_theta_tradeoff_nocap.png"))
    # Theta sweeps (tau)
    fig = plt.figure()
    for th, d in theta_results_tau.items():
        plt.plot(d['serR'].index, np.cumprod(d['serR'].values), label=f"θ={th}")
    plt.legend(); plt.title("EAPO θ‑Sweep (τ=0.20): Cumulative Wealth"); plt.xlabel("Date"); plt.ylabel("Wealth")
    savefig(os.path.join(outdir, "R2_theta_wealth_tau.png"))
    fig = plt.figure()
    for th, d in theta_results_tau.items():
        serI = d['serI']
        plt.plot(serI.resample('M').mean().index, serI.resample('M').mean().values, label=f"θ={th}")
    plt.legend(); plt.title("EAPO θ‑Sweep (τ=0.20): Scope‑1 Intensity (Monthly Avg)"); plt.xlabel("Date"); plt.ylabel("tCO2e / USD mm")
    savefig(os.path.join(outdir, "R2_theta_intensity_tau.png"))
    fig = plt.figure()
    theta_tau = pd.DataFrame({th: d['metrics'] for th,d in theta_results_tau.items()}).T.sort_index()
    plt.plot(theta_tau['Avg_Intensity'], theta_tau['Ann_Return'], marker='o')
    for th in theta_tau.index:
        plt.annotate(f"θ={th}", (theta_tau.loc[th,'Avg_Intensity'], theta_tau.loc[th,'Ann_Return']))
    plt.xlabel("Average Scope‑1 Intensity (tCO2e / USD mm)"); plt.ylabel("Annualized Return (%)"); plt.title("θ‑Sweep Trade‑off (τ=0.20)"); plt.grid(True, alpha=0.3)
    savefig(os.path.join(outdir, "R2_theta_tradeoff_tau.png"))
    # Rolling Sharpe for baseline
    fig = plt.figure()
    for k,ser in ret_series.items():
        r = ser.values-1.0
        if len(r)>126:
            roll_m = pd.Series(r, index=ser.index).rolling(126).mean()
            roll_s = pd.Series(r, index=ser.index).rolling(126).std(ddof=1)
            rs = (roll_m/roll_s*np.sqrt(252)).replace([np.inf, -np.inf], np.nan)
            plt.plot(rs.index, rs.values, label=k)
    plt.legend(); plt.title("Rolling 6‑Month Sharpe (126 trading days)"); plt.xlabel("Date"); plt.ylabel("Sharpe (rolling)")
    savefig(os.path.join(outdir, "R2_rolling_sharpe.png"))
    # Drawdowns
    fig = plt.figure()
    eapo_key = [k for k in ret_series.keys() if k.startswith('EAPO-θ')][0]
    for k in ['Equal Weight', eapo_key]:
        wealth = np.cumprod(ret_series[k].values)
        peak = np.maximum.accumulate(wealth)
        dd = wealth/peak - 1.0
        plt.plot(ret_series[k].index, dd, label=k)
    plt.legend(); plt.title("Drawdowns (EW vs EAPO θ=0.5)"); plt.xlabel("Date"); plt.ylabel("Drawdown")
    savefig(os.path.join(outdir, "R2_drawdowns.png"))
    # Histogram of daily excess
    fig = plt.figure()
    diff = (ret_series[eapo_key]-1.0) - (ret_series['Equal Weight']-1.0)
    plt.hist(diff.values, bins=50); plt.title("Distribution of Daily Excess Return: EAPO − EW"); plt.xlabel("Daily Excess Return"); plt.ylabel("Frequency")
    savefig(os.path.join(outdir, "R2_hist_eapo_minus_ew.png"))
    # Weighted CDF of holdings' intensities (monthly rebalances)
    fig = plt.figure()
    def weighted_cdf(weights_df, L, label):
        vals = []
        for t,row in weights_df.iterrows():
            lam = L.loc[t, row.index].values
            w = row.values
            reps = np.clip((w*10000).astype(int), 0, None)
            if reps.sum() == 0:
                continue
            m = np.repeat(lam, reps)
            vals.extend(m.tolist())
        if len(vals)>0:
            vals = np.sort(np.array(vals))
            y = np.linspace(0,1,len(vals))
            plt.plot(vals, y, label=label)
    weighted_cdf(weights_ts['Equal Weight'], L, 'EW')
    weighted_cdf(weights_ts[eapo_key], L, 'EAPO θ=0.5')
    plt.legend(); plt.title("Weighted CDF of Holdings' Scope‑1 Intensities (Monthly Rebalances)"); plt.xlabel("tCO2e / USD mm"); plt.ylabel("Cumulative weight")
    savefig(os.path.join(outdir, "R2_intensity_weighted_cdf.png"))
    # Turnover series
    fig = plt.figure()
    turn = list(turnovers_tau.values())[0] if len(turnovers_tau)>0 else None
    if turn is not None:
        plt.plot(turn.index, turn.values)
    plt.title("Turnover per Monthly Rebalance — EAPO θ=0.5 (τ=0.20)"); plt.xlabel("Date"); plt.ylabel("L1 Turnover")
    savefig(os.path.join(outdir, "R2_turnover_series.png"))

def main():
    # Load panels
    R, L, e1, years_range = build_panels()
    # Baseline
    ret_series, int_series, weights_ts, turnovers, summary = run_baseline(R, L, e1, years_range, lookback=252, Gamma=3.5, theta=0.5, cost_rate=0.0002)
    summary.to_csv("R2_summary_with_theta.csv")
    # Pareto
    mu_grid = np.geomspace(1e-7, 5e-4, 14)
    frontier = pareto_frontier(R, L, mu_grid, lookback=252, cost_rate=0.0002).sort_values('Avg_Intensity').reset_index(drop=True)
    frontier.to_csv("R2_pareto_frontier_points.csv", index=False)
    # Theta sweeps
    theta_vals = [0.0, 0.25, 0.5, 1.0]
    theta_results_no, theta_results_tau = {}, {}
    for th in theta_vals:
        r,i,t,w,Ms = run_eapo_theta(R, L, theta=th, lookback=252, Gamma=3.5, tau=None,   cost_rate=0.0002)
        theta_results_no[th]  = {'serR': r, 'serI': i, 'metrics': Ms}
        r,i,t,w,Ms = run_eapo_theta(R, L, theta=th, lookback=252, Gamma=3.5, tau=0.20,  cost_rate=0.0002)
        theta_results_tau[th] = {'serR': r, 'serI': i, 'metrics': Ms}
    pd.DataFrame({th: d['metrics'] for th,d in theta_results_no.items()}).T.sort_index().round(3).to_csv("R2_theta_sweep_no_turnover.csv")
    pd.DataFrame({th: d['metrics'] for th,d in theta_results_tau.items()}).T.sort_index().round(3).to_csv("R2_theta_sweep_with_turnover.csv")
    # HAC / Newey–West
    eapo_key = [k for k in ret_series if k.startswith('EAPO-θ')][0]
    daily_simple = {k: ret_series[k]-1.0 for k in ret_series}
    comparisons = []
    for base in ['Equal Weight', 'GMV (inv-var)', 'EMW (1/Scope1)']:
        diff = daily_simple[eapo_key] - daily_simple[base]
        se = newey_west_se_mean(diff.values, L=20)
        mean = np.nanmean(diff.values)
        tstat = mean / se if se>0 else np.nan
        comparisons.append({'Comparison': f'{eapo_key} − {base}', 'Mean daily (bp)': 1e4*mean, 'SE (bp)': 1e4*se, 't-stat': tstat})
    pd.DataFrame(comparisons).round(3).to_csv("R2_newey_west.csv", index=False)
    # Plots
    plot_all(ret_series, int_series, theta_results_no, theta_results_tau, frontier, weights_ts, 
             turnovers_tau={0.5: theta_results_tau[0.5]['serI']*0 + 0.0}, outdir='.')
    print("Done. Figures (R2_*.png) and CSVs written.")

if __name__ == "__main__":
    main()
