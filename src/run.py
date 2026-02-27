"""
run.py

End-to-end demo:
tickers -> prices -> returns -> mu/cov -> Markowitz weights -> metrics
"""

from src.data import fetch_prices, clean_prices
from src.returns import compute_returns, estimate_mean_returns, estimate_covariance
from src.markowitz import minimum_variance_weights, max_sharpe_weights
from src.indicators import portfolio_return, portfolio_volatility, sharpe_ratio
from src.dashboard import show_dashboard
from src.frontier import efficient_frontier
from src.plotting import plot_frontier
from src.risk_profile import ask_risk_profile, risk_score, choose_strategy, blend_weights



def main():
    tickers = ["ODL.OL", "VEND.OL", "STB.OL"] #hvilke aksjer vi vil investere i
    years = 3 #antall år vi vil gå tilbake i tid
    rf = 0.04 #risikofri rente. Det som gir oss avkastning hvis vi hadde satt pengene i banken. 
    

    prices, fetch_report = fetch_prices(tickers, years=years, progress=False) 
    prices, clean_report = clean_prices(prices, method="dropna", min_obs=252)

    if prices.empty or prices.shape[1] < 2:
        print(fetch_report)
        print(clean_report)
        print("Not enough usable price series to continue.")
        return

    rets = compute_returns(prices, kind="log")
    mu = estimate_mean_returns(rets, periods=252) #forventingsverdi for antall åpne dager på børsen
    cov = estimate_covariance(rets, periods=252)

    # Markowitz weights
    w_min = minimum_variance_weights(cov) 
    w_tan = max_sharpe_weights(mu, cov, rf=rf)

    # Metrics
    min_ret = portfolio_return(w_min.values, mu)
    min_vol = portfolio_volatility(w_min.values, cov)
    min_sr = sharpe_ratio(w_min.values, mu, cov, rf=rf)

    tan_ret = portfolio_return(w_tan.values, mu)
    tan_vol = portfolio_volatility(w_tan.values, cov)
    tan_sr = sharpe_ratio(w_tan.values, mu, cov, rf=rf)

    # Risk profiling
    profile = ask_risk_profile()
    score = risk_score(profile)
    decision = choose_strategy(score)

    if decision.strategy == "min_variance":
        w_rec = w_min
    elif decision.strategy == "max_sharpe":
        w_rec = w_tan
    else:
        # balanced: alpha in [0,1] depending on score in [5.0, 7.5]
        alpha = (score - 5.0) / (7.5 - 5.0)
        w_rec = blend_weights(w_min, w_tan, alpha=alpha)

    rec_ret = portfolio_return(w_rec.values, mu)
    rec_vol = portfolio_volatility(w_rec.values, cov)
    rec_sr = sharpe_ratio(w_rec.values, mu, cov, rf=rf)

    risk_info = {
        "time_horizon_years": profile.time_horizon_years,
        "loss_tolerance": profile.loss_tolerance,
        "experience": profile.experience,
        "score": score,
        "strategy": decision.strategy,
        "explanation": decision.explanation,
        "rec_ret": rec_ret,
        "rec_vol": rec_vol,
        "rec_sr": rec_sr,
        "w_rec": w_rec,
    }

    # Pretty output
    show_dashboard(
        tickers=list(prices.columns),
        mu=mu.loc[prices.columns],
        w_min=w_min.loc[prices.columns],
        w_tan=w_tan.loc[prices.columns],
        min_ret=min_ret,
        min_vol=min_vol,
        min_sr=min_sr,
        tan_ret=tan_ret,
        tan_vol=tan_vol,
        tan_sr=tan_sr,
        rf=rf,
        years=years,
        risk_info=risk_info
    )

    frontier_df, _ = efficient_frontier(mu, cov, n_points=50)
    plot_frontier(
        frontier_df,
        mu=mu,
        cov=cov,
        min_var=(min_vol, min_ret),
        max_sharpe=(tan_vol, tan_ret),
        rf=rf,
        title="Efficient Frontier",
    )


if __name__ == "__main__":
    main()