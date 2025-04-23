import pandas as pd, numpy as np, matplotlib.pyplot as plt

load = lambda f: pd.read_csv(f"data/{f}.csv", names=["DATE", f], header=0, parse_dates=["DATE"]).dropna().set_index("DATE").apply(pd.to_numeric, errors="coerce")
files = ["embDiv", "embPrice", "GS10", "DEXUSEU", "CPI", "SP500", "USSTHPI"]
data = {f: load(f) for f in files}; data["USSTHPI"].columns = ["house_price"]

df = pd.concat(data.values(), axis=1).resample("QE-DEC").mean(numeric_only=True)
pct = lambda s, n=1: s.pct_change(n, fill_method=None)
comp = pd.DataFrame(index=df.index)
comp["real_equity_return"] = pct(df["SP500"],4) - pct(df["CPI"],4)
comp["equity_vol"] = pct(df["SP500"]).rolling(3).std()
comp["fx_vol"] = pct(df["DEXUSEU"]).rolling(3).std()
comp["bond_yield_vol"] = pct(df["GS10"]).rolling(3).std()
comp["fx_spread"] = df["embDiv"] / df["embPrice"]
comp["real_house_price_growth"] = pct(df["house_price"],4) - pct(df["CPI"],4)

p = comp.dropna().rank(pct=True) * 100
for c in ["equity_vol","fx_vol","bond_yield_vol","fx_spread"]: p[c] = 100 - p[c]
p["Mispricing_Risk"] = p.mean(1); r = p["2015":"2024"]

r["Mispricing_Risk"].plot(figsize=(12,6), lw=2, color="blue", title="U.S. Mispricing Risk")
plt.axhline(66.7, color="orange", ls="--", lw=1.5, label="66.7 Early Warning")
plt.axhline(80, color="red", ls="--", lw=1.5, label="80 High Risk")
plt.ylabel("Percentile"); plt.grid(); plt.legend(); plt.tight_layout(); plt.show()