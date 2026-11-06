"""
Track E: Case Study - Battery Energy Storage System (BESS) Dispatch Optimization
Problem Formulation:
- Horizon: T = 24 hours (1-hour dispatch slots)
- Inputs: Dynamic wholesale price ($/MWh), Solar generation profile (MW), Carbon intensity (gCO2/kWh)
- Decision Variables:
    P_ch(t): Charge power [0, P_max]
    P_dis(t): Discharge power [0, P_max]
    SoC(t): Battery State of Charge [SoC_min, SoC_max]
- Constraints:
    1. SoC dynamic conservation: SoC(t+1) = SoC(t) + (eta_ch * P_ch(t) - P_dis(t)/eta_dis) * dt / E_cap
    2. Operational bounds: SoC_min <= SoC(t) <= SoC_max
    3. Power limits: 0 <= P_ch(t) <= P_max, 0 <= P_dis(t) <= P_max
    4. End-of-day condition: SoC(T) >= SoC(0)
    5. Grid feeder limit: P_dis(t) - P_ch(t) + P_solar(t) <= P_feeder
- Objective:
    Minimize Total Net Cost = sum_{t} [ Price(t) * (P_ch(t) - P_dis(t)) + C_deg * (P_ch(t) + P_dis(t)) + Carbon_cost(t) ]
    (Net Profit = - Net Cost)
"""

import numpy as np

class BESSProblem:
    def __init__(self, seed=42):
        self.T = 24  # 24 hours
        self.dt = 1.0  # hour
        self.E_cap = 100.0  # MWh battery storage capacity
        self.P_max = 25.0   # MW maximum charge/discharge rate (C/4 rate)
        self.eta_ch = 0.95  # Charging efficiency
        self.eta_dis = 0.95 # Discharging efficiency
        self.SoC_min = 0.10 # Minimum safe SoC (10%)
        self.SoC_max = 0.90 # Maximum safe SoC (90%)
        self.SoC_0 = 0.50   # Initial SoC (50%)
        self.C_deg = 3.5    # Degradation wear cost ($/MWh throughput)
        self.P_feeder = 40.0 # Maximum substation grid injection limit (MW)

        rng = np.random.default_rng(seed)
        # Synthetic CAISO wholesale price profile ($/MWh): low at night/midday solar, peak in evening (17:00-21:00)
        hours = np.arange(self.T)
        base_price = 25.0 + 15.0 * np.sin((hours - 4) * np.pi / 12.0)
        # Duck curve midday dip + evening peak
        duck_curve = np.zeros(self.T)
        duck_curve[10:16] = -12.0  # Solar surplus depression
        duck_curve[17:22] = 45.0   # Evening peak ramping
        self.prices = np.clip(base_price + duck_curve + rng.normal(0, 2.0, self.T), 10.0, 120.0)

        # Solar PV generation (MW)
        solar = np.zeros(self.T)
        solar[6:19] = 30.0 * np.sin(np.linspace(0, np.pi, 13))
        self.solar_profile = solar

        # Grid carbon intensity (kg CO2 / MWh)
        self.carbon_intensity = 400.0 - 250.0 * (solar / 30.0)
        self.carbon_price = 0.03 # $30/ton = $0.03/kg

    def evaluate_dispatch(self, P_ch, P_dis):
        """
        Simulates battery physics and evaluates feasibility, net cost, and profit.
        """
        P_ch = np.array(P_ch, dtype=float)
        P_dis = np.array(P_dis, dtype=float)

        soc = np.zeros(self.T + 1)
        soc[0] = self.SoC_0

        penalty = 0.0
        for t in range(self.T):
            # Power bounds
            if P_ch[t] < 0 or P_ch[t] > self.P_max:
                penalty += 1000.0 * max(0, P_ch[t] - self.P_max)
            if P_dis[t] < 0 or P_dis[t] > self.P_max:
                penalty += 1000.0 * max(0, P_dis[t] - self.P_max)

            # Feeder injection limit
            grid_net = P_dis[t] - P_ch[t] + self.solar_profile[t]
            if grid_net > self.P_feeder:
                penalty += 500.0 * (grid_net - self.P_feeder)

            # SoC update
            d_energy = (self.eta_ch * P_ch[t] - (P_dis[t] / self.eta_dis)) * self.dt
            soc[t + 1] = soc[t] + (d_energy / self.E_cap)

            if soc[t + 1] < self.SoC_min:
                penalty += 2000.0 * (self.SoC_min - soc[t + 1])
            elif soc[t + 1] > self.SoC_max:
                penalty += 2000.0 * (soc[t + 1] - self.SoC_max)

        # End of day balance
        if soc[-1] < self.SoC_0:
            penalty += 1500.0 * (self.SoC_0 - soc[-1])

        # Financial cost
        energy_cost = np.sum(self.prices * (P_ch - P_dis) * self.dt)
        deg_cost = self.C_deg * np.sum((P_ch + P_dis) * self.dt)
        carbon_cost = np.sum(self.carbon_price * self.carbon_intensity * np.maximum(0, P_ch - self.solar_profile) * self.dt)

        total_net_cost = energy_cost + deg_cost + carbon_cost + penalty
        net_profit = -total_net_cost

        is_feasible = (penalty < 1e-3)
        return total_net_cost, net_profit, is_feasible, soc

