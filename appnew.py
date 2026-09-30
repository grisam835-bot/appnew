import streamlit as st
import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.stats import poisson, nbinom, erlang, wasserstein_distance
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pure Market Engine 9x9 (Optimized + Copula + Reverse Eng)", layout="wide")

class PureMarketEngine9x9:
    def __init__(self, max_goals=8):
        self.max_goals = max_goals
        self.weights = self._build_weight_matrix()

    def _build_weight_matrix(self):
        W = np.full((self.max_goals + 1, self.max_goals + 1), 0.02)
        anchor_scores = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (2,1), (1,2), (3,0), (0,3)]
        for h, a in anchor_scores:
            if h <= self.max_goals and a <= self.max_goals:
                W[h, a] = 1.0
        secondary_scores = [(3,1), (1,3), (2,2), (3,2), (2,3), (4,0), (0,4), (4,1), (1,4), (4,2), (2,4)]
        for h, a in secondary_scores:
            if h <= self.max_goals and a <= self.max_goals:
                W[h, a] = 0.45
        return W

    def _power_unmargin_binary(self, odd_1, odd_2):
        if odd_1 <= 1.0 or odd_2 <= 1.0:
            return None, None
        p1_raw, p2_raw = 1.0 / odd_1, 1.0 / odd_2
        try:
            res = root_scalar(lambda k: (p1_raw ** k + p2_raw ** k) - 1.0, bracket=[0.001, 5.0], method='brentq')
            return (p1_raw ** res.root), (p2_raw ** res.root)
        except:
            clean = self._shin_unmargin({'1': p1_raw, '2': p2_raw})
            return clean['1'], clean['2']

    def _shin_unmargin(self, raw_probs):
        n = len(raw_probs)
        if n == 0: return {}
        sum_pi = sum(raw_probs.values())
        if sum_pi <= 1.0: return {k: v / sum_pi for k, v in raw_probs.items()}
        z = (sum_pi - 1.0) / max(1, n - 1)
        clean_probs = {}
        for k, pi in raw_probs.items():
            num = np.sqrt(z**2 + 4 * (1 - z) * (pi**2 / sum_pi)) - z
            den = 2 * (1 - z)
            clean_probs[k] = max(1e-12, num / max(den, 1e-12))
        total_clean = sum(clean_probs.values())
        return {k: v / total_clean for k, v in clean_probs.items()}

    def _shin_unmargin_asymmetric(self, raw_probs, z_mri_x=0.0):
        if z_mri_x <= 1.8:
            return self._shin_unmargin(raw_probs)
        
        n = len(raw_probs)
        if n == 0: return {}
        sum_pi = sum(raw_probs.values())
        if sum_pi <= 1.0: return {k: v / sum_pi for k, v in raw_probs.items()}
        
        z_base = (sum_pi - 1.0) / max(1, n - 1)
        clean_probs = {}
        asym_boost = 1.0 + 0.12 * min(z_mri_x - 1.8, 2.5)
        
        for k, pi in raw_probs.items():
            z_eff = z_base * (asym_boost if (isinstance(k, tuple) and k[0] == k[1]) else 1.0)
            num = np.sqrt(z_eff**2 + 4 * (1 - z_eff) * (pi**2 / sum_pi)) - z_eff
            den = 2 * (1 - z_eff)
            clean_probs[k] = max(1e-12, num / max(den, 1e-12))
            
        total_clean = sum(clean_probs.values())
        return {k: v / total_clean for k, v in clean_probs.items()}

    def calculate_x_stress_z_score(self, kl_div, x_gap, mri_index):
        mri_x_component = mri_index * (x_gap / max(x_gap + kl_div + 0.1, 1e-4))
        z_mri_x = (mri_x_component - 15.0) / 12.0
        return float(max(0.0, z_mri_x))

    def _apply_shannon_bayes_noise_filter(self, raw_probs):
        if not raw_probs: return {}
        sum_pi = sum(raw_probs.values())
        payout = 1.0 / sum_pi if sum_pi > 0 else 1.0
        if payout < 0.92:
            filtered_probs = {}
            margin_penalty = (0.92 - payout) * 2.0
            for k, p in raw_probs.items():
                sigmoid_weight = 1.0 / (1.0 + np.exp(-15.0 * (p - 0.03)))
                adj_p = p * (1.0 - margin_penalty * (1.0 - sigmoid_weight))
                filtered_probs[k] = max(1e-12, adj_p)
            return filtered_probs
        return raw_probs

    def _get_shin_alpha(self, raw_probs):
        n = len(raw_probs)
        if n <= 1: return 0.0
        sum_pi = sum(raw_probs.values())
        if sum_pi <= 1.0: return 0.0
        return float((sum_pi - 1.0) / max(1, n - 1))

    def _dixon_coles_adjustment(self, h, a, lambda_h, mu_a, rho):
        xg_ratio = lambda_h / max(mu_a, 1e-5)
        if xg_ratio > 2.5:
            asym_factor = 1.0 + 0.15 * np.tanh(xg_ratio - 2.5)
        elif xg_ratio < 0.4:
            asym_factor = 1.0 + 0.15 * np.tanh((1.0 / xg_ratio) - 2.5)
        else:
            asym_factor = 1.0

        eff_rho = rho * asym_factor
        if h == 0 and a == 0: return 1.0 - (lambda_h * mu_a * eff_rho)
        elif h == 1 and a == 0: return 1.0 + (mu_a * eff_rho)
        elif h == 0 and a == 1: return 1.0 + (lambda_h * eff_rho)
        elif h == 1 and a == 1: return 1.0 - eff_rho
        return 1.0

    def _apply_copula_density(self, u, v, copula_type="Frank", theta=1.5):
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        v = np.clip(v, 1e-6, 1.0 - 1e-6)
        
        if copula_type == "Frank":
            if abs(theta) < 1e-4:
                return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
            num = -theta * (np.exp(-theta) - 1.0) * np.exp(-theta * (u + v))
            den = ((np.exp(-theta * u) - 1.0) * (np.exp(-theta * v) - 1.0) + (np.exp(-theta) - 1.0)) ** 2
            return np.maximum(1e-6, num / np.maximum(den, 1e-12))
            
        elif copula_type == "Gumbel":
            if theta <= 1.0:
                return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
            x = -np.log(u)
            y = -np.log(v)
            A = (x**theta + y**theta)**(1.0 / theta)
            C = np.exp(-A)
            density = (C / (u * v)) * ((x * y)**(theta - 1.0) / (x**theta + y**theta)**(2.0 - 1.0/theta)) * (A + theta - 1.0)
            return np.maximum(1e-6, density)

        elif copula_type == "Clayton":
            if theta <= 0.0:
                return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
            density = (1.0 + theta) * ((u * v) ** (-1.0 - theta)) * ((u ** (-theta) + v ** (-theta) - 1.0) ** (-2.0 - 1.0 / theta))
            return np.maximum(1e-6, density)
            
        return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0

    def _get_marginal_pmf_cdf(self, mu, k_val, phi=0.0):
        if phi <= 1e-4:
            pmf = poisson.pmf(k_val, mu)
            cdf = poisson.cdf(k_val, mu)
        else:
            r = 1.0 / phi
            p_param = r / (r + mu)
            pmf = nbinom.pmf(k_val, r, p_param)
            cdf = nbinom.cdf(k_val, r, p_param)
        return pmf, cdf

    def _generate_matrix(self, lambda_h, mu_a, rho=0.0, pi_zero=0.0, copula_type="Fără", copula_theta=1.5, phi_dispersion=0.0):
        h_arr = np.arange(self.max_goals + 1)
        a_arr = np.arange(self.max_goals + 1)
        
        p_h, cdf_h = self._get_marginal_pmf_cdf(lambda_h, h_arr, phi_dispersion)
        p_a, cdf_a = self._get_marginal_pmf_cdf(mu_a, a_arr, phi_dispersion)
        
        P_H, P_A = np.meshgrid(p_h, p_a, indexing='ij')
        CDF_H, CDF_A = np.meshgrid(cdf_h, cdf_a, indexing='ij')
        
        xg_ratio = lambda_h / max(mu_a, 1e-5)
        if xg_ratio > 2.5:
            asym_factor = 1.0 + 0.15 * np.tanh(xg_ratio - 2.5)
        elif xg_ratio < 0.4:
            asym_factor = 1.0 + 0.15 * np.tanh((1.0 / xg_ratio) - 2.5)
        else:
            asym_factor = 1.0
        eff_rho = rho * asym_factor

        adj = np.ones((self.max_goals + 1, self.max_goals + 1))
        if self.max_goals >= 1:
            adj[0, 0] = 1.0 - (lambda_h * mu_a * eff_rho)
            adj[1, 0] = 1.0 + (mu_a * eff_rho)
            adj[0, 1] = 1.0 + (lambda_h * eff_rho)
            adj[1, 1] = 1.0 - eff_rho

        base_matrix = P_H * P_A * adj
        
        if copula_type != "Fără":
            u_mid = np.clip(CDF_H - 0.5 * P_H, 1e-6, 1.0 - 1e-6)
            v_mid = np.clip(CDF_A - 0.5 * P_A, 1e-6, 1.0 - 1e-6)
            c_density = self._apply_copula_density(u_mid, v_mid, copula_type, copula_theta)
            base_matrix *= c_density

        diag_mask = np.eye(self.max_goals + 1, dtype=bool)
        diag_boost = 1.0 + (pi_zero * np.exp(-0.5 * h_arr))
        
        matrix = np.where(diag_mask, base_matrix * diag_boost, (1.0 - pi_zero * 0.2) * base_matrix)
        matrix = np.maximum(1e-12, matrix)
                
        total_p = np.sum(matrix)
        return matrix / total_p if total_p > 0 else matrix

    def _huber_loss(self, y_true, y_pred, delta=0.001):
        error = y_pred - y_true
        return np.where(np.abs(error) <= delta, 0.5 * (error ** 2), delta * (np.abs(error) - 0.5 * delta))

    # --- CALCUL EROARE REZIDUALĂ PENTRU AIC ---
    def _calculate_model_residual_error(self, matrix, cs_odds, main_ah, main_ou):
        sse = 0.0
        # 1. Reziduuri pe Correct Score
        for (h, a), odd in cs_odds.items():
            if h <= self.max_goals and a <= self.max_goals and odd > 1.0:
                p_implied = 1.0 / odd
                p_model = matrix[h, a]
                sse += (p_implied - p_model) ** 2

        # 2. Reziduu pe Asian Handicap
        if main_ah and main_ah.get('home_odd', 0) > 1.0:
            p_ah_clean, _ = self._power_unmargin_binary(main_ah['home_odd'], main_ah['away_odd'])
            if p_ah_clean:
                p_ah_model = self.calculate_ah_probability(matrix, main_ah['home_line'], is_home=True)
                sse += (p_ah_clean - p_ah_model) ** 2

        # 3. Reziduu pe Over/Under
        if main_ou and main_ou.get('over_odd', 0) > 1.0:
            p_ou_clean, _ = self._power_unmargin_binary(main_ou['over_odd'], main_ou['under_odd'])
            if p_ou_clean:
                p_ou_model = self.calculate_ou_probability(matrix, main_ou['line'], is_over=True)
                sse += (p_ou_clean - p_ou_model) ** 2

        return max(sse, 1e-12)

    # --- NEW MODULE 6: MARKET TOOTHPRINTS & STRESS ---
    def calculate_vig_squeeze_matrix(self, matrix_decoupled, cs_odds_close):
        squeeze_mat = np.ones((self.max_goals + 1, self.max_goals + 1))
        for (h, a), odd in cs_odds_close.items():
            if h <= self.max_goals and a <= self.max_goals and odd > 1.0:
                p_close = 1.0 / odd
                p_decoupled = matrix_decoupled[h, a]
                squeeze_mat[h, a] = p_close / max(p_decoupled, 1e-12)
        return squeeze_mat

    def calculate_asymmetric_pressure_field(self, matrix):
        diag = np.diag(matrix)
        curvature = np.gradient(np.gradient(diag)) if len(diag) > 2 else np.zeros_like(diag)
        return float(np.mean(curvature))

    # --- NEW MODULE 7: REVERSE ENGINEERING ON STATE VECTOR ---
    def reverse_engineer_bookmaker_state(self, cs_odds, main_ah, main_ou, main_x):
        copula_candidates = ["Frank", "Gumbel", "Clayton"]
        best_fit = None
        min_aic = float('inf')
        n_data_points = len(cs_odds) + 3

        for c_type in copula_candidates:
            l, m, r, pi, _, _, _, _, _, _, mat, theta_fit, phi_fit = self.extract_pure_xg(
                cs_odds, main_ah, main_ou, main_x,
                mode="Anchored (5.0 Weight)",
                copula_type=c_type,
                auto_fit_copula=True,
                use_nbinom=True
            )
            sse = self._calculate_model_residual_error(mat, cs_odds, main_ah, main_ou)
            k_params = 6
            aic = n_data_points * np.log(max(sse / n_data_points, 1e-12)) + 2 * k_params

            if aic < min_aic:
                min_aic = aic
                best_fit = {
                    'copula_type': c_type,
                    'lambda_home': round(l, 3),
                    'mu_away': round(m, 3),
                    'dixon_coles_rho': round(r, 4),
                    'copula_theta': round(theta_fit, 3),
                    'nbinom_phi': round(phi_fit, 4),
                    'aic_score': round(aic, 2),
                    'fit_quality': "EXCELENT" if sse < 0.005 else "MEDIU / ZGOMOT"
                }

        return best_fit

    # --- MODULUL 1-5 & METRICI EXISTENTE ---
    def calculate_vector_field_flow(self, mat_open, mat_close):
        delta_p = mat_close - mat_open
        v_y, u_x = np.gradient(delta_p)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(delta_p, cmap="coolwarm", origin="upper", vmin=-0.03, vmax=0.03)
        x, y = np.meshgrid(np.arange(self.max_goals + 1), np.arange(self.max_goals + 1))
        ax.quiver(x, y, u_x, v_y, color="black", angles="xy", scale_units="xy", scale=0.5, pivot="middle")
        ax.set_title("Vector Field Flow (ΔP & Gradient Stream)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Goluri Oaspeți")
        ax.set_ylabel("Goluri Gazde")
        ax.set_xticks(range(self.max_goals + 1))
        ax.set_yticks(range(self.max_goals + 1))
        fig.colorbar(cax, ax=ax, label="Shift Probabilitate (ΔP)")
        plt.tight_layout()
        plt.close(fig)
        return u_x, v_y, fig

    def calculate_surface_laplacian(self, mat_p):
        kernel = np.array([[0,  1, 0], [1, -4, 1], [0,  1, 0]])
        laplacian_map = ndimage.convolve(mat_p, kernel, mode='constant', cval=0.0)
        return laplacian_map, float(np.max(np.abs(laplacian_map))), float(np.std(laplacian_map))

    def calculate_topological_bimodal_index(self, mat_p, threshold_relative=0.25):
        local_max = ndimage.maximum_filter(mat_p, size=3) == mat_p
        max_p = np.max(mat_p)
        significant_peaks = local_max & (mat_p >= (max_p * threshold_relative))
        coords = np.argwhere(significant_peaks)
        return len(coords) >= 2, len(coords), [(int(c[0]), int(c[1]), float(mat_p[c[0], c[1]])) for c in coords]

    def calculate_bivariate_moments(self, mat_p):
        grid = np.arange(self.max_goals + 1)
        x_grid, y_grid = np.meshgrid(grid, grid)
        lambda_h = np.sum(mat_p * y_grid)
        mu_a = np.sum(mat_p * x_grid)
        sigma_h = max(np.sqrt(np.sum(mat_p * ((y_grid - lambda_h) ** 2))), 1e-6)
        sigma_a = max(np.sqrt(np.sum(mat_p * ((x_grid - mu_a) ** 2))), 1e-6)
        coskew_ha = np.sum(mat_p * (y_grid - lambda_h) * ((x_grid - mu_a) ** 2)) / (sigma_h * (sigma_a ** 2))
        
        if coskew_ha > 0.10: regime, desc = "Meci Răzbunător (Tit-for-Tat)", "Un gol primit determină o replică ofensivă."
        elif coskew_ha < -0.10: regime, desc = "Meci de Blocaj Defensiv", "Un gol marcat închide jocul."
        else: regime, desc = "Meci Simetric / Echilibrat", "Evoluția scorului urmează dinamica Poisson."
        return float(coskew_ha), regime, desc

    def calculate_first_passage_time(self, lambda_h, mu_a):
        rate = lambda_h + mu_a
        if rate <= 0: return 90.0, 0.0, 0.0
        return float((1.0 / rate) * 90.0), float((1.0 - np.exp(-rate * (15/90))) * 100), float((1.0 - np.exp(-rate * (30/90))) * 100)

    def calculate_svi_9x9(self, mat_open, mat_close, grad_volatilitate_t=0.15):
        delta = mat_close - mat_open
        return float(np.std(delta) * (1.0 + grad_volatilitate_t)), delta

    def calculate_residual_heatmap_9x9(self, mat_open, mat_close):
        res_mat = mat_close - mat_open
        max_dev = float(np.max(res_mat))
        idx = np.unravel_index(np.argmax(res_mat), (9, 9))
        score = (int(idx[0]), int(idx[1]))
        contaminated = max_dev > 0.04
        msg = f"⚠️ Anomalie Detectată la {score[0]}-{score[1]} (+{max_dev*100:.2f}%)" if contaminated else "Piață 9x9 Structurată"
        return {"matrice_reziduuri": res_mat, "max_devier": max_dev, "scor_anomalie": score, "este_contaminat": contaminated, "mesaj": msg}

    def calculate_micro_price_skew(self, mat_close, main_ou):
        p_over = self.calculate_ou_probability(mat_close, main_ou['line'], is_over=True)
        return float(1.0 / max(p_over, 1e-5))

    def calculate_obi_s(self, mass_analysis, main_ou):
        delta_p = abs(mass_analysis.get('under_shift_pct', 0.0)) / 100.0
        delta_odd = abs(main_ou['over_odd'] - 1.90) / 1.90
        return float(delta_p / max(delta_odd, 1e-4))

    def calculate_phantom_shift(self, mat_open, mat_close, main_ou):
        p_mat_open = self.calculate_ou_probability(mat_open, main_ou['line'], is_over=True)
        p_mat_close = self.calculate_ou_probability(mat_close, main_ou['line'], is_over=True)
        p_clean, _ = self._power_unmargin_binary(main_ou['over_odd'], main_ou['under_odd'])
        return float(abs((p_mat_close - p_mat_open) - ((p_clean if p_clean else 0.5) - 0.5)))

    def calculate_marginal_overround_asymmetry(self, main_ou):
        if not main_ou or main_ou.get('over_odd', 0) <= 1.0 or main_ou.get('under_odd', 0) <= 1.0: return 0.0
        p_over_raw, p_under_raw = 1.0 / main_ou['over_odd'], 1.0 / main_ou['under_odd']
        tot = (p_over_raw + p_under_raw) - 1.0
        if tot <= 0: return 0.0
        p_clean, _ = self._power_unmargin_binary(main_ou['over_odd'], main_ou['under_odd'])
        return float((p_over_raw - (p_clean if p_clean else p_over_raw)) / max(tot, 1e-6))

    def calculate_wasserstein_emd_1d(self, mat_open, mat_close):
        grid = np.arange(self.max_goals + 1)
        emd_g = wasserstein_distance(grid, grid, np.sum(mat_open, axis=1), np.sum(mat_close, axis=1))
        diff_grid = np.arange(-self.max_goals, self.max_goals + 1)
        d_open, d_close = np.zeros(len(diff_grid)), np.zeros(len(diff_grid))
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                idx = (h - a) + self.max_goals
                d_open[idx] += mat_open[h, a]
                d_close[idx] += mat_close[h, a]
        return float(emd_g), float(wasserstein_distance(diff_grid, diff_grid, d_open, d_close))

    def calculate_price_elasticity_compression(self, under_shift_pct, main_ou):
        dp = abs(under_shift_pct) / 100.0
        if dp < 1e-4: return 0.0
        return float((abs(main_ou['under_odd'] - 1.90) / 1.90) / dp)

    def calculate_conditional_entropy(self, matrix, main_ah_line):
        h_grid, a_grid = np.indices(matrix.shape)
        ah_mask = (h_grid - a_grid) >= main_ah_line
        p_x1, p_x0 = np.sum(matrix[ah_mask]), 1.0 - np.sum(matrix[ah_mask])
        if p_x1 <= 1e-6 or p_x0 <= 1e-6: return 0.0
        ou_mask = (h_grid + a_grid) > np.median(h_grid + a_grid)
        p_y1_x1 = np.sum(matrix[ah_mask & ou_mask]) / p_x1
        p_y1_x0 = np.sum(matrix[(~ah_mask) & ou_mask]) / p_x0
        hy1 = - (p_y1_x1 * np.log2(max(p_y1_x1, 1e-12)) + (1-p_y1_x1) * np.log2(max(1-p_y1_x1, 1e-12)))
        hy0 = - (p_y1_x0 * np.log2(max(p_y1_x0, 1e-12)) + (1-p_y1_x0) * np.log2(max(1-p_y1_x0, 1e-12)))
        return float(p_x1 * hy1 + p_x0 * hy0)

    def calculate_gini_index(self, matrix):
        flat = np.sort(matrix.flatten())
        n = len(flat)
        return float((2 * np.sum(np.arange(1, n + 1) * flat)) / (n * np.sum(flat)) - (n + 1) / n)

    def calculate_top3_density(self, matrix):
        return float(np.sum(np.sort(matrix.flatten())[::-1][:3]) * 100.0)

    def calculate_modal_skewness(self, lambda_h, mu_a):
        return float((1.0 / np.sqrt(max(lambda_h, 1e-5))) - (1.0 / np.sqrt(max(mu_a, 1e-5))))

    def calculate_mvi(self, lambda_h, mu_a):
        return float(abs(lambda_h - mu_a) / max(lambda_h + mu_a, 1e-5))

    def calculate_kl_divergence(self, mat_open, mat_close):
        p, q = np.clip(mat_open.flatten(), 1e-12, 1.0), np.clip(mat_close.flatten(), 1e-12, 1.0)
        return float(np.sum(p * np.log(p / q)))

    def calculate_jsd(self, mat_open, mat_close):
        p, q = np.clip(mat_open.flatten(), 1e-12, 1.0), np.clip(mat_close.flatten(), 1e-12, 1.0)
        m = 0.5 * (p + q)
        return float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))

    def calculate_market_refractive_index(self, jsd_div, ah_line_open, ah_line_close):
        return float(jsd_div / (abs(ah_line_close - ah_line_open) + 1e-4))

    def calculate_erlang_adjusted_srp(self, jsd_div, ah_line_open, ah_line_close, total_xg):
        dl = abs(ah_line_close - ah_line_open)
        return float((jsd_div / (dl + 1e-4)) * (1.0 + erlang.pdf(dl + 0.1, 2, scale=max(0.1, total_xg / 2.0))))

    def calculate_bivariate_skewness_tensor(self, matrix):
        h_grid, a_grid = np.indices(matrix.shape)
        mh, ma = np.sum(h_grid * matrix), np.sum(a_grid * matrix)
        vh = np.sum(((h_grid - mh) ** 2) * matrix)
        va = np.sum(((a_grid - ma) ** 2) * matrix)
        return float((np.sum(((h_grid - mh) ** 3) * matrix) / (vh ** 1.5 + 1e-6)) - (np.sum(((a_grid - ma) ** 3) * matrix) / (va ** 1.5 + 1e-6)))

    def calculate_analytical_tail_dependence(self, matrix, min_goals=3):
        h_grid, a_grid = np.indices(matrix.shape)
        mask = (h_grid + a_grid) >= min_goals
        if not np.any(mask): return 0.0
        w = matrix[mask] / max(np.sum(matrix[mask]), 1e-12)
        h_vals, a_vals = h_grid[mask], a_grid[mask]
        mh, ma = np.sum(h_vals * w), np.sum(a_vals * w)
        cov = np.sum((h_vals - mh) * (a_vals - ma) * w)
        denom = np.sqrt(np.sum(((h_vals - mh)**2) * w) * np.sum(((a_vals - ma)**2) * w))
        return float(cov / denom) if denom > 1e-6 else 0.0

    def calculate_shin_alpha_variance(self, cs_odds_dict, main_ah_input, main_ou_input, main_x_odd):
        alphas = []
        valid_cs = {k: 1.0 / v for k, v in cs_odds_dict.items() if 1.0 < v <= 100.0}
        if valid_cs: alphas.append(self._get_shin_alpha(valid_cs))
        if main_ah_input and main_ah_input.get('home_odd', 0) > 1.0:
            alphas.append(self._get_shin_alpha({'h': 1.0/main_ah_input['home_odd'], 'a': 1.0/main_ah_input['away_odd']}))
        if main_ou_input and main_ou_input.get('over_odd', 0) > 1.0:
            alphas.append(self._get_shin_alpha({'o': 1.0/main_ou_input['over_odd'], 'u': 1.0/main_ou_input['under_odd']}))
        return float(np.var(alphas)) if len(alphas) >= 2 else 0.0

    def calculate_cross_market_stress_index(self, kl_div, shin_var, x_gap):
        return float(np.clip((0.40 * min(kl_div * 10.0, 1.0)) + (0.35 * min(shin_var * 1000.0, 1.0)) + (0.25 * min(x_gap / 3.0, 1.0)), 0.0, 1.0))

    def calculate_ah_probability(self, matrix, ah_line, is_home=True):
        rem = abs(ah_line) % 0.5
        if abs(rem - 0.25) < 1e-4:
            p_w1, p_p1, _ = self._single_line_ah_outcomes(matrix, ah_line - 0.25, is_home)
            p_w2, p_p2, _ = self._single_line_ah_outcomes(matrix, ah_line + 0.25, is_home)
            return float(np.clip(0.5 * (p_w1 + 0.5 * p_p1 + p_w2 + 0.5 * p_p2), 1e-5, 1.0))
        else:
            p_w, p_p, _ = self._single_line_ah_outcomes(matrix, ah_line, is_home)
            denom = 1.0 - p_p
            return float(p_w / denom) if denom > 0 else 0.5

    def _single_line_ah_outcomes(self, matrix, ah_line, is_home=True):
        h_grid, a_grid = np.indices(matrix.shape)
        diff = ((h_grid - a_grid) if is_home else (a_grid - h_grid)) + ah_line
        return np.sum(matrix[diff > 1e-5]), np.sum(matrix[np.abs(diff) <= 1e-5]), np.sum(matrix[diff < -1e-5])

    def calculate_ou_probability(self, matrix, ou_line, is_over=True):
        rem = abs(ou_line) % 0.5
        if abs(rem - 0.25) < 1e-4:
            p_w1, p_p1, _ = self._single_line_ou_outcomes(matrix, ou_line - 0.25, is_over)
            p_w2, p_p2, _ = self._single_line_ou_outcomes(matrix, ou_line + 0.25, is_over)
            return float(np.clip(0.5 * (p_w1 + 0.5 * p_p1 + p_w2 + 0.5 * p_p2), 1e-5, 1.0))
        else:
            p_w, p_p, _ = self._single_line_ou_outcomes(matrix, ou_line, is_over)
            denom = 1.0 - p_p
            return float(p_w / denom) if denom > 0 else 0.5

    def _single_line_ou_outcomes(self, matrix, ou_line, is_over=True):
        h_grid, a_grid = np.indices(matrix.shape)
        diff = ((h_grid + a_grid) - ou_line) if is_over else (ou_line - (h_grid + a_grid))
        return np.sum(matrix[diff > 1e-5]), np.sum(matrix[np.abs(diff) <= 1e-5]), np.sum(matrix[diff < -1e-5])

    def calculate_draw_probability(self, matrix):
        return float(np.trace(matrix))

    def analyze_mass_shifts(self, cs_open_dict, cs_close_dict, ou_line=2.5, z_mri_x=0.0):
        open_clean = self._shin_unmargin_asymmetric(self._apply_shannon_bayes_noise_filter({k: 1.0/v for k, v in cs_open_dict.items() if 1.0 < v <= 100.0}), z_mri_x)
        close_clean = self._shin_unmargin_asymmetric(self._apply_shannon_bayes_noise_filter({k: 1.0/v for k, v in cs_close_dict.items() if 1.0 < v <= 100.0}), z_mri_x)
        shift_report, th, ta, tu = {}, 0.0, 0.0, 0.0

        for score, p_open in open_clean.items():
            if score in close_clean:
                dp = close_clean[score] - p_open
                shift_report[score] = dp
                h, a = score
                if h > a: th += dp
                elif a > h: ta += dp
                if (h + a) < ou_line: tu += dp

        return {
            'home_shift_pct': round(th * 100, 2),
            'away_shift_pct': round(ta * 100, 2),
            'under_shift_pct': round(tu * 100, 2),
            'top_inflows': [(f"{s[0]}-{s[1]}", round(d * 100, 2)) for s, d in sorted(shift_report.items(), key=lambda x: x[1], reverse=True)[:3]],
            'top_outflows': [(f"{s[0]}-{s[1]}", round(d * 100, 2)) for s, d in sorted(shift_report.items(), key=lambda x: x[1])[:3]]
        }

    def extract_pure_xg(self, cs_odds_dict, main_ah_input=None, main_ou_input=None, main_x_odd=None, mode="Decoupled", copula_type="Fără", copula_theta=1.5, auto_fit_copula=False, use_nbinom=False, z_mri_x=0.0):
        clean_probs = self._shin_unmargin_asymmetric(self._apply_shannon_bayes_noise_filter({k: 1.0 / v for k, v in cs_odds_dict.items() if 1.0 < v <= 100.0}), z_mri_x)

        target_p_ah, target_p_ou, target_p_x = None, None, None
        if main_ah_input and main_ah_input.get('home_odd', 0) > 1.0:
            target_p_ah, _ = self._power_unmargin_binary(main_ah_input['home_odd'], main_ah_input['away_odd'])
        if main_ou_input and main_ou_input.get('over_odd', 0) > 1.0:
            target_p_ou, _ = self._power_unmargin_binary(main_ou_input['over_odd'], main_ou_input['under_odd'])
        if main_x_odd and main_x_odd > 1.0:
            target_p_x = (1.0 / main_x_odd) / 1.05

        def loss_function(params):
            lh, ma, rh, pz = params[:4]
            ct = params[4] if auto_fit_copula and len(params) > 4 else copula_theta
            cp = params[5] if use_nbinom and len(params) > 5 else 0.0
            theo = self._generate_matrix(lh, ma, rh, pz, copula_type, ct, cp)
            st_sub = sum(theo[h, a] for (h, a) in clean_probs.keys() if h <= self.max_goals and a <= self.max_goals)
            if st_sub <= 0: return 1e6
                
            loss = sum(self.weights[h, a] * self._huber_loss(pc, theo[h, a] / st_sub) for (h, a), pc in clean_probs.items() if h <= self.max_goals and a <= self.max_goals)

            if mode == "Anchored (5.0 Weight)":
                if target_p_ou and main_ou_input:
                    loss += 5.0 * ((self.calculate_ou_probability(theo, main_ou_input['line'], is_over=True) - target_p_ou) ** 2)
                if target_p_ah and main_ah_input:
                    loss += 5.0 * ((self.calculate_ah_probability(theo, main_ah_input['home_line'], is_home=True) - target_p_ah) ** 2)
            if target_p_x:
                loss += 2.5 * ((self.calculate_draw_probability(theo) - target_p_x) ** 2)
            return loss

        init_guess, bounds = [1.40, 1.10, 0.0, 0.01], [(0.1, 4.5), (0.1, 4.5), (-0.25, 0.25), (0.0, 0.25)]
        if auto_fit_copula and copula_type != "Fără":
            init_guess.append(copula_theta)
            bounds.append((1.01 if copula_type == "Gumbel" else (0.01 if copula_type == "Clayton" else 0.1), 5.0 if copula_type != "Frank" else 10.0))
        if use_nbinom:
            init_guess.append(0.01)
            bounds.append((0.0, 0.5))
        
        res = minimize(loss_function, init_guess, bounds=bounds, method='L-BFGS-B')
        if not res.success or res.x[0] >= 4.3 or res.x[1] >= 4.3:
            res = minimize(loss_function, init_guess, bounds=bounds, method='Nelder-Mead')

        l_p, m_p, r_p, pi_p = res.x[:4]
        idx = 4
        f_theta = res.x[idx] if (auto_fit_copula and copula_type != "Fără") else copula_theta
        if auto_fit_copula and copula_type != "Fără": idx += 1
        f_phi = res.x[idx] if use_nbinom else 0.0

        mat = self._generate_matrix(l_p, m_p, r_p, pi_p, copula_type, f_theta, f_phi)
        top4 = np.sum(np.sort(mat.flatten())[-4:])
        flat_m = mat.flatten()[mat.flatten() > 0]
        
        return l_p, m_p, r_p, pi_p, top4 / (1.0 - top4 + 1e-6), -np.sum(flat_m * np.log2(flat_m)), self.calculate_gini_index(mat), self.calculate_top3_density(mat), self.calculate_modal_skewness(l_p, m_p), self.calculate_mvi(l_p, m_p), mat, f_theta, f_phi

    def decode_comparative(self, cs_open, cs_close, main_ah_open, main_ah_close, main_ou_open, main_ou_close, main_x_open=None, main_x_close=None, mode="Decoupled", copula_type="Fără", copula_theta=1.5, auto_fit_copula=False, use_nbinom=False):
        l_op_pre, m_op_pre, _, _, _, _, _, _, _, _, mat_op_pre, _, _ = self.extract_pure_xg(cs_open, main_ah_open, main_ou_open, main_x_open, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x=0.0)
        l_cl_pre, m_cl_pre, _, _, _, _, _, _, _, _, mat_cl_pre, _, _ = self.extract_pure_xg(cs_close, main_ah_close, main_ou_close, main_x_close, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x=0.0)

        kl_pre = self.calculate_kl_divergence(mat_op_pre, mat_cl_pre)
        jsd_pre = self.calculate_jsd(mat_op_pre, mat_cl_pre)
        mri_pre = self.calculate_market_refractive_index(jsd_pre, main_ah_open['home_line'], main_ah_close['home_line'])
        fair_x_pre = 1.0 / max(self.calculate_draw_probability(mat_cl_pre), 1e-5)
        x_gap_pre = abs(fair_x_pre - (main_x_close if main_x_close and main_x_close > 1.0 else fair_x_pre))
        z_mri_x = self.calculate_x_stress_z_score(kl_pre, x_gap_pre, mri_pre)

        l_open, m_open, r_open, pi_open, cs_ratio_open, ent_open, gini_open, top3_open, skew_open, mvi_open, mat_open, theta_open, phi_open = self.extract_pure_xg(cs_open, main_ah_open, main_ou_open, main_x_open, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x)
        l_close, m_close, r_close, pi_close, cs_ratio_close, ent_close, gini_close, top3_close, skew_close, mvi_close, mat_close, theta_close, phi_close = self.extract_pure_xg(cs_close, main_ah_close, main_ou_close, main_x_close, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x)

        delta_xg = (l_close + m_close) - (l_open + m_open)
        kl_div = self.calculate_kl_divergence(mat_open, mat_close)
        jsd_div = self.calculate_jsd(mat_open, mat_close)
        mri_index = self.calculate_market_refractive_index(jsd_div, main_ah_open['home_line'], main_ah_close['home_line'])
        ear_srp = self.calculate_erlang_adjusted_srp(jsd_div, main_ah_open['home_line'], main_ah_close['home_line'], l_close + m_close)
        biv_skew = self.calculate_bivariate_skewness_tensor(mat_close)
        tail_corr = self.calculate_analytical_tail_dependence(mat_close)
        shin_var = self.calculate_shin_alpha_variance(cs_close, main_ah_close, main_ou_close, main_x_close)

        svi_score_9x9, matrice_delta_9x9 = self.calculate_svi_9x9(mat_open, mat_close)
        res_heatmap_9x9 = self.calculate_residual_heatmap_9x9(mat_open, mat_close)

        u_x, v_y, fig_quiver = self.calculate_vector_field_flow(mat_open, mat_close)
        lap_map, max_curv, stiffness = self.calculate_surface_laplacian(mat_close)
        is_bimodal, num_peaks, peaks = self.calculate_topological_bimodal_index(mat_close)
        coskew_ha, regime_name, regime_desc = self.calculate_bivariate_moments(mat_close)
        exp_min, p15, p30 = self.calculate_first_passage_time(l_close, m_close)

        moa_index = self.calculate_marginal_overround_asymmetry(main_ou_close)
        emd_goals, emd_diff = self.calculate_wasserstein_emd_1d(mat_open, mat_close)
        mass_analysis = self.analyze_mass_shifts(cs_open, cs_close, ou_line=main_ou_close['line'], z_mri_x=z_mri_x)
        pec_index = self.calculate_price_elasticity_compression(mass_analysis['under_shift_pct'], main_ou_close)
        cond_entropy = self.calculate_conditional_entropy(mat_close, main_ah_close['home_line'])

        # CALCUL MODULUL 6 & MODULUL 7
        squeeze_mat = self.calculate_vig_squeeze_matrix(mat_close, cs_close)
        apf_score = self.calculate_asymmetric_pressure_field(mat_close)
        reverse_eng_results = self.reverse_engineer_bookmaker_state(cs_close, main_ah_close, main_ou_close, main_x_close)

        micro_skew = self.calculate_micro_price_skew(mat_close, main_ou_close)
        obi_s_val = self.calculate_obi_s(mass_analysis, main_ou_close)
        phantom_val = self.calculate_phantom_shift(mat_open, mat_close, main_ou_close)

        fair_ah_h = 1.0 / max(self.calculate_ah_probability(mat_close, main_ah_close['home_line'], is_home=True), 1e-5)
        edge_ah_h = (main_ah_close['home_odd'] / fair_ah_h) - 1.0 if main_ah_close['home_odd'] > 1.0 else 0.0

        fair_ah_a = 1.0 / max(self.calculate_ah_probability(mat_close, -main_ah_close['home_line'], is_home=False), 1e-5)
        edge_ah_a = (main_ah_close['away_odd'] / fair_ah_a) - 1.0 if main_ah_close['away_odd'] > 1.0 else 0.0

        fair_ou_o = 1.0 / max(self.calculate_ou_probability(mat_close, main_ou_close['line'], is_over=True), 1e-5)
        edge_ou_o = (main_ou_close['over_odd'] / fair_ou_o) - 1.0 if main_ou_close['over_odd'] > 1.0 else 0.0

        fair_ou_u = 1.0 / max(self.calculate_ou_probability(mat_close, main_ou_close['line'], is_over=False), 1e-5)
        edge_ou_u = (main_ou_close['under_odd'] / fair_ou_u) - 1.0 if main_ou_close['under_odd'] > 1.0 else 0.0

        fair_x_odd = 1.0 / max(self.calculate_draw_probability(mat_close), 1e-5)
        edge_x_odd = (main_x_close / fair_x_odd) - 1.0 if main_x_close and main_x_close > 1.0 else 0.0

        x_gap = abs(fair_x_odd - (main_x_close if main_x_close and main_x_close > 1.0 else fair_x_odd))
        stress_index = self.calculate_cross_market_stress_index(kl_div, shin_var, x_gap)

        scenario = "Scenariul D: Sharp Re-evaluation (Piață Recalibrată)"
        signal = "✅ PIAȚĂ ECHILIBRATĂ"
        explanation = f"Piața s-a recalibrat natural. xG Total s-a mutat cu {round(delta_xg, 2)} goluri."

        if stress_index > 0.75:
            scenario, signal = "Scenariul S: Cross-Market Anomaly", "🚨 ALERTĂ MAXIMĂ: INEFIENȚĂ STRUCTURALĂ DE PIAȚĂ"
            explanation = f"CMSI a atins {round(stress_index, 2)}. Ruptură masivă între CS, AH și 1X2."
        elif x_gap > 2.50 and main_x_close and main_x_close > 1.0:
            scenario, signal = "Scenariul E: Structural Cement Fracture", "⚠️ CAPCANĂ PE FAVORIT / PRĂPASTIE STRUCTURALĂ"
            explanation = f"Prăpastia dintre Cotă Egal Fair ({round(fair_x_odd, 2)}) și Cotă Egal Piață ({main_x_close}) este uriașă ({round(x_gap, 2)})."
        elif abs(delta_xg) < 0.15 and (abs(edge_ou_o) > 0.035 or abs(edge_ah_h) > 0.035):
            scenario, signal = "Scenariul A: Liquidity Distortion", "🔥 VALOARE PE COTE UMFLATE DE BANI"
            explanation = "Linia s-a mutat din motive de lichiditate, în timp ce xG-ul pur a rămas nemișcat."

        return {
            'open_l': round(l_open, 2), 'open_m': round(m_open, 2), 'open_xg': round(l_open+m_open, 2),
            'close_l': round(l_close, 2), 'close_m': round(m_close, 2), 'close_xg': round(l_close+m_close, 2),
            'delta_l': round(l_close - l_open, 2), 'delta_m': round(m_close - m_open, 2), 'delta_xg': round(delta_xg, 2),
            'cs_ratio_close': round(cs_ratio_close, 2), 'delta_cs': round(cs_ratio_close - cs_ratio_open, 2),
            'ent_close': round(ent_close, 2), 'delta_ent': round(ent_close - ent_open, 2),
            'gini_close': round(gini_close, 3), 'delta_gini': round(gini_close - gini_open, 3),
            'top3_close': round(top3_close, 1), 'delta_top3': round(top3_close - top3_open, 1),
            'skew_close': round(skew_close, 2), 'mvi_close': round(mvi_close, 2),
            'kl_div': round(kl_div, 4), 'jsd_div': round(jsd_div, 4), 'mri_index': round(mri_index, 4),
            'ear_srp': round(ear_srp, 4), 'biv_skew': round(biv_skew, 3),
            'tail_corr': round(tail_corr, 3), 'shin_var': round(shin_var, 5), 'x_gap': round(x_gap, 2),
            'stress_index': round(stress_index, 3), 'fitted_theta': round(theta_close, 2), 'fitted_phi': round(phi_close, 3),
            'moa_index': round(moa_index, 3), 'emd_goals': round(emd_goals, 3), 'emd_diff': round(emd_diff, 3),
            'pec_index': round(pec_index, 3), 'cond_entropy': round(cond_entropy, 3),
            'micro_skew': round(micro_skew, 2), 'obi_s_val': round(obi_s_val, 3), 'phantom_val': round(phantom_val, 4),
            'z_mri_x': round(z_mri_x, 2),
            'svi_score_9x9': round(svi_score_9x9, 4),
            'res_heatmap_9x9': res_heatmap_9x9,
            'matrice_delta_9x9': matrice_delta_9x9,
            'fair_ah_home': round(fair_ah_h, 2), 'edge_ah_home': round(edge_ah_h * 100, 2),
            'fair_ah_away': round(fair_ah_a, 2), 'edge_ah_away': round(edge_ah_a * 100, 2),
            'fair_ou_over': round(fair_ou_o, 2), 'edge_ou_over': round(edge_ou_o * 100, 2),
            'fair_ou_under': round(fair_ou_u, 2), 'edge_ou_under': round(edge_ou_u * 100, 2),
            'fair_x_odd': round(fair_x_odd, 2), 'edge_x_odd': round(edge_x_odd * 100, 2),
            'scenario': scenario, 'signal': signal, 'explanation': explanation, 'matrix_close': mat_close,
            'mass_analysis': mass_analysis,
            'fig_quiver': fig_quiver, 'lap_map': lap_map, 'max_curv': round(max_curv, 4), 'stiffness': round(stiffness, 4),
            'is_bimodal': is_bimodal, 'num_peaks': num_peaks, 'peaks': peaks,
            'coskew_ha': round(coskew_ha, 4), 'regime_name': regime_name, 'regime_desc': regime_desc,
            'exp_min': round(exp_min, 1), 'p15': round(p15, 1), 'p30': round(p30, 1),
            # NOILE DATE EXTRACTE PENTRU MODULUL 6 & 7
            'squeeze_mat': squeeze_mat, 'apf_score': round(apf_score, 6), 'reverse_eng_results': reverse_eng_results
        }


# ==========================================
# STREAMLIT CACHE & INTERFAȚĂ
# ==========================================

@st.cache_data(show_spinner="Calculare matrice și optimizare în curs...")
def run_cached_engine_decode(cs_open_input, cs_close_input, main_ah_open, main_ah_close, main_ou_open, main_ou_close, main_x_open, main_x_close, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom):
    engine = PureMarketEngine9x9()
    return engine.decode_comparative(
        cs_open_input, cs_close_input, main_ah_open, main_ah_close, main_ou_open, main_ou_close, main_x_open, main_x_close, 
        mode=mode, copula_type=copula_type, copula_theta=copula_theta, 
        auto_fit_copula=auto_fit_copula, use_nbinom=use_nbinom
    )

st.title("🕵️ Pure Market Engine 9x9 (Optimized + Copula + Reverse Eng)")

ah_options = [round(x, 2) for x in np.arange(-3.50, 3.75, 0.25)]
ou_options = [round(x, 2) for x in np.arange(1.25, 5.25, 0.25)]

cs_main = {
    (0,0): 12.0, (1,1): 6.8, (2,2): 13.5, (3,3): 50.0,
    (1,0): 9.0, (2,0): 15.0, (2,1): 11.0, (3,0): 35.0, (3,1): 22.0, (3,2): 28.0, (4,0): 80.0, (4,1): 65.0, (4,2): 70.0,
    (0,1): 8.5, (0,2): 10.5, (1,2): 8.35, (0,3): 24.0, (1,3): 15.0, (2,3): 20.0, (0,4): 60.0, (1,4): 50.0, (2,4): 55.0
}
cs_ext_home = {(5,0): 90.0, (5,1): 75.0, (5,2): 85.0, (6,0): 100.0, (6,1): 100.0, (6,2): 100.0}
cs_ext_away = {(0,5): 90.0, (1,5): 75.0, (2,5): 85.0, (0,6): 100.0, (1,6): 100.0, (2,6): 100.0}

st.sidebar.header("⚙️ Setări Motor Optimizare")
engine_mode = st.sidebar.radio("Mod Optimizare Solver:", ["Decoupled (Pure CS Matrix)", "Anchored (5.0 Weight)"])

st.sidebar.header("🔬 Setări Avansate Modelare")
use_nbinom = st.sidebar.checkbox("Activare Negative Binomial (Overdispersion)", value=False)

st.sidebar.header("🌀 Strat Copula (Dependență Non-Liniară)")
copula_type = st.sidebar.selectbox("Selectează Model Copula:", ["Fără", "Frank", "Gumbel", "Clayton"])
auto_fit_copula = st.sidebar.checkbox("Auto-Fit Optim Parametru Θ (Copula)", value=True)
copula_theta = 1.5

if copula_type != "Fără" and not auto_fit_copula:
    min_t, max_t, default_t = (1.01, 5.0, 1.2) if copula_type == "Gumbel" else ((0.1, 5.0, 1.0) if copula_type == "Clayton" else (0.1, 10.0, 1.5))
    copula_theta = st.sidebar.slider(f"Parametru Manual Θ ({copula_type})", min_value=float(min_t), max_value=float(max_t), value=float(default_t), step=0.1)

st.sidebar.header("🎯 1. Matrice Scor Corect (Open vs Close)")
fav_option = st.sidebar.radio("Extensie Favorit Extrem:", ["Fără", "Favorit Gazde (5-0..6-2)", "Favorit Oaspeți (0-5..2-6)"])

cs_open_input, cs_close_input = {}, {}

with st.sidebar.expander("📌 Correct Score OPEN", expanded=False):
    for score, default_odd in cs_main.items():
        cs_open_input[score] = st.number_input(f"Open {score[0]}-{score[1]}", value=default_odd, step=0.25, key=f"op_m_{score}")
    if fav_option == "Favorit Gazde (5-0..6-2)":
        for score, default_odd in cs_ext_home.items():
            cs_open_input[score] = st.number_input(f"Open {score[0]}-{score[1]}", value=default_odd, step=0.5, key=f"op_eh_{score}")
    elif fav_option == "Favorit Oaspeți (0-5..2-6)":
        for score, default_odd in cs_ext_away.items():
            cs_open_input[score] = st.number_input(f"Open {score[0]}-{score[1]}", value=default_odd, step=0.5, key=f"op_ea_{score}")

with st.sidebar.expander("📌 Correct Score CLOSE", expanded=True):
    for score, default_odd in cs_main.items():
        cs_close_input[score] = st.number_input(f"Close {score[0]}-{score[1]}", value=default_odd, step=0.25, key=f"cl_m_{score}")
    if fav_option == "Favorit Gazde (5-0..6-2)":
        for score, default_odd in cs_ext_home.items():
            cs_close_input[score] = st.number_input(f"Close {score[0]}-{score[1]}", value=default_odd, step=0.5, key=f"cl_eh_{score}")
    elif fav_option == "Favorit Oaspeți (0-5..2-6)":
        for score, default_odd in cs_ext_away.items():
            cs_close_input[score] = st.number_input(f"Close {score[0]}-{score[1]}", value=default_odd, step=0.5, key=f"cl_ea_{score}")

st.sidebar.header("⚖️ 2. Anchors Sharp OPEN & CLOSE (AH, O/U & 1X2 X)")
col_a1, col_a2 = st.sidebar.columns(2)

with col_a1:
    st.markdown("#### 🔓 Deschidere (OPEN)")
    home_ah_line_op = st.selectbox("AH Gazde Open", ah_options, index=13, key="ah_line_op")
    home_ah_odd_op = st.number_input("Cotă AH Gazde Open", value=1.95, step=0.01, key="ah_h_op")
    away_ah_odd_op = st.number_input("Cotă AH Oaspeți Open", value=1.95, step=0.01, key="ah_a_op")
    main_ou_line_op = st.selectbox("O/U Open", ou_options, index=5, key="ou_line_op")
    main_ou_odd_op = st.number_input("Cotă Over Open", value=1.90, step=0.01, key="ou_o_op")
    under_ou_odd_op = st.number_input("Cotă Under Open", value=1.90, step=0.01, key="ou_u_op")
    main_x_odd_op = st.number_input("Cotă X Open", value=3.40, step=0.05, key="x_op")

with col_a2:
    st.markdown("#### 🔒 Închidere (CLOSE)")
    home_ah_line_cl = st.selectbox("AH Gazde Close", ah_options, index=13, key="ah_line_cl")
    home_ah_odd_cl = st.number_input("Cotă AH Gazde Close", value=1.95, step=0.01, key="ah_h_cl")
    away_ah_odd_cl = st.number_input("Cotă AH Oaspeți Close", value=1.95, step=0.01, key="ah_a_cl")
    main_ou_line_cl = st.selectbox("O/U Close", ou_options, index=5, key="ou_line_cl")
    main_ou_odd_cl = st.number_input("Cotă Over Close", value=1.90, step=0.01, key="ou_o_cl")
    under_ou_odd_cl = st.number_input("Cotă Under Close", value=1.90, step=0.01, key="ou_u_cl")
    main_x_odd_cl = st.number_input("Cotă X Close", value=3.40, step=0.05, key="x_cl")

main_ah_open = {'home_line': home_ah_line_op, 'home_odd': home_ah_odd_op, 'away_odd': away_ah_odd_op}
main_ah_close = {'home_line': home_ah_line_cl, 'home_odd': home_ah_odd_cl, 'away_odd': away_ah_odd_cl}

main_ou_open = {'line': main_ou_line_op, 'over_odd': main_ou_odd_op, 'under_odd': under_ou_odd_op}
main_ou_close = {'line': main_ou_line_cl, 'over_odd': main_ou_odd_cl, 'under_odd': under_ou_odd_cl}

res = run_cached_engine_decode(
    cs_open_input, cs_close_input, 
    main_ah_open, main_ah_close, 
    main_ou_open, main_ou_close, 
    main_x_odd_op, main_x_odd_cl, 
    engine_mode, copula_type, copula_theta, 
    auto_fit_copula, use_nbinom
)

if res['is_bimodal']:
    st.error(f"⚠️ **Meci Bifurcat (Schizofrenie de Piață)**: Două Scenarii Incompatibile Cotate Simultanal ({res['num_peaks']} vârfuri detectate)!")
    st.caption(f"Vârfuri de probabilitate identificate la scorurile: {[(p[0], p[1]) for p in res['peaks']]}")

st.subheader("1. Metricile xG Pure & Structură (Shin Unmargined)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("xG Pur Gazde (λ)", f"{res['close_l']}", delta=f"{res['delta_l']} vs Open")
c2.metric("xG Pur Oaspeți (μ)", f"{res['close_m']}", delta=f"{res['delta_m']} vs Open")
c3.metric("xG Pur Total", f"{res['close_xg']}", delta=f"{res['delta_xg']} vs Open")
c4.metric("CS Ratio (Concentrare)", f"{res['cs_ratio_close']}", delta=f"{res['delta_cs']} vs Open")
c5.metric("Entropie Shannon (Haos)", f"{res['ent_close']}", delta=f"{res['delta_ent']} vs Open")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Index Gini (Inegalitate)", f"{res['gini_close']}", delta=f"{res['delta_gini']} vs Open")
m2.metric("Densitate Top 3 Scoruri", f"{res['top3_close']}%", delta=f"{res['delta_top3']}% vs Open")
m3.metric("Asimetrie Coadă (Skew)", f"{res['skew_close']}")
m4.metric("Indice MVI (Margin/Var)", f"{res['mvi_close']}")

st.markdown("---")

st.subheader("2. Indici Avansați de Analiză Spectrală, Informațională & Protecție")
s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Divergență KL", f"{res['kl_div']}")
s2.metric("Divergență JSD", f"{res['jsd_div']}")
s3.metric("Indice MRI", f"{res['mri_index']}")
s4.metric("Ear SRP Index", f"{res['ear_srp']}")
s5.metric("Asimetrie Tensor", f"{res['biv_skew']}")
s6.metric("Dependență Cozi", f"{res['tail_corr']}")

s7, s8, s9, s10, s11, s12 = st.columns(6)
s7.metric("Variație Alpha Shin", f"{res['shin_var']}")
s8.metric("ΔX Gap (Prăpastie X)", f"{res['x_gap']}")
s9.metric("🔥 Cross-Market Stress", f"{res['stress_index']}")
s10.metric("Asimetrie Marjă (MOA)", f"{res['moa_index']}")
s11.metric("Entropie Cond. H(Y|X)", f"{res['cond_entropy']}")
s12.metric("Z-Score MRI X (Metoda C)", f"{res['z_mri_x']}")

e1, e2, e3 = st.columns(3)
e1.metric("EMD Wasserstein Goluri (Y)", f"{res['emd_goals']}")
e2.metric("EMD Wasserstein Dif. (X)", f"{res['emd_diff']}")
e3.metric("Compresie Elasticitate (PEC)", f"{res['pec_index']}")

st.markdown("---")

# AFISARE TAB-URI MODULE NOI (1, 2, 4, 5, 6, 7)
st.subheader("🌊 Analiza Dinamică, Toothprints & Reverse Engineering (Modulele 1 - 7)")
tab_mod1, tab_mod2, tab_mod4, tab_mod5, tab_mod6, tab_mod7 = st.tabs([
    "Modulul 1: Vector Field Flow", 
    "Modulul 2: Surface Laplacian", 
    "Modulul 4: Bivariate Co-Skewness", 
    "Modulul 5: First Passage Time",
    "Modulul 6: Market Toothprints & Stress",
    "Modulul 7: Bookmaker State Reverse Calibration"
])

with tab_mod1:
    st.pyplot(res['fig_quiver'])

with tab_mod2:
    mc1, mc2 = st.columns(2)
    mc1.metric("Max Local Curvature (Tensiune Max)", f"{res['max_curv']}")
    mc2.metric("Surface Stiffness Index", f"{res['stiffness']}")
    st.write("**Matricea Laplaciană 9x9 (Zone de Tensiune Concentrată):**")
    st.dataframe(np.round(res['lap_map'], 4), use_container_width=True)

with tab_mod4:
    st.metric("CoSkewness HA Score", f"{res['coskew_ha']:+.4f}")
    if res['coskew_ha'] > 0.10: st.warning(f"**Regim Detectat:** {res['regime_name']}\n\n_{res['regime_desc']}_")
    elif res['coskew_ha'] < -0.10: st.info(f"**Regim Detectat:** {res['regime_name']}\n\n_{res['regime_desc']}_")
    else: st.success(f"**Regim Detectat:** {res['regime_name']}\n\n_{res['regime_desc']}_")

with tab_mod5:
    fpt1, fpt2, fpt3 = st.columns(3)
    fpt1.metric("Minut Așteptat Prim Gol", f"{res['exp_min']}'")
    fpt2.metric("Probabilitate Gol 0-15 Min", f"{res['p15']}%")
    fpt3.metric("Probabilitate Gol 0-30 Min", f"{res['p30']}%")

with tab_mod6:
    st.markdown("### 🔍 Modulul 6: Market Toothprints & Vig Squeeze")
    col_g1, col_g2 = st.columns(2)
    col_g1.metric("Cross-Market Stress Index (CMSI)", f"{res['stress_index']}")
    col_g2.metric("Asymmetric Pressure Field (APF Curvature)", f"{res['apf_score']}")
    
    st.write("**Matrice Heatmap Vig Squeeze (Tăiere Dinamică a Marjei):**")
    st.dataframe(np.round(res['squeeze_mat'], 3), use_container_width=True)
    st.caption("Celulele cu valori > 1.25 reprezintă scorurile unde casa de pariuri a scumpit artificial cota (amprenta defensivă).")

with tab_mod7:
    st.markdown("### 🧪 Modulul 7: Bookmaker State Engine Reverse Calibration")
    rev = res['reverse_eng_results']
    if rev:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Copula Detectată", f"{rev['copula_type']}")
        r2.metric("AIC Score", f"{rev['aic_score']}")
        r3.metric("Copula Theta (Θ)", f"{rev['copula_theta']}")
        r4.metric("Calitate Potrivire", f"{rev['fit_quality']}")
        
        st.json(rev)
    else:
        st.info("Nu s-au putut extrage datele de reverse engineering.")

st.markdown("---")

st.subheader("🌐 Analiză Volatilitate & Anomali Suprafețe 9x9 (Metoda 1 & Metoda 3)")
met1_col, met3_col = st.columns(2)

with met1_col:
    st.markdown("**Metoda 1: Surface Volatility Index (SVI 9x9)**")
    st.metric("SVI Score 9x9 (Vibrație Suprafață)", f"{res['svi_score_9x9']}")

with met3_col:
    st.markdown("**Metoda 3: Residual Heatmap & Detecție Anomalii**")
    res_info = res['res_heatmap_9x9']
    if res_info['este_contaminat']: st.error(res_info['mesaj'])
    else: st.success(res_info['mesaj'])
    st.caption(f"Abatere maximă: **+{res_info['max_devier']*100:.2f}%** la **{res_info['scor_anomalie'][0]}-{res_info['scor_anomalie'][1]}**")

st.markdown("---")

st.subheader("📊 Synthetic Order Book & Depth Radar")
sb1, sb2, sb3 = st.columns(3)
sb1.metric("μ-Price Skew (Cotă Gravitațională)", f"{res['micro_skew']}")
sb2.metric("OBI-S (Synthetic Imbalance)", f"{res['obi_s_val']}")
sb3.metric("Phantom Shift (Spoofing Index)", f"{res['phantom_val']}")

if res['z_mri_x'] > 1.8:
    st.warning(f"⚡ **Recalibrare Asimetrică Activată (Metoda C):** Z_MRI_X a atins **{res['z_mri_x']}** (> 1.8).")

st.markdown("---")

st.subheader("3. Analiza Fluxului Real de Masă (ΔP Shift)")
ma = res['mass_analysis']
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("ΔP Victorie Gazde", f"{ma['home_shift_pct']}%")
col_m2.metric("ΔP Victorie Oaspeți", f"{ma['away_shift_pct']}%")
col_m3.metric(f"ΔP Under {main_ou_close['line']}", f"{ma['under_shift_pct']}%")

st.write(f"**Top Inflows:** {ma['top_inflows']}")
st.write(f"**Top Outflows:** {ma['top_outflows']}")

st.markdown("---")

st.subheader("4. Validare Linii Principale Sharp (AH, O/U & Egal X)")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Asian Handicap Principal**")
    st.metric(f"Fair AH Gazde ({home_ah_line_cl:+.2f})", res['fair_ah_home'], delta=f"Edge: {res['edge_ah_home']}%")
    st.metric(f"Fair AH Oaspeți ({-home_ah_line_cl:+.2f})", res['fair_ah_away'], delta=f"Edge: {res['edge_ah_away']}%")

with col2:
    st.markdown("**Over / Under Principal**")
    st.metric(f"Fair Over ({main_ou_line_cl})", res['fair_ou_over'], delta=f"Edge: {res['edge_ou_over']}%")
    st.metric(f"Fair Under ({main_ou_line_cl})", res['fair_ou_under'], delta=f"Edge: {res['edge_ou_under']}%")

with col3:
    st.markdown("**Ancoră Secundară Egal (X)**")
    st.metric("Fair Cotă Egal (X)", res['fair_x_odd'], delta=f"Edge: {res['edge_x_odd']}%")

st.markdown("---")

st.subheader("5. Decizie Tactică & Scenariu")
st.success(f"**SCENARIU DETECTAT:** {res['scenario']}")
st.info(f"**SEMNAL:** {res['signal']}\n\n**Explicație:** {res['explanation']}")

st.markdown("---")

st.subheader("6. Matricea Pură 9x9 Fără Marjă (%)")
st.dataframe(np.round(res['matrix_close'] * 100, 2), use_container_width=True)