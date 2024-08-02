import mistlib as mist
import numpy as np
import os
from scipy import special
from scipy import optimize

class Consts:

    def set_Tf(self):
        self.Tf = mat.properties["solidus_eutectic_temperature"].value
    def set_Tf_noneq(self):
        self.Tf_noneq = mat.properties["solidus_eutectic_temperature"].value
    def set_Gamma(self, phases, mat):
        self.Gamma = mat.phase_properties[phases[0]].properties['gibbs_thomson_coeff'].value
    def set_k_0(self, mat, matrix_phase):
        self.k_0 = mat.phase_properties[matrix_phase].properties['partition_coefficient_alpha'].value
    def set_m(self, mat, matrix_phase):
        self.m = mat.phase_properties[matrix_phase].properties['liquidus_slope'].value
    def set_D_l(mat):
        mat.phase_properties['liquid'].properties['solute_diffusivities']
    def set_C_0(self):
        self.C_0 = input
    def set_L(self):
        self.L = 168
        
path_to_example_data = os.path.join(os.path.dirname(__file__), '../../mist/examples/AlCu.json')
mat = mist.core.MaterialInformation(path_to_example_data)
print (Consts.set_D_l(mat))
# --------------------------------------------------------------------------------
# Linear Phase Diagram
# Model taken from:
# ??
# --------------------------------------------------------------------------------
class TestLinearPhaseDiagramPt:

    def linear_phase_diagram_pt(Consts):
            # If compositions are with respect to the minor element, then m is expected to be negative and T_melting is at C=0
            T_ref = Consts.set_Tf
            C_ref = 0
            k = Consts.set_k_0
            m = Consts.set_m
            T_melting = T_ref - m * C_ref # assuming T_ref is the reference temp which we assumed was the eutectic temp
            T_liquidus = T_melting + m * C
            T_solidus = T_melting + (m / k) * C 
            
            print(T_melting)
            
            print (T_liquidus, T_solidus)

# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Hunt Dendrite Arm Spacing
# Model taken from:
# Hunt, Kurtz-Fisher, & Trivedi
# --------------------------------------------------------------------------------
def get_hunt_primary_spacing(G, V, Consts):
    Tf = Consts.set_Tf
    k = Consts.set_k_0
    D_l = Consts.set_D_l
    Gamma = Consts.set_Gamma
    
    return 2.0 * np.sqrt(2.0) * np.sqrt(1.0/G) * (D_l * Gamma * k * Tf / V)**0.25
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Kurz Fisher Dendrite Arm Spacing
# Model taken from:
# Hunt, Kurtz-Fisher, & Trivedi
# --------------------------------------------------------------------------------
def kurz_fisher_primary_spacing(G, V, Consts):
    Tf = Consts.set_Tf
    Tf_noneq = Tf  # Because I still havent separated the solidus and eutectic temp, this is == to Tf
    L = 168
    k = Consts.set_k_0
    D_l = Consts.set_D_l
    Gamma = Consts.set_Gamma
    return 4.3 * np.sqrt(Tf_noneq/G) * (D_l * Gamma /(V*k*Tf))**0.25
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# KGT Model for Planar-Cellular-Dendritic Transitions
#
def KGT3(R,V,G, Consts):
   
    Gamma = Consts.Gamma
    k_0 = Consts.k_0
    m = Consts.m
    D_l = Consts.D_l
    C_0 = Consts.C_0
    
    P = R * V/ (2.*D_l)
    
    A = np.pi**2 * Gamma/(P**2 * D_l**2)
    
    np.seterr(all='raise')
    try:
        Iv = P * np.exp(P) * special.exp1(P)
    except:
        return 1e10
    
    xi_c = 1.0 - 2. * k_0 /( np.sqrt(1+(2.0*np.pi/P)**2)-1.+2.*k_0)
    B = m * C_0 * (1.-k_0) * xi_c / (D_l * (1.-(1.-k_0) * Iv))
    return V**2 * A + V * B + G
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 
def R_V_curve_both(V,G,consts,tol=1e-10,first_R_guess=2e-4):
    R_lower = np.zeros(V.shape)
    R_upper = np.zeros(V.shape)
    
    upper_bound_R = 1e3
    lower_bound_R = 1e-8
    
    for i, Vval in enumerate(V):
        
        # Find the R at the min objective function value
        Rtest = np.logspace(np.log10(lower_bound_R), np.log10(upper_bound_R), 300)
        result = np.zeros(Rtest.shape)

        for j, Rval in enumerate(Rtest):
            result[j] = KGT3(Rval,Vval,G,consts)
    

        min_index = np.nanargmin(result)
        R_at_min = Rtest[min_index]
        
        # Find the lower branch
        min_R = lower_bound_R
        max_R = R_at_min
        
        min_bracket = KGT3(min_R, V[i], G, consts)
        max_bracket = KGT3(max_R, V[i], G, consts)

        root_found = False
        if (np.sign(min_bracket) != np.sign(max_bracket)):
            res = optimize.root_scalar(KGT3, args=(V[i],G, consts), method='bisect', xtol=tol, bracket=[min_R, max_R])
            if (res.converged):
                R_lower[i] = res.root
                root_found = True
                
        if (not root_found):
            R_lower[i] = np.nan
            
        # Find the upper branch
        min_R = R_at_min
        max_R = upper_bound_R
        
        min_bracket = KGT3(min_R, V[i], G, consts)
        max_bracket = KGT3(max_R, V[i], G, consts)                                                                                 
                                                                                                    
        root_found = False
        R_upper_temp = 0.0
        if (np.sign(min_bracket) != np.sign(max_bracket)):
            res = optimize.root_scalar(KGT3, args=(V[i],G, consts), method='bisect', xtol=tol, bracket=[min_R, max_R])
            R_upper_temp = res.root
            if (res.converged):
                R_upper[i] = res.root
                root_found = True
                
        if (not root_found):
            R_upper[i] = np.nan
                        
    return (R_lower, R_upper)


def get_Vmax_Vrmin_both(G,consts,V_log_bounds, V_sample_num, tol=1e-10,first_R_guess=2e-4):
    V = np.logspace(V_log_bounds[0], V_log_bounds[1], V_sample_num)

    R_lower,R_upper = R_V_curve_both(V,G,consts,tol=1e-10,first_R_guess=2e-4)

    max_V_lower, V_at_min_R_lower = get_Vmax_Vrmin_single(R_lower,V)
    max_V_upper, V_at_min_R_upper = get_Vmax_Vrmin_single(R_upper,V)
    
    # Doing the upper branch by hand
    V_at_max_R_upper = np.nan
    min_V_upper = np.nan
    if (not np.isnan(R_upper).all()):
        max_index = np.nanargmax(R_upper)
        V_at_max_R = V[max_index]
        V_at_max_R_upper = V_at_max_R

        min_V = np.inf
        for i, val in enumerate(V):
            if (not np.isnan(R_upper[i])):
                if val < min_V:
                    min_V = val
        min_V_upper= min_V
    
    # Doing the lower branch by hand
    min_V_lower = np.nan
    if (not np.isnan(R_lower).all()):
        min_V = np.inf
        for i, val in enumerate(V):
            if (not np.isnan(R_lower[i])):
                if val < min_V:
                    min_V = val
        min_V_lower = min_V
    
    # Progress print statement
    print(G, max_V_lower, V_at_min_R_lower, V_at_max_R_upper, min_V_upper)
                
    return (max_V_lower, V_at_min_R_lower, max_V_upper, V_at_min_R_upper, min_V_lower, min_V_upper, V_at_max_R_upper)

def get_Vmax_Vrmin_single(R,V):
    if (np.isnan(R).all()):
        return (np.nan, np.nan)
    else:
        min_index = np.nanargmin(R)
        min_R = R[min_index]
        V_at_min_R = V[min_index]

        max_V = -np.inf
        for i, val in enumerate(V):
            if (not np.isnan(R[i])):
                if val > max_V:
                    max_V = val

        return (max_V, V_at_min_R)

material_system = "AlCu"
path_to_example_data = os.path.join(os.path.dirname(__file__), "../../mist/examples/AlCu.JSON")
mat = mist.core.MaterialInformation(path_to_example_data)

print("Data plotted for:", material_system)

# G = 1e7
# V = 5e-3

# V_kf = G * D_l/(Tf*k_0)
# print("The velocity of boundary for 'low' and 'high K-F velocities: ", V_kf)


# lambda_trivedi = trivedi_primary_spacing(G,V,consts) # Low confidence in this functional form
# lambda_kf = kurz_fisher_primary_spacing(G,V,consts)
# lambda_hunt = hunt_primary_spacing(G,V,consts)

# print(lambda_kf, lambda_hunt)