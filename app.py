import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from scipy.special import erf
from scipy.ndimage import gaussian_filter, label

# ==============================================================================
# 1. DYNAMIC IN-MEMORY METALLURGICAL LOGO GENERATOR (ZERO EXTERNAL FILE LOAD)
# ==============================================================================
def create_dynamic_metallurgy_logo():
    """Generates a professional 3D metallic crystal unit cell icon dynamically in memory."""
    img_size = (300, 300)
    img = Image.new("RGBA", img_size, (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    
    # Outer Rounded Frame
    draw.rounded_rectangle([10, 10, 290, 290], radius=40, outline="#3b82f6", width=6)
    
    # 3D Unit Cell Wireframe Coordinates
    nodes = [(80, 80), (220, 80), (220, 220), (80, 220), 
             (120, 50), (260, 50), (260, 190), (120, 190)]
    
    # Edges
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    for e in edges:
        draw.line([nodes[e[0]], nodes[e[1]]], fill="#64748b", width=3)
        
    # (111) Crystallographic Plane Highlight
    draw.polygon([nodes[3], nodes[1], nodes[4]], fill=(225, 29, 72, 120), outline="#f43f5e")
    
    # Atomic Positions
    for x, y in nodes:
        draw.ellipse([x-12, y-12, x+12, y+12], fill="#38bdf8", outline="#ffffff", width=2)
        
    # Interstitial Atom
    draw.ellipse([170-15, 135-15, 170+15, 135+15], fill="#f59e0b", outline="#ffffff", width=3)
    
    return img

dynamic_logo = create_dynamic_metallurgy_logo()

# ==============================================================================
# 2. APPLICATION CONFIGURATION & GLASSMORPHISM STYLING
# ==============================================================================
st.set_page_config(
    page_title="Materials Core Engine | Muhammad Mudassar",
    page_icon=dynamic_logo,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    .stMetric { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background: linear-gradient(90deg, #2563eb, #3b82f6); color: white; }
    .history-card { background: rgba(255, 255, 255, 0.03); padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# Session State Initializations
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "history" not in st.session_state:
    st.session_state["history"] = []

def log_history(activity_type, details):
    st.session_state["history"].append({
        "Type": activity_type,
        "Details": details,
        "Module": st.session_state.get("current_module", "General")
    })

# ==============================================================================
# 3. USER AUTHENTICATION & ACCESS PORTAL
# ==============================================================================
if not st.session_state["authenticated"]:
    st.markdown("<h1 style='text-align: center;'>⚙️ Materials Core Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Engineered by <b>Muhammad Mudassar</b></p>", unsafe_allow_html=True)
    
    col_login1, col_login2, col_login3 = st.columns([1, 1.2, 1])
    with col_login2:
        st.image(dynamic_logo, use_container_width=True)
        st.subheader("🔑 Engineer Portal Access")
        auth_mode = st.radio("Select Action:", ["Login", "Register Account"], horizontal=True)
        user_input = st.text_input("Engineer Username / ID:")
        pass_input = st.text_input("Password:", type="password")
        
        if st.button("Access Dashboard"):
            if user_input.strip() != "":
                st.session_state["authenticated"] = True
                st.session_state["username"] = user_input
                log_history("User Session", f"Logged in as {user_input}")
                st.rerun()
            else:
                st.error("Please enter a valid Username / ID.")
    st.stop()

# ==============================================================================
# 4. SIDEBAR NAVIGATION & HISTORY TRACKER
# ==============================================================================
st.sidebar.image(dynamic_logo, use_container_width=True)
st.sidebar.title("Materials Core Engine")
st.sidebar.caption("Engineered by **Muhammad Mudassar**")
st.sidebar.markdown(f"👤 **Active User**: `{st.session_state['username']}`")

if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

st.sidebar.markdown("---")

CHOICE_M1 = "🔬 Module 1: Automated Microstructure Characterization"
CHOICE_M2 = "🧮 Module 2: Alloy Selection & Kinetics Calculator"
CHOICE_M3 = "🧊 Module 3: Dynamic 3D Crystal Lattice & Planes"

module_choice = st.sidebar.radio(
    "Select Engineering Module:",
    [CHOICE_M1, CHOICE_M2, CHOICE_M3]
)
st.session_state["current_module"] = module_choice

st.sidebar.markdown("---")
with st.sidebar.expander("📜 Search & Calculation History"):
    if len(st.session_state["history"]) == 0:
        st.write("No session records found.")
    else:
        for item in reversed(st.session_state["history"]):
            st.markdown(f"""
            <div class="history-card">
                <small><b>{item['Type']}</b> ({item['Module']})</small><br>
                <span>{item['Details']}</span>
            </div>
            """, unsafe_allow_html=True)

st.title("⚙️ Materials Core Engine")
st.caption("Developed by **Muhammad Mudassar** | Advanced Metallurgy & Materials Design Platform")
st.divider()

# ==============================================================================
# HELPER MATHEMATICAL ALGORITHMS
# ==============================================================================
def numpy_otsu(gray_arr):
    """Pure NumPy implementation of Otsu's Global Thresholding."""
    hist, bin_edges = np.histogram(gray_arr, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2, 1))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    return bin_centers[idx]

def get_miller_plane_mesh(h, k, l):
    """Computes 3D surface mesh vertices for Miller Plane (h k l)."""
    if h == 0 and k == 0 and l == 0:
        return None
    def find_pts(c_val):
        pts = []
        if h != 0:
            for y_e in [0.0, 1.0]:
                for z_e in [0.0, 1.0]:
                    x = (c_val - k * y_e - l * z_e) / h
                    if 0.0 <= x <= 1.0: pts.append((x, y_e, z_e))
        if k != 0:
            for x_e in [0.0, 1.0]:
                for z_e in [0.0, 1.0]:
                    y = (c_val - h * x_e - l * z_e) / k
                    if 0.0 <= y <= 1.0: pts.append((x_e, y, z_e))
        if l != 0:
            for x_e in [0.0, 1.0]:
                for y_e in [0.0, 1.0]:
                    z = (c_val - h * x_e - k * y_e) / l
                    if 0.0 <= z <= 1.0: pts.append((x_e, y_e, z))
        unique = []
        for p in pts:
            p_round = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
            if p_round not in unique: unique.append(p_round)
        return unique

    unique_pts = find_pts(1.0)
    if len(unique_pts) < 3:
        c_alt = (1.0 if h > 0 else 0.0) + (1.0 if k > 0 else 0.0) + (1.0 if l > 0 else 0.0)
        c_alt = c_alt * 0.5 if c_alt > 0 else 0.5
        unique_pts = find_pts(c_alt)
    if len(unique_pts) < 3: return None
        
    pts = np.array(unique_pts)
    centroid = np.mean(pts, axis=0)
    normal = np.array([h, k, l], dtype=float)
    norm_val = np.linalg.norm(normal)
    if norm_val == 0: return None
    normal /= norm_val
    v1 = pts[0] - centroid
    if np.linalg.norm(v1) < 1e-5: v1 = pts[1] - centroid
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    angles = [np.arctan2(np.dot(p - centroid, v2), np.dot(p - centroid, v1)) for p in pts]
    pts = pts[np.argsort(angles)]
    i_list, j_list, k_list = [], [], []
    for idx in range(1, len(pts) - 1):
        i_list.append(0); j_list.append(idx); k_list.append(idx + 1)
    return pts[:, 0], pts[:, 1], pts[:, 2], i_list, j_list, k_list

# ==============================================================================
# MODULE 1: MICROSTRUCTURE ANALYSIS & HEAT TREATMENT
# ==============================================================================
if module_choice == CHOICE_M1:
    st.header("Module 1: Automated Microstructure Characterization & Heat Treatment Advisory")
    st.caption("Perform quantitative grain sizing (ASTM E112), yield strength prediction, and isothermal transformation kinetics.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Microstructure Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        use_sample = st.checkbox("Or use Synthetic Test Microstructure Image", value=True if uploaded_file is None else False)
        microns_per_pixel = st.number_input("Microscope Scale Factor (μm / pixel):", min_value=0.01, value=0.50, step=0.05)
        
    img_rgb = None
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert('RGB')
        img_rgb = np.array(pil_img)
        log_history("Microstructure Analysis", f"Processed upload: {uploaded_file.name}")
    elif use_sample:
        sample_img = np.zeros((400, 400), dtype=np.uint8)
        np.random.seed(42)
        for _ in range(35):
            cx, cy, r = np.random.randint(40, 360), np.random.randint(40, 360), np.random.randint(20, 50)
            y_grid, x_grid = np.ogrid[:400, :400]
            mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= r**2
            sample_img[mask] = 255
        img_rgb = np.stack([sample_img]*3, axis=-1)

    if img_rgb is not None:
        gray = np.dot(img_rgb[...,:3], [0.2989, 0.5870, 0.1140])
        blur = gaussian_filter(gray, sigma=1.5)
        t_val = numpy_otsu(blur)
        binary = blur < t_val
        labeled_array, num_features = label(binary)
        
        with col1:
            st.image(img_rgb, caption="Input Micrograph", use_container_width=True)
        with col2:
            fig_cont = px.imshow(labeled_array, color_continuous_scale="Viridis", title="Grain Boundary Segmentation Heatmap")
            st.plotly_chart(fig_cont, use_container_width=True)
            
        grain_count = max(num_features, 12)
        total_img_area_mm2 = (img_rgb.shape[0] * img_rgb.shape[1] * (microns_per_pixel ** 2)) / 1e6
        mean_area_um = (img_rgb.shape[0] * img_rgb.shape[1] * (microns_per_pixel ** 2)) / grain_count
        mean_ecd_um = 2 * np.sqrt(mean_area_um / np.pi)
        
        N_A = (grain_count / total_img_area_mm2) * 0.01
        astm_G = (np.log(max(N_A, 1.0)) / np.log(2)) + 1.0
        
        sigma_0, k_y = 70.0, 0.74
        d_meters = mean_ecd_um * 1e-6
        predicted_sigma_y = sigma_0 + (k_y / np.sqrt(d_meters)) / 1e3
        
        st.subheader("Quantitative Stereological Results (ASTM E112 Standard)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Grains Counted", f"{grain_count}")
        m2.metric("Mean Grain Diameter", f"{mean_ecd_um:.2f} μm")
        m3.metric("ASTM Grain Size (G)", f"{astm_G:.1f}")
        m4.metric("Predicted Yield Strength", f"{predicted_sigma_y:.1f} MPa")
        
        st.divider()
        st.subheader("Microstructure Diagnosis & Heat Treatment Advisory")
        
        if astm_G < 5.0:
            st.warning("⚠️ **Diagnosis**: Coarse Grain Structure detected ($G < 5$).")
            st.markdown("""
            * **Limitations**: Low impact toughness, elevated ductile-brittle transition temperature.
            * **Remedial Action (Normalizing)**: Heat component to **$880^\circ\text{C} - 920^\circ\text{C}$** and cool in still air.
            * **Target Result**: Refines ASTM grain size to $G > 8$, restoring yield strength and impact energy.
            """)
        else:
            st.success("✅ **Diagnosis**: Fine Grain Structure detected ($G \ge 5$).")
            st.markdown("""
            * **Strengths**: Superior yield strength via Hall-Petch boundary strengthening.
            * **Optimization (Stress Relief Annealing)**: Heat to **$550^\circ\text{C} - 650^\circ\text{C}$** for $2\text{ hours}$.
            """)
            
        st.subheader("JMAK Transformation Kinetics Simulation")
        t = np.linspace(0.1, 100, 200)
        k_param = st.slider("Kinetic Rate Constant (k):", 0.001, 0.1, 0.01, format="%.3f")
        n_param = st.slider("Avrami Growth Exponent (n):", 1.0, 4.0, 2.5, 0.1)
        
        f_t = (1 - np.exp(-k_param * (t ** n_param))) * 100
        fig_jmak = px.line(x=t, y=f_t, labels={'x': 'Time (seconds)', 'y': 'Transformed Phase Fraction (%)'},
                           title="Isothermal Recrystallization / Phase Growth Kinetics")
        fig_jmak.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% Transformed")
        st.plotly_chart(fig_jmak, use_container_width=True)

# ==============================================================================
# MODULE 2: ALLOY DATABASE & CALCULATOR HUB
# ==============================================================================
elif module_choice == CHOICE_M2:
    st.header("Module 2: Engineering Alloy Database & Kinetics Calculator Hub")
    st.caption("Search multi-class alloy profiles, calculate stress-strain plasticity, and model diffusion parameters.")
    
    alloy_dict = {
        'AISI 1010 (Low Carbon Steel)': {
            'type': 'Plain Carbon Steel', 'density': 7.87, 'yield': 200, 'tensile': 360, 'youngs': 200, 'elongation': 30, 'hardness': '105 HB',
            'mechanism': 'Solid solution strengthening by Carbon & Manganese in a ductile Ferrite matrix.',
            'heat_treatment': 'Full Annealing at 890°C - 920°C, air cool for normalization.',
            'apps': 'Automotive body panels, deep-drawn components, structural tubing.'
        },
        'AISI 1045 (Medium Carbon Steel)': {
            'type': 'Medium Carbon Steel', 'density': 7.85, 'yield': 370, 'tensile': 625, 'youngs': 206, 'elongation': 16, 'hardness': '180 HB',
            'mechanism': 'Balanced Ferrite-Pearlite dual phase matrix.',
            'heat_treatment': 'Normalize at 870°C. Water quench from 840°C and temper at 400°C - 600°C.',
            'apps': 'Crankshafts, transmission gears, axles, high-strength bolts.'
        },
        'AISI 1095 (High Carbon Steel)': {
            'type': 'High Carbon Steel', 'density': 7.84, 'yield': 500, 'tensile': 760, 'youngs': 208, 'elongation': 10, 'hardness': '240 HB',
            'mechanism': 'High volume fraction of lamellar Pearlite and cementite networks.',
            'heat_treatment': 'Spheroidize anneal at 720°C. Oil quench from 800°C, temper at 200°C.',
            'apps': 'High-strength springs, industrial blades, wear-resistant plates.'
        },
        'AISI 4140 (Chromoly Alloy Steel)': {
            'type': 'Low-Alloy Steel', 'density': 7.85, 'yield': 750, 'tensile': 1000, 'youngs': 210, 'elongation': 19, 'hardness': '320 HB',
            'mechanism': 'Chromium-Molybdenum carbide precipitation & tempered Martensite.',
            'heat_treatment': 'Oil quench from 845°C. Temper at 540°C - 680°C.',
            'apps': 'Oilfield drill collars, aircraft engine mounts, hydraulic shafts.'
        },
        'AISI 316L (Austenitic Stainless Steel)': {
            'type': 'Stainless Steel', 'density': 8.00, 'yield': 290, 'tensile': 580, 'youngs': 193, 'elongation': 50, 'hardness': '150 HB',
            'mechanism': 'FCC Austenite phase stabilized by Nickel and Molybdenum.',
            'heat_treatment': 'Solution anneal at 1040°C - 1120°C followed by rapid water quenching.',
            'apps': 'Chemical hardware, marine structures, medical implants.'
        },
        'Ti-6Al-4V (Grade 5 Titanium)': {
            'type': 'Titanium Alloy', 'density': 4.43, 'yield': 880, 'tensile': 950, 'youngs': 114, 'elongation': 14, 'hardness': '36 HRC',
            'mechanism': 'Alpha-Beta dual phase strengthening with Al and V additions.',
            'heat_treatment': 'Solution treat at 950°C, water quench, age at 540°C for 4 hours.',
            'apps': 'Aerospace airframes, turbine blades, orthopedic bone implants.'
        },
        'Al 6061-T6 (Precipitation Hardened Aluminum)': {
            'type': 'Aluminum Alloy', 'density': 2.70, 'yield': 276, 'tensile': 310, 'youngs': 68.9, 'elongation': 12, 'hardness': '95 HB',
            'mechanism': 'Precipitation hardening via coherent Mg2Si needles.',
            'heat_treatment': 'Solutionize at 530°C, water quench, artificial age at 160°C for 18 hours.',
            'apps': 'Bicycle frames, aircraft wings, structural frameworks.'
        },
        'Inconel 718 (Nickel Superalloy)': {
            'type': 'Nickel Superalloy', 'density': 8.19, 'yield': 1100, 'tensile': 1375, 'youngs': 205, 'elongation': 15, 'hardness': '40 HRC',
            'mechanism': 'Gamma prime γ" [Ni3Nb] coherent precipitation strengthening.',
            'heat_treatment': 'Solution anneal at 980°C, duplex age at 720°C and 620°C.',
            'apps': 'Rocket propulsion engines, gas turbine disks, nuclear reactors.'
        }
    }
    
    tab_step1, tab_step2, tab_step3 = st.tabs([
        "Step 1: Alloy Search & Deep Technical Profile",
        "Step 2: Stress-Strain & Plasticity Calculator",
        "Step 3: Diffusion & Kinetics Calculator (Arrhenius / Fick)"
    ])
    
    with tab_step1:
        st.subheader("🔍 Search Alloy Database")
        selected_alloy_name = st.selectbox("Select Material Grade:", list(alloy_dict.keys()))
        alloy_info = alloy_dict[selected_alloy_name]
        log_history("Alloy Query", f"Fetched profile for {selected_alloy_name}")
        
        st.markdown(f"### Material Profile: **{selected_alloy_name}** (`{alloy_info['type']}`)")
        
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Yield Stress (σy)", f"{alloy_info['yield']} MPa")
        m2.metric("Tensile Stress (σu)", f"{alloy_info['tensile']} MPa")
        m3.metric("Elongation (%)", f"{alloy_info['elongation']} %")
        m4.metric("Young's Modulus", f"{alloy_info['youngs']} GPa")
        m5.metric("Hardness", f"{alloy_info['hardness']}")
        m6.metric("Density (ρ)", f"{alloy_info['density']} g/cm³")
        
        st.divider()
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.info("🧬 **Microstructure & Strengthening Mechanism**")
            st.write(alloy_info['mechanism'])
        with col_d2:
            st.warning("🔥 **Heat Treatment Procedure**")
            st.write(alloy_info['heat_treatment'])
        with col_d3:
            st.success("⚙️ **Industrial Applications & Uses**")
            st.write(alloy_info['apps'])

    with tab_step2:
        st.subheader("📊 Stress-Strain & Plasticity Calculator")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            eng_stress = st.number_input("Engineering Stress (σ_eng in MPa):", min_value=1.0, value=350.0, step=10.0)
            eng_strain = st.number_input("Engineering Strain (ε_eng as decimal):", min_value=0.001, value=0.150, step=0.01)
            youngs_m = st.number_input("Young's Modulus (E in GPa):", min_value=10.0, value=200.0, step=5.0)
            
            K_strength = st.number_input("Strength Coefficient (K in MPa):", min_value=100.0, value=800.0, step=50.0)
            n_hardening = st.slider("Strain Hardening Exponent (n):", 0.05, 0.50, 0.20, 0.01)

        true_stress = eng_stress * (1 + eng_strain)
        true_strain = np.log(1 + eng_strain)
        resilience_Ur = ((eng_stress ** 2) / (2 * youngs_m * 1000))
        
        if st.button("Save Stress-Strain Calculation"):
            log_history("Stress Calc", f"σ_true={true_stress:.1f} MPa, ε_true={true_strain:.3f}")
            st.success("Logged calculation into history!")

        with col_c2:
            st.success(f"**True Stress ($\sigma_{{true}}$)**: `{true_stress:.2f} MPa`")
            st.success(f"**True Strain ($\epsilon_{{true}}$)**: `{true_strain:.4f}`")
            st.info(f"**Modulus of Resilience ($U_r$)**: `{resilience_Ur:.4f} MJ/m³`")
            
            eps_range = np.linspace(0.001, 0.40, 200)
            sigma_hollomon = K_strength * (eps_range ** n_hardening)
            
            fig_ss = px.line(x=eps_range, y=sigma_hollomon, labels={'x': 'True Strain (ε)', 'y': 'True Stress (MPa)'},
                             title=f"Hollomon Power Law: σ = {K_strength} × ε^({n_hardening})")
            fig_ss.add_scatter(x=[true_strain], y=[true_stress], mode='markers+text',
                               marker=dict(size=12, color='red'), text=["  Operating Point"], textposition="top right")
            st.plotly_chart(fig_ss, use_container_width=True)

    with tab_step3:
        st.subheader("🧪 Diffusion Coefficient & Fick's Law Kinetics Solver")
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("### 1. Arrhenius Diffusion Solver")
            D0 = st.number_input("Frequency Factor (D₀ in m²/s):", min_value=1e-7, value=2.3e-5, format="%.2e")
            Q_activation = st.number_input("Activation Energy (Q in J/mol):", min_value=10000.0, value=148000.0, step=5000.0)
            temp_C = st.number_input("Temperature (°C):", min_value=100.0, value=927.0, step=10.0)
            
            R_gas = 8.314
            temp_K = temp_C + 273.15
            D_coeff = D0 * np.exp(-Q_activation / (R_gas * temp_K))
            st.success(f"**Calculated Diffusion Coefficient (D)**: `{D_coeff:.4e} m²/s`")

        with col_k2:
            st.markdown("### 2. Fick's 2nd Law Concentration Solver")
            C_surface = st.number_input("Surface Carbon Conc. (C_s wt%):", min_value=0.1, value=1.20, step=0.05)
            C_initial = st.number_input("Initial Carbon Conc. (C₀ wt%):", min_value=0.01, value=0.20, step=0.05)
            carburizing_hours = st.number_input("Carburizing Time (t in Hours):", min_value=0.1, value=4.0, step=0.5)
            
            t_seconds = carburizing_hours * 3600
            depth_x_mm = np.linspace(0.01, 3.0, 100)
            depth_x_m = depth_x_mm * 1e-3
            
            conc_profile = C_surface - (C_surface - C_initial) * erf(depth_x_m / (2 * np.sqrt(D_coeff * t_seconds)))
            
            fig_diff = px.line(x=depth_x_mm, y=conc_profile, labels={'x': 'Depth below surface (mm)', 'y': 'Carbon Conc. (wt%)'},
                               title=f"Diffusion Profile after {carburizing_hours} Hours at {temp_C}°C")
            st.plotly_chart(fig_diff, use_container_width=True)

# ==============================================================================
# MODULE 3: DYNAMIC 3D CRYSTAL LATTICE & PLANES ENGINE
# ==============================================================================
elif module_choice == CHOICE_M3:
    st.header("Module 3: Dynamic 3D Crystal Lattice, Planes & Directions Engine")
    st.caption("Interactive 3D rendering of unit cell sites, Miller Planes (h k l), and direction vectors [u v w].")
    
    col_config1, col_config2 = st.columns([1, 1])
    with col_config1:
        lattice_type = st.selectbox("Select Crystal Structure:", ["FCC (Face-Centered Cubic)", "BCC (Body-Centered Cubic)", "HCP (Hexagonal Close-Packed)"])
        dislocation_shift = st.slider("Apply Shear Strain (Dislocation Glide):", 0.0, 0.5, 0.0, 0.05)

    st.subheader("📐 Miller Indices Controls")
    col_p, col_d = st.columns(2)
    with col_p:
        st.markdown("**Miller Plane Indices $(h\,k\,l)$**")
        show_plane_custom = st.checkbox("Show Custom Miller Plane", value=True)
        cp1, cp2, cp3 = st.columns(3)
        h_val = cp1.number_input("h index", min_value=-3, max_value=3, value=1, step=1)
        k_val = cp2.number_input("k index", min_value=-3, max_value=3, value=1, step=1)
        l_val = cp3.number_input("l index", min_value=-3, max_value=3, value=1, step=1)

    with col_d:
        st.markdown("**Miller Direction Indices $[u\,v\,w]$**")
        show_dir_custom = st.checkbox("Show Custom Direction Vector", value=True)
        cd1, cd2, cd3 = st.columns(3)
        u_val = cd1.number_input("u index", min_value=-3, max_value=3, value=1, step=1)
        v_val = cd2.number_input("v index", min_value=-3, max_value=3, value=1, step=1)
        w_val = cd3.number_input("w index", min_value=-3, max_value=3, value=1, step=1)

    atoms = []
    if lattice_type.startswith("FCC"):
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    atoms.append([x + (z * dislocation_shift if z == 1 else 0), y, z, 'Corner Site'])
        faces = [[0.5, 0.5, 0], [0.5, 0.5, 1], [0.5, 0, 0.5], [0.5, 1, 0.5], [0, 0.5, 0.5], [1, 0.5, 0.5]]
        for f in faces:
            atoms.append([f[0] + (f[2] * dislocation_shift if f[2] == 1 else 0), f[1], f[2], 'Face-Center Site'])
            
    elif lattice_type.startswith("BCC"):
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    atoms.append([x + (z * dislocation_shift if z == 1 else 0), y, z, 'Corner Site'])
        atoms.append([0.5 + 0.5 * dislocation_shift, 0.5, 0.5, 'Body-Center Site'])
        
    else:
        for z in [0, 1]:
            atoms.append([0, 0, z, 'Hex-Center Site'])
            for angle in np.linspace(0, 2*np.pi, 7)[:-1]:
                atoms.append([np.cos(angle), np.sin(angle), z, 'Hex-Corner Site'])
        atoms.append([0.33, 0.58, 0.5, 'Mid-Layer Site'])
        atoms.append([-0.33, 0.58, 0.5, 'Mid-Layer Site'])
        atoms.append([0, -0.58, 0.5, 'Mid-Layer Site'])

    atoms_df = pd.DataFrame(atoms, columns=['X', 'Y', 'Z', 'Site'])
    
    fig_3d = px.scatter_3d(
        atoms_df, x='X', y='Y', z='Z', color='Site', size_max=30,
        title=f"3D Lattice Visualizer: {lattice_type}", opacity=0.9
    )
    fig_3d.update_traces(marker=dict(size=18, line=dict(width=2, color='DarkSlateGrey')))
    
    if not lattice_type.startswith("HCP"):
        for x in [0, 1]:
            for y in [0, 1]:
                fig_3d.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[0, 1], mode='lines', line=dict(color='black', width=3), showlegend=False))
                fig_3d.add_trace(go.Scatter3d(x=[x, x], y=[0, 1], z=[y, y], mode='lines', line=dict(color='black', width=3), showlegend=False))
                fig_3d.add_trace(go.Scatter3d(x=[0, 1], y=[x, x], z=[y, y], mode='lines', line=dict(color='black', width=3), showlegend=False))
                
    if show_plane_custom and not (h_val == 0 and k_val == 0 and l_val == 0):
        plane_res = get_miller_plane_mesh(h_val, k_val, l_val)
        if plane_res is not None:
            px_m, py_m, pz_m, i_m, j_m, k_m = plane_res
            fig_3d.add_trace(go.Mesh3d(
                x=px_m, y=py_m, z=pz_m,
                i=i_m, j=j_m, k=k_m,
                color='crimson', opacity=0.5,
                name=f'Plane ({h_val} {k_val} {l_val})',
                showlegend=True
            ))

    if show_dir_custom and not (u_val == 0 and v_val == 0 and w_val == 0):
        max_c = max(abs(u_val), abs(v_val), abs(w_val))
        u_s, v_s, w_s = u_val / max_c, v_val / max_c, w_val / max_c
        
        fig_3d.add_trace(go.Scatter3d(
            x=[0, u_s], y=[0, v_s], z=[0, w_s],
            mode='lines+markers+text',
            line=dict(color='gold', width=8),
            marker=dict(size=8, color='darkorange', symbol='diamond'),
            text=["(0,0,0)", f"  [{u_val} {v_val} {w_val}]"],
            textposition="top right",
            name=f'Direction [{u_val} {v_val} {w_val}]',
            showlegend=True
        ))

    fig_3d.update_layout(scene=dict(aspectmode='data'), height=650)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("""
    **Crystallographic Mechanics Breakdown**:
    * **Miller Plane Notation $(h\,k\,l)$**: Represents reciprocal intercepts of a plane with unit cell axes ($x, y, z$).
    * **Miller Direction Notation $[u\,v\,w]$**: Vector direction starting from origin $(0,0,0)$ through coordinates $(u, v, w)$.
    * **FCC**: Primary close-packed slip plane is $\{111\}$ along $\langle110\rangle$ directions.
    * **BCC**: Primary slip plane is $\{110\}$ along $\langle111\rangle$ directions.
    """)