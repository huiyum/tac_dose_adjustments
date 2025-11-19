import torch
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from torch import nn


torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DRUG_INTERACTION_TEXT: str = """
**Drug Interactions**

i. **Aminoglycosides, amphotericin B, and non-steroidal anti-inflammatory drugs** (NSAIDs) may exacerbate the nephrotoxicity of tacrolimus.

ii. **Drugs that inhibit cytochrome P450 3A4/5 (CYP3A4/5) and p-glycoprotein (P-gp)** decrease the metabolism of tacrolimus and/or increase the absorption of tacrolimus. Therefore, they will **increase blood levels** of tacrolimus. These medications include macrolide antibiotics (notably **erythromycin and clarithromycin**), azole antifungal agents (**fluconazole, itraconazole, ketoconazole, posaconazole, and voriconazole**), and some calcium channel blockers (**nicardipine, diltiazem, and verapamil**). Patients receiving these medications may need to have their tacrolimus dose **decreased**.

iii. **Drugs that induce CYP3A4/5 and P-gp** increase the metabolism of tacrolimus and/or decrease the absorption of tacrolimus. Therefore, they will **decrease blood levels** of tacrolimus. These medications include **rifampin, carbamazepine, phenobarbital, phenytoin, and St. John's wort**. Patients receiving these medications will need to have their tacrolimus dose **increased**.

iv. **Grapefruit and pomelo** inhibit CYP3A4 and P-gp and must be **avoided**. For other fruits, very limited information is available. Some studies suggest that the following fruits may also cause fluctuations in drug levels: **papaya, pomegranate, and star fruit**.

v. **Sirolimus** may interfere with the absorption of tacrolimus or increase its clearance, which will **decrease the blood levels** of tacrolimus. This interaction appears more prominent at higher concentrations of sirolimus. Tacrolimus does not affect sirolimus levels.
"""


def preprocess(demo_df, preprocessing_info):

    df = demo_df.copy()

    # === 1. Handle race recoding ===
    if 'race' in df.columns:
        df['race'] = df['race'].where(df['race'].isin(['C', 'AA']), 'others')

    # === 2. Convert object columns to numeric ===
    numeric_cols = preprocessing_info['static_num_features'] + preprocessing_info['time_varying_features']
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # === 3. Fill missing values with median ===
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # === 4. One-hot encode using trained encoder ===
    encoder = preprocessing_info['encoders']['categorical']
    cat_cols = preprocessing_info['static_cat_features']
    encoded_cats = encoder.transform(df[cat_cols])
    
    encoded_feature_names = []
    for i, feature in enumerate(cat_cols):
        categories = encoder.categories_[i][1:]  # drop='first'
        encoded_feature_names.extend([f"{feature}_{cat}" for cat in categories])
    
    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=df.index)
    for col in encoded_feature_names:
        df[col] = encoded_df[col]

    # === 5. Scale numerical features ===
    static_scaler = preprocessing_info['scalers']['static']
    dynamic_scaler = preprocessing_info['scalers']['dynamic']

    static_num_features = preprocessing_info['static_num_features']
    time_varying_features = preprocessing_info['time_varying_features']

    df[static_num_features] = static_scaler.transform(df[static_num_features])
    df[time_varying_features] = dynamic_scaler.transform(df[time_varying_features])

    return df


def prepare_inputs(demo_df, preprocessing_info):

    """
    Prepare model inputs for a single patient using the same logic as PKDataset.
    """

    seq_len = 50

    demo_df = demo_df.sort_values("hours_from_first_dose").copy()

    # Extract static features
    static = demo_df.iloc[0][preprocessing_info['all_static_features']].values.astype(np.float32)
    static = torch.tensor(static).unsqueeze(0).to(device)  # [1, static_dim]

    # Extract sequences
    doses = demo_df["Dose"].values.astype(np.float32)
    times = demo_df["hours_from_first_dose"].values.astype(np.float32)
    concs = demo_df["C_whole"].values.astype(np.float32)

    time_var_features = preprocessing_info['time_varying_features']
    time_var_values = [demo_df[col].values.astype(np.float32) for col in time_var_features]
    time_var_values = np.stack(time_var_values, axis=1)  # [T, D]

    last_idx = len(demo_df)

    # Find previous known concentration
    previous_conc = 0.0
    for j in range(last_idx - 1, -1, -1):
        if not np.isnan(concs[j]):
            previous_conc = concs[j]
            break

    # Use data up to and including that previous known concentration
    seq_end = last_idx
    seq_start = max(0, seq_end - seq_len)
    pad_len = seq_len - (seq_end - seq_start)

    dose_seq = doses[seq_start:seq_end]
    conc_seq = concs[seq_start:seq_end]
    time_var_seq = time_var_values[seq_start:seq_end]
    time_seq = times[seq_start:seq_end]

    dose_seq = np.nan_to_num(dose_seq)
    conc_seq = np.nan_to_num(conc_seq)

    # Pad sequences
    dose_seq = np.pad(dose_seq, (0, pad_len), mode='constant', constant_values=0)
    conc_seq = np.pad(conc_seq, (0, pad_len), mode='constant', constant_values=0)
    time_var_seq = np.pad(time_var_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
    time_seq = np.pad(time_seq, (0, pad_len), mode='constant', constant_values=0)

    # Previous concentration sequence (same value repeated)
    prev_conc_seq = np.full_like(conc_seq, previous_conc)

    # Stack sequence inputs
    dose_seq = torch.tensor(dose_seq).unsqueeze(-1)
    conc_seq = torch.tensor(conc_seq).unsqueeze(-1)
    prev_conc_seq = torch.tensor(prev_conc_seq).unsqueeze(-1)
    time_var_seq = torch.tensor(time_var_seq)

    x_seq = torch.cat([dose_seq, conc_seq, prev_conc_seq, time_var_seq], dim=-1).unsqueeze(0).to(device)  # [1, seq_len, D]
    time_seq = torch.tensor(time_seq).unsqueeze(0).to(device)  # [1, seq_len]
    pad_len = torch.tensor([pad_len], dtype=torch.long).to(device)

    return static, x_seq, dose_seq, time_seq, pad_len


def calculate_c_whole(ke, A, dose_times, dose_values, current_time):
    """
    Calculate C_whole at a given current_time.

    Args:
        ke (float): Elimination rate constant.
        A (float): Scaling factor.
        dose_times (list or np.array): Times at which doses were administered.
        dose_values (list or np.array): Corresponding dose values.
        current_time (float): Time at which to compute C_whole.

    Returns:
        float: Estimated C_whole at current_time.
    """
    ke = float(ke)
    A = float(A)
    dose_times = np.array(dose_times)
    dose_values = np.array(dose_values)

    # Only consider doses given before or at current_time
    mask = dose_times <= current_time
    dose_times = dose_times[mask]
    dose_values = dose_values[mask]
    dose_values = np.nan_to_num(dose_values)

    # Time since each dose
    delta_t = current_time - dose_times

    # Sum contribution from each dose
    concentrations = A * dose_values * np.exp(-ke * delta_t)
    c_whole = np.sum(concentrations)

    return c_whole


def solve_dose_linear(ke, A, dose_times_fixed, dose_values_fixed, added_dose_times, target_c, current_time):
    # Fixed dose contribution
    delta_t_fixed = current_time - np.array(dose_times_fixed)
    c_fixed = A * np.sum(np.array(dose_values_fixed) * np.exp(-ke * delta_t_fixed))

    # Weights for equal added dose
    delta_t_added = current_time - np.array(added_dose_times)
    weights = A * np.exp(-ke * delta_t_added)
    total_weight = np.sum(weights)

    # Solve for x
    x = (target_c - c_fixed) / total_weight
    return x


def display_interaction_checker():

    with st.expander("Potential Drug-Drug Interactions"):

        st.markdown(DRUG_INTERACTION_TEXT)


class PKModel(nn.Module):
    
    def __init__(self, static_dim, seq_input_dim=3, hidden_dim=32):
        
        super().__init__()

        self.up_proj = nn.Linear(seq_input_dim + static_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.d_embeddings = nn.Embedding(2, hidden_dim)
        self.c_embeddings = nn.Embedding(2, hidden_dim)

        self.hidden_to_ke = nn.Linear(hidden_dim, 1)
        self.hidden_to_A = nn.Linear(hidden_dim, 1)

    def forward(self, x_static, x_seq, dose_seq, time_seq, pad_len):

        is_d_missing = (x_seq[:, :, 0] == 0.0).long()

        hidden_states = torch.cat(
            [x_seq, x_static.unsqueeze(1).expand(-1, x_seq.shape[1], -1)], 
            dim=-1
        )

        hidden_states = self.up_proj(hidden_states)

        hidden_states = hidden_states + self.d_embeddings(is_d_missing)

        hidden_states = self.rnn(hidden_states)[0]

        index = hidden_states.size(1) - pad_len - 1

        # Reshape index to [B, 1, 1] to broadcast over H
        index_expanded = index.view(hidden_states.size(0), 1, 1).expand(-1, 1, hidden_states.size(2))  # [B, 1, H]

        # Use gather to select the correct vectors
        last_hidden = torch.gather(hidden_states, dim=1, index=index_expanded)  # [B, 1, H]
        last_hidden = last_hidden.squeeze(1)  # [B, H]

        ke = torch.nn.functional.softplus(self.hidden_to_ke(last_hidden))  # [B, 1]
        ke = ke.clamp(min=0.017, max=0.23)
        
        A = self.hidden_to_A(last_hidden)
        A = torch.nn.functional.softplus(A)

        return ke, A


@st.cache_resource
def load_model_and_preprocessing():

    preprocessing_info = joblib.load("gru/preprocessing_info.pkl")

    # Reconstruct the model
    model = PKModel(
        static_dim=len(preprocessing_info['all_static_features']),
        seq_input_dim=len(preprocessing_info['time_varying_features']) + 3,
        hidden_dim=128,
    )

    model.load_state_dict(torch.load("gru/model.pt", map_location=device))
    model.to(device)

    model.eval()

    return model, preprocessing_info


def main():

    # Streamlit app
    st.title("Tacrolimus Dose Predictor")

    st.markdown("""
    This application predicts the tacrolimus trough concentration and the required dose to achieve a target trough based on patient features.
    Enter the patient details below to get the predicted trough concentration and recommended dose. This tool is intended for research purposes and should not replace clinical judgment.
    """)

    # Load model and preprocessing info
    try:
        model, preprocessing_info = load_model_and_preprocessing()
    except FileNotFoundError:
        st.error("Model or preprocessing files not found. Please ensure 'gru/model.pt' and 'gru/preprocessing_info.pkl' are available.")
        st.stop()
    
    # === Drug Interaction Checker Section (Simplified) ===
    display_interaction_checker()
    st.markdown("---") # Separator line

    # Input form
    st.header("Patient Information")

    st.markdown("### Target Trough Concentration (ng/mL)")
    target_trough = st.number_input("Target Trough", min_value=0.0, max_value=30.0, value=10.0, format="%.1f", step=0.1)
    target_hours = st.number_input("Target Hours", min_value=12, max_value=128, value=24, step=12)

    with st.form("patient_form"):

        st.markdown("### Required Fields")
        
        col1, col2 = st.columns(2)
        
        with col1:

            age = st.number_input("Age (years)*", min_value=18, max_value=120, value=52)
            
            race_options = [
                {"display": "European", "value": "C"},
                {"display": "African American", "value": "AA"},
                {"display": "Others", "value": "Others"}
            ]

            race_selected = st.selectbox(
                "Race*", 
                options=race_options,
                format_func=lambda x: x["display"]
            )

            hct = st.number_input("Hematocrit (hct, %)*", min_value=15.0, max_value=80.0, value=28.0)

            weight = st.number_input("Weight (kg)*", min_value=10.0, max_value=300.0, value=87.0, format="%.1f", step=0.1)
            height = st.number_input("Height (cm)*", min_value=30.0, max_value=250.0, value=172.0, format="%.1f", step=0.1)
        
        with col2:

            prev_dose = st.number_input("Previous Dose (mg)*", min_value=0.5, max_value=50.0, value=6.0, format="%.1f", step=0.1)
            prev_c_whole = st.number_input("Previous Trough Concentration (ng/ml)*", min_value=0.5, max_value=150.0, value=8.0, format="%.1f", step=0.1)
            
            alb = st.number_input("Albumin (alb, g/dL)*", min_value=0.0, value=3.5, format="%.1f", step=0.1)
            sex = st.selectbox("Sex*", options=['M', 'F'])

            hours_after_transplant = st.number_input("Time Post-transplant (hour)*", min_value=12.0, max_value=168.0, value=72.0, format="%.1f", step=0.1)


        st.markdown("### Optional Fields")

        col3, col4 = st.columns(2)
        
        with col3:
            creat = st.number_input("Creatinine (mg/dL)", min_value=0.0, value=1.5, format="%.2f", step=0.01)
            bsa = st.number_input("BSA (m²)", min_value=0.0, value=2.0, format="%.2f", step=0.01)
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, value=30.0, format="%.1f", step=0.1)

        with col4:
            alt = st.number_input("ALT (U/L)", min_value=0.0, value=30.0, format="%.1f", step=0.1)
            ast = st.number_input("AST (U/L)", min_value=0.0, value=30.0, format="%.1f", step=0.1)
        
        submitted = st.form_submit_button("Predict")

    if submitted:

        # Validate required fields
        required_fields_empty = False
        
        if age <= 0:
            st.error("Please enter a valid age (required).")
            required_fields_empty = True
        
        if prev_dose <= 0:
            st.error("Please enter a valid previous dose (required).")
            required_fields_empty = True
        
        if prev_c_whole <= 0:
            st.error("Please enter a valid previous trough concentration (required).")
            required_fields_empty = True
            
        if hours_after_transplant <= 0:
            st.error("Please enter valid hours after transplant (required).")
            required_fields_empty = True
            
        if hct <= 0:
            st.error("Please enter a valid hematocrit value (required).")
            required_fields_empty = True
        
        # If required fields are missing, don't proceed with prediction
        if required_fields_empty:
            st.stop()
        
        input_data = {
            'age': age,
            'WeightKG': weight,
            'HeightCM': height,
            'sex': sex,
            'race': race_selected["value"],
            'prev_Dose': prev_dose,
            'prev_C_whole': prev_c_whole,
            'hours_after_transplant': hours_after_transplant,
            'hct': hct,
            'alb': alb,
            'alt': alt,
            'ast': ast,
            'creat': creat,
            'hours_from_first_dose': hours_after_transplant,  # Approximate as same for simplicity
            'hours_from_last_dose': 12.0
        }
        
        # Print input data for debugging
        print("Input data before preprocessing:")
        print(input_data)


        ######## HERE ########
        num_doses = target_hours // 12
        target_hours = num_doses * 12

        demo_df = pd.DataFrame.from_dict({
            'Dose': [prev_dose, prev_dose, prev_dose, prev_dose, None],
            'C_whole': [None, None, None, None, prev_c_whole],
            'hours_after_transplant': [
               hours_after_transplant - 48, hours_after_transplant - 36,hours_after_transplant - 24, hours_after_transplant - 12, hours_after_transplant
            ],
            'hours_from_first_dose': [0, 12, 24, 36, 48],
            'hours_from_last_dose': [0, 12, 12, 12, 12],
            'sex': [sex] * 5,
            'race': [race_selected["value"]] * 5,
            'age': [age] * 5,
            'HeightCM': [height] * 5,
            'WeightKG': [weight] * 5,
            'BSA': [bsa] * 5,
            'BMI': [bmi] * 5,
            'hct': [hct] * 5,
            'alb': [alb] * 5,
            'creat': [creat] * 5,
            'alt': [alt] * 5,
            'ast': [ast] * 5,
            'hours_from_last_trough': [None] * 5,
        })

        demo_df = preprocess(demo_df, preprocessing_info)

        (x_static, x_seq, dose_seq, time_seq, pad_len) = prepare_inputs(demo_df, preprocessing_info)

        ke, A = model(x_static, x_seq, dose_seq, time_seq, pad_len)

        current_time = 24 + num_doses * 12
        added_dose_times = [24 + max(1, 12 * i) for i in range(num_doses)]

        recommended_dose = solve_dose_linear(
            ke.item(), A.item(),
            dose_times_fixed=np.nan_to_num(demo_df['hours_from_first_dose'].values),
            dose_values_fixed=np.nan_to_num(demo_df['Dose'].values.tolist()),
            added_dose_times=added_dose_times,
            target_c=target_trough,
            current_time=current_time,
        )

        # round recommended dose to 0.5 mg
        recommended_dose = round(recommended_dose / 0.5) * 0.5

        C_whole = calculate_c_whole(
            ke=ke,
            A=A,
            dose_times=(
                demo_df['hours_from_first_dose'].values.tolist() + added_dose_times
            ),
            dose_values=(
                demo_df['Dose'].values.tolist() + [recommended_dose] * num_doses
            ),
            current_time=current_time,
        )

        ########## HERE ##########
        
        # Display results
        st.header("Prediction Results")

        # (1) RENAL WARNING MECHANISM
        if creat > 2.0:

            st.warning(
                f"⚠️ **RENAL WARNING: Elevated Creatinine** ⚠️\n\n"
                f"The patient's serum creatinine is **{creat:.2f} mg/dL**, "
                f"which exceeds the common threshold of 2.0 mg/dL."
            )

        # (2) IMPLEMENT SAFETY ALERT
        if recommended_dose > 1.5 * prev_dose:

            st.error(
                f"⚠️ **DOSE ALERT: Recommended Dose Change Exceeds 50%** ⚠️\n\n"
                f"The **Recommended Dose ({recommended_dose:.2f} mg)** is more than "
                f"1.5 times the **Previous Dose ({prev_dose:.1f} mg)**.\n\n"
                f"Please **carefully review** this significant dose adjustment."
            )

        # Display with red and bold formatting
        st.markdown(f"""
        <div style='color: green; font-weight: bold;'>
        Recommended Tacrolimus Daily Dose for Target Trough {target_trough:.1f} ng/mL: {recommended_dose:.2f} mg
        </div>
        <div style='color: green; font-weight: bold;'>
        Predicted Ke and A: {ke.item():.2f}, {A.item():.2f}, C_whole: {C_whole.item():.2f}
        </div>
        """, unsafe_allow_html=True)
        
        # Display input summary
        with st.expander("View Input Summary"):
            st.write(pd.DataFrame([input_data]).T.rename(columns={0: 'Value'}))

    # Add some information about the model
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This app uses a GRU model trained on data from 1,624 patients to predict tacrolimus trough concentrations within the first week after a kidney transplant.
    The model considers patient demographics, lab results, previous dosing history and trough levels.
    The predicted individual parameters are then used to determine the appropriate dose to achieve a predefined target tacrolimus trough concentration.
    """)


if __name__ == "__main__":

    main()