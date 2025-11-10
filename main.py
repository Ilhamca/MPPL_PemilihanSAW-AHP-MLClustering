import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Sistem Pemilihan Laptop Mahasiswa",
    page_icon="💻",
    layout="wide"
)

# Title and description
st.title("🎓 Sistem Pendukung Keputusan Pemilihan Laptop Mahasiswa")
st.markdown("### Metode Hybrid: AHP + SAW dengan Machine Learning (K-Means)")

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Menu Navigasi",
    ["Home", "Upload Data Laptop","Clustering (K-Means)", "Pembobotan Kriteria (AHP)", 
     "Perankingan (SAW)", "Hasil Rekomendasi"]
)

# Global Var
laptop_features = ['Harga', 'Prosesor_Score', 'RAM', 'Storage', 'GPU_Score', 'Baterai', 'Bobot']
weights = None


# Initialize session state
if 'laptops_data' not in st.session_state:
    st.session_state.laptops_data = pd.DataFrame()
if 'ahp_weights' not in st.session_state:
    st.session_state.ahp_weights = {}
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'saw_results' not in st.session_state:
    st.session_state.saw_results = None
# category holds important variables used by KMeans clustering so they are globally accessible
if 'category' not in st.session_state:
    st.session_state.category = {
        'numeric_cols': laptop_features.copy(),
        'n_clusters': 3,
        'cluster_names': [],
        'available_cols': []
    }

# ============= HOME PAGE =============
if menu == "Home":
    st.header("Selamat Datang! 👋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Tentang Sistem")
        st.write("""
        Sistem ini membantu mahasiswa dalam memilih laptop yang sesuai dengan kebutuhan mereka
        menggunakan metode ilmiah dan machine learning.
        
        **Metode yang Digunakan:**
        - **K-Means Clustering**: Pengelompokan laptop berdasarkan karakteristik
        - **AHP (Analytical Hierarchy Process)**: Pembobotan kriteria
        - **SAW (Simple Additive Weighting)**: Perankingan alternatif
        """)
    
    with col2:
        st.subheader("🎯 Kriteria Pemilihan")
        st.write("""
        Kriteria yang dipertimbangkan:
        - 💰 **Harga**: Budget mahasiswa
        - ⚡ **Prosesor**: Performa CPU
        - 🧠 **RAM**: Memori sistem
        - 💾 **Storage**: Kapasitas penyimpanan
        - 🎨 **GPU**: Kemampuan grafis
        - 🔋 **Baterai**: Daya tahan
        - ⚖️ **Bobot**: Portabilitas
        """)
    
    st.info("👈 Gunakan menu sidebar untuk mulai menggunakan sistem")

# ============= INPUT DATA LAPTOP =============
elif menu == "Upload Data Laptop":
    st.subheader("Upload Data CSV")
    st.write("Format CSV: Company,TypeName,Inches,ScreenResolution,Cpu,Ram,Memory,Gpu,OpSys,Weight,Price")
    st.write("Contoh format data:")
    st.table(pd.DataFrame({
        "Company": ["Company A", "Company B"],
        "TypeName": ["Gaming", "Ultrabook"],
        "Inches": [15.6, 13.3],
        "ScreenResolution": ["IPS 1920x1080", "FULL HD 2560x1600"],
        "Cpu": ["Intel Core i5 2.3GHz", "AMD A9-Series A9-9420 3GHz"],
        "Ram": ["8GB", "16GB"],
        "Memory": ["512 HDD", "1024 SSD"],
        "Gpu": ["Nvidia GeForce GTX 1050 Ti", "AMD Radeon RX 580"],
        "OpSys": ["Windows 10", "Linux"],
        "Weight": ["1.5 kg", "2.0 kg"],
        "Price (Euro)": ["69210.72", "213.12"]
    }))
    
    # More general information on data upload
    st.write("""
    Catatan:
    - Semua RAM Asumsi dalam generasi yang sama
    """)

    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            st.session_state.laptops_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Berhasil memuat {len(st.session_state.laptops_data)} data laptop!")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display current data
    if not st.session_state.laptops_data.empty:
        st.subheader("📊 Data Laptop Saat Ini")
        st.dataframe(st.session_state.laptops_data, use_container_width=True)
        
        if st.button("🗑️ Hapus Semua Data"):
            st.session_state.laptops_data = pd.DataFrame()
            st.rerun()

# ============= K-MEANS CLUSTERING =============
elif menu == "Clustering (K-Means)":
    st.header("🔍 K-Means Clustering")
    
    if st.session_state.laptops_data.empty:
        st.warning("⚠️ Belum ada data laptop. Silakan input data terlebih dahulu.")
    else:
        pd_original = st.session_state.laptops_data.copy()  # keep original data
        pd = st.session_state.laptops_data.copy()
        pd = pd.loc[:, ~pd.columns.str.startswith("Unnamed")]  # drop unnamed columns if any
        
        # initialize top-level keys so widgets are controlled; default EMPTY on first open
        if 'selected_companies' not in st.session_state:
            st.session_state['selected_companies'] = []
        if 'selected_types' not in st.session_state:
            st.session_state['selected_types'] = []

        # callbacks to sync nested category without trying to write the widget key
        def _sync_selected_companies():
            st.session_state.category['selected_companies'] = st.session_state.get('selected_companies', [])

        def _sync_selected_types():
            st.session_state.category['selected_types'] = st.session_state.get('selected_types', [])

        # Do NOT assign the multiselect return directly into st.session_state (avoids modifying widget key)
        st.multiselect(
            "Pilih Company untuk Clustering",
            options=pd['Company'].unique().tolist(),
            key='selected_companies',
            on_change=_sync_selected_companies
        )

        st.multiselect(
            "Pilih TypeName untuk Clustering",
            options=pd['TypeName'].unique().tolist(),
            key='selected_types',
            on_change=_sync_selected_types
        )

        # Select All and Clear Filter buttons placed side-by-side
        col1, col2 = st.columns(2, gap="small", vertical_alignment="bottom", width=260)

        # use on_click callbacks (safe to mutate widget keys here)
        def _select_all():
            st.session_state['selected_companies'] = pd_original['Company'].unique().tolist()
            st.session_state['selected_types'] = pd_original['TypeName'].unique().tolist()
            st.session_state.category['selected_companies'] = st.session_state['selected_companies']
            st.session_state.category['selected_types'] = st.session_state['selected_types']

        def _clear_all():
            st.session_state['selected_companies'] = []
            st.session_state['selected_types'] = []
            st.session_state.category['selected_companies'] = []
            st.session_state.category['selected_types'] = []

        col1.button("Select All", key="select_all", on_click=_select_all)
        col2.button("Clear All Filter", key="clear_filter", on_click=_clear_all)

        # Quick lil info
        pd.info()
        pd.dtypes

        st.divider()
        
        # Remove all empty data
        pd = pd.dropna()

        # Apply filters only for columns that exist to avoid empty/no-op filtering
        if 'Company' in pd.columns and 'TypeName' in pd.columns:
            pd = pd[
                pd['Company'].isin(st.session_state.category.get('selected_companies', [])) &
                pd['TypeName'].isin(st.session_state.category.get('selected_types', []))
            ]
        elif 'Company' in pd.columns:
            pd = pd[pd['Company'].isin(st.session_state.category.get('selected_companies', []))]
        elif 'TypeName' in pd.columns:
            pd = pd[pd['TypeName'].isin(st.session_state.category.get('selected_types', []))]
        
        # Convert columns into possible float64
        # Ram column
        # TODO: Separate DDR4 and DDR5 if needed
        pd['Ram'] = pd['Ram'].str.replace('GB', '').astype(float)
        pd.rename(columns={'Ram': 'Ram (GB)'}, inplace=True)
        
        # Weight column (Only takes number)
        pd['Weight'] = pd['Weight'].str.replace('kg', '').astype(float)
        pd.rename(columns={'Weight': 'Weight (kg)'}, inplace=True)

        # Memory column
        pd['Memory'] = pd['Memory'].str.replace('GB', '').astype(float)
        pd.rename(columns={'Memory': 'Memory (GB)'}, inplace=True)

        # Show converted Data
        if pd.empty:
            st.warning("Tidak ada data yang cocok dengan filter yang diterapkan.")
        else:
            st.subheader("Filtered Data Laptop")
            st.dataframe(pd, width='stretch')
            pd.info()
            pd.dtypes

        # Select features for clustering (use global category in session_state)
        numeric_cols = st.session_state.category.get('numeric_cols', ['Harga', 'Prosesor_Score', 'RAM', 'Storage', 'GPU_Score', 'Baterai', 'Bobot'])
        available_cols = [col for col in numeric_cols if col in pd.columns]
        # persist available cols
        st.session_state.category['available_cols'] = available_cols

        if len(available_cols) < 3:
            st.error("Data tidak lengkap untuk clustering. Pastikan semua kolom tersedia.")
        else:
            n_clusters = st.slider("Jumlah Cluster", 2, 5, st.session_state.category.get('n_clusters', 3))
            # persist selected n_clusters
            st.session_state.category['n_clusters'] = n_clusters

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Kategori Cluster")
                cluster_names = []
                for i in range(n_clusters):
                    default_name = st.session_state.category.get('cluster_names', [f"Kategori {j+1}" for j in range(n_clusters)])[i] if st.session_state.category.get('cluster_names') and len(st.session_state.category.get('cluster_names'))>=n_clusters else f"Kategori {i+1}"
                    name = st.text_input(f"Nama Cluster {i}", default_name, key=f"cluster_{i}")
                    cluster_names.append(name)
                # persist cluster names
                st.session_state.category['cluster_names'] = cluster_names

            if st.button("Jalankan Clustering"):
                # Prepare data
                X = pd[available_cols].fillna(0)

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Apply K-Means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                # Add cluster labels to local df and persist back to session_state
                pd['Cluster'] = clusters
                pd['Kategori'] = [st.session_state.category['cluster_names'][c] for c in clusters]
                st.session_state.laptops_data = pd
                st.session_state.clusters = kmeans
                # persist category state
                st.session_state.category['n_clusters'] = n_clusters
                st.session_state.category['available_cols'] = available_cols

                st.success(f"✅ Clustering berhasil! Laptop dikelompokkan ke dalam {n_clusters} kategori.")

            with col2:
                if 'Cluster' in pd.columns:
                    st.subheader("Visualisasi Cluster")

                    # 3D scatter plot
                    fig = px.scatter_3d(
                        pd,
                        x='Harga', y='RAM', z='Prosesor_Score',
                        color='Kategori',
                        hover_data=['Nama'],
                        title='Visualisasi Clustering Laptop'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Display clustered data
            if 'Cluster' in pd.columns:
                st.subheader("📊 Hasil Clustering")
                st.dataframe(pd, use_container_width=True)

                # Cluster statistics
                st.subheader("📈 Statistik per Cluster")
                for i in range(n_clusters):
                    with st.expander(f"{cluster_names[i]} - {len(pd[pd['Cluster']==i])} laptop"):
                        cluster_data = pd[pd['Cluster']==i]
                        st.write(cluster_data[['Nama', 'Harga', 'RAM', 'Storage']].describe())

# ============= AHP WEIGHTING =============
elif menu == "Pembobotan Kriteria (AHP)":
    st.header("⚖️ Analytical Hierarchy Process (AHP)")
    
    st.write("""
    Tentukan tingkat kepentingan relatif antar kriteria menggunakan skala perbandingan berpasangan.
    
    **Skala AHP:**
    - 1: Sama penting
    - 3: Sedikit lebih penting
    - 5: Lebih penting
    - 7: Sangat lebih penting
    - 9: Mutlak lebih penting
    """)
    
    criteria = ['Harga', 'Prosesor', 'RAM', 'Storage', 'GPU', 'Baterai', 'Bobot']
    n = len(criteria)
    st.write(f"Jumlah Kriteria: {n}")
    # Create pairwise comparison matrix
    st.subheader("Matriks Perbandingan Berpasangan")
    
    comparison_matrix = np.ones((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**{criteria[i]}**")
            with col2:
                value = st.slider(f"{i}_{j}", 1, 9, 1, key=f"ahp_{i}_{j}", 
                                label_visibility="collapsed")
            with col3:
                st.write(f"**{criteria[j]}**")
            
            comparison_matrix[i][j] = value
            comparison_matrix[j][i] = 1/value
    
    # Calculate weights using eigenvector method
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues.real)
    principal_eigenvector = eigenvectors[:, max_eigenvalue_index].real
    
    # Consistency Index and Ratio calculation
    # Consistency check (CI and CR)
    # Random Index (RI) values for matrix sizes 1..10 (Saaty's table)
    RI = {1:0.0, 2:0.0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}

    # Largest eigenvalue (real part)
    lambda_max = eigenvalues.real[max_eigenvalue_index]
    if n > 1:
        CI = (lambda_max - n) / (n - 1)
    else:
        CI = 0.0

    RI_n = RI.get(n, None)
    if RI_n is None or RI_n == 0:
        CR = 0.0
    else:
        CR = CI / RI_n

    st.write(f"• Lambda max: {lambda_max:.4f}")
    st.write(f"• Consistency Index (CI): {CI:.4f}")
    st.write(f"• Consistency Ratio (CR): {CR:.4f}")

    if RI_n is None:
        st.warning("RI belum tersedia untuk ukuran matriks ini; tidak dapat mengevaluasi CR secara akurat.")
    elif CR > 0.1:
        st.error("⚠️ Konsistensi perbandingan berpasangan rendah (CR > 0.1). Pertimbangkan untuk meninjau nilai perbandingan.")
        # Set st.button("Hitung Bobot AHP") to disabled state
        st.session_state.ahp_calculated = False
    else:
        st.success("✅ Konsistensi perbandingan berpasangan OK (CR <= 0.1).")
        if st.button("Hitung Bobot AHP"):
            # Normalize to get weights
            weights = principal_eigenvector / principal_eigenvector.sum()
            
            # Store weights
            st.session_state.ahp_weights = {criteria[i]: weights[i] for i in range(n)}
            
            # Display results
            st.subheader("📊 Hasil Pembobotan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                weights_df = pd.DataFrame({
                    'Kriteria': criteria,
                    'Bobot': [weights[i] for i in range(n)],
                    'Persentase': [f"{weights[i]*100:.2f}%" for i in range(n)]
                })
                st.dataframe(weights_df, use_container_width=True)
            
            with col2:
                fig = px.pie(weights_df, values='Bobot', names='Kriteria', 
                            title='Distribusi Bobot Kriteria')
                st.plotly_chart(fig, use_container_width=True)

# ============= SAW RANKING =============
elif menu == "Perankingan (SAW)":
    st.header("📊 Simple Additive Weighting (SAW)")
    
    if st.session_state.laptops_data.empty:
        st.warning("⚠️ Belum ada data laptop. Silakan input data terlebih dahulu.")
    elif not st.session_state.ahp_weights:
        st.warning("⚠️ Belum ada bobot kriteria. Silakan lakukan pembobotan AHP terlebih dahulu.")
    else:
        st.write("Melakukan perankingan laptop berdasarkan bobot kriteria dari AHP.")
        
        # User preferences
        st.subheader("🎯 Preferensi Pengguna")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_budget = st.number_input("Budget Maksimal (Juta)", 
                                        min_value=0.0, 
                                        value=float(st.session_state.laptops_data['Harga'].max()) 
                                        if 'Harga' in st.session_state.laptops_data.columns else 20.0)
        
        with col2:
            min_ram = st.number_input("RAM Minimal (GB)", min_value=4, value=8, step=4)
        
        with col3:
            if 'Kategori' in st.session_state.laptops_data.columns:
                selected_category = st.selectbox("Kategori Laptop", 
                                                 ["Semua"] + list(st.session_state.laptops_data['Kategori'].unique()))
            else:
                selected_category = "Semua"
        
        if st.button("Hitung Peringkat SAW"):
            # Filter data based on preferences
            filtered_data = st.session_state.laptops_data.copy()
            
            if 'Harga' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Harga'] <= max_budget]
            if 'RAM' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['RAM'] >= min_ram]
            if selected_category != "Semua" and 'Kategori' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Kategori'] == selected_category]
            
            if len(filtered_data) == 0:
                st.error("❌ Tidak ada laptop yang memenuhi kriteria.")
            else:
                # Normalize criteria
                criteria_mapping = {
                    'Harga': 'Harga',
                    'Prosesor': 'Prosesor_Score',
                    'RAM': 'RAM',
                    'Storage': 'Storage',
                    'GPU': 'GPU_Score',
                    'Baterai': 'Baterai',
                    'Bobot': 'Bobot'
                }
                
                normalized = filtered_data.copy()
                saw_score = np.zeros(len(filtered_data))
                
                for criteria_name, weight in st.session_state.ahp_weights.items():
                    if criteria_name in criteria_mapping:
                        col_name = criteria_mapping[criteria_name]
                        
                        if col_name in filtered_data.columns:
                            values = filtered_data[col_name].values
                            
                            # Cost criteria (lower is better): Harga, Bobot
                            if criteria_name in ['Harga', 'Bobot']:
                                norm_values = values.min() / (values + 0.0001)
                            # Benefit criteria (higher is better)
                            else:
                                norm_values = values / (values.max() + 0.0001)
                            
                            saw_score += weight * norm_values
                
                filtered_data['SAW_Score'] = saw_score
                filtered_data = filtered_data.sort_values('SAW_Score', ascending=False).reset_index(drop=True)
                filtered_data['Peringkat'] = range(1, len(filtered_data) + 1)
                
                st.session_state.saw_results = filtered_data
                
                st.success(f"✅ Berhasil! Ditemukan {len(filtered_data)} laptop yang sesuai.")
        
        # Display results
        if st.session_state.saw_results is not None:
            st.subheader("🏆 Hasil Perankingan")
            
            display_cols = ['Peringkat', 'Nama', 'Harga', 'RAM', 'Storage', 'SAW_Score']
            if 'Kategori' in st.session_state.saw_results.columns:
                display_cols.insert(2, 'Kategori')
            
            available_display_cols = [col for col in display_cols if col in st.session_state.saw_results.columns]
            st.dataframe(st.session_state.saw_results[available_display_cols], use_container_width=True)
            
            # Visualization
            fig = px.bar(st.session_state.saw_results.head(10), 
                        x='Nama', y='SAW_Score',
                        title='Top 10 Laptop Berdasarkan Skor SAW',
                        labels={'SAW_Score': 'Skor SAW', 'Nama': 'Nama Laptop'},
                        color='SAW_Score',
                        color_continuous_scale='Viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# ============= RECOMMENDATIONS =============
elif menu == "Hasil Rekomendasi":
    st.header("🎯 Rekomendasi Laptop")
    
    if st.session_state.saw_results is None or st.session_state.saw_results.empty:
        st.warning("⚠️ Belum ada hasil perankingan. Silakan lakukan perankingan SAW terlebih dahulu.")
    else:
        st.subheader("🏆 Top 5 Rekomendasi Laptop")
        
        top_5 = st.session_state.saw_results.head(5)
        
        for idx, row in top_5.iterrows():
            with st.expander(f"#{int(row['Peringkat'])} - {row['Nama']} (Skor: {row['SAW_Score']:.4f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Spesifikasi:**")
                    if 'Harga' in row:
                        st.write(f"💰 Harga: Rp {row['Harga']:.1f} Juta")
                    if 'Prosesor' in row:
                        st.write(f"⚡ Prosesor: {row['Prosesor']}")
                    if 'RAM' in row:
                        st.write(f"🧠 RAM: {row['RAM']} GB")
                    if 'Storage' in row:
                        st.write(f"💾 Storage: {row['Storage']} GB")
                
                with col2:
                    st.write("**Detail Tambahan:**")
                    if 'GPU' in row:
                        st.write(f"🎨 GPU: {row['GPU']}")
                    if 'Baterai' in row:
                        st.write(f"🔋 Baterai: {row['Baterai']} Wh")
                    if 'Bobot' in row:
                        st.write(f"⚖️ Bobot: {row['Bobot']} kg")
                    if 'Kategori' in row:
                        st.write(f"📁 Kategori: {row['Kategori']}")
        
        # Comparison chart
        st.subheader("📊 Perbandingan Visual")
        
        if len(top_5) > 0:
            criteria_cols = ['RAM', 'Storage', 'Prosesor_Score', 'GPU_Score']
            available_criteria = [col for col in criteria_cols if col in top_5.columns]
            
            if available_criteria:
                fig = go.Figure()
                
                for idx, row in top_5.iterrows():
                    values = [row[col] for col in available_criteria if col in row]
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=available_criteria,
                        fill='toself',
                        name=row['Nama']
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True,
                    title="Perbandingan Spesifikasi Top 5 Laptop"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.subheader("📥 Ekspor Hasil")
        csv = st.session_state.saw_results.to_csv(index=False)
        st.download_button(
            label="Download Hasil Rekomendasi (CSV)",
            data=csv,
            file_name="rekomendasi_laptop.csv",
            mime="text/csv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Sistem Pemilihan Laptop Mahasiswa**  
Metode: AHP + SAW + K-Means  
Developed with Streamlit 🚀
""")
