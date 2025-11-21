import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import app.data_utils as data_utils
from app.clustering_utils import get_available_numeric_cols, run_kmeans
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Sistem Pemilihan Laptop Mahasiswa",
    page_icon="💻",
    layout="wide"
)

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Menu Navigasi",
    ["Upload dan Filter", "Upload Data Laptop + Filtering","Clustering (K-Means)", "Pembobotan Kriteria (AHP)", 
     "Perankingan (SAW)", "Hasil Rekomendasi"]
)

# Global Var
laptop_features = ['Inches', 'Ram (GB)', 'Memory (GB)', 'Prosesor_Score', 'GPU_Score', 'Memory_Value', 'berat (KG)']
berat = None
uploaded_file = None
filtered_company_data = None
filtered_typename_data = None
filtered_os_data = None
df_filtered_result = pd.DataFrame()

# Initialize session state
if 'laptops_data' not in st.session_state:
    st.session_state.laptops_data = pd.DataFrame()
if 'ahp_berat' not in st.session_state:
    st.session_state.ahp_berat = {}
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

contoh = pd.DataFrame(columns=["ID", "Nama", "Merek", "Tipe Laptop", "Sistem Operasi", "Ukuran (Inches)", "Resolusi Layar", "CPU", "RAM", "Memory", "GPU", "Berat (KG)", "Harga"])
st.session_state.laptop_categorized_data = contoh

# Title and description
st.html("<h1 style='color:white;text-align:center;'>🎓 Sistem Pendukung Keputusan Pemilihan Laptop Mahasiswa</h1><h3 style='text-align:center;'>Pencarian Efektif Menggunakan AHP + SAW dengan Machine Learning (K-Means)</h3><hr>")

# ============= HOME PAGE =============
if menu == "Upload dan Filter":
    if st.session_state.laptops_data.empty:
        uploaded_file = st.file_uploader(label="", type=['csv'], help="Unggah file CSV berisi data laptop dengan format yang sesuai.", key="home_file_uploader")
    else:
        st.button("Hapus Data Laptop", type="primary", on_click=lambda: st.session_state.update({'laptops_data': pd.DataFrame()}))
    if uploaded_file is not None:
        try:
            st.session_state.laptops_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Berhasil memuat {len(st.session_state.laptops_data)} data laptop!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")   
    
    if st.session_state.laptops_data.empty:
        st.html("<p style='text-align:center;'>Upload file CSV untuk mulai</p>")
    else:
        # Choose mapping mode
        mode = st.radio("Mode Pemetaan Kolom", ["Otomatis", "Manual"], index=0, horizontal=True)

        if mode == "Otomatis":
            st.markdown("Otomatisasi pemetaan kolom berdasarkan nama kolom umum. Direkomendasikan jika kolom sudah sesuai format.")
        else:
            st.markdown("Pilih kolom yang akan digunakan untuk analisis lebih lanjut.")
            st.html("<h4>Data untuk diukur (Minimal 3)</h4>")

            column_options = ["None"] + st.session_state.laptops_data.columns.tolist()

            # Define manual selection fields (label -> session key)
            manual_fields = {
            "Pilih kolom Ukuran (Inches)": "inchesColSelection",
            "Pilih kolom Resolusi Layar": "screenResolutionColSelection",
            "Pilih kolom CPU": "cpuColSelection",
            "Pilih kolom GPU": "gpuColSelection",
            "Pilih kolom Memory": "memoryColSelection",
            "Pilih kolom RAM": "ramColSelection",
            "Pilih kolom berat": "beratColSelection",
            "Pilih kolom harga": "hargaColSelection",
            "Pilih kolom ID": "idColSelection",
            "Pilih kolom Nama": "nameColSelection",
            "Pilih kolom Company": "companyColSelection",
            "Pilih kolom Laptop Type": "laptopTypeColSelection",
            "Pilih kolom Operating System": "operatingSystemColSelection"
            }

            # Initialize shadow keys and helper to get saved index
            for key in manual_fields.values():
                shadow = f"shadow_{key}"
                if shadow not in st.session_state:
                    st.session_state[shadow] = "None"

            def saved_index(key):
                shadow = f"shadow_{key}"
                saved = st.session_state.get(shadow, "None")
                try:
                    return column_options.index(saved)
                except ValueError:
                    return 0

            # Render selectboxes in rows of up to 4 per row
            items = list(manual_fields.items())
            for i in range(0, len(items), 4):
                cols = st.columns(min(4, len(items) - i))
                for col, (label, skey) in zip(cols, items[i:i+4]):
                    with col:
                        choice = st.selectbox(label, options=column_options, key=skey, index=saved_index(skey))
                        st.session_state[f"shadow_{skey}"] = choice

            st.info("Apabila ID tidak dipilih, ID akan otomatis diinputkan.", icon="ℹ️")

        # Copy original data for filtering and UI
        df_original = st.session_state.laptops_data.copy()

        # Initialize selection lists in session_state
        for k in ('selected_companies', 'selected_types', 'selected_operating_systems'):
            if k not in st.session_state:
                st.session_state[k] = []

        def _sync_selected(key, state_key):
            st.session_state.category[state_key] = st.session_state.get(key, [])

        # Filtering UI (guard for missing columns)
        st.write('## Filter Data Laptop')
        if 'Company' in df_original.columns:
            st.multiselect("Pilih Company untuk Filtering", options=df_original['Company'].unique().tolist(), key='selected_companies', on_change=_sync_selected, args=('selected_companies', 'selected_companies'))
        if 'TypeName' in df_original.columns:
            st.multiselect("Pilih Tipe Laptop untuk Filtering", options=df_original['TypeName'].unique().tolist(), key='selected_types', on_change=_sync_selected, args=('selected_types', 'selected_types'))
        if 'OpSys' in df_original.columns:
            st.multiselect("Pilih Operating System untuk Filtering", options=df_original['OpSys'].unique().tolist(), key='selected_operating_systems', on_change=_sync_selected, args=('selected_operating_systems', 'selected_operating_systems'))

        col1, col2, col3 = st.columns(3, gap="small")

        if filtered_company_data is not None and 'Company' in df_original.columns:
            with col1:
                st.write(f"**Data setelah filter Company ({len(filtered_company_data)} baris)**")
                st.dataframe(filtered_company_data, use_container_width=True)

        if filtered_typename_data is not None and 'TypeName' in df_original.columns:
            with col2:
                st.write(f"**Data setelah filter Tipe Laptop ({len(filtered_typename_data)} baris)**")
                st.dataframe(filtered_typename_data, use_container_width=True)

        if filtered_os_data is not None and 'OpSys' in df_original.columns:
            with col3:
                st.write(f"**Data setelah filter Operating System ({len(filtered_os_data)} baris)**")
                st.dataframe(filtered_os_data, use_container_width=True)

        def _select_all():
            if 'Company' in df_original.columns:
                st.session_state['selected_companies'] = df_original['Company'].unique().tolist()
            if 'TypeName' in df_original.columns:
                st.session_state['selected_types'] = df_original['TypeName'].unique().tolist()
            if 'OpSys' in df_original.columns:
                st.session_state['selected_operating_systems'] = df_original['OpSys'].unique().tolist()

        def _clear_all():
            st.session_state['selected_companies'] = []
            st.session_state['selected_types'] = []
            st.session_state['selected_operating_systems'] = []

        st.container()
        cb1, cb2 = st.columns(2, gap="small", width=260)
        cb1.button("Select All", on_click=_select_all)
        cb2.button("Clear All Filter", on_click=_clear_all)

        # Effective selections (empty => all)
        sel_companies = st.session_state.get('selected_companies') or (df_original['Company'].unique().tolist() if 'Company' in df_original.columns else [])
        sel_types = st.session_state.get('selected_types') or (df_original['TypeName'].unique().tolist() if 'TypeName' in df_original.columns else [])
        sel_ops = st.session_state.get('selected_operating_systems') or (df_original['OpSys'].unique().tolist() if 'OpSys' in df_original.columns else [])

        # Apply filters
        df_filtered = df_original.copy()
        if 'Company' in df_filtered.columns and sel_companies:
            df_filtered = df_filtered[df_filtered['Company'].isin(sel_companies)]
        if 'TypeName' in df_filtered.columns and sel_types:
            df_filtered = df_filtered[df_filtered['TypeName'].isin(sel_types)]
        if 'OpSys' in df_filtered.columns and sel_ops:
            df_filtered = df_filtered[df_filtered['OpSys'].isin(sel_ops)]

        st.divider()

        if st.button("Filter Data Laptop"):
            if mode == "Otomatis":
                df_filtered_result = data_utils.automaticColumnTable(df_filtered)
            else:
                # Collect manual mapping choices (use shadows to avoid missing keys)
                manual_mapping = {}
                for label, key in manual_fields.items():
                    manual_mapping[key] = st.session_state.get(f"shadow_{key}", "None")
                df_filtered_result = data_utils.manualColumnTable(df_filtered, mapping=manual_mapping)

        st.markdown("<h4>Data Laptop Saat Ini</h4>", unsafe_allow_html=True)

        if df_filtered_result is None or df_filtered_result.empty:
            st.warning("Tidak ada data yang cocok dengan filter yang diterapkan.")
        else:
            st.dataframe(df_filtered_result, use_container_width=True)

    

    


# ============= INPUT DATA LAPTOP =============
elif menu == "Upload Data Laptop + Filtering":
    st.subheader("Upload Data CSV")
    st.write("Format CSV: Company,TypeName,Inches,ScreenResolution,Cpu,Ram,Memory,Gpu,OpSys,Weight,harga")
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
        "harga (Euro)": ["69210.72", "213.12"]
    }))
    
    # More general information on data upload
    st.write("""
    Catatan:
    - Semua RAM Asumsi dalam generasi yang sama
    """)
    
    # Display current data
    if not st.session_state.laptops_data.empty:
        st.subheader("📊 Data Laptop Saat Ini")
        
        if st.button("🗑️ Hapus Semua Data"):
            st.session_state.laptops_data = pd.DataFrame()
            st.rerun()
            
    st.markdown("---")

    if st.session_state.laptops_data.empty:
        st.warning("⚠️ Belum ada data laptop. Silakan input data terlebih dahulu.")
    else:
        df_original = st.session_state.laptops_data.copy()

        # Multiselect UI (controlled via top-level session_state keys)
        if 'selected_companies' not in st.session_state:
            st.session_state['selected_companies'] = []
        if 'selected_types' not in st.session_state:
            st.session_state['selected_types'] = []

        def _sync_selected_companies():
            st.session_state.category['selected_companies'] = st.session_state.get('selected_companies', [])

        def _sync_selected_types():
            st.session_state.category['selected_types'] = st.session_state.get('selected_types', [])

        # Filtering UI
        st.write('## Filter Data Laptop')
        st.multiselect("Pilih Company untuk Filtering", options=df_original['Company'].unique().tolist(), key='selected_companies', on_change=_sync_selected_companies)
        st.multiselect("Pilih TypeName untuk Filtering", options=df_original['TypeName'].unique().tolist(), key='selected_types', on_change=_sync_selected_types)

        col1, col2 = st.columns(2, gap="small")

        if filtered_company_data is not None:
            with col1:
                st.write(f"**Data setelah filter Company ({len(filtered_company_data)} baris)**")
                st.dataframe(filtered_company_data, use_container_width=True)

        if filtered_typename_data is not None:
            with col2:
                st.write(f"**Data setelah filter TypeName ({len(filtered_typename_data)} baris)**")
                st.dataframe(filtered_typename_data, use_container_width=True)

        def _select_all():
            st.session_state['selected_companies'] = df_original['Company'].unique().tolist()
            st.session_state['selected_types'] = df_original['TypeName'].unique().tolist()

        def _clear_all():
            st.session_state['selected_companies'] = []
            st.session_state['selected_types'] = []

        st.container()
        col1, col2 = st.columns(2, gap="small", width=260)
        col1.button("Select All", on_click=_select_all)
        col2.button("Clear All Filter", on_click=_clear_all)

        # determine effective selections (empty means all)
        sel_companies = st.session_state.get('selected_companies', [])
        sel_types = st.session_state.get('selected_types', [])
        if not sel_companies:
            sel_companies = df_original['Company'].unique().tolist()
        if not sel_types:
            sel_types = df_original['TypeName'].unique().tolist()

        # apply filters to a working copy
        df_filtered = df_original.copy()
        if 'Company' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Company'].isin(sel_companies)]
        if 'TypeName' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['TypeName'].isin(sel_types)]

        # clean and cache data
        if 'cleaned_df' not in st.session_state:
            st.session_state.cleaned_df = None

        
        if st.button("🧹 Bersihkan Data"):
            filtered_company_data = df_filtered[df_filtered['Company'].isin(sel_companies)]
            filtered_typename_data = df_filtered[df_filtered['TypeName'].isin(sel_types)]
            df = data_utils.clean_laptops_df(df_filtered)
            st.session_state.cleaned_df = df

        st.divider()
        if df_filtered.empty:
            try:
                if st.session_state.cleaned_df.empty:
                    st.warning("Tidak ada data yang cocok dengan filter yang diterapkan.")
                else:
                    st.subheader("Filtered Data Laptop")
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Tekan tombol ""'Bersihkan Data'"" untuk memulai pembersihan")
        else:
            st.subheader("Filtered Data Laptop")
            st.dataframe(st.session_state.cleaned_df, use_container_width=True)

# ============= K-MEANS CLUSTERING =============
elif menu == "Clustering (K-Means)":
    try:
        st.header("🔍 K-Means Clustering")
        st.dataframe(st.session_state.cleaned_df, use_container_width=True)
        
        if st.session_state.cleaned_df.empty:
            st.warning("⚠️ Belum ada data laptop atau data belum di filter. Silakan input data dan filter terlebih dahulu.")
        else:
            df = st.session_state.cleaned_df.copy()
            numeric_cols = st.session_state.category.get('numeric_cols', laptop_features)
            available_cols = get_available_numeric_cols(df, numeric_cols)
            st.session_state.category['available_cols'] = available_cols
            
            st.write(f"Kolom numerik tersedia untuk clustering: {', '.join(available_cols)}")
            st.write(df.info())
            st.write(df.describe())
            st.write(df.dtypes)

            if len(available_cols) < 3:
                st.error("Data tidak lengkap untuk clustering. Pastikan semua kolom tersedia.")
            else:
                st.info("💡 **K-Means Clustering** akan mengelompokkan laptop secara otomatis berdasarkan karakteristik yang relevan dengan kebutuhan mahasiswa.")
                
                # Predefined categories for student usage
                st.subheader("🎯 Kategori Penggunaan Mahasiswa")
                
                usage_profiles = {
                    "Budget Friendly": {
                        "description": "Laptop terjangkau untuk penggunaan umum (browsing, office, streaming)",
                        "icon": "💰",
                        "features": ["harga", "Ram (GB)", "Memory (GB)"]
                    },
                    "Programming & Development": {
                        "description": "Laptop untuk coding, compile, dan development software",
                        "icon": "💻",
                        "features": ["Prosesor_Score", "Ram (GB)", "Memory (GB)"]
                    },
                    "Desain Grafis & Multimedia": {
                        "description": "Laptop untuk editing foto/video, 3D modeling, rendering",
                        "icon": "🎨",
                        "features": ["GPU_Score", "Prosesor_Score", "Ram (GB)", "Memory (GB)"]
                    },
                    "Gaming & High Performance": {
                        "description": "Laptop untuk gaming, machine learning, dan aplikasi berat",
                        "icon": "🎮",
                        "features": ["GPU_Score", "Prosesor_Score", "Ram (GB)"]
                    },
                    "Portabilitas & Mobilitas": {
                        "description": "Laptop ringan dan portable untuk mobilitas tinggi",
                        "icon": "🎒",
                        "features": ["Weight (KG)", "Ram (GB)", "Memory (GB)"]
                    }
                }
                
                # Number of clusters
                n_clusters = st.slider(
                    "Jumlah Kategori Laptop", 
                    2, 5, 
                    st.session_state.category.get('n_clusters', 3),
                    help="Pilih berapa kategori laptop yang ingin dihasilkan dari clustering"
                )
                st.session_state.category['n_clusters'] = n_clusters
                
                # Display category profiles
                cols = st.columns(min(3, n_clusters))
                for idx, (profile_name, profile_data) in enumerate(list(usage_profiles.items())[:n_clusters]):
                    with cols[idx % 3]:
                        st.markdown(f"""
                        **{profile_data['icon']} {profile_name}**  
                        {profile_data['description']}
                        """)
                
                with st.expander("⚙️ Konfigurasi Clustering", expanded=False):
                    st.write("**Kriteria yang Digunakan untuk Clustering:**")
                    
                    # Allow users to select which features to prioritize
                    selected_features = st.multiselect(
                        "Pilih fitur untuk clustering",
                        options=available_cols,
                        default=available_cols,
                        help="Fitur yang dipilih akan digunakan untuk mengelompokkan laptop"
                    )
                    
                    if not selected_features:
                        st.warning("⚠️ Pilih minimal satu fitur untuk clustering!")
                        selected_features = available_cols
                    
                    # Custom naming
                    st.write("**Penamaan Kategori:**")
                    cluster_names = []
                    default_names = list(usage_profiles.keys())[:n_clusters]
                    
                    for i in range(n_clusters):
                        default_name = default_names[i] if i < len(default_names) else f"Kategori {i+1}"
                        if st.session_state.category.get('cluster_names') and len(st.session_state.category.get('cluster_names')) >= n_clusters:
                            default_name = st.session_state.category.get('cluster_names')[i]
                        
                        name = st.text_input(
                            f"Nama Kategori {i+1}", 
                            default_name, 
                            key=f"cluster_{i}",
                            help=f"Beri nama untuk kategori laptop yang akan dihasilkan"
                        )
                        cluster_names.append(name)
                    
                    st.session_state.category['cluster_names'] = cluster_names

                if st.button("🚀 Jalankan Analisis Clustering", type="primary"):
                    with st.spinner("Menganalisis data dan mengelompokkan laptop..."):
                        # Run K-Means clustering
                        kmeans, clusters, X_scaled = run_kmeans(df, selected_features, n_clusters)
                        
                        # Add cluster results to dataframe
                        df['Cluster'] = clusters
                        df['Kategori'] = [st.session_state.category['cluster_names'][c] for c in clusters]
                        
                        # Pastikan kolom 'No' tetap ada
                        if 'No' not in df.columns:
                            df.insert(0, 'No', range(1, len(df) + 1))
                        
                        # Calculate cluster centers for interpretation
                        cluster_centers = kmeans.cluster_centers_
                        
                        # Store results
                        st.session_state.laptops_data = df
                        st.session_state.clusters = kmeans
                        st.session_state.cluster_centers = cluster_centers
                        st.session_state.selected_features = selected_features
                        
                        st.success(f"✅ Clustering berhasil! Laptop telah dikelompokkan ke dalam {n_clusters} kategori berdasarkan kebutuhan mahasiswa.")

                # Display results if clustering has been performed
                if 'Cluster' in df.columns:
                    st.divider()
                    
                    # Cluster distribution
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("📊 Distribusi Kategori")
                        cluster_counts = df['Kategori'].value_counts()
                        fig_pie = px.pie(
                            values=cluster_counts.values,
                            names=cluster_counts.index,
                            title="Proporsi Laptop per Kategori"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.subheader("📈 Jumlah Laptop per Kategori")
                        fig_bar = px.bar(
                            x=cluster_counts.index,
                            y=cluster_counts.values,
                            labels={'x': 'Kategori', 'y': 'Jumlah Laptop'},
                            title="Distribusi Laptop",
                            color=cluster_counts.index
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # 3D Visualization
                    st.subheader("🎯 Visualisasi Clustering 3D")
                    
                    # Select axes for visualization
                    viz_col1, viz_col2, viz_col3 = st.columns(3)
                    with viz_col1:
                        x_axis = st.selectbox("Sumbu X", options=available_cols, index=0 if 'harga' not in available_cols else available_cols.index('harga'))
                    with viz_col2:
                        y_axis = st.selectbox("Sumbu Y", options=available_cols, index=1 if len(available_cols) > 1 else 0)
                    with viz_col3:
                        z_axis = st.selectbox("Sumbu Z", options=available_cols, index=2 if len(available_cols) > 2 else 0)
                    
                    fig_3d = px.scatter_3d(
                        df,
                        x=x_axis, 
                        y=y_axis, 
                        z=z_axis,
                        color='Kategori',
                        hover_data=['Company', 'No'],
                        title=f'Clustering Laptop: {x_axis} vs {y_axis} vs {z_axis}'
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # Detailed results per cluster
                    st.subheader("📋 Detail Kategori Laptop")
                    for i in range(n_clusters):
                        cluster_data = df[df['Cluster'] == i]
                        category_name = st.session_state.category['cluster_names'][i]
                        
                        with st.expander(f"**{category_name}** - {len(cluster_data)} laptop"):
                            # Summary statistics
                            st.write("**Karakteristik Rata-rata:**")
                            summary_cols = ['harga', 'Ram (GB)', 'Memory (GB)', 'Prosesor_Score', 'GPU_Score', 'Weight (KG)']
                            available_summary = [col for col in summary_cols if col in cluster_data.columns]
                            
                            if available_summary:
                                summary_df = cluster_data[available_summary].describe().loc[['mean', 'min', 'max']].T
                                summary_df.columns = ['Rata-rata', 'Minimum', 'Maximum']
                                st.dataframe(summary_df, use_container_width=True)
                            
                            # Sample laptops in this cluster
                            st.write("**Sample Laptop dalam Kategori:**")
                            display_cols = ['Company', 'TypeName', 'harga', 'Ram (GB)', 'Memory (GB)', 'Prosesor_Score']
                            available_display = [col for col in display_cols if col in cluster_data.columns]
                            st.dataframe(cluster_data[available_display].head(5), use_container_width=True)
                    
                    # Full clustered data
                    st.subheader("🗂️ Data Lengkap dengan Kategori")
                    st.dataframe(df, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Mohon untuk fileter dan bersihkan data terlebih dahulu sebelum melakukan clustering.")



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
            
            # Pastikan kolom 'No' ada di posisi pertama untuk tracking
            if 'No' not in filtered_data.columns:
                if 'No' in st.session_state.cleaned_df.columns:
                    # Copy kolom 'No' dari cleaned_df
                    filtered_data.insert(0, 'No', st.session_state.cleaned_df['No'].values[:len(filtered_data)])
                else:
                    # Buat kolom No baru
                    filtered_data.insert(0, 'No', range(1, len(filtered_data) + 1))
            
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
                
                # Sorting berdasarkan SAW_Score (descending)
                filtered_data = filtered_data.sort_values('SAW_Score', ascending=False).reset_index(drop=True)
                
                # Tambahkan kolom Peringkat di posisi pertama
                filtered_data.insert(0, 'Peringkat', range(1, len(filtered_data) + 1))
                
                st.session_state.saw_results = filtered_data
                
                st.success(f"✅ Berhasil! Ditemukan {len(filtered_data)} laptop yang sesuai.")
        
        # Display results
        if st.session_state.saw_results is not None:
            st.subheader("🏆 Hasil Perankingan")
            
            # Info box
            if 'No' in st.session_state.saw_results.columns:
                st.info("💡 **Kolom 'No'** menunjukkan nomor urut laptop dalam dataset asli untuk memudahkan pelacakan.")
            
            display_cols = ['Peringkat', 'No', 'Company', 'TypeName', 'harga', 'Ram (GB)', 'Memory (GB)', 'Prosesor_Score', 'GPU_Score', 'SAW_Score']
            if 'Kategori' in st.session_state.saw_results.columns:
                display_cols.insert(2, 'Kategori')
            
            available_display_cols = [col for col in display_cols if col in st.session_state.saw_results.columns]
            st.dataframe(st.session_state.saw_results[available_display_cols], use_container_width=True)
            
            # Visualization
            top_10 = st.session_state.saw_results.head(10).copy()
            top_10['Label'] = top_10['No'].astype(str) if 'No' in top_10.columns else top_10['Peringkat'].astype(str)
            
            fig = px.bar(top_10, 
                        x='Label', y='SAW_Score',
                        title='Top 10 Laptop Berdasarkan Skor SAW',
                        labels={'SAW_Score': 'Skor SAW', 'Label': 'No. Laptop'},
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
            laptop_title = f"#{int(row['Peringkat'])} - {row['Company']} {row['TypeName']}"
            if 'No' in row:
                laptop_title += f" (Laptop No. {int(row['No'])})"
            laptop_title += f" | Skor: {row['SAW_Score']:.4f}"
            
            with st.expander(laptop_title):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Identifikasi:**")
                    if 'No' in row:
                        st.write(f"🔢 No. Laptop: {int(row['No'])}")
                    if 'Company' in row:
                        st.write(f"🏢 Brand: {row['Company']}")
                    if 'TypeName' in row:
                        st.write(f"📱 Tipe: {row['TypeName']}")
                    
                    st.write("\n**Spesifikasi:**")
                    if 'harga' in row:
                        st.write(f"💰 Harga: €{row['harga']:.2f}")
                    if 'Ram (GB)' in row:
                        st.write(f"🧠 RAM: {int(row['Ram (GB)'])} GB")
                    if 'Memory (GB)' in row:
                        st.write(f"💾 Storage: {int(row['Memory (GB)'])} GB")
                
                with col2:
                    st.write("**Performa:**")
                    if 'Prosesor_Score' in row:
                        st.write(f"⚡ CPU Score: {int(row['Prosesor_Score'])}")
                    if 'GPU_Score' in row:
                        st.write(f"🎨 GPU Score: {int(row['GPU_Score'])}")
                    if 'Weight (KG)' in row:
                        st.write(f"⚖️ Berat: {row['Weight (KG)']:.2f} kg")
                    if 'Kategori' in row:
                        st.write(f"\n📁 Kategori: **{row['Kategori']}**")
        
        # Comparison chart
        st.subheader("📊 Perbandingan Visual")
        
        if len(top_5) > 0:
            criteria_cols = ['RAM', 'Storage', 'Prosesor_Score', 'GPU_Score']
            available_criteria = [col for col in criteria_cols if col in top_5.columns]
            
            if available_criteria:
                fig = go.Figure()
                
                for idx, row in top_5.iterrows():
                    values = [row[col] for col in available_criteria if col in row]
                    # Buat label yang informatif
                    label = f"#{int(row['Peringkat'])}"
                    if 'No' in row:
                        label += f" - No.{int(row['No'])}"
                    if 'Company' in row:
                        label += f" {row['Company']}"
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=available_criteria,
                        fill='toself',
                        name=label
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
