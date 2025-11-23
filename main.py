import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import app.data_utils as data_utils
from app.clustering_utils import get_available_numeric_cols, run_kmeans
import re
import plotly.express as px
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="Sistem Pemilihan Laptop Mahasiswa",
    page_icon="💻",
    layout="wide"
)

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Menu Navigasi",
    ["Upload dan Filter","Clustering (K-Means)", "Pembobotan Kriteria (AHP)", 
     "Perankingan (SAW)", "Hasil Rekomendasi"]
)

# Global Var
laptop_features = ['Ukuran (Inches)', 'Resolusi Layar_value', 'RAM (GB)', 'Memory (GB)', 'CPU_Score', 'GPU_Score', 'Memory_Value', 'Weight (KG)', 'Harga']
berat = None
uploaded_file = None
filtered_company_data = None
filtered_typename_data = None
filtered_os_data = None
df_filtered_result = pd.DataFrame()

# Initialize session state
if 'laptops_data' not in st.session_state:
    st.session_state.laptops_data = pd.DataFrame()
if 'original_uploaded_data' not in st.session_state:
    st.session_state.original_uploaded_data = pd.DataFrame()
if 'ahp_berat' not in st.session_state:
    st.session_state.ahp_berat = {}
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'saw_results' not in st.session_state:
    st.session_state.saw_results = None
if 'df_filtered_result' not in st.session_state:
    st.session_state.df_filtered_result = pd.DataFrame()
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
        st.button("Hapus Data Laptop", type="primary", on_click=lambda: st.session_state.update({'laptops_data': pd.DataFrame(), 'original_uploaded_data': pd.DataFrame()}))
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.session_state.laptops_data = uploaded_df.copy()
            st.session_state.original_uploaded_data = uploaded_df.copy()
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
        elif mode == "Manual":
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
        # Gunakan original_uploaded_data jika ada, jika tidak gunakan laptops_data
        if not st.session_state.original_uploaded_data.empty:
            df_original = st.session_state.original_uploaded_data.copy()
        else:
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

        col1filter, col2filter, col3filter = st.columns(3, gap="small")

        if filtered_company_data is not None and 'Company' in df_original.columns:
            with col1filter:
                st.write(f"**Data setelah filter Company ({len(filtered_company_data)} baris)**")
                st.dataframe(filtered_company_data, use_container_width=True)

        if filtered_typename_data is not None and 'TypeName' in df_original.columns:
            with col2filter:
                st.write(f"**Data setelah filter Tipe Laptop ({len(filtered_typename_data)} baris)**")
                st.dataframe(filtered_typename_data, use_container_width=True)

        if filtered_os_data is not None and 'OpSys' in df_original.columns:
            with col3filter:
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
                st.session_state.df_filtered_result = data_utils.automaticColumnTable(df_filtered)
            elif mode == "Manual":
                # Collect manual mapping choices (use shadows to avoid missing keys)
                manual_mapping = {}
                for label, key in manual_fields.items():
                    manual_mapping[key] = st.session_state.get(f"shadow_{key}", "None")
                st.session_state.df_filtered_result = data_utils.manualColumnTable(df_filtered, mapping=manual_mapping)
            
            # Remove duplicate columns if any exist
            if not st.session_state.df_filtered_result.empty:
                st.session_state.df_filtered_result = st.session_state.df_filtered_result.loc[:, ~st.session_state.df_filtered_result.columns.duplicated()]

        st.markdown("<h4>Data Laptop Saat Ini</h4>", unsafe_allow_html=True)

        if st.session_state.df_filtered_result.empty:
            st.warning("Tidak ada data yang cocok dengan filter yang diterapkan.")
        else:
            # Remove duplicate columns before display
            display_df = st.session_state.df_filtered_result.loc[:, ~st.session_state.df_filtered_result.columns.duplicated()]
            st.dataframe(display_df, use_container_width=True)
            
            # Clean the dataframe (also remove duplicates before cleaning)
            df_to_clean = st.session_state.df_filtered_result.loc[:, ~st.session_state.df_filtered_result.columns.duplicated()]
            cleaned_df = data_utils.clean_laptops_df(df_to_clean)
            
            # Remove duplicate columns from cleaned dataframe
            cleaned_df = cleaned_df.loc[:, ~cleaned_df.columns.duplicated()]
            
            # Example usage
            with st.status("Memproses Data Laptop...", expanded=True) as status:
                st.write("Mapping kolom otomatis...")
                time.sleep(1) # Simulating work

                st.write("Membersihkan data RAM dan Weight...")
                time.sleep(1)

                st.write("Mengkonversi CPU dan GPU ke Numerik...")
                # actual_function_call()
                time.sleep(1)

                # Update status when done
                status.update(label="Data selesai diproses!", state="complete", expanded=False)
            
            st.markdown("<h4>Data Setelah dikonversikan ke numerik</h4>", unsafe_allow_html=True)
            st.dataframe(cleaned_df, use_container_width=True)
            st.session_state.cleaned_df = cleaned_df

# ============= K-MEANS CLUSTERING =============
elif menu == "Clustering (K-Means)":
    try:
        st.header("🔍 K-Means Clustering")
        
        # Check if cleaned_df exists
        if 'cleaned_df' not in st.session_state:
            st.error("⚠️ **Anda harus memfilter data anda terlebih dahulu!**")
            st.info("📌 Silakan ke menu **'Upload dan Filter'** untuk mengupload dan memfilter data laptop terlebih dahulu.")
        else:
            st.dataframe(st.session_state.cleaned_df, use_container_width=True)
            
            if st.session_state.cleaned_df.empty:
                st.warning("⚠️ Belum ada data laptop atau data belum di filter. Silakan input data dan filter terlebih dahulu.")
            else:
                df = st.session_state.cleaned_df.copy()
                numeric_cols = st.session_state.category.get('numeric_cols', laptop_features)
                available_cols = get_available_numeric_cols(df, numeric_cols)
                st.session_state.category['available_cols'] = available_cols
                
                # Visual check for required numeric columns (show ✅ / ❌)
                required = [
                    "Ukuran (Inches)", "Resolusi Layar", "CPU", "RAM (GB)",
                    "Memory (GB)", "GPU", "Berat (KG)", "Harga"
                ]

                # Normalize helper
                def normalize(s):
                    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

                # helper predicates to match common variants in available_cols
                def matches(col, keywords):
                    ncol = normalize(col)
                    for k in keywords:
                        if normalize(k) in ncol or normalize(k) in ' '.join(ncol.split()):
                            return True
                    return False

                def is_available(field):
                    kws = {
                        "Ukuran (Inches)": ["inch", "ukuran", "inches"],
                        "Resolusi Layar": ["resolusi", "resolution", "layar", "resolusi layar value", "resolusi layar_value"],
                        "CPU": ["cpu", "prosesor", "processor", "cpu score", "cpu_score"],
                        "RAM (GB)": ["ram"],
                        "Memory (GB)": ["memory", "storage", "memory_value", "mem"],
                        "GPU": ["gpu", "grafik", "vga", "gpu score", "gpu_score"],
                        "Berat (KG)": ["berat", "weight", "kg", "berat kg", "berat (kg)"],
                        "Harga": ["harga", "price", "cost"]
                    }.get(field, [field])
                    matched = [c for c in available_cols if matches(c, kws)]
                    return matched

                st.write("Kolom numerik untuk clustering (cek ketersediaan):")
                available_count = 0
                for field in required:
                    matched = is_available(field)
                    if matched:
                        available_count += 1
                        st.markdown(f"✅ **{field}** — Tersedia sebagai: {', '.join(matched)}")
                    else:
                        st.markdown(f"❌ **{field}** — Tidak ditemukan")


                # Optional: show dataframe overview inside an expander for debugging
                with st.expander("Tampilkan info dan statistik dataframe"):
                    st.write("Columns available for clustering:", ", ".join(available_cols))
                    st.write(df.dtypes)
                    st.write(df.describe(include='all'))
                    
                # Availability rule: at least 4 fields present => overall available
                threshold = 4
                if not available_count >= threshold:
                    st.warning(f"{available_count}/{len(required)} kolom ditemukan. Minimal {threshold} diperlukan untuk menjalankan clustering.")
                    all_ok = False
                else:
                    st.success(f"{available_count}/{len(required)} kolom ditemukan. Cukup untuk clustering (>= {threshold}).")
                    all_ok = True
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
                            
                            # Store results (laptops_data akan berisi data + cluster)
                            # original_uploaded_data tetap menyimpan data asli tanpa cluster
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
                            hover_data=['Merek', 'No'],
                            title=f'Clustering Laptop: {x_axis} vs {y_axis} vs {z_axis}'
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    
                        # Detailed results per cluster
                        st.subheader("📋 Detail Kategori Laptop")
                        for i in range(n_clusters):
                            cluster_data = df[df['Cluster'] == i]
                            category_name = st.session_state.category['cluster_names'][i]
                            category_name_str = str(category_name) if pd.notna(category_name) else 'Unknown'
                            
                            with st.expander(f"**{category_name_str}** - {len(cluster_data)} laptop"):
                                # Summary statistics
                                st.write("**Karakteristik Rata-rata:**")
                                summary_cols = ['Harga', 'RAM (GB)', 'Memory (GB)', 'CPU_Score', 'GPU_Score', 'Weight (KG)']
                                available_summary = [col for col in summary_cols if col in cluster_data.columns]
                                
                                if available_summary:
                                    summary_df = cluster_data[available_summary].describe().loc[['mean', 'min', 'max']].T
                                    summary_df.columns = ['Rata-rata', 'Minimum', 'Maximum']
                                    st.dataframe(summary_df, use_container_width=True)
                                
                                # Sample laptops in this cluster
                                st.write("**Sample Laptop dalam Kategori:**")
                                display_cols = ['Merek', 'Tipe Laptop', 'Harga', 'RAM (GB)', 'Memory (GB)', 'CPU_Score']
                                available_display = [col for col in display_cols if col in cluster_data.columns]
                                st.dataframe(cluster_data[available_display].head(5), use_container_width=True)
                        
                        # Full clustered data
                        st.subheader("🗂️ Data Lengkap dengan Kategori")
                        st.dataframe(df, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Mohon untuk filter dan bersihkan data terlebih dahulu sebelum melakukan clustering. Kesalahan: {e}")



# ============= AHP WEIGHTING =============
elif menu == "Pembobotan Kriteria (AHP)":
    st.header("⚖️ Analytical Hierarchy Process (AHP)")
    
    if 'cleaned_df' not in st.session_state:
        st.error("⚠️ **Anda harus memfilter data anda terlebih dahulu!**")
        st.info("📌 Silakan ke menu **'Upload dan Filter'** untuk mengupload dan memfilter data laptop terlebih dahulu.")
    else:
    # Check if clustering has been performed
        has_clusters = 'Cluster' in st.session_state.laptops_data.columns if not st.session_state.laptops_data.empty else False
        
        if has_clusters:
            st.info("💡 **Clustering terdeteksi!** Anda dapat membuat bobot kriteria yang berbeda untuk setiap kategori laptop.")
            
            # Get available categories as-is
            categories = st.session_state.laptops_data['Kategori'].unique().tolist()
            
            # Mode selection
            weighting_mode = st.radio(
                "Mode Pembobotan",
                ["Global (Semua Kategori)", "Per Kategori (Spesifik)"],
                help="Global: Satu bobot untuk semua laptop. Per Kategori: Bobot berbeda untuk setiap kategori laptop."
            )
            
            if weighting_mode == "Per Kategori (Spesifik)":
                # Display categories as strings but keep original value
                category_display = [str(cat) if pd.notna(cat) else 'Unknown' for cat in categories]
                selected_display = st.selectbox(
                    "Pilih Kategori untuk Pembobotan",
                    category_display,
                    help="Buat bobot kriteria khusus untuk kategori ini"
                )
                # Map back to original category value
                selected_category = categories[category_display.index(selected_display)]
                selected_category_str = selected_display
                st.write(f"📁 **Membuat bobot untuk kategori: {selected_category_str}**")
            else:
                selected_category = "Global"
                st.write("🌐 **Membuat bobot global untuk semua kategori**")
        else:
            st.info("ℹ️ Belum ada clustering. Bobot akan diterapkan secara global untuk semua laptop.")
            selected_category = "Global"
            weighting_mode = "Global (Semua Kategori)"
        
        st.write("""
        **Skala AHP:**
        - 1: Sama penting
        - 3: Sedikit lebih penting
        - 5: Lebih penting
        - 7: Sangat lebih penting
        - 9: Mutlak lebih penting
        """)
        
        criteria = ['Harga', 'Prosesor', 'RAM', 'Storage', 'GPU', 'Baterai', 'Bobot']
        n = len(criteria)
        
        # Create pairwise comparison matrix
        # Pastikan selected_category_str sudah didefinisikan
        if weighting_mode != "Per Kategori (Spesifik)":
            selected_category_str = str(selected_category)
        st.subheader(f"Matriks Perbandingan Berpasangan - {selected_category_str}")
        
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
            # Always convert category to string for dictionary key
            selected_category_key = str(selected_category) if pd.notna(selected_category) else 'Unknown'
            if st.button(f"Hitung Bobot AHP - {selected_category_key}", type="primary"):
                # Normalize to get weights
                weights = principal_eigenvector / principal_eigenvector.sum()
                
                # Initialize ahp_weights structure if not exists
                if 'ahp_weights_per_category' not in st.session_state:
                    st.session_state.ahp_weights_per_category = {}
                
                # Store weights per category (gunakan string key untuk konsistensi)
                category_weights = {criteria[i]: weights[i] for i in range(n)}
                st.session_state.ahp_weights_per_category[selected_category_key] = category_weights
                
                # Also store in old format for backward compatibility
                st.session_state.ahp_weights = category_weights
                
                # Display results
                st.subheader(f"📊 Hasil Pembobotan - {selected_category_key}")
                
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
                                title=f'Distribusi Bobot - {selected_category_key}')
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Bobot untuk '{selected_category_key}' berhasil disimpan!")
    
    # Display summary of all category weights
    if 'ahp_weights_per_category' in st.session_state and st.session_state.ahp_weights_per_category:
        st.divider()
        st.subheader("📋 Ringkasan Bobot Semua Kategori")
        
        summary_data = []
        for cat, weights_dict in st.session_state.ahp_weights_per_category.items():
            row = {'Kategori': cat}
            row.update(weights_dict)
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Show which categories still need weights
            if has_clusters and weighting_mode == "Per Kategori (Spesifik)":
                weighted_cats = set(st.session_state.ahp_weights_per_category.keys())
                all_cats = set(categories)
                missing_cats = all_cats - weighted_cats
                
                if missing_cats:
                    st.warning(f"⚠️ Kategori yang belum diberi bobot: {', '.join(missing_cats)}")
                else:
                    st.success("✅ Semua kategori sudah memiliki bobot!")

# ============= SAW RANKING =============
elif menu == "Perankingan (SAW)":
    st.header("📊 Simple Additive Weighting (SAW)")
    
    if 'cleaned_df' not in st.session_state:
        st.error("⚠️ **Anda harus memfilter data anda terlebih dahulu!**")
        st.info("📌 Silakan ke menu **'Upload dan Filter'** untuk mengupload dan memfilter data laptop terlebih dahulu.")
    else:
        try:
            # Check if clustering exists
            has_clusters = 'Cluster' in st.session_state.laptops_data.columns
            has_category_weights = ('ahp_weights_per_category' in st.session_state and 
                                   st.session_state.ahp_weights_per_category is not None and 
                                   len(st.session_state.ahp_weights_per_category) > 0)
            
            if has_clusters and has_category_weights:
                st.info("💡 **Clustering dan Bobot Per Kategori Terdeteksi!** SAW akan menggunakan bobot spesifik untuk setiap kategori.")
                
                # Show weight summary
                with st.expander("📋 Lihat Bobot Per Kategori"):
                    if 'ahp_weights_per_category' in st.session_state and st.session_state.ahp_weights_per_category:
                        for cat, weights in st.session_state.ahp_weights_per_category.items():
                            st.write(f"**{cat}:**")
                            weight_str = ", ".join([f"{k}: {v:.3f}" for k, v in weights.items()])
                            st.write(weight_str)
                    else:
                        st.warning("Belum ada bobot per kategori. Gunakan bobot default.")
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
                    # Get categories and create display names
                    categories_raw = st.session_state.laptops_data['Kategori'].unique()
                    categories_display = [str(cat) if pd.notna(cat) else 'Unknown' for cat in categories_raw]
                    selected_display = st.selectbox("Kategori Laptop", 
                                                    ["Semua"] + categories_display)
                    # Map back to original value for filtering
                    if selected_display != "Semua":
                        selected_category_idx = categories_display.index(selected_display)
                        selected_category = categories_raw[selected_category_idx]
                    else:
                        selected_category = "Semua"
                else:
                    selected_category = "Semua"
            
            # Show ranking mode
            if has_clusters:
                ranking_mode = st.radio(
                    "Mode Perankingan",
                    ["Per Kategori (Terpisah)", "Global (Gabungan Semua)"],
                    help="Per Kategori: Ranking terpisah untuk setiap kategori. Global: Ranking semua laptop digabung."
                )
            else:
                ranking_mode = "Global (Gabungan Semua)"
            
            if st.button("Hitung Peringkat SAW", type="primary"):
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
                if 'RAM (GB)' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['RAM (GB)'] >= min_ram]
                if selected_category != "Semua" and 'Kategori' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['Kategori'] == selected_category]
                
                if len(filtered_data) == 0:
                    st.error("❌ Tidak ada laptop yang memenuhi kriteria.")
                else:
                    # Normalize criteria
                    criteria_mapping = {
                        'Harga': 'Harga',
                        'Prosesor': 'CPU_Score',
                        'RAM': 'RAM (GB)',
                        'Storage': 'Memory (GB)',
                        'GPU': 'GPU_Score',
                        'Baterai': 'Baterai',
                        'Bobot': 'Weight (KG)'
                    }
                    
                    # Calculate SAW scores based on ranking mode
                    if ranking_mode == "Per Kategori (Terpisah)" and 'Kategori' in filtered_data.columns:
                        # Calculate SAW per category with category-specific weights
                        all_results = []
                        
                        for kategori in filtered_data['Kategori'].unique():
                            # Konversi kategori ke string untuk keamanan
                            kategori_key = str(kategori) if pd.notna(kategori) else 'Unknown'
                            # Gunakan kategori asli untuk filtering (bisa float), tapi kategori_key untuk display
                            category_data = filtered_data[filtered_data['Kategori'] == kategori].copy()
                            
                            # Get weights for this category (fallback to global if not available)
                            if has_category_weights and kategori_key in st.session_state.ahp_weights_per_category:
                                weights_to_use = st.session_state.ahp_weights_per_category[kategori_key]
                                st.write(f"✓ Menggunakan bobot spesifik untuk kategori: {kategori_key}")
                            elif has_category_weights and 'Global' in st.session_state.ahp_weights_per_category:
                                weights_to_use = st.session_state.ahp_weights_per_category['Global']
                                st.write(f"⚠️ Menggunakan bobot Global untuk kategori: {kategori_key}")
                            elif 'ahp_weights' in st.session_state and st.session_state.ahp_weights:
                                weights_to_use = st.session_state.ahp_weights
                                st.write(f"⚠️ Menggunakan bobot default untuk kategori: {kategori_key}")
                            else:
                                st.error("❌ Tidak ada bobot AHP yang tersedia. Silakan buat bobot terlebih dahulu di menu 'Pembobotan Kriteria (AHP)'.")
                                st.stop()
                            
                            saw_score = np.zeros(len(category_data))
                            
                            for criteria_name, weight in weights_to_use.items():
                                criteria_name_str = str(criteria_name)
                                
                                if criteria_name_str in criteria_mapping:
                                    col_name = criteria_mapping[criteria_name_str]
                                    
                                    if col_name in category_data.columns:
                                        values = category_data[col_name].values
                                        
                                        # Cost criteria (lower is better): Harga, Bobot
                                        if criteria_name_str in ['Harga', 'Bobot']:
                                            min_val = float(values.min())
                                            norm_values = min_val / (values.astype(float) + 0.0001)
                                        # Benefit criteria (higher is better)
                                        else:
                                            norm_values = values / (values.max() + 0.0001)
                                        
                                        saw_score += weight * norm_values
                            
                            category_data['SAW_Score'] = saw_score
                            all_results.append(category_data)
                        
                        # Combine all categories
                        filtered_data = pd.concat(all_results, ignore_index=True)
                        
                    else:
                        # Global ranking - use global weights or default
                        if has_category_weights and 'Global' in st.session_state.ahp_weights_per_category:
                            weights_to_use = st.session_state.ahp_weights_per_category['Global']
                            st.write("✓ Menggunakan bobot Global")
                        elif 'ahp_weights' in st.session_state and st.session_state.ahp_weights:
                            weights_to_use = st.session_state.ahp_weights
                            st.write("✓ Menggunakan bobot default")
                        else:
                            st.error("❌ Tidak ada bobot AHP yang tersedia. Silakan buat bobot terlebih dahulu di menu 'Pembobotan Kriteria (AHP)'.")
                            st.stop()
                        
                    saw_score = np.zeros(len(filtered_data))
                    
                    for criteria_name, weight in weights_to_use.items():
                        criteria_name_str = str(criteria_name)
                        
                        if criteria_name_str in criteria_mapping:
                            col_name = criteria_mapping[criteria_name_str]
                            
                            if col_name in filtered_data.columns:
                                values = filtered_data[col_name].values
                                
                                # Cost criteria (lower is better): Harga, Bobot
                                if criteria_name_str in ['Harga', 'Bobot']:
                                    min_val = float(values.min())
                                    norm_values = min_val / (values.astype(float) + 0.0001)
                                # Benefit criteria (higher is better)
                                else:
                                    norm_values = values / (values.max() + 0.0001)
                                
                                saw_score += weight * norm_values
                    
                    filtered_data['SAW_Score'] = saw_score
                    
                    # Sorting berdasarkan SAW_Score (descending)
                    st.write("🔎 Mengurutkan laptop berdasarkan skor SAW...")
                filtered_data = filtered_data.sort_values('SAW_Score', ascending=False).reset_index(drop=True)
                
                # Tambahkan kolom Peringkat di posisi pertama
                filtered_data.insert(0, 'Peringkat', range(1, len(filtered_data) + 1))
                
                st.session_state.saw_results = filtered_data
                
                st.success(f"✅ Berhasil! Ditemukan {len(filtered_data)} laptop yang sesuai.")
            
            # Display results
            st.write("---")
            
            if st.session_state.saw_results is not None:
                st.divider()
                st.subheader("🏆 Hasil Perankingan")
                
                # Info box
                if 'No' in st.session_state.saw_results.columns:
                    st.info("💡 **Kolom 'No'** menunjukkan nomor urut laptop dalam dataset asli untuk memudahkan pelacakan.")
                
                # Display by category if available
                if 'Kategori' in st.session_state.saw_results.columns and ranking_mode == "Per Kategori (Terpisah)":
                    st.write("### 📊 Peringkat Per Kategori")
                    
                    for kategori in st.session_state.saw_results['Kategori'].unique():
                        kategori_str = str(kategori) if pd.notna(kategori) else 'Unknown'
                        with st.expander(f"🏅 Kategori: {kategori_str}", expanded=True):
                            category_results = st.session_state.saw_results[
                                st.session_state.saw_results['Kategori'] == kategori
                            ].copy()
                            
                            # Re-rank within category
                            category_results = category_results.sort_values('SAW_Score', ascending=False).reset_index(drop=True)
                            category_results['Peringkat_Kategori'] = range(1, len(category_results) + 1)
                            
                            display_cols = ['Peringkat_Kategori', 'No', 'Merek', 'Tipe Laptop', 'Harga', 'RAM (GB)', 'Memory (GB)', 'CPU_Score', 'GPU_Score', 'SAW_Score']
                            available_display_cols = [col for col in display_cols if col in category_results.columns]
                            
                            st.dataframe(category_results[available_display_cols], use_container_width=True)
                            
                            # Top 3 in this category
                            st.write(f"**🥇 Top 3 di {kategori_str}:**")
                            top_3 = category_results.head(3)
                            for idx, row in top_3.iterrows():
                                medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else "🏅"
                                merek = str(row['Merek']) if 'Merek' in row else 'Unknown'
                                tipe = str(row['Tipe Laptop']) if 'Tipe Laptop' in row else 'Unknown'
                                st.write(f"{medal} {merek} {tipe} - Skor: {row['SAW_Score']:.4f}")
                else:
                    # Global display
                    display_cols = ['Peringkat', 'No', 'Merek', 'Tipe Laptop', 'Harga', 'RAM (GB)', 'Memory (GB)', 'CPU_Score', 'GPU_Score', 'SAW_Score']
                    if 'Kategori' in st.session_state.saw_results.columns:
                        display_cols.insert(2, 'Kategori')
                    
                    available_display_cols = [col for col in display_cols if col in st.session_state.saw_results.columns]
                    st.dataframe(st.session_state.saw_results[available_display_cols], use_container_width=True)
                
                # Visualization
                top_10 = st.session_state.saw_results.head(10).copy()
                if 'No' in top_10.columns:
                    top_10['Label'] = top_10['No'].astype(str)
                elif 'Peringkat' in top_10.columns:
                    top_10['Label'] = top_10['Peringkat'].astype(str)
                else:
                    top_10['Label'] = [str(i) for i in range(1, len(top_10) + 1)]
                
                fig = px.bar(top_10, 
                            x='Label', y='SAW_Score',
                            title='Top 10 Laptop Berdasarkan Skor SAW',
                            labels={'SAW_Score': 'Skor SAW', 'Label': 'No. Laptop'},
                            color='SAW_Score',
                            color_continuous_scale='Viridis')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:  
            st.error(f"❌ Terjadi kesalahan saat perankingan: {e}")


# ============= RECOMMENDATIONS =============
elif menu == "Hasil Rekomendasi":
    st.header("🎯 Rekomendasi Laptop")
    
    if 'cleaned_df' not in st.session_state:
        st.error("⚠️ **Anda harus memfilter data anda terlebih dahulu!**")
        st.info("📌 Silakan ke menu **'Upload dan Filter'** untuk mengupload dan memfilter data laptop terlebih dahulu.")
    elif 'saw_results' not in st.session_state or st.session_state.saw_results is None:
        st.error("⚠️ **Anda harus melakukan perankingan SAW terlebih dahulu!**")
        st.info("📌 Silakan ke menu **'Perankingan (SAW)'** untuk melakukan perankingan laptop.")
    else:
        # Check if we have clustering results
        has_categories = 'Kategori' in st.session_state.saw_results.columns
        
        if has_categories:
            st.info("💡 **Rekomendasi berdasarkan kategori clustering** - Menampilkan laptop terbaik dari setiap kategori untuk membantu Anda memilih sesuai kebutuhan.")
            
            # Display mode selection
            display_mode = st.radio(
                "Mode Tampilan",
                ["Top Overall (5 Terbaik Keseluruhan)", "Best Per Category (Terbaik dari Setiap Kategori)"],
                help="Pilih bagaimana Anda ingin melihat rekomendasi"
            )
            
            if display_mode == "Best Per Category (Terbaik dari Setiap Kategori)":
                st.subheader("🌟 Laptop Terbaik per Kategori")
                
                categories = st.session_state.saw_results['Kategori'].unique()
                
                # Create tabs for each category
                tabs = st.tabs([f"🏅 {str(cat)}" for cat in categories])
                
                for tab, kategori in zip(tabs, categories):
                    with tab:
                        kategori_str = str(kategori) if pd.notna(kategori) else 'Unknown'
                        category_results = st.session_state.saw_results[
                            st.session_state.saw_results['Kategori'] == kategori
                        ].sort_values('SAW_Score', ascending=False)
                        
                        st.write(f"### Rekomendasi untuk: {kategori_str}")
                        
                        # Top 3 in this category
                        top_3_category = category_results.head(3)
                        
                        for idx, (_, row) in enumerate(top_3_category.iterrows()):
                            medal = ["🥇", "🥈", "🥉"][idx]
                            merek = str(row['Merek']) if 'Merek' in row else 'Unknown'
                            tipe = str(row['Tipe Laptop']) if 'Tipe Laptop' in row else 'Unknown'
                            laptop_title = f"{medal} {merek} {tipe}"
                            if 'No' in row:
                                laptop_title += f" (No. {int(row['No'])})"
                            laptop_title += f" - Skor: {row['SAW_Score']:.4f}"
                            
                            with st.expander(laptop_title, expanded=(idx==0)):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**📋 Identifikasi:**")
                                    if 'No' in row:
                                        st.write(f"🔢 No. Laptop: {int(row['No'])}")
                                    if 'Merek' in row:
                                        st.write(f"🏢 Brand: {str(row['Merek'])}")
                                    if 'Tipe Laptop' in row:
                                        st.write(f"📱 Tipe: {str(row['Tipe Laptop'])}")
                                    if 'Kategori' in row:
                                        st.write(f"📁 Kategori: **{str(row['Kategori'])}**")
                                    
                                    st.write("\n**💰 Spesifikasi:**")
                                    if 'Harga' in row:
                                        st.write(f"💵 Harga: €{row['Harga']:.2f}")
                                    if 'RAM (GB)' in row:
                                        st.write(f"🧠 RAM: {int(row['RAM (GB)'])} GB")
                                    if 'Memory (GB)' in row:
                                        st.write(f"💾 Storage: {int(row['Memory (GB)'])} GB")
                                
                                with col2:
                                    st.write("**⚡ Performa:**")
                                    if 'CPU_Score' in row:
                                        st.write(f"🖥️ CPU Score: {int(row['CPU_Score'])}")
                                    if 'GPU_Score' in row:
                                        st.write(f"🎨 GPU Score: {int(row['GPU_Score'])}")
                                    if 'Weight (KG)' in row and pd.notna(row['Weight (KG)']):
                                        st.write(f"⚖️ Berat: {float(row['Weight (KG)']):.2f} kg")
                                    
                                    st.write("\n**📊 SAW Score:**")
                                    st.metric("Skor Total", f"{row['SAW_Score']:.4f}")
                        
                        # Show summary chart for this category
                        if len(category_results) >= 3:
                            st.write("### 📊 Perbandingan Top 3")
                            comparison_data = top_3_category[['Merek', 'CPU_Score', 'GPU_Score', 'RAM (GB)', 'SAW_Score']].copy()
                            comparison_data['Label'] = comparison_data['Merek'].astype(str) + " " + top_3_category['Tipe Laptop'].astype(str)
                            
                            fig = px.bar(comparison_data, x='Label', y='SAW_Score',
                                       title=f'Perbandingan Skor SAW - {kategori_str}',
                                       color='SAW_Score',
                                       color_continuous_scale='Viridis')
                            st.plotly_chart(fig, use_container_width=True)
            else:
                # Top Overall mode
                st.subheader("🏆 Top 5 Rekomendasi Laptop Overall")
                
                top_5 = st.session_state.saw_results.head(5)
                
                for idx, row in top_5.iterrows():
                    medal = ["🥇", "🥈", "🥉", "🏅", "🏅"][idx] if idx < 5 else "🏅"
                    merek = str(row['Merek']) if 'Merek' in row else 'Unknown'
                    tipe = str(row['Tipe Laptop']) if 'Tipe Laptop' in row else 'Unknown'
                    laptop_title = f"{medal} #{int(row['Peringkat'])} - {merek} {tipe}"
                    if 'No' in row:
                        laptop_title += f" (No. {int(row['No'])})"
                    laptop_title += f" | Skor: {row['SAW_Score']:.4f}"
                    
                    with st.expander(laptop_title, expanded=(idx==0)):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📋 Identifikasi:**")
                            if 'No' in row:
                                st.write(f"🔢 No. Laptop: {int(row['No'])}")
                            if 'Merek' in row:
                                st.write(f"🏢 Brand: {str(row['Merek'])}")
                            if 'Tipe Laptop' in row:
                                st.write(f"📱 Tipe: {str(row['Tipe Laptop'])}")
                            if 'Kategori' in row:
                                st.write(f"📁 Kategori: **{str(row['Kategori'])}**")
                            
                            st.write("\n**💰 Spesifikasi:**")
                            if 'Harga' in row:
                                st.write(f"💵 Harga: €{row['Harga']:.2f}")
                            if 'RAM (GB)' in row:
                                st.write(f"🧠 RAM: {int(row['RAM (GB)'])} GB")
                            if 'Memory (GB)' in row:
                                st.write(f"💾 Storage: {int(row['Memory (GB)'])} GB")
                        
                        with col2:
                            st.write("**⚡ Performa:**")
                            if 'CPU_Score' in row:
                                st.write(f"🖥️ CPU Score: {int(row['CPU_Score'])}")
                            if 'GPU_Score' in row:
                                st.write(f"🎨 GPU Score: {int(row['GPU_Score'])}")
                            if 'Weight (KG)' in row and pd.notna(row['Weight (KG)']):
                                st.write(f"⚖️ Berat: {float(row['Weight (KG)']):.2f} kg")
                            
                            st.write("\n**📊 SAW Score:**")
                            st.metric("Skor Total", f"{row['SAW_Score']:.4f}")
        else:
            # No categories - original behavior
            st.subheader("🏆 Top 5 Rekomendasi Laptop")
            
            top_5 = st.session_state.saw_results.head(5)
            
            for idx, row in top_5.iterrows():
                merek = str(row['Merek']) if 'Merek' in row else 'Unknown'
                tipe = str(row['Tipe Laptop']) if 'Tipe Laptop' in row else 'Unknown'
                laptop_title = f"#{int(row['Peringkat'])} - {merek} {tipe}"
                if 'No' in row:
                    laptop_title += f" (Laptop No. {int(row['No'])})"
                laptop_title += f" | Skor: {row['SAW_Score']:.4f}"
                
                with st.expander(laptop_title):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Identifikasi:**")
                        if 'No' in row:
                            st.write(f"🔢 No. Laptop: {int(row['No'])}")
                        if 'Merek' in row:
                            st.write(f"🏢 Brand: {str(row['Merek'])}")
                        if 'Tipe Laptop' in row:
                            st.write(f"📱 Tipe: {str(row['Tipe Laptop'])}")
                        
                        st.write("\n**Spesifikasi:**")
                        if 'Harga' in row:
                            st.write(f"💰 Harga: €{row['Harga']:.2f}")
                        if 'RAM (GB)' in row:
                            st.write(f"🧠 RAM: {int(row['RAM (GB)'])} GB")
                        if 'Memory (GB)' in row:
                            st.write(f"💾 Storage: {int(row['Memory (GB)'])} GB")
                    
                    with col2:
                        st.write("**Performa:**")
                        if 'CPU_Score' in row:
                            st.write(f"⚡ CPU Score: {int(row['CPU_Score'])}")
                        if 'GPU_Score' in row:
                            st.write(f"🎨 GPU Score: {int(row['GPU_Score'])}")
                        if 'Weight (KG)' in row and pd.notna(row['Weight (KG)']):
                            st.write(f"⚖️ Berat: {float(row['Weight (KG)']):.2f} kg")
                        if 'Kategori' in row:
                            st.write(f"\n📁 Kategori: **{row['Kategori']}**")
        
        # Comparison chart
        st.divider()
        st.subheader("📊 Perbandingan Visual")
        
        top_5 = st.session_state.saw_results.head(5)
        if len(top_5) > 0:
            criteria_cols = ['RAM (GB)', 'Memory (GB)', 'CPU_Score', 'GPU_Score']
            available_criteria = [col for col in criteria_cols if col in top_5.columns]
            
            if available_criteria:
                fig = go.Figure()
                
                for idx, row in top_5.iterrows():
                    values = [row[col] for col in available_criteria if col in row]
                    # Buat label yang informatif
                    label = f"#{int(row['Peringkat'])}"
                    if 'No' in row:
                        label += f" - No.{int(row['No'])}"
                    if 'Merek' in row:
                        label += f" {str(row['Merek'])}"
                    
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
