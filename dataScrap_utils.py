import requests
from bs4 import BeautifulSoup

# URL dari halaman yang ingin Anda scrape
url = "https://www.cpubenchmark.net/cpu_list.php"

#CPU Benchmark Scraping
try:
    # 1. Mengirim permintaan HTTP untuk mendapatkan konten halaman
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Ini akan memberi error jika request gagal

    # 2. Mem-parsing konten HTML menggunakan BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # 3. Menemukan tabel data
    # (Setelah diperiksa, tabel utamanya memiliki ID "cputable")
    cpu_table = soup.find('table', id='cputable')

    if cpu_table:
        # Menemukan semua baris <tr> di dalam badan tabel <tbody>
        rows = cpu_table.find('tbody').find_all('tr')

        print("--- Menampilkan semua CPU dari tabel ---")

        # 4. Melakukan loop pada setiap baris dan mengekstrak data
        for row in rows:
            # Menemukan semua sel <td> di dalam baris
            cols = row.find_all('td')
            
            # Mengambil teks dari sel
            # Sel pertama (indeks 0) adalah nama CPU
            # Sel kedua (indeks 1) adalah CPU Mark
            if len(cols) > 1:
                cpu_name = cols[0].get_text(strip=True)
                cpu_mark = cols[1].get_text(strip=True)
                
                print(f"Nama: {cpu_name}, Mark: {cpu_mark}")

    else:
        print("Tidak dapat menemukan tabel dengan ID 'cputable'.")

    # 5. Hilangkan '@' dari nama CPU jika ada (opsional)
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 1:
            cpu_name = cols[0].get_text(strip=True).replace('@', '').strip()
            cols[0].string = cpu_name  # Memperbarui teks di elemen <td>

    # 6. Ganti '-' menjadi ' ' di nama CPU (opsional)
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 1:
            cpu_name = cols[0].get_text(strip=True).replace('-', ' ').strip()
            cols[0].string = cpu_name  # Memperbarui teks di elemen <td>

    # 5. Mengubah data menjadi CSV (opsional)
    import csv
    with open('csv/cpu_data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Nama CPU', 'CPU Mark'])  # Menulis header
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                cpu_name = cols[0].get_text(strip=True)
                cpu_mark = cols[1].get_text(strip=True)
                writer.writerow([cpu_name, cpu_mark])  # Menulis data CPU
                

                
    print("Data CPU telah disimpan ke 'cpu_data.csv'.")
except requests.exceptions.RequestException as e:
    print(f"Terjadi kesalahan saat mengambil URL: {e}")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
    
# GPU Benchmark Scraping
url = "https://www.videocardbenchmark.net/gpu_list.php"

try:
    # 1. Mengirim permintaan HTTP untuk mendapatkan konten halaman
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Memberi error jika request gagal

    # 2. Mem-parsing konten HTML menggunakan BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # 3. Menemukan tabel data
    # (Setelah diperiksa, tabel utamanya memiliki ID "gputable")
    gpu_table = soup.find('table', id='cputable')

    if gpu_table:
        # Menemukan semua baris <tr> di dalam badan tabel <tbody>
        rows = gpu_table.find('tbody').find_all('tr')

        print("--- Menampilkan semua GPU dari tabel ---")

        # 4. Melakukan loop pada setiap baris dan mengekstrak data
        for row in rows:
            # Menemukan semua sel <td> di dalam baris
            cols = row.find_all('td')
            
            # Mengambil teks dari sel
            # Sel pertama (indeks 0) adalah nama GPU
            # Sel kedua (indeks 1) adalah G3D Mark
            if len(cols) > 1:
                gpu_name = cols[0].get_text(strip=True)
                gpu_mark = cols[1].get_text(strip=True)
                
                print(f"Nama: {gpu_name}, G3D Mark: {gpu_mark}")

    else:
        print("Tidak dapat menemukan tabel dengan ID 'gputable'.")
        
    # 5. Mengubah data menjadi CSV (opsional)
    with open('csv/gpu_data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Nama GPU', 'GPU Mark'])  # Menulis header
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                gpu_name = cols[0].get_text(strip=True)
                gpu_mark = cols[1].get_text(strip=True)
                writer.writerow([gpu_name, gpu_mark])  # Menulis data GPU
    print("Data GPU telah disimpan ke 'gpu_data.csv'.")
except requests.exceptions.RequestException as e:
    print(f"Terjadi kesalahan saat mengambil URL: {e}")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")