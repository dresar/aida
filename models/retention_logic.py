def get_retention_strategy(risk_category, customer_data):
    """
    Menghasilkan strategi retensi yang dipersonalisasi berdasarkan risiko dan penggunaan pelanggan.
    """
    strategies = []
    
    # Strategi Dasar berdasarkan Risiko
    if risk_category == "High":
        strategies.append("🚨 **Prioritas:** Intervensi Segera Diperlukan.")
        strategies.append("- **Penawaran:** Diskon 20% untuk 6 bulan ke depan jika memperbarui kontrak.")
        
        if customer_data.get('Contract') == 'Month-to-month':
            strategies.append("- **Tindakan:** Dorong untuk beralih ke kontrak 1 tahun dengan tarif terkunci yang lebih murah.")
            
        if customer_data.get('TechSupport') == 'No':
            strategies.append("- **Layanan:** Tawarkan gratis 3 bulan Dukungan Teknis Premium.")
            
    elif risk_category == "Medium":
        strategies.append("⚠️ **Prioritas:** Pantau & Libatkan.")
        strategies.append("- **Penawaran:** Tingkatkan kecepatan internet dengan harga sama selama 3 bulan.")
        
        if customer_data.get('StreamingTV') == 'No' or customer_data.get('StreamingMovies') == 'No':
            strategies.append("- **Bundling:** Tawarkan paket Streaming dengan diskon 50%.")
            
    else: # Low
        strategies.append("✅ **Prioritas:** Jaga Kepuasan.")
        strategies.append("- **Loyalitas:** Kirim ucapan 'Terima Kasih' dengan bonus kuota 5GB atau keuntungan kecil lainnya.")
        strategies.append("- **Upsell:** Rekomendasikan penambahan Proteksi Perangkat atau Backup Online.")

    # Spesifik Metode Pembayaran
    if customer_data.get('PaymentMethod') == 'Electronic check':
        strategies.append("- **Optimasi:** Dorong untuk beralih ke Pembayaran Otomatis (Kartu Kredit/Transfer Bank) untuk mendapatkan kredit tagihan $5 (mengurangi risiko churn).")
        
    return "\n".join(strategies)
