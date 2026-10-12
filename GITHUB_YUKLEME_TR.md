# GitHub'a manuel yükleme

Bu klasör seçilmiş bilimsel paylaşım içindir. Ana metin, hakem mektubu, kişisel çalışma klasörleri, tüm deney arşivleri veya API anahtarı içermez. Supplementary olarak yine yalnızca S1 metni gönderilecektir; bu GitHub deposu ayrı bir kod/veri erişimidir.

1. GitHub'da paylaşmak istediğiniz görünürlükte yeni bir depo oluşturun. Hakemlerin giriş yapmadan erişmesi isteniyorsa depo public olmalıdır. Hazır README oluşturmayın; bu klasördeki README kullanılacaktır.
2. Depoda **Add file > Upload files** seçin. `mechatronics_github_minimal` klasörünün **içeriğini** (alt klasörlerle birlikte) sürükleyin. Tek ZIP dosyası yerine klasör yapısının görünmesi denetlenebilirliği kolaylaştırır. `.gitignore` dosyasını da ekleyin. Yüklenen dosya sayısını `UPLOAD_CHECK.json` ile karşılaştırın.
3. Commit işlemini tamamlayın. README ve `data`, `simulation`, `analysis`, `tools` klasörlerinin depoda göründüğünü doğrulayın. Depo adresini buraya paylaşın; gerçek adres daha sonra ana metnin erişim paragrafına, S1'e ve hakem yanıtına eklenecektir.

Hazırlanan her dosya GitHub'ın tarayıcı yüklemesi için 25 MiB sınırının altındadır ve toplam dosya sayısı tek seferde 100 dosya sınırının altındadır. Kaynak: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository?platform=windows

`audit_result.json` yalnızca yerel doğrulama çıktısıdır ve paylaşım manifestinin dışında tutulur. Yükleme klasörüne otomatik eklenmez. `analysis/audit_release.py` aynı raporu depoyu indiren kişi için üretir. Canlı Gemini çağrısı yapmadan sonuçların tutarlılığı kontrol edilebilir.

Bu seçilmiş paket bütün makale/ek dosyaların tam veri deposu değildir. README'de belirtilen sınırları değiştirmeyin: fiziksel robotun koşu bazlı ham kayıtları, ek donanım düzenleri ve ikincil model çalışmalarının tam dosyaları bu paylaşımda yoktur. Yerel klasörün oluşturulması tek başına halka açık erişim anlamına gelmez.
