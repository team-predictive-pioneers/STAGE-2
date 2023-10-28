# Team Predictive Pioneers

## STAGE 2 Homework Data Preprocessing

## 1. Data Cleansing 

**Source code import library :**
```code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```

**Explorasi Data**
```code
data = pd.read_csv(r'D:\FILE_IRFAN_KARIM\Rakamin DS MSIB\Dataset/Churn_Modelling.csv')
data.head()
```

Berdasarkan eksplorasi awal, dataset ini berisi beberapa fitur tentang pelanggan sebuah bank, dan kolom target "Exited" yang menunjukkan apakah pelanggan tersebut churn (bernilai 1) atau tidak (bernilai 0).

**Fitur-fitur yang tersedia di dataset ini meliputi :**

1. **RowNumber** : Nomor baris
2. **CustomerId** : ID pelanggan
3. **Surname** : Nama belakang pelanggan
4. **CreditScore** : Skor kredit pelanggan
5. **Geography** : Negara asal pelanggan
6. **Gender** : Jenis kelamin pelanggan
7. **Age** : Umur pelanggan
8. **Tenure** : Lamanya pelanggan menjadi nasabah bank
9. **Balance** : Saldo rekening pelanggan
10. **NumOfProducts** : Jumlah produk yang dimiliki pelanggan di bank
11. **HasCrCard** : Apakah pelanggan memiliki kartu kredit (1 = Ya, 0 = Tidak)
12. **IsActiveMember** : Apakah pelanggan aktif (1 = Ya, 0 = Tidak)
13. **EstimatedSalary** : Gaji estimasi pelanggan
14. **Exited** : Apakah pelanggan churn (1 = Ya, 0 = Tidak)

**A. Handle Missing Values**

**#Check for missing values in the dataset**
```code
missing_values = data.isnull().sum()
missing_values
```

Hasil diatas menjawab pertanyaan bahwa tidak ada nilai yang hilang di setiap kolom. Oleh karena itu, tidak perlu mengambil tindakan apa pun terkait dengan nilai yang hilang atau menggunakan metode lain untuk memanipulasi keluaran isnull() dan menggunakan fungsi sebaliknya, notna(), yang mengembalikan jumlah nilai yang terisi dalam kerangka data.

**B. Handle Duplicated Data**

**#Check for duplicated rows in the dataset**
```code
duplicated_rows = data.duplicated().sum()
duplicated_rows
```

**Penjelasan dari Handle Duplicated Data:** 
    
Tidak ada baris yang duplikat dalam dataset ini. Oleh karena itu, kita tidak perlu mengambil tindakan apa pun terkait dengan data duplikat Kemudian, kita akan lanjutkan ke langkah 

**C. Handle Outliers**
    
Kita akan fokus pada fitur numerik untuk memeriksa adanya outlier. Untuk memudahkan identifikasi, kita akan menggunakan visualisasi berupa boxplot.

**#List of numeric features to check for outliers**
```code
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
```
**#Plot boxplots for each numeric feature to visualize outliers**
```code
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=data[feature])
    plt.title(feature)
    plt.tight_layout()

plt.show()
```

![stg2.1](images_2/stg2.1.jpg)

**Dari boxplot di atas, kita dapat mengamati beberapa hal :**

1. **CreditScore** : Terdapat beberapa nilai yang lebih rendah daripada whisker bawah, yang dapat dianggap sebagai outlier.
2. **Age** : Terdapat beberapa nilai yang lebih tinggi daripada whisker atas, yang dapat dianggap sebagai outlier.
3. **Tenure** : Tidak tampak adanya outlier.
4. **Balance** : Tidak tampak adanya outlier.
5. **NumOfProducts** : Terdapat beberapa nilai yang lebih tinggi daripada whisker atas, yang dapat dianggap sebagai outlier.
6. **EstimatedSalary** : Tidak tampak adanya outlier.

Meskipun kita dapat mengidentifikasi beberapa outlier, keputusan untuk menanganinya tergantung pada konteks bisnis dan tujuan analisis. Dalam banyak kasus, outlier mungkin mengandung informasi yang penting. 

Sebagai contoh, dalam analisis churn, pelanggan dengan perilaku yang "tidak biasa" (mis. skor kredit yang sangat rendah atau usia yang sangat tinggi) mungkin justru adalah segmen yang penting untuk dipahami. Tetapi, outlier bisa saja mengakibatkan model yang akan dibuat menghasilkan hasil yang kurang memuaskan.

Untuk saat ini, kami akan mengatasi outlier tersebut dengan menggunakan metode interquantile range (IQR) untuk mengurangi outlier yang ada.

**#Handle Outlier pada kolom CreditScore**
```code
Q1 = data['CreditScore'].quantile(0.25)
Q3 = data['CreditScore'].quantile(0.75)

IQR = Q3 - Q1
low_limit = Q1 - (1.5 * IQR)
high_limit = Q3 + (1.5 * IQR)
filtered_entries = ((data['CreditScore'] >= low_limit) & (data['CreditScore'] <= high_limit)) 
data = data[filtered_entries]
```

**#Handle Outlier pada kolom Age**
```code
Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)

IQR = Q3 - Q1
low_limit = Q1 - (1.5 * IQR)
high_limit = Q3 + (1.5 * IQR)
filtered_entries = ((data['Age'] >= low_limit) & (data['Age'] <= high_limit)) 
data = data[filtered_entries]
```

**#Handle Outlier pada kolom NumOfProducts**
```code
Q1 = data['NumOfProducts'].quantile(0.25)
Q3 = data['NumOfProducts'].quantile(0.75)

IQR = Q3 - Q1
low_limit = Q1 - (1.5 * IQR)
high_limit = Q3 + (1.5 * IQR)
filtered_entries = ((data['NumOfProducts'] >= low_limit) & (data['NumOfProducts'] <= high_limit)) 
data = data[filtered_entries]
```

**#Cek hasil handling outlier pada kolom yang teridentifikasi memiliki outlier**
```code
numeric_features = ['CreditScore', 'Age', 'NumOfProducts']
```
**#Plot boxplots for each numeric feature to visualize outliers**
```code
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=data[feature])
    plt.title(feature)
    plt.tight_layout()

plt.show()
```

![stg2.2](images_2/stg2.2.jpg)

**D. Feature Transformation**

Transformasi fitur dapat meningkatkan performa model dengan mengubah distribusi atau skala data. Beberapa metode transformasi populer meliputi normalisasi, standarisasi, dan transformasi logaritmik. Pertama, kita lihat distribusi dari fitur numerik untuk memutuskan apakah transformasi diperlukan.

**#Plot histograms for each numeric feature to visualize their distribution**
```code
numeric_features_2 = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features_2, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[feature], kde=True, bins=30)
    plt.title(feature)
    plt.tight_layout()
```
#plt.savefig('D:\FILE_IRFAN_KARIM\Rakamin DS MSIB\Tugas\Homework Week 11\Homework\PLOT/hist.png')
```code
plt.show()
```

![stg2.3](images_2/stg2.3.jpg)

**Melakukan Normalisasi Data**
```code
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['Balance','CreditScore', 'EstimatedSalary']] = scaler.fit_transform(data[['Balance','CreditScore', 'EstimatedSalary']
```

**#Plot Setelah di Normalisasi**
```code
numeric_features_2 = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features_2, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[feature], kde=True, bins=30)
    plt.title(feature)
    plt.tight_layout()
```
#plt.savefig(r'D:\FILE_IRFAN_KARIM\Rakamin DS MSIB\Tugas\Homework Week 11\Homework\PLOT/hist_after_normalisasi.png')
```code
plt.show()
```

![stg2.4](images_2/stg2.4.jpg)

**Berdasarkan histogram di atas :**

1. **CreditScore** : Distribusi tampaknya mendekati normal.
2. **Age** : Distribusi sedikit condong ke kanan.
3. **Tenure** : Distribusi tampak multi-modal, dengan beberapa puncak.
4. **Balance** : Terdapat dua puncak, salah satunya di nol yang menunjukkan banyak pelanggan dengan saldo nol.
5. **NumOfProducts** : Distribusi adalah kategorikal dengan beberapa nilai yang dominan.
6. **EstimatedSalary** : Distribusi tampaknya seragam.

Pada Feature-Feature yang memiliki rentang yang jauh kami melakukan transformasi feature menggunakan minmaxscaler untuk mengubah rentang datanya menjadi 0-1 tujuan nya agar model yang nantinya kami buat dapat menghasilkan nilai yang optimal.


**E. Feature Encoding**

Encoding adalah proses konversi fitur kategorikal menjadi format yang dapat dimengerti oleh algoritma machine learning. Kita akan memeriksa tipe data dari setiap kolom dan menentukan apakah ada fitur kategorikal yang perlu di-encode.

**#Check the data types of each column**
```code
data_types = data.dtypes
data_types
```

**Berdasarkan tipe data, kita memiliki beberapa fitur kategorikal :**

1. **Surname**
2. **Geography**
3. **Gender**

Namun, Surname adalah fitur yang unik untuk setiap pelanggan dan mungkin tidak memiliki kegunaan dalam prediksi churn. Oleh karena itu, kolom ini dihapus pada awal pengolahan data.

Untuk fitur Geography, Exited dan Gender, kita perlu melakukan encoding. Ada berbagai metode encoding seperti One-Hot Encoding, Label Encoding, dan lainnya. Untuk tujuan ini, kita akan menggunakan Label Encoding.

```code
v = data.head()
x = data['Geography'].value_counts()
y = data['Gender'].value_counts()
z = data['Exited'].value_counts()
display(v)
print(x)
print(y)
print(z)
```

**## Menggunakan Label Encoder**
```code
from sklearn.preprocessing import LabelEncoder
```
**# Buat objek LabelEncoder**
```code
label_encoder = LabelEncoder()
```
**# Lakukan label encoding pada kolom "Geography" dan "Gender"**
```code
data['Geography'] = label_encoder.fit_transform(data['Geography'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Exited'] = label_encoder.fit_transform(data['Exited'])
```
**# Hasil label encoding**
```code
display(data[['Geography','Gender', 'Exited']])
```

**#Hasil dari label encoding**
```code
x = data['Geography'].value_counts()
y = data['Gender'].value_counts()
z =  data['Exited'].value_counts()
print('Hasil encoding kolom geography:\n',x)
print('Hasil encoding kolom gender:\n',y)
print('Hasil encoding kolom exited:\n',z)
```

Setelah melakukan label encoding terhadap 3 kolom categorik yang penting, kami mendapatkan hasil dibawah ini :

Pada kolom **Geography :** 0 = Germany; 1 = France; 2 = Spain.

Pada kolom **Gender :** 1 = Male; 0 = Female.

Pada kolom **Exited :** 0 = No; 1 = Yes.

Fitur Geography, Exited dan Gender telah di-encode menggunakan Label Encoding. Selanjutnya, kita akan melanjutkan ke langkah F. Handle Class Imbalance. Kita akan memeriksa distribusi kelas target (Exited) untuk melihat apakah ada ketidakseimbangan kelas yang perlu diatasi.

**F. Handle Class Imbalance**

Sebelum melakukan imbalancing terhadap data, ada baiknya kita melihat terlebih dahulu berapa persen kelas mayoritas dan minoritas pada label target yang akan kita imbalance.

**#Check the distribution of the target class 'Exited'**
```code
class_distribution = data['Exited'].value_counts(normalize=True)
class_distribution
```

![stg2.6](images_2/stg2.6.jpg)

**Distribusi kelas target (Exited) adalah sebagai berikut :**

1. Kelas **0** (Tidak Churn): 80.23%
2. Kelas **1** (Churn): 19.76%

Terdapat ketidakseimbangan kelas di mana kelas 0 memiliki representasi yang jauh lebih tinggi dibandingkan dengan kelas 1. Ketidakseimbangan ini dapat mempengaruhi performa model, terutama dalam menilai kelas minoritas.

**Ada beberapa metode untuk menangani ketidakseimbangan kelas, seperti :**

1. **Resampling:** Teknik ini melibatkan penambahan atau pengurangan sampel dari kelas tertentu untuk mencapai distribusi yang lebih seimbang.
2. **Menggunakan metrik evaluasi yang tepat:** Akurasi mungkin bukan metrik yang ideal dalam kasus ketidakseimbangan kelas. Metrik lain seperti F1-score, AUC-ROC, atau precision dan recall mungkin lebih informatif.
3. **Penggunaan algoritma yang mendukung penimbangan kelas:** Beberapa algoritma memungkinkan penimbangan kelas saat pelatihan, yang memberikan penalti lebih tinggi untuk kesalahan pada kelas minoritas.
4. **Penggunaan teknik ensemble seperti Random OverSampling Boost (ROSB) atau Synthetic Minority Over-sampling Technique (SMOTE).**

Keputusan tentang bagaimana menangani ketidakseimbangan kelas tergantung pada tujuan analisis dan model yang akan digunakan. Jika kita ingin fokus pada identifikasi pelanggan yang mungkin churn (kelas 1), maka mungkin perlu mempertimbangkan resampling atau teknik lain untuk meningkatkan sensitivitas model terhadap kelas tersebut.

Untuk saat ini, asumsi kita akan menggunakan algoritma yang mendukung penimbangan kelas dan metrik evaluasi yang sesuai saat pelatihan model.

Ketidakseimbangan kelas akan di-handle dengan menggunakan algoritma yang mendukung penimbangan kelas dan metrik evaluasi yang tepat. Resampling dilakukan dengan menetapkan threshold 0.5 untuk memperbanyak label yes.

![stg2.7](images_2/stg2.7.jpg)

**Imbalancing terhadap data target dengan menetapkan threshold 50%**

```code
data['Exited'] = data['Exited'] > 0.8
print(data['Exited'].value_counts())
```

```code
X = data[[col for col in data.columns if col not in['Exited']]].values
y = data['Exited'].values
print(X.shape)
print(y.shape)
```

```code
from imblearn import under_sampling, over_sampling
X_over_SMOTE, y_over_SMOTE = over_sampling.SMOTE(sampling_strategy=0.5).fit_resample(X,y)
```

```code
print(pd.Series(y_over_SMOTE).value_counts())
```

**# Buat DataFrame baru untuk hasil oversampling**
```code
oversampled_data = pd.DataFrame(X_over_SMOTE, columns=data.drop('Exited',axis=1).columns)
```
#Gunakan kolom yang sesuai dengan DataFrame awal

```code
oversampled_data['Exited'] = y_over_SMOTE  
```
#Tambahkan kolom 'Exited' yang telah dihasilkan oleh SMOTE

**# Sekarang, "oversampled_data" berisi data oversampling yang sesuai dengan DataFrame asli dengan kolom "Exited" yang sudah di-oversample**

**# Cetak beberapa baris pertama sebagai contoh**
```code
display(oversampled_data.head())
```

```code
oversampled_data
```

```code
oversampled_data.to_csv('Churn_Fix.csv', index=False)
```


## 2. Feature Engineering

### A.) Feature selection (membuang feature yang kurang relevan atau redundan

**Source code import library :**
```code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```

**Baca dataset :**
```code
data = pd.read_csv(r'D:\FILE_IRFAN_KARIM\Rakamin DS MSIB\Tugas\Homework Week 11\Homework/Churn_Fix.csv')
data.head()
```

![stg2.8](images_2/stg2.8.jpg)

Pada data diatas, sebelumnya kami sudah membuang beberapa kolom yang tidak terlalu dibutuhkan seperti (Surname, CustomerId dan RowNumber) sehingga total kolom nya sekarang hanya ada 11.

Karena kolom Exited masih berupa nilai categorik kita ubah menjadi numerik terlebih dahulu

**Menggunakan Label Encoder**
```code
from sklearn.preprocessing import LabelEncoder
```

**#Buat objek LabelEncoder**
```code
label_encoder = LabelEncoder()
data['Exited'] = label_encoder.fit_transform(data['Exited'])
```

**Hasil dari encoding**

```code
data['Exited'].value_counts()
```

![stg2.9](images_2/stg2.9.jpg)
**0 = NO , 1 = YES**

Sebelum kita memasuki seleksi fitur ada baiknya kita melihat korelasi antar fitur terlebih dahulu dengan menggunakan heatmap agar lebih jelas.

**Cek korelasi dengan tabel :**
```code
data.corr(method='pearson')
```

**Cek korelasi menggunakan heatmap :**
```code
cek_corr = data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(cek_corr, cmap='PiYG', annot=True)
plt.show()
```

![stg2.10](images_2/stg2.10.jpg)

**Insight Feature Selection:**

Dapat dilihat pada visualisasi diatas, tidak ditemukannya fitur yang redundan. Artinya kita tidak perlu membuang suatu fitur, fitur-fitur di dalam dataset ini bisa dipakai dan tinggal menambahkan feature baru agar bisa mengambil insight lebih banyak dari dataset ini.


### B.) Feature extraction (membuat feature baru dari feature yang sudah ada)

```code
data['Balance_to_EstimatedSalary'] = data['Balance'] / data['EstimatedSalary']
data['Age_to_Tenure'] = data['Age'] / data['Tenure']
data['CreditScore_to_Age'] = data['CreditScore'] / data['Age']
data['CreditScore_to_Balance'] = data['CreditScore'] / (data['Balance'] + 1)
data['NumOfProducts_to_Age'] = data['NumOfProducts'] / data['Age']
```

**Penjelasan fitur yang dibuat :**

**1. Balance_to_EstimatedSalary :** Fitur ini menghitung rasio antara saldo rekening pelanggan dan gaji estimasi pelanggan. Ini dapat memberikan indikasi seberapa besar persentase dari gaji pelanggan yang disimpan di rekening bank. Fitur ini mungkin berguna untuk mengidentifikasi pola-pola yang berkaitan dengan besarnya saldo rekening relatif terhadap gaji.

**2. Age_to_Tenure :** Fitur ini menghitung rasio antara usia pelanggan dan lamanya pelanggan menjadi nasabah bank. Ini mencoba mengukur seberapa lama pelanggan telah menjadi nasabah dalam konteks usianya. Hal ini bisa membantu dalam memahami apakah pelanggan yang lebih muda atau lebih tua cenderung menjadi pelanggan baru atau setia.

**3. CreditScore_to_Age :** Fitur ini menghitung rasio antara skor kredit pelanggan dan usia pelanggan. Ini mencoba melihat hubungan antara skor kredit dan usia, apakah skor kredit cenderung berbeda antara kelompok usia yang berbeda.

**4. CreditScore_to_Balance :** Fitur ini menghitung rasio antara skor kredit pelanggan dan saldo rekening pelanggan. Ini dapat memberikan gambaran tentang apakah ada korelasi antara skor kredit dan seberapa banyak uang yang disimpan di rekening.

**5. NumOfProducts_to_Age :** Fitur ini menghitung rasio antara jumlah produk yang dimiliki pelanggan dan usia pelanggan. Ini bisa membantu dalam memahami sejauh mana pelanggan yang lebih muda atau lebih tua cenderung memiliki lebih banyak produk perbankan.

**Cek korelasi setelah adanya fitur baru :**
```code
cek_after_engineering = data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(cek_after_engineering, annot=True)
#plt.savefig(r'D:\FILE_IRFAN_KARIM\Rakamin DS MSIB\Tugas\Homework Week 11\Homework\PLOT/Heatmap2.png')
plt.tight_layout()
plt.show()
```

![stg2.11](images_2/stg2.11.jpg)

**Dapat dilihat pada visualisasi diatas, tidak ditemukannya fitur yang redundan. Artinya kita tidak perlu membuang suatu fitur, fitur-fitur di dalam dataset ini bisa dipakai dan tinggal menambahkan feature baru agar bisa mengambil insight lebih banyak serta membantu meningkatkan performa model.**

**#Cek informasi kolom setelah ditambahkan**
```code
data.info()
```

**Lampiran Feature Engineering**

```code
# 1. Age Groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0,30,40,50,100], labels=['Young','Middle-Aged','Senior','Elderly'])

# 2. Balance Categories
df['Balance_Category'] = pd.cut(df['Balance'], bins=[0,50000,100000,150000,250000], labels=['Low','Medium','High','Very High'])

# 3. Credit Score Ranges
df['CreditScore_Range'] = pd.cut(df['CreditScore'], bins=[350,500,650,750,850], labels=['Poor', 'Fair', 'Good', 'Excellent'])

# 4. Tenure Categories
df['Tenure_Category'] = pd.cut(df['Tenure'], bins=[0,2,5,10], labels=['Short Term','Medium Term', 'Long Term'])

# 5. NumOfProducts Range
df['NumOfProducts_category'] = pd.cut(df['NumOfProducts'], bins=[0,1,3,4], labels=['Low','Medium','High'])

# 6. Salary Range
df['Salary_Range'] = pd.cut(df['EstimatedSalary'], bins=[0,50000,100000, 150000, 200000], labels=['Low','Medium','High','Very High'])

# 7. Customer Relationship (combining IsActiveMember and HasCrCard)
df['Customer_Relationship'] = df['IsActiveMember'].astype(str) + df['HasCrCard'].astype(str)

# 8. Combine NumProduct to Age Group
df['NumProduct_AgeGroup'] = df['NumOfProducts_category'].astype(str) + '_' + df['Age_Group'].astype(str)

# 9. Salary to Balance Ratio
df['Salary_to_Balance_Ratio'] = df['EstimatedSalary'] / (df['Balance'] +  1) #Adding 1 to avoid division by zero(0)

# 10. Interaction between Age and Balance
df['Balance_AgeGroup'] = df['Balance_Category'].astype(str) + '_' + df['Age_Group'].astype(str)
# 11. Interaction between CreditScore and NumOfProducts
df['CreditScore_NumProducts'] = df['CreditScore_Range'].astype(str) + '_' + df['NumOfProducts_category'].astype(str)

# 12. Interaction between Age and CreditScore
df['CreditScore_AgeGroup'] = df['CreditScore_Range'].astype(str) + '_' + df['Age_Group'].astype(str)

# 13. Interaction between Geography and Gender
df['Geography_Gender_Interact'] = df['Geography'] + '_' + df['Gender']

# 14. Tenure and Number of Products Interaction
df['Tenure_NumProducts'] = df['Tenure_Category'].astype(str) + '_' + df['NumOfProducts_category'].astype(str)

# 15. Interaction Balance & Product 
df['Balance_NumProducts'] = df['Balance_Category'].astype(str) + '_' + df['NumOfProducts_category'].astype(str)

# 16. Salary to Credit Score Ratio
df['Salary_NumProducts'] = df['Salary_Range'].astype(str) + '_' + df['CreditScore_Range'].astype(str)

# 17. Age and Balance Interaction
df['Balance_AgeGroup'] = df['Balance_Category'].astype(str) + '_' + df['Age_Group'].astype(str)

# 18. Percentage of Products Used
df['Product_Usage_Percentage'] = (df['NumOfProducts'] / 4) * 100 # Assuming 4 is the maximum number of products

# 19. Customer Lifetime
df['Customer_Lifetime'] = df['Age'] * (1 / (1 - df['Exited']))

# 20. Balance to Age Ratio
df['Balance_to_Age_Ratio'] = df['Balance'] / df['Age']

# 21. Average Credit Score in Age Group
age_group_avg_credit = df.groupby('Age_Group')['CreditScore'].mean().reset_index()
age_group_avg_credit.rename(columns={'CreditScore':'Avg_CreditScore_in_Age_Group'}, inplace=True)
df = df.merge(age_group_avg_credit, on='Age_Group', how='left')

# 22. Tenure to Age Ratio
df['Tenure_to_Age_Ratio'] = df['Tenure'] / df['Age']

# 23. Salary Rank
df['Salary_Rank'] = df['EstimatedSalary'].rank(pct=True)

# 24. Credit Score to Age Ratio
df['CreditScore_to_Age_Ratio'] = df['CreditScore'] / df['Age']

# Transform category into object
# List of column names with category data type
category_columns = ['Age_Group', 'Balance_Category', 'CreditScore_Range', 'Tenure_Category', 'NumOfProducts_category', 'Salary_Range']

# Convert the specified columns from category to object
df[category_columns] = df[category_columns].astype('object')
```
