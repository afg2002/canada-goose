import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st

st.title('Project Canada Goose')
st.write('Mempertahankan brand "canada goose" agar tetap menjadi penjualan tertinggi (untuk 1 tahun kedepan) dengan metode time series forecasting')
st.markdown('# All Data')
@st.cache
def load_csv_data():
    data = pd.read_csv('Final_Data_Sales.csv')

    # Convert data yang bukan datetime yang seperti 0000-0000 ke Datetime agar hasilnya NaT
    data['sold_at'] = pd.to_datetime(data['sold_at'], errors='coerce')
    data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')
    data['shipped_at'] = pd.to_datetime(data['shipped_at'], errors='coerce')
    data['delivered_at'] = pd.to_datetime(data['delivered_at'], errors='coerce')
    data['returned_at'] = pd.to_datetime(data['returned_at'], errors='coerce')

    # Ambil data date dari data setelahnya.
    data.fillna(method='bfill', inplace=True)
    return data

data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_csv_data()
st.dataframe(data)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Ini adalah data keseluruhan dari data csv")

total_data = data.shape
st.write(f'Total Datanya adalah : {total_data}')

# Data Cleaning
data = data.dropna()
st.write("Jumlah data setelah menghapus missing value:", len(data))

#Statistika Deskriptif
st.markdown('## Statistika Deskriptif')
analisis = data.copy()
analisis = analisis[['sale_price', 'cost']]
st.table(analisis.describe())

#Perbandingan Shipped, Processing, Cancelled, Complete dan Returned
st.markdown("## Perbandingan Shipped, Processing, Cancelled, Complete dan Returned")
# plt.figure(figsize=(10,5))
# plt.pie(data['status'].value_counts(), labels=data['status'].unique(), autopct='%.2f%%')
# plt.show()
fig1, ax1 = plt.subplots()
ax1.pie(data['status'].value_counts(), labels=data['status'].unique(), autopct='%.2f%%')
st.pyplot(fig1)

#Brand Terlaris
st.markdown("## Brand Terlaris")
st.write("Ini adalah top 5 brand terlaris ")
brand = data[['product_id','product_brand', 'sale_price']]
brand = brand.groupby(['product_id','product_brand'], as_index=False)['sale_price'].sum()
brand = brand.sort_values('sale_price', ascending=False)
st.table(brand.head(5))

#Penjualan Tertinggi Berdasarkan Product Brand
st.markdown("## Penjualan Tertinggi Berdasarkan Product Brand")
def perbandingan(w, a, x, y, z):
    plt.figure(figsize=(20, 8))

    plt.subplot(221)
    plt.grid()
    plt.bar(w[a], w['sale_price'], label="Sale Price")
    plt.title(y)

    plt.subplot(222)
    plt.grid()
    plt.bar(x[a], x['sale_price'], label="Sale Price")
    plt.title(z)
    st.pyplot(plt)

product_brand = brand
pb = product_brand[['product_brand', 'sale_price']]
sh = pb.sort_values('sale_price').tail(5)
sl = pb.sort_values('sale_price').head(5)

perbandingan(sh, 'product_brand', sl, 'Penjualan Tertinggi Berdasarkan Product Brand', 'Penjualan Terendah Berdasarkan Product Brand')

#Visualisasi Data Sale Price
st.markdown(' # Visualisasi Data Sale Price Khusus Untuk Canada Goose')
cg = data.copy()
cg= cg[['created_at','product_brand','sale_price']]
cg_f = cg.loc[cg['product_brand'] == 'Canada Goose'] #Ambil data Canada Goose Saja
cg_f = cg_f.sort_values('created_at')
st.write('Sorting berdasarkan tanggal pada created_at')
st.dataframe(cg_f)

#Resampling Data to Monthly
st.markdown('## Resampling data perbulan')
st.write('Data sale_price disini ditampilkan dalam perbulan')

cg_e = cg_f[['created_at','sale_price']] ## Ambil created at dan sale price
cg_e = cg_e.sort_values('created_at')
y = cg_e.set_index('created_at').resample('M').mean() ## Rata rata sale price /bulan agar data tidak lebih 'noisy' (m yang dimaksud adalah month end frequency)
y = y.dropna() #Hapus Value Kosong
y = y.rename_axis(None, axis=1).rename_axis('Date', axis=0) #Ubah index yang tadinya 'created_at' menjadi 'Date'

st.dataframe(y.head(10)) #Tampilkan 10 data teratas saja

# Classic Time Series Decomposition -> 1920
st.markdown('## Classic Time Series Decomposition -> 1920')
st.markdown('''
Teknik untuk memisahkan time series menjadi trend, seasonal, dan residual menggunakan movie average, ada 2 tipe:

*Additive = Trend + Seasonal + Residual*\n
*Multiplicative = Trend * Seasonal * Residual*\n

Additive dipakai **untuk trend dan seasonal yang tidak terlalu bervariasi**\n
Multiplicative dipakai **untuk trend dan seasonal yang berubah seiring jalannya waktu**
''')
rcParams['figure.figsize'] = 10, 5 #Besar Figur
decomposition = seasonal_decompose(y.copy(), model='additive',period=12)

fig = decomposition.plot()
st.pyplot(fig)

#Model

y_train, y_test = y[:28], y[-7:] # Pisah data untuk keperlaun model dengan 80% train dan 20% test

st.markdown('# Model')
st.markdown('## ProphetFB Model')
from prophet import Prophet #Import Prophet FB Model

m = Prophet()
d = y.copy()
d= d.reset_index()
d = d.rename(columns={'Date' : 'ds', 'sale_price' : 'y'})

model = m.fit(d)
future = m.make_future_dataframe(periods=14, freq='M') #bisa setting periode untuk setting seberapa jauh untuk diprediksi (dalam bulan)
forecast = m.predict(future)
forecast = forecast.set_index('ds')
d = d.set_index('ds')
final_forecast = forecast['yhat']

fig = plt.figure(figsize=(15,5))
plt.title("Prediksi untuk 1 tahun kedepan dengan ProphetFB Model")
plt.plot(d, label="Actual")
plt.plot(final_forecast, label="Predicted")
plt.legend(loc = 'upper left')
st.pyplot(fig)

#Arima Model
st.markdown("## ARIMA Model")
from pmdarima import auto_arima
arima = auto_arima(y_train,start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                             start_P=0, seasonal=True, d=1, D=1, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)

n_forecast = len(y_test) + 8
pred= arima.predict(n_forecast,D=1,seasonal=(1,0,0))
dates = pd.date_range(y_test.index[-1],periods=n_forecast, freq='M')
pred= pd.Series(pred, index=dates)

fig = plt.figure(figsize=(15,5))
plt.title("Prediksi menurut arima untuk 1 tahun kedepan")
plt.plot(y_train,label="Training")
plt.plot(y_test,label="Test")
plt.plot(pred,label="Pred")
plt.legend(loc = 'upper left')
st.pyplot(fig)







