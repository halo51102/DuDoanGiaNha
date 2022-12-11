import streamlit as st

def app():
    st.set_page_config(page_title="Information", page_icon="🧾")

    st.markdown("""
# Dự đoán giá nhà

## Mô tả

"Dự đoán về giá nhà", xây dựng trên Streamlit framework"

## Mục tiêu

Mục tiêu của dự án này là dự đoán giá của một ngôi nhà ở Ames bằng cách sử dụng các thuộc tính do bộ dữ liệu cung cấp.

## Thuộc tính

Bộ dữ liệu chứa các thuộc tính sau:

* **OverallQual**: Chất lượng tổng thể của ngôi nhà
* **GrLivArea**: Diện tích sinh hoạt trên tầng (mặt đất) feet vuông
* **GarageCars**: Số lượng gara ô tô
* **TotalBsmtSF**: Tổng diện tích thước vuông diện tích tầng hầm
* **FullBath**: Số lượng bồn tắm
* **YearBuilt**: Năm xây nhà
* **TotRmsAbvGrd**: Tổng số phòng trên cấp (không bao gồm phòng tắm và tủ quần áo)
* **Fireplaces**: Số lượng lò sưởi
* **BedroomAbvGr**: Số phòng ngủ trên tầng
* **GarageYrBlt**: Năm ga ra được xây dựng
* **LowQualFinSF**: Diện tích phần chất lượng thấp nhất đã hoàn thành
* **LotFrontage**: Diện tích lô đất mặt tiền
* **MasVnrArea**: Diện tích gạch ốp tường
* **WoodDeckSF**: Diện tích sàn gỗ
* **OpenPorchSF**: Diện tích hiên mở
* **EnclosedPorch**: Diện tích hiên nhà kín
* **3SsnPorch**: Diện tích hiên nhà ba mùa
* **ScreenPorch**: Diện tích hiên nhà mặt tiền
* **PoolArea**: Diện tích hồ bơi
* **MiscVal**: Giá trị khác
* **MoSold**: Month house was sold
* **YrSold**: Năm bán nhà
* **SalePrice**: Giá khuyến mãi

## Cách sử dụng

```bash
# clone the repo
git clone https://github.com/halo51102/DuDoanGiaNha.git

# change to the repo directory
cd DuDoanGiaNHa

# if virtualenv is not installed, install it
#pip install virtualenv

# create a virtualenv
virtualenv -p python3 venv

# install dependencies
pip install -r requirements.txt

# run the script
streamlit run 1_💻_Trang_Chủ.py
```

## Mô hình phát triển

### Mô hình
Dựa trên thuật toán Grid Search Cross Validation

### Training

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(max_depth=3, 
                    n_estimators = 100, 
                    learning_rate = 0.2,
                    min_child_samples = 10)
model.fit(x_train, y_train)
```

Grid Search Cross Validation được sử dụng cho các tham số của mô hình.

```python
from sklearn.model_selection import GridSearchCV

params = [{"max_depth":[3, 5], 
            "n_estimators" : [50, 100], 
            "learning_rate" : [0.1, 0.2],
            "min_child_samples" : [20, 10]}]

gs_knn = GridSearchCV(model,
                      param_grid=params,
                      cv=5)

gs_knn.fit(x_train, y_train)
gs_knn.score(x_train, y_train)

pred_y_train = model.predict(x_train)
pred_y_test = model.predict(x_test)

r2_train = metrics.r2_score(y_train, pred_y_train)
r2_test = metrics.r2_score(y_test, pred_y_test)

msle_train =metrics.mean_squared_log_error(y_train, pred_y_train)
msle_test =metrics.mean_squared_log_error(y_test, pred_y_test)

print(f"Train r2 = {r2_train:.2f} \nTest r2 = {r2_test:.2f}")
print(f"Train msle = {msle_train:.2f} \nTest msle = {msle_test:.2f}")

print(gs_knn.best_params_)
```

### Evaluation

```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_log_error

y_pred = model.predict(x_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean Squared Log Error:', mean_squared_log_error(y_test, y_pred))
print('Explained Variance Score:', explained_variance_score(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))
```

## Deployment

Phân phối mô hình đơn giản được thực hiện bằng Streamlit.


""")

app()