from math import ceil, floor
import pickle
import numpy as np
import streamlit as st
import pandas as pd
import datetime

thedate = datetime.date.today()
def app():
    # đọc file CSV 
    df = pd.read_csv(r"data/house_price.csv")

    
    dropColumns = ["Id", "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2", "BldgType", "OverallCond", "RoofStyle",
                "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "Electrical", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath"] + ["SaleCondition", "SaleType", "YrSold", "MoSold", "MiscVal", "MiscFeature", "Fence", "PoolQC", "PoolArea", "ScreenPorch", "3SsnPorch", "EnclosedPorch", "OpenPorchSF", "WoodDeckSF", "PavedDrive", "GarageCond", "GarageQual", "GarageType", "FireplaceQu", "Functional", "KitchenAbvGr", "BedroomAbvGr"]

    #Bỏ các cột (có tên là phần tử của mảng dropColumns) trong dataframe 
    droppedDf = df.drop(columns=dropColumns, axis=1)

    #Điền các giá trị còn thiếu (null)
    droppedDf.isnull().sum().sort_values(ascending=False)
    droppedDf["Alley"].fillna("NO", inplace=True)
    droppedDf["LotFrontage"].fillna(df.LotFrontage.mean(), inplace=True)
    droppedDf["GarageFinish"].fillna("NO", inplace=True)
    droppedDf["GarageYrBlt"].fillna(df.GarageYrBlt.mean(), inplace=True)
    droppedDf["BsmtQual"].fillna("NO", inplace=True)
    droppedDf["MasVnrArea"].fillna(0, inplace=True)
        # Chia thuộc tính "MasVnrAreaCatg" thành 3 loại khác nhau.
    droppedDf['MasVnrAreaCatg'] = np.where(droppedDf.MasVnrArea > 1000, 'BIG',
                                    np.where(droppedDf.MasVnrArea > 500, 'MEDIUM',
                                    np.where(droppedDf.MasVnrArea > 0, 'SMALL', 'NO')))

    # Chuẩn bị dữ liệu đầu vào cho mô hình
    droppedDf = droppedDf.drop(['SalePrice'], axis=1)
    inputDf = droppedDf.iloc[[0]].copy()

    for i in inputDf:
        if inputDf[i].dtype == "object":
            inputDf[i] = droppedDf[i].mode()[0]
        elif inputDf[i].dtype == "int64" or inputDf[i].dtype == "float64":
            inputDf[i] = droppedDf[i].mean()

    # Chuyển đổi dữ liệu loại "object" thành "category" cho mô hình GridSearchCrossValidation.
    obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
    for feature in obj_feat:
        inputDf[feature] = inputDf[feature].astype('category')

    # Load trọng số mô hình và dự đoán mục tiêu
    modelName = r"trained_model.model"
    loaded_model = pickle.load(open(modelName, 'rb'))

    # %% STREAMLIT FRONT-END 
    st.title("Dự đoán giá nhà")
    st.write("##### Đây là một mô hình đơn giản để dự đoán giá nhà.")

    st.sidebar.title("Tùy chỉnh thông số mô hình")
    
    expander= st.sidebar.expander("Các thuộc tính của mô hình")
    expander.write("## Các thuộc tính quan trọng")
    
    # Lấy những thuộc tính quan trọng của mô hình 
    # để làm tiêu chí dự đoán
    featureImportances = pd.Series(loaded_model.feature_importances_,index = droppedDf.columns).sort_values(ascending=False)[:20]
    
    inputDict = dict(inputDf)

    # Thêm các tiêu chí lựa chọn vào expander trên giao diện
    for idx, i in enumerate(featureImportances.index):
        if droppedDf[i].dtype == "object":
            variables = droppedDf[i].drop_duplicates().to_list()
            inputDict[i] = expander.selectbox(i, options=variables, key=idx)
        elif droppedDf[i].dtype == "int64" or droppedDf[i].dtype == "float64":
            inputDict[i] = expander.slider(i, ceil(droppedDf[i].min()),
                                                floor(droppedDf[i].max()), int(droppedDf[i].mean()), key=idx)
        else:
            expander.write(i)


    for key, value in inputDict.items():
        inputDf[key] = value

    obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
    for feature in obj_feat:
        inputDf[feature] = inputDf[feature].astype('category')

    # Dự đoán
    prediction = loaded_model.predict(inputDf)

    st.write("###### Giá dự đoán của ngôi nhà dựa vào các thuộc tính bạn đã chọn: $", prediction.item())

    st.markdown("------")

    st.write("###### Ngày: ", thedate)
    
st.set_page_config(page_title="Dự đoán", page_icon="📈")

app()
