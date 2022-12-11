import streamlit as st
from streamlit.logger import get_logger
import streamlit.components.v1 as components
import datetime

LOGGER = get_logger(__name__)

thedate = datetime.date.today()
def run():
    st.image(r'./resources/enter_page_image.jpg', use_column_width=True)

    st.write("""
    # Đồ án Học máy: Dự đoán giá nhà
    """) 

    st.markdown(
        """
    "Dự đoán về giá nhà", xây dựng trên Streamlit framework, được phát triển bằng cách sử dụng bộ dữ liệu Kaggle 'House Prices - Advanced Regression Techniques'.
    
    
    ------

    ###### Sinh viên thực hiện:

    Đỗ Duy Nhựt - 20110298

    Ngô Vũ Nhật Nguyên

    Hồ Hà Thanh Lâm - 20110667

    """
    )
    st.write("###### Ngày: ", thedate)
                                        
                                          
                            

if __name__ == "__main__":
    run()