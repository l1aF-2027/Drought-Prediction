import streamlit as st
from datetime import timedelta, date
import pydeck as pdk
import pandas as pd
import geopandas as gpd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Drought Monitor", page_icon="🌵", layout="wide")
# Thêm CSS cho sidebar cố định
st.markdown("""
<style>
    .css-1d391kg {
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        width: 14rem;
        overflow-y: auto;
        background-color: #f0f2f6;
        z-index: 100;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .css-18e3th9 {
        padding-left: 16rem !important;
    }
    
    .css-1lsmgbg, .block-container, header[data-testid="stHeader"] {
        padding-left: 2rem !important;
    }
    
    @media (max-width: 768px) {
        .css-1d391kg {
            width: 100%;
            height: auto;
            position: relative;
        }
        .css-18e3th9 {
            padding-left: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)



# Cache các hàm đọc dữ liệu để chỉ đọc một lần
@st.cache_data
def load_data():
    dict_map = pd.read_csv('https://huggingface.co/spaces/l1aF2027/Drought-Monitor/raw/main/src/data/dict_map.csv')
    counties_gdf = gpd.read_file('data/counties.geojson')
    counties_gdf['fips'] = counties_gdf['GEOID'].astype(str).str.zfill(5)
    dict_map['fips'] = dict_map['fips'].astype(str).str.zfill(5)
    dict_map['date'] = pd.to_datetime(dict_map['date'])
    return dict_map, counties_gdf

def get_drought_color(level=0):
    colors = {
        0: [166, 249, 166, 160],
        1: [255, 255, 190, 160],
        2: [255, 211, 127, 160],
        3: [255, 170, 0, 160],
        4: [230, 0, 0, 160],
        5: [115, 0, 0, 160]
    }
    return colors.get(level, [166, 249, 166, 160])

def get_drought_description(level=0):
    descriptions = {
        0: "Không có hạn hán",
        1: "Khô hạn nhẹ (D0)",
        2: "Hạn trung bình (D1)",
        3: "Hạn nghiêm trọng (D2)",
        4: "Hạn cực kỳ nghiêm trọng (D3)",
        5: "Hạn thảm khốc (D4)"
    }
    return descriptions.get(level, "Không xác định")

def get_drought_levels():
    return {
        0: "Không có hạn hán",
        1: "Khô hạn nhẹ (D0)",
        2: "Hạn trung bình (D1)",
        3: "Hạn nghiêm trọng (D2)",
        4: "Hạn cực kỳ nghiêm trọng (D3)",
        5: "Hạn thảm khốc (D4)"
    }

def get_previous_monday(date_val):
    days_to_subtract = (date_val.weekday() - 0) % 7
    if days_to_subtract == 0:
        return date_val
    else:
        return date_val - timedelta(days=days_to_subtract)

def calculate_accuracy(df):
    if df.empty:
        return 0.0
    correct_predictions = (df['y_pred_rounded'] == df['y_true_rounded']).sum()
    total_predictions = len(df)
    return (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

def prepare_map_data(current_data, counties_gdf, is_prediction=True):
    """Tiền xử lý dữ liệu cho bản đồ để tránh lặp lại mã"""
    if is_prediction:
        data = current_data.copy()
        data['drought_level'] = data['y_pred_rounded']
    else:
        data = current_data.copy()
        data['drought_level'] = data['y_true_rounded']
    
    # Áp dụng bộ lọc mức độ hạn hán nếu có lựa chọn
    if st.session_state.selected_drought_levels:
        data = data[data['drought_level'].isin(st.session_state.selected_drought_levels)]
    
    merged = counties_gdf.merge(data, on='fips', how='inner')
    if merged.empty:
        return None
    
    merged['line_width'] = 0
    merged['line_color'] = [[0, 0, 0, 0]] * len(merged)
    merged['opacity'] = 0.8  # Độ mờ mặc định cao hơn

    # Tô đậm viền vùng sai
    merged.loc[merged['is_prediction_wrong'], 'line_width'] = 2.0 if is_prediction else 0.5
    for idx in merged.index[merged['is_prediction_wrong']]:
        merged.at[idx, 'line_color'] = [0, 0, 0, 255]

    # Highlight FIPS được tìm
    if st.session_state.searched_fips:
        merged.loc[merged['fips'] == st.session_state.searched_fips, 'line_width'] = 4.0
        merged.loc[merged['fips'] == st.session_state.searched_fips, 'opacity'] = 1.0
        for idx in merged.index[merged['fips'] == st.session_state.searched_fips]:
            merged.at[idx, 'line_color'] = [255, 0, 0, 255]

    merged['color'] = merged['drought_level'].apply(get_drought_color)
    merged['drought_description'] = merged['drought_level'].apply(get_drought_description)
    merged = merged.to_crs('EPSG:4326')
    for col in merged.select_dtypes(include=['datetime64']).columns:
        merged[col] = merged[col].astype(str)
    
    return merged

def create_deck(merged_data, fips_location):
    """Tạo deck map để tránh lặp lại mã"""
    if merged_data is None:
        return None
        
    merged_json = json.loads(merged_data.to_json())
    view_state = pdk.ViewState(
        latitude=fips_location['latitude'] if fips_location else st.session_state.view_state['latitude'],
        longitude=fips_location['longitude'] if fips_location else st.session_state.view_state['longitude'],
        zoom=fips_location['zoom'] if fips_location else st.session_state.view_state['zoom'],
        pitch=0,
        bearing=0
    )
    
    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        data=merged_json,
        stroked=True,
        filled=True,
        get_fill_color="properties.color",
        get_line_color="properties.line_color",
        get_line_width="properties.line_width",
        get_opacity="properties.opacity",
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=[0, 0, 0, 50], 
    )
    tooltip = {
        "html": "<div><b>FIPS:</b> {fips}<br><b>Hạt:</b> {NAME}<br><b>Mức hạn hán:</b> {drought_description}</div>",
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "border": "1px solid black",
            "padding": "10px",
            "borderRadius": "5px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.15)"
        }
    }
    return pdk.Deck(
        layers=[polygon_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip=tooltip
    )

def create_statistics_chart(pred_counts, true_counts, all_levels):
    """Tạo biểu đồ thống kê để tránh lặp lại mã"""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Vị trí của các cột
    x = np.arange(len(all_levels))
    width = 0.35
    
    # Điều chỉnh số lượng theo mức độ
    pred_values = [pred_counts.get(level, 0) for level in all_levels]
    true_values = [true_counts.get(level, 0) for level in all_levels]
    
    # Vẽ cột
    rects1 = ax.bar(x - width/2, pred_values, width, label='Dự đoán', alpha=0.7, color='#3498db')
    rects2 = ax.bar(x + width/2, true_values, width, label='Thực tế', alpha=0.7, color='#e74c3c')
    
    # Thêm nhãn, tiêu đề và chú thích
    ax.set_xlabel('Mức độ hạn hán', fontsize=16)
    ax.set_ylabel('Số lượng vùng', fontsize=16)
    ax.set_title('Phân bố mức độ hạn hán', fontsize=20)
    ax.set_xticks(x)
    labels = ["Không hạn", "D0", "D1", "D2", "D3", "D4"]
    ax.set_xticklabels(labels, fontsize=16)
    ax.legend(fontsize=16)
    
    # Hiển thị số lượng trên các cột nếu giá trị > 0
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=16)
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    return fig

def create_confusion_matrix(filtered_data):
    """Tạo confusion matrix để tránh lặp lại mã"""
    # Xác định các nhãn dựa trên dữ liệu đã lọc
    unique_levels = sorted(set(filtered_data['y_true_rounded'].unique()) | 
                          set(filtered_data['y_pred_rounded'].unique()))
    
    if len(unique_levels) > 0:
        # Tạo confusion matrix với các nhãn phù hợp
        cm = confusion_matrix(
            filtered_data['y_true_rounded'], 
            filtered_data['y_pred_rounded'],
            labels=unique_levels
        )

        # Tạo nhãn đẹp
        label_names = ["0", "D0", "D1", "D2", "D3", "D4"]
        cm_labels = [label_names[level] if level < len(label_names) else f"Level {level}" for level in unique_levels]
        
        # Tạo bảng DataFrame có nhãn đẹp
        cm_df = pd.DataFrame(
            cm, 
            index=[f"Thực tế {label}" for label in cm_labels],
            columns=[f"Dự đoán {label}" for label in cm_labels]
        )

        # Tạo heatmap cho confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        plt.title('Confusion Matrix', fontsize=15)
        plt.tight_layout()
        
        return fig
    
    return None

# Khởi tạo session state
if 'view_state' not in st.session_state:
    st.session_state.view_state = {
        'latitude': 39.8283,
        'longitude': -98.5795,
        'zoom': 3.5,
        'pitch': 0,
        'bearing': 0
    }

if 'searched_fips' not in st.session_state:
    st.session_state.searched_fips = None

if 'hovered_fips' not in st.session_state:
    st.session_state.hovered_fips = None

if 'fips_location' not in st.session_state:
    st.session_state.fips_location = None

if 'selected_drought_levels' not in st.session_state:
    st.session_state.selected_drought_levels = []

# UI phần sidebar
with st.sidebar:
    selected_date = st.date_input(
        "Chọn ngày",
        date(2019, 1, 1),  
        min_value=date(2018, 6, 25),
        max_value=date(2020, 12, 31)
    )
    selected_weeks_to_show = st.number_input("Chọn số tuần dự đoán", value=1, min_value=1, max_value=6)
    search_fips = st.text_input("Nhập mã FIPS (5 chữ số)", "", help="FIPS(County) là mã định danh từng hạt của Mỹ")
    
    if search_fips:
        st.session_state.searched_fips = search_fips.zfill(5)
        st.info(f"Đang tìm FIPS: {st.session_state.searched_fips}")
    
    # Thêm nút để xóa kết quả tìm kiếm
    if not search_fips:
        st.session_state.searched_fips = None
        st.session_state.fips_location = None
    
    drought_levels = get_drought_levels()
    options = list(drought_levels.values())
    default = []  # Không chọn gì mặc định = hiển thị tất cả
    
    selected_drought_labels = st.multiselect(
        "Chọn mức độ hạn hán để hiển thị",
        options=options,
        default=default,
        help="Để trống sẽ hiển thị tất cả mức độ hạn hán"
    )
    
    # Chuyển đổi các nhãn đã chọn thành mã số
    selected_drought_level_codes = []
    for label in selected_drought_labels:
        for code, desc in drought_levels.items():
            if desc == label:
                selected_drought_level_codes.append(code)
    
    st.session_state.selected_drought_levels = selected_drought_level_codes
    
    st.header("Chú thích mức độ hạn hán")
    for level in range(6):
        st.markdown(
            f'<div style="display:flex;align-items:center;">'
            f'<div style="width:20px;height:20px;background-color:rgba{tuple(get_drought_color(level))};margin-right:10px;"></div>'
            f'<div>{get_drought_description(level)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# Phần nội dung chính
st.title("🌵 Drought Monitor")
st.markdown("Bản đồ trực quan hóa dữ liệu hạn hán dựa trên tọa độ địa lý")

dict_map, counties_gdf = load_data()

# Tìm FIPS một lần duy nhất
if st.session_state.searched_fips:
    fips_data = counties_gdf[counties_gdf['fips'] == st.session_state.searched_fips]
    if not fips_data.empty:
        fips_centroid = fips_data.geometry.centroid.iloc[0]
        st.session_state.fips_location = {
            'latitude': fips_centroid.y,
            'longitude': fips_centroid.x,
            'zoom': 8
        }
        st.success(f"Đã tìm thấy FIPS: {st.session_state.searched_fips}")
    else:
        st.error(f"Không tìm thấy FIPS: {st.session_state.searched_fips}")
        st.session_state.searched_fips = None
        st.session_state.fips_location = None

# Lấy vị trí FIPS từ session_state
fips_location = st.session_state.fips_location

# Tính ngày bắt đầu tuần một lần
target_monday = get_previous_monday(selected_date)

# Pre-filter data cho toàn bộ range tuần được chọn để cải thiện hiệu suất
start_date = target_monday
end_date = target_monday + timedelta(days=7*(selected_weeks_to_show-1))
date_range_data = dict_map[(dict_map['date'] >= pd.Timestamp(start_date)) & 
                            (dict_map['date'] <= pd.Timestamp(end_date))]

# Hiển thị từng tuần
for week in range(selected_weeks_to_show):
    current_date = target_monday + timedelta(days=7*week)
    last_date = current_date + timedelta(days=6)
    st.header(f"Tuần {week+1}: {current_date.strftime('%d/%m/%Y')} - {last_date.strftime('%d/%m/%Y')}")
    
    # Lọc dữ liệu cho tuần hiện tại từ dữ liệu đã được lọc trước đó (nhanh hơn)
    current_data = date_range_data[(date_range_data['date'] == pd.Timestamp(current_date)) & 
                                    (date_range_data['week'] == week)]
    
    if current_data.empty:
        st.warning(f"Không có dữ liệu cho ngày {current_date.strftime('%d/%m/%Y')} và tuần {week}")
        continue
    
    # Tạo bản sao dữ liệu gốc trước khi áp dụng bất kỳ bộ lọc nào
    unfiltered_data = current_data.copy()
    unfiltered_data['is_prediction_wrong'] = unfiltered_data['y_pred_rounded'] != unfiltered_data['y_true_rounded']
    
    # Đánh dấu các dự đoán đúng và sai
    current_data['is_prediction_wrong'] = current_data['y_pred_rounded'] != current_data['y_true_rounded']
    
    # Chuyển đổi date thành string một lần duy nhất
    current_data_str_date = current_data.copy()
    current_data_str_date['date'] = current_data_str_date['date'].dt.strftime('%Y-%m-%d')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bản đồ dự đoán")
        # Chuẩn bị dữ liệu bản đồ dự đoán
        merged_pred = prepare_map_data(current_data_str_date, counties_gdf, is_prediction=True)
        if merged_pred is not None:
            deck_pred = create_deck(merged_pred, fips_location)
            st.pydeck_chart(deck_pred)
        else:
            st.info("Không có dữ liệu phù hợp với các mức độ hạn hán đã chọn")
    
    with col2:
        st.subheader("Bản đồ thực tế")
        # Chuẩn bị dữ liệu bản đồ thực tế
        merged_true = prepare_map_data(current_data_str_date, counties_gdf, is_prediction=False)
        if merged_true is not None:
            deck_true = create_deck(merged_true, fips_location)
            st.pydeck_chart(deck_true)
        else:
            st.info("Không có dữ liệu phù hợp với các mức độ hạn hán đã chọn")
    
    # Tạo hai cột cho metrics và confusion matrix
    metrics_col, cm_col = st.columns(2)
    
    with metrics_col:
        # Hiển thị thống kê về các mức độ hạn hán
        st.subheader("Thống kê và đánh giá mô hình")
        
        # Tạo dữ liệu đã lọc theo drought levels (nếu có)
        if st.session_state.selected_drought_levels:
            # Lọc dữ liệu theo drought levels đã chọn (cho cả pred và true)
            filtered_data_for_metrics = current_data[
                (current_data['y_pred_rounded'].isin(st.session_state.selected_drought_levels)) | 
                (current_data['y_true_rounded'].isin(st.session_state.selected_drought_levels))
            ]
        else:
            # Nếu không có bộ lọc, sử dụng toàn bộ dữ liệu
            filtered_data_for_metrics = current_data
        
        # Chỉ hiển thị các chỉ số nếu có dữ liệu
        if not filtered_data_for_metrics.empty:
            # Tính accuracy dựa trên dữ liệu đã lọc
            accuracy = calculate_accuracy(filtered_data_for_metrics)
            
            # Hiển thị accuracy với màu sắc phù hợp
            accuracy_color = "#2f855a" if accuracy >= 80 else "#c05621" if accuracy >= 50 else "#c53030"
            
            # Nếu đang áp dụng bộ lọc, cũng tính accuracy của dữ liệu không lọc để so sánh
            if st.session_state.selected_drought_levels:
                unfiltered_accuracy = calculate_accuracy(unfiltered_data)
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h5 style="margin: 0; color: black;">Accuracy (đã lọc): <span style="color: {accuracy_color};">{accuracy:.2f}%</span></h5>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h5 style="margin: 0; color: black;">Accuracy: <span style="color: {accuracy_color};">{accuracy:.2f}%</span></h5>
                </div>
                """, unsafe_allow_html=True)
        
        # Chuẩn bị dữ liệu cho biểu đồ
        # Lấy dữ liệu cho bản đồ dự đoán và thực tế
        pred_data = current_data_str_date.copy()
        pred_data['drought_level'] = pred_data['y_pred_rounded']
        true_data = current_data_str_date.copy()
        true_data['drought_level'] = true_data['y_true_rounded']
        
        # Áp dụng bộ lọc
        if st.session_state.selected_drought_levels:
            pred_data = pred_data[pred_data['drought_level'].isin(st.session_state.selected_drought_levels)]
            true_data = true_data[true_data['drought_level'].isin(st.session_state.selected_drought_levels)]
        
        # Lấy số lượng từ dữ liệu
        if not pred_data.empty:
            pred_counts = pred_data['drought_level'].value_counts().to_dict()
        else:
            pred_counts = {}
        
        if not true_data.empty:
            true_counts = true_data['drought_level'].value_counts().to_dict()
        else:
            true_counts = {}
        
        # Tạo biểu đồ
        if not (pred_data.empty and true_data.empty):
            all_levels = range(6)
            fig = create_statistics_chart(pred_counts, true_counts, all_levels)
            st.pyplot(fig)
        else:
            st.info("Không có dữ liệu phù hợp với bộ lọc để hiển thị biểu đồ")
    
    with cm_col:
        # Áp dụng bộ lọc cho confusion matrix
        if st.session_state.selected_drought_levels:
            # Lọc dữ liệu theo drought levels đã chọn (cho cả pred và true)
            filtered_data_for_cm = current_data[
                (current_data['y_pred_rounded'].isin(st.session_state.selected_drought_levels)) | 
                (current_data['y_true_rounded'].isin(st.session_state.selected_drought_levels))
            ]
        else:
            # Nếu không có bộ lọc, sử dụng toàn bộ dữ liệu
            filtered_data_for_cm = current_data
        
        # Chỉ hiển thị các chỉ số nếu có dữ liệu
        if not filtered_data_for_cm.empty:
            # Tạo confusion matrix
            st.subheader("Confusion Matrix")
            cm_fig = create_confusion_matrix(filtered_data_for_cm)
            if cm_fig:
                st.pyplot(cm_fig)
            else:
                st.warning("Không đủ dữ liệu để tạo confusion matrix với bộ lọc hiện tại")
        else:
            st.warning("Không có dữ liệu phù hợp với bộ lọc để tính các chỉ số")

# Thêm CSS cho hiệu ứng làm mờ
st.markdown("""
<style>
.highlight-region {
    filter: none !important;
}
.blur-region {
    filter: blur(2px) opacity(0.7);
    transition: filter 0.3s ease;
}
</style>
""", unsafe_allow_html=True)