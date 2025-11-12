"""多次考试成绩 & 排名可视化看板

特性：
1. 支持多 Excel 同时上传 (accept_multiple_files=True)
2. 可自定义排序（考试时间 / 批次顺序）与自定义考试标签
3. 单学生：
   - 各科分数柱状图（来自最新一次考试）
   - 多次考试总分排名变化折线图（校次排名）
   - 多次考试雷达图对比：展示各科校次排名（可选多次考试叠加）
4. 多学生：可选择若干学生对比总分排名变化折线图

假设：
- 原始列模式：准考证号, 班级, 姓名, 总分, 总分校次, 总分班次, 语文, 语文校次, 语文班次, 数学, 数学校次, 数学班次, ...
- 实际文件可能只有一部分列，脚本按已知顺序重命名前若干列。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import hashlib
import io

try:
    # 可选：拖拽排序支持
    from streamlit_sortables import sort_items  # type: ignore
    HAS_SORTABLES = True
except Exception:
    HAS_SORTABLES = False

st.set_page_config(page_title="成绩可视化看板", layout="wide")
st.title("📊 成绩可视化看板")
st.markdown("**By Ga1axy v1.0**")

# ====== 打印优化：注入防分页 CSS ======
print_css = """
<style>
/* 避免图表在浏览器打印时被分页拆分 */
@media print {
    .block-container {padding-top: 0 !important;}
    /* 每个 st 元素外层容器 */
    div[data-testid="stVerticalBlock"] > div {page-break-inside: avoid;}
    /* Plotly 图表容器 */
    .js-plotly-plot, .plotly-graph-div {page-break-inside: avoid !important;}
    /* 通用卡片/表格 */
    .stDataFrame, .stTable {page-break-inside: avoid !important;}
    /* 移除交互控件在打印时的多余空白 */
    .stButton, .stCheckbox, .stTextInput, .stSelectbox, .stMultiSelect {page-break-inside: avoid !important;}
    /* 让页面背景为白色 */
    body { -webkit-print-color-adjust: exact; print-color-adjust: exact; background: #ffffff; }
}
/* 限制最大宽度，保证打印居中 */
@page { size: A4 portrait; margin: 10mm; }
</style>
"""
st.markdown(print_css, unsafe_allow_html=True)

# =====================================
# 配置 & 常量
# =====================================
DEFAULT_SUBJECTS = ["语文", "数学", "英语", "政治", "历史", "地理"]
# 复合科目（只提供排名，无单独分数）
COMPOSITE_SUBJECTS = ["语数英", "7选3"]
# 可选扩展科目列表（加入复合科目作为可选项）
ALL_SUBJECT_OPTIONS = DEFAULT_SUBJECTS + ["物理", "化学", "生物", "技术"] + COMPOSITE_SUBJECTS

# ========= 通用数值格式化 =========
def _fmt_one_decimal(v):
    """若有小数保留一位；若为整数则不显示小数；空值返回空串。"""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, (np.floating,)) and np.isnan(v)):
        return ""
    try:
        f = float(v)
    except Exception:
        return str(v)
    f1 = round(f, 1)
    if float(f1).is_integer():
        return str(int(f1))
    return f"{f1:.1f}"

@st.cache_data(show_spinner=False)
def standardize_columns(df: pd.DataFrame, subjects: List[str]) -> pd.DataFrame:
    """将“宁外期末表头”格式映射为统一列名。

    输入表头示例（第一行是考试标签，已在读取时剔除）：
    班级 学号 姓名 语文 语班 语年 数学 数班 数年 英语 英班 英年 物赋 物班 物年 化赋 化班 化年 生赋 生班 生年 政赋 政班 政年 史赋 史班 史年 地赋 地班 地年 技赋 技班 技年 语数外 班排 年排 7选3 班排 年排 总分 班级排名 年级排名

    目标统一列：
    - 基础：班级, 准考证号, 姓名, 总分, 总分_班次(班级排名), 总分_校次(年级排名)
    - 学科：<科目, 科目_班次, 科目_校次>
    """
    df2 = df.copy()

    # 列名去空白
    df2.columns = [str(c).strip() for c in df2.columns]

    # 基础字段映射
    base_map = {
        "班级": "班级",
        "学号": "准考证号",
        "姓名": "姓名",
        "总分": "总分",
        "班级排名": "总分_班次",
        "年级排名": "总分_校次",
    }

    # 学科字段映射（分数/班排/年排）
    # 注意：表中理化生政史地技的分数字段使用“赋”字样
    subject_source_map = {
        "语文": ("语文", "语班", "语年"),
        "数学": ("数学", "数班", "数年"),
        "英语": ("英语", "英班", "英年"),
        "物理": ("物赋", "物班", "物年"),
        "化学": ("化赋", "化班", "化年"),
        "生物": ("生赋", "生班", "生年"),
        "政治": ("政赋", "政班", "政年"),
        "历史": ("史赋", "史班", "史年"),
        "地理": ("地赋", "地班", "地年"),
        "技术": ("技赋", "技班", "技年"),
    }

    rename_map: Dict[str, str] = {}
    # 应用基础映射（存在才映射）
    for src, dst in base_map.items():
        if src in df2.columns:
            rename_map[src] = dst

    # 应用学科映射
    for std_subj, (score_col, cls_rank_col, grd_rank_col) in subject_source_map.items():
        if score_col in df2.columns:
            rename_map[score_col] = std_subj
        if cls_rank_col in df2.columns:
            rename_map[cls_rank_col] = f"{std_subj}_班次"
        if grd_rank_col in df2.columns:
            rename_map[grd_rank_col] = f"{std_subj}_校次"

    # 复合科目：语数英（源：语数外 班排/年排），7选3（源：7选3 班排/年排）
    # 兼容“无空格”和“有空格”两种写法
    composite_candidates = [
        ("语数英", ["语数英", "语数外"], ["班排", "年排"]),
        ("7选3", ["7选3", "七选三"], ["班排", "年排"]),
    ]

    def _col_exists(*names: str) -> Optional[str]:
        for n in names:
            if n in df2.columns:
                return n
        return None

    for std_name, bases, rank_words in composite_candidates:
        # 尝试匹配：<base>班排 或 "<base> 班排"
        # 年排同理
        for base in bases:
            cls_variants = [f"{base}班排", f"{base} 班排"]
            grd_variants = [f"{base}年排", f"{base} 年排"]
            # 也兼容“<base> 班级排名/年级排名”的极端情况
            cls_variants.extend([f"{base}班级排名", f"{base} 班级排名"])
            grd_variants.extend([f"{base}年级排名", f"{base} 年级排名"])
            cls_col = _col_exists(*cls_variants)
            grd_col = _col_exists(*grd_variants)
            if cls_col:
                rename_map[cls_col] = f"{std_name}_班次"
            if grd_col:
                rename_map[grd_col] = f"{std_name}_校次"

    df2 = df2.rename(columns=rename_map)

    # 仅保留分析所需列，去除未映射的原始列（如“语数外/7选3 的 班排/年排”等），避免重复列名导致 concat 失败
    # 构建保留列：基础 + 所有学科与复合科目的存在列（分数/班次/校次任一存在即可）
    all_std_subjects = list(subject_source_map.keys()) + ["语数英", "7选3"]
    keep_cols_order = [
        "班级", "准考证号", "姓名", "总分", "总分_班次", "总分_校次",
    ]
    for s in all_std_subjects:
        # 分数列不一定存在（如复合科目通常只有排名），因此分别判断
        if s in df2.columns:
            keep_cols_order.append(s)
        if f"{s}_班次" in df2.columns:
            keep_cols_order.append(f"{s}_班次")
        if f"{s}_校次" in df2.columns:
            keep_cols_order.append(f"{s}_校次")
    # 实际存在的列
    keep_cols_order = [c for c in keep_cols_order if c in df2.columns]
    if keep_cols_order:
        df2 = df2[keep_cols_order].copy()

    # 数值化：分数与排名字段
    numeric_like_cols = [c for c in df2.columns if (c == "总分" or c.endswith("_班次") or c.endswith("_校次") or c in list(subject_source_map.keys()))]

    def _coerce_numeric(val):
        if isinstance(val, bytes):
            try:
                val = val.decode("utf-8", "ignore")
            except Exception:
                return pd.NA
        return val

    for c in numeric_like_cols:
        if c in df2.columns:
            try:
                df2[c] = pd.to_numeric(df2[c].map(_coerce_numeric), errors="coerce")
            except Exception:
                pass

    # 班级/姓名等转字符串，避免分类问题
    for c in ["班级", "姓名", "准考证号"]:
        if c in df2.columns:
            try:
                df2[c] = df2[c].astype(str)
            except Exception:
                pass

    return df2

def _read_excel_bytes(file_obj) -> bytes:
    """将上传对象读取为 bytes，支持 Streamlit UploadedFile/文件句柄/bytes。"""
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue()
    if hasattr(file_obj, "read"):
        return file_obj.read()
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    # 兜底：当传入为路径字符串时
    with open(file_obj, "rb") as f:
        return f.read()

def _parse_label_and_dataframe(xlsx_bytes: bytes) -> Tuple[pd.DataFrame, Optional[str]]:
    """从 Excel bytes 中抽取考试标签与数据表。

    规则：
    - 第一行（索引0）为考试标签（可能是合并单元格，取该行所有非空单元格拼接）
    - 找到包含“班级”和“姓名”的行作为表头行，从下一行开始为数据
    """
    raw = pd.read_excel(io.BytesIO(xlsx_bytes), header=None)
    exam_label: Optional[str] = None
    if not raw.empty:
        first_row_vals = [str(x).strip() for x in raw.iloc[0].tolist() if pd.notna(x) and str(x).strip() != "nan"]
        if first_row_vals:
            exam_label = " ".join(first_row_vals)

    # 寻找表头行（包含关键列）
    header_row_idx = None
    for i in range(min(len(raw), 10)):  # 前10行内搜寻
        row_vals = [str(x).strip() for x in raw.iloc[i].tolist()]
        if ("班级" in row_vals) and ("姓名" in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        # 回退：假设第1行为标签，第2行为表头
        header_row_idx = 1 if len(raw) > 1 else 0

    header_vals = [str(x).strip() for x in raw.iloc[header_row_idx].tolist()]
    data = raw.iloc[header_row_idx + 1 :].copy()
    data.columns = header_vals
    # 丢弃全空列
    data = data.loc[:, ~(data.isna().all())]
    # 丢弃全空行
    data = data.dropna(how="all").reset_index(drop=True)
    return data, exam_label

def build_exam_dataframe(file, fallback_label: str, order: int, subjects: List[str]) -> pd.DataFrame:
    """读取单个 Excel，使用第一行作为考试标签；如缺失则回退为传入标签。"""
    xbytes = _read_excel_bytes(file)
    data_df, label_in_file = _parse_label_and_dataframe(xbytes)
    df_std = standardize_columns(data_df, subjects)
    df_std["考试标签"] = label_in_file or fallback_label
    df_std["考试顺序"] = order
    # ===== 复合科目分数与排名自动计算 =====
    df_std = _compute_composite_scores_and_ranks(df_std)
    return df_std

def _compute_composite_scores_and_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """为单次考试数据增加 语数英 与 7选3 的分数及班/校排名（若尚不存在）。
    语数英 = 语文 + 数学 + 英语（存在则求和）
    7选3 = 在 [物理, 化学, 生物, 政治, 历史, 地理, 技术] 中取分数最高的 3 科求和（>=3 科才计算，否则求和可用科目）
    排名：descending 分数越高排名越靠前，使用 method='min' 获得稳定名次。
    """
    required_cols_triple = ["语文", "数学", "英语"]
    has_triple = all(c in df.columns for c in required_cols_triple)
    if "语数英" not in df.columns and has_triple:
        df["语数英"] = df[required_cols_triple].sum(axis=1, min_count=1)
    # 7选3计算
    elective_cols = [c for c in ["物理", "化学", "生物", "政治", "历史", "地理", "技术"] if c in df.columns]
    if "7选3" not in df.columns and elective_cols:
        def _top3_sum(row):
            vals = [row[c] for c in elective_cols if pd.notna(row[c])]
            if not vals:
                return np.nan
            vals_sorted = sorted(vals, reverse=True)
            if len(vals_sorted) >= 3:
                return sum(vals_sorted[:3])
            return sum(vals_sorted)  # 不足3科则求和全部
        df["7选3"] = df.apply(_top3_sum, axis=1)
    # 班级 / 年级排名（按考试标签分组）
    if "考试标签" in df.columns:
        # 语数英排名
        if "语数英" in df.columns:
            if "语数英_校次" not in df.columns:
                df["语数英_校次"] = df.groupby("考试标签")["语数英"].rank(method="min", ascending=False)
            if "语数英_班次" not in df.columns and "班级" in df.columns:
                df["语数英_班次"] = df.groupby(["考试标签", "班级"])["语数英"].rank(method="min", ascending=False)
        # 7选3排名
        if "7选3" in df.columns:
            if "7选3_校次" not in df.columns:
                df["7选3_校次"] = df.groupby("考试标签")["7选3"].rank(method="min", ascending=False)
            if "7选3_班次" not in df.columns and "班级" in df.columns:
                df["7选3_班次"] = df.groupby(["考试标签", "班级"])["7选3"].rank(method="min", ascending=False)
    return df

def extract_exam_label_from_file(file) -> Optional[str]:
    """仅抽取考试标签（第一行）。"""
    try:
        xbytes = _read_excel_bytes(file)
        raw = pd.read_excel(io.BytesIO(xbytes), header=None, nrows=1)
        if raw.empty:
            return None
        vals = [str(x).strip() for x in raw.iloc[0].tolist() if pd.notna(x) and str(x).strip() != "nan"]
        return " ".join(vals) if vals else None
    except Exception:
        return None

def extract_rank_time_series(all_df: pd.DataFrame, subjects: List[str]) -> pd.DataFrame:
    """提取所有学生所有考试的总分及科目校次排名 (长表)。"""
    rank_cols = ["总分_校次"] + [f"{s}_校次" for s in subjects if f"{s}_校次" in all_df.columns]
    cols_needed = ["姓名", "考试标签", "考试顺序"] + rank_cols
    exist_cols = [c for c in cols_needed if c in all_df.columns]
    long_df = all_df[exist_cols].melt(id_vars=["姓名", "考试标签", "考试顺序"], var_name="项目", value_name="校次排名")
    return long_df

def transform_rank_for_radar(sub_df: pd.DataFrame) -> pd.DataFrame:
    """雷达图希望“面积越大越好”，将排名(名次越小越好)反转归一。
    简单策略：value = max_rank + 1 - rank
    """
    # 只针对数值行
    sub_df_num = sub_df.dropna(subset=["校次排名"]).copy()
    if sub_df_num.empty:
        return sub_df
    sub_df["雷达值"] = sub_df["校次排名"].apply(lambda x: (grade_total + 1 - x) if pd.notna(x) else None)
    return sub_df

# =====================================
# 侧边栏：上传与排序
# =====================================
st.sidebar.header("⚙️ 数据与排序设置")
uploaded_files = st.sidebar.file_uploader("上传多个考试 Excel 文件", type=["xlsx"], accept_multiple_files=True)
st.sidebar.write("---")

subjects = st.sidebar.multiselect(
    "**选择参与分析的科目**",
    options=ALL_SUBJECT_OPTIONS,
    default=DEFAULT_SUBJECTS,
    help="未选中的科目将不会出现在后续图表与表格中。"
)
grade_total = st.sidebar.number_input(
    "年级总人数",
    min_value=1,
    max_value=5000,
    value=560,
    step=1,
    help="输入当前年级的学生总人数，可用于后续添加排名百分比等指标。"
)
st.sidebar.write("---")


# 分数图Y轴起点自动调整设置
st.sidebar.write("**调整Y轴**")

auto_y_start = st.sidebar.checkbox("分数图自动调整Y轴起点", value=True, help="开启后，分数类柱状图的Y轴将从接近最小分数处开始，以放大差异。")
offset_y = st.sidebar.number_input("Y轴起点下移幅度", min_value=0, max_value=100, value=10, step=1, help="在最小分数基础上再下移的幅度。仅当启用自动调整时生效。")
st.sidebar.write("---")

if uploaded_files:
    # 收集导出图表
    export_figs: Dict[str, go.Figure] = {}
    # 构造排序/标签编辑表
    meta_rows = []
    for idx, f in enumerate(uploaded_files, start=1):
        file_label = extract_exam_label_from_file(f) or f.name.rsplit('.', 1)[0]
        meta_rows.append({"文件名": f.name, "默认顺序": idx, "自定义顺序": idx, "考试标签": file_label})
    meta_df = pd.DataFrame(meta_rows)
    # 新增“可视”布尔列，默认全部可见
    if "可视" not in meta_df.columns:
        meta_df["可视"] = True
    # 基于当前上传文件名生成稳定摘要，用于重置拖拽/编辑组件状态（当增删文件时重建控件）
    names_for_hash = [f.name for f in uploaded_files]
    files_digest = hashlib.md5("|".join(sorted(names_for_hash)).encode("utf-8")).hexdigest()

    # 基础表（先根据拖拽更新顺序，再渲染编辑表）
    work_meta = meta_df.copy()

    # 拖拽排序（可选）
    if HAS_SORTABLES:
        with st.sidebar:
            st.markdown("**拖拽排序**：拖动下列项目改变顺序，从上到下为考试时间顺序")
            # 仅使用考试标签作为拖拽项，并按当前“自定义顺序”显示
            items = [f"{row['考试标签']}" for _, row in work_meta.sort_values("自定义顺序").iterrows()]
            try:
                # 将文件列表摘要纳入 key，确保当文件增删时，拖拽组件会刷新
                sorted_items = sort_items(items, direction="vertical", key=f"exam_drag_order_{files_digest}")
                # 基于“考试标签”构建 新顺序映射：标签 -> 顺序编号
                def _extract_label(s: str) -> str:
                    return str(s)
                new_order_map = { _extract_label(name): idx + 1 for idx, name in enumerate(sorted_items) }
                # 根据“考试标签”写入新的“自定义顺序”
                work_meta["自定义顺序"] = work_meta["考试标签"].map(new_order_map).fillna(work_meta["自定义顺序"]).astype(int)
            except Exception as e:
                st.info(f"拖拽排序组件不可用，已回退为表格内手动输入顺序。({e})")
    else:
        st.sidebar.caption("如需拖拽排序，请安装 streamlit-sortables，并重启应用。")

    st.sidebar.write("---")
    # 渲染可编辑表（已经按当前顺序排序后展示）
    # 同理，为编辑表设置动态 key，列表变化时重建编辑控件
    # 侧边栏只显示：考试标签 + 可视（顺序基于拖拽后的 自定义顺序）
    st.sidebar.write("**可视选项**：在下表中可查看考试展示顺序编辑可视状态。")
    simplified_df = work_meta.sort_values("自定义顺序")["考试标签 可视".split()]
    edited_meta = st.sidebar.data_editor(
        simplified_df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"meta_editor_{files_digest}"
    )
    # 将可视状态写回 work_meta
    try:
        visibility_map = dict(zip(edited_meta["考试标签"], edited_meta["可视"]))
        work_meta["可视"] = work_meta["考试标签"].map(visibility_map).fillna(True)
    except Exception:
        pass

    # 根据自定义顺序排序
    # 使用 work_meta（已写回可视状态）继续；自定义顺序来自拖拽结果
    try:
        edited_meta_sorted = work_meta.sort_values("自定义顺序")
    except Exception:
        edited_meta_sorted = work_meta

    visible_meta = edited_meta_sorted[edited_meta_sorted["可视"].fillna(True)] if "可视" in edited_meta_sorted.columns else edited_meta_sorted

    label_map: Dict[str, Dict] = {row["文件名"]: {"标签": row["考试标签"], "顺序": row["自定义顺序"]} for _, row in visible_meta.iterrows()}
    # 供图表使用的考试标签顺序（仅来自可视的考试）
    exam_label_order = list(visible_meta["考试标签"].astype(str).values)

    # 组合所有考试数据
    exam_dfs = []
    visible_files = set(visible_meta["文件名"].astype(str).tolist())
    for f in uploaded_files:
        if f.name in visible_files:
            info = label_map[f.name]
            exam_dfs.append(build_exam_dataframe(f, info["标签"], info["顺序"], subjects))
    if not exam_dfs:
        st.warning("所有考试均被设为不可视，暂无数据。请在侧边栏勾选‘可视’后继续。")
        st.stop()
    all_exams_df = pd.concat(exam_dfs, ignore_index=True)
    all_exams_df.sort_values(["考试顺序"], inplace=True)

    # ================= 班级筛选 =================
    st.sidebar.write("---")
    if "班级" in all_exams_df.columns:
        classes = sorted([c for c in all_exams_df["班级"].dropna().astype(str).unique()])
    else:
        classes = []
    if classes:
        selected_classes = st.sidebar.multiselect("**筛选班级**", classes, default=classes)
        filtered_df = all_exams_df[all_exams_df["班级"].astype(str).isin(selected_classes)] if selected_classes else all_exams_df.iloc[0:0]
    else:
        filtered_df = all_exams_df

    # ================= 单学生选择 =================
    all_students = sorted(filtered_df["姓名"].dropna().unique())
    if not all_students:
        st.warning("未检测到任何学生姓名，请检查列名或文件内容。")
        st.stop()

    col_a, col_b = st.columns([1,1])
    with col_a:
        student_name = st.selectbox("**选择单个学生**", all_students)
    # 折线图对比选项已移动到“排名时间序列”的科目选择下方

   
    # ================= 该学生全部考试成绩明细 =================
    st.subheader("📄 该学生全部考试明细")
    score_cols_exist = (["总分"] if "总分" in filtered_df.columns else []) + [c for c in subjects if c in filtered_df.columns]
    # 单科与总分区分开：明细表仍可同时展示
    score_cols_subjects_only = [c for c in subjects if c in filtered_df.columns]
    score_cols_exist = (["总分"] if "总分" in filtered_df.columns else []) + score_cols_subjects_only
    if score_cols_exist:
        score_long = filtered_df[filtered_df["姓名"] == student_name][["考试标签", "考试顺序", "姓名"] + score_cols_exist].copy()
        score_long["考试标签"] = pd.Categorical(score_long["考试标签"], categories=exam_label_order, ordered=True)
        score_pivot = score_long.sort_values("考试顺序").set_index("考试标签")[score_cols_exist]
        # 静态格式化：一位小数（仅需要时）
        score_pivot_fmt = score_pivot.applymap(lambda v: ("" if pd.isna(v) else (str(int(round(float(v),1))) if round(float(v),1).is_integer() else f"{round(float(v),1):.1f}")) if pd.api.types.is_number(v) else v)
        st.table(score_pivot_fmt)
    else:
        st.info("无科目成绩列可供展示。")
    st.markdown("---")

    # ================= 单学生所有排名明细表 =================
    ts_long = extract_rank_time_series(filtered_df, subjects)
    # 使用考试标签作为列，更直观
    student_all_raw = ts_long[ts_long["姓名"] == student_name].copy()
    # 保持标签顺序
    student_all_raw["考试标签"] = pd.Categorical(student_all_raw["考试标签"], categories=exam_label_order, ordered=True)
    student_all = student_all_raw.pivot_table(index=["项目"], columns="考试标签", values="校次排名")
    # 重新排序行：总分优先，其次各科
    desired_rows = ["总分_校次"] + [f"{s}_校次" for s in subjects if f"{s}_校次" in student_all.index]
    student_all = student_all.reindex(desired_rows)
    student_all_fmt = student_all.applymap(lambda v: ("" if pd.isna(v) else (str(int(round(float(v),1))) if round(float(v),1).is_integer() else f"{round(float(v),1):.1f}")) if pd.api.types.is_number(v) else v)
    st.table(student_all_fmt)

    # ================= 排名时间序列 =================
    # 可选择查看的项目：总分 或 各科（基于存在的“*_校次”项目）
    proj_keys = [p for p in ts_long["项目"].dropna().unique().tolist() if isinstance(p, str) and p.endswith("_校次")]
    # 将内部键映射为展示名（总分_校次 -> 总分；语文_校次 -> 语文）
    def _proj_disp(k: str) -> str:
        return "总分" if k == "总分_校次" else k.replace("_校次", "")
    options_disp = [_proj_disp(k) for k in proj_keys]
    # 为了稳定顺序，按照 subjects 与“总分”优先的顺序重排
    ordered_keys = []
    if "总分_校次" in proj_keys:
        ordered_keys.append("总分_校次")
    for s in subjects:
        k = f"{s}_校次"
        if k in proj_keys and k not in ordered_keys:
            ordered_keys.append(k)
    # 补充任何未覆盖的键
    for k in proj_keys:
        if k not in ordered_keys:
            ordered_keys.append(k)
    ordered_disp = [_proj_disp(k) for k in ordered_keys]
    default_disp = "总分" if "总分_校次" in ordered_keys else (ordered_disp[0] if ordered_disp else "")
    selected_disp = st.multiselect(
        "选择查看项目（总分或科目）(可多选)",
        ordered_disp,
        default=([default_disp] if default_disp else []),
        help="可选择一个或多个项目进行折线对比"
    ) if ordered_disp else []
    # 将“折线图对比多个学生”的选项移动至此（紧跟科目/总分选择）
    # ---- 同步多学生对比选择逻辑（避免 default 与 session_state 同时设置冲突） ----
    # 原因：当使用固定 key 时，若在 Session State 中已经设置了该 key 的值，同时又在组件上提供了 default，会触发冲突提示。
    # 方案：
    #  - 首次渲染：仅通过 default 提供初值（不预先写 session_state[list_key]）。
    #  - 之后渲染：如需调整，先更新 session_state[list_key]，再创建组件且不传 default。
    anchor_key = "_anchor_student_for_multiline"
    list_key = "multi_students_for_line"

    if list_key in st.session_state:
        # 已初始化过：根据当前主学生与可选项动态维护列表
        if st.session_state.get(anchor_key) != student_name:
            st.session_state[list_key] = [student_name]
            st.session_state[anchor_key] = student_name
        else:
            # 清理掉已经不在候选中的学生
            st.session_state[list_key] = [s for s in st.session_state[list_key] if s in all_students]
            # 确保主学生在列表中
            if student_name not in st.session_state[list_key]:
                st.session_state[list_key].insert(0, student_name)

        multi_students = st.multiselect(
            "折线图对比多个学生 (可选)",
            all_students,
            key=list_key,
            help="当上面选择的主学生改变时，此列表会自动同步包含该学生。"
        )
    else:
        # 首次渲染：通过 default 设置初值，同时记录锚定学生。
        st.session_state[anchor_key] = student_name
        multi_students = st.multiselect(
            "折线图对比多个学生 (可选)",
            all_students,
            default=[student_name],
            key=list_key,
            help="当上面选择的主学生改变时，此列表会自动同步包含该学生。"
        )
    # 新增：选择查看内容（分数 / 校次排名 / 班次排名）
    view_options = ["校次排名", "班次排名", "分数"]
    view_choice = st.selectbox("选择查看内容", view_options, index=0, key="series_view_type")

    fig_line = go.Figure()
    if selected_disp:
        # 根据查看内容选择数据列与来源（支持多项目）
        y_label = ""
        reverse_y = False
        if view_choice == "校次排名":
            selected_keys = [("总分_校次" if disp == "总分" else f"{disp}_校次") for disp in selected_disp]
            df_tmp = ts_long[ts_long["项目"].isin(selected_keys)].copy()
            df_tmp = df_tmp[df_tmp["姓名"].isin(multi_students)]
            df_tmp["值"] = df_tmp["校次排名"]
            df_tmp["项目显示名"] = df_tmp["项目"].apply(lambda k: "总分" if k == "总分_校次" else str(k).replace("_校次", ""))
            line_df = df_tmp[["考试标签", "考试顺序", "姓名", "项目显示名", "值"]]
            y_label = "校次排名"
            reverse_y = True
        elif view_choice == "班次排名":
            selected_cols = [("总分_班次" if disp == "总分" else f"{disp}_班次") for disp in selected_disp]
            exist_cols = [c for c in selected_cols if c in filtered_df.columns]
            if exist_cols:
                df_tmp = filtered_df[["考试标签", "考试顺序", "姓名"] + exist_cols].copy()
                df_tmp = df_tmp[df_tmp["姓名"].isin(multi_students)]
                melted = df_tmp.melt(id_vars=["考试标签", "考试顺序", "姓名"], value_vars=exist_cols, var_name="项目", value_name="值")
                melted["项目显示名"] = melted["项目"].apply(lambda k: "总分" if k == "总分_班次" else str(k).replace("_班次", ""))
                line_df = melted[["考试标签", "考试顺序", "姓名", "项目显示名", "值"]]
            else:
                line_df = pd.DataFrame(columns=["考试标签", "考试顺序", "姓名", "项目显示名", "值"])  # 空
            y_label = "班次排名"
            reverse_y = True
        else:  # 分数
            selected_cols = [("总分" if disp == "总分" else disp) for disp in selected_disp]
            exist_cols = [c for c in selected_cols if c in filtered_df.columns]
            if exist_cols:
                df_tmp = filtered_df[["考试标签", "考试顺序", "姓名"] + exist_cols].copy()
                df_tmp = df_tmp[df_tmp["姓名"].isin(multi_students)]
                melted = df_tmp.melt(id_vars=["考试标签", "考试顺序", "姓名"], value_vars=exist_cols, var_name="项目显示名", value_name="值")
                line_df = melted[["考试标签", "考试顺序", "姓名", "项目显示名", "值"]]
            else:
                line_df = pd.DataFrame(columns=["考试标签", "考试顺序", "姓名", "项目显示名", "值"])  # 空
            y_label = "分数"
            reverse_y = False

        # 使用考试标签作为 X 轴，但保持顺序
        if not line_df.empty:
            line_df["考试标签"] = pd.Categorical(line_df["考试标签"], categories=exam_label_order, ordered=True)

        # 生成图或提示
        if line_df.empty:
            st.warning("所选学生/科目没有可用的数据。")
        else:
            line_df = line_df.copy()
            line_df["显示值"] = line_df["值"].apply(_fmt_one_decimal)
            # 同时按学生与项目区分曲线
            fig_line = px.line(
                line_df.sort_values("考试顺序"),
                x="考试标签",
                y="值",
                # 颜色区分科目/项目（含“总分”）
                color="项目显示名",
                # 线型区分学生
                line_dash="姓名",
                # 同步使用符号区分学生，提升辨识度
                symbol="姓名",
                text="显示值",
                markers=True,
                category_orders={"考试标签": exam_label_order},
                title=f"{','.join(selected_disp)} {y_label}变化"
            )
            fig_line.update_traces(mode="lines+markers+text", texttemplate="%{text}", textposition="top center")
            if reverse_y:
                fig_line.update_yaxes(autorange="reversed")
            export_figs[f"{','.join(selected_disp)} {y_label}变化折线图"] = fig_line
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
    # ================= 雷达图（各科校次排名对比） =================
    st.subheader("🕸️ 雷达图：各科校次排名对比")
    # 选考试标签（多选）
    # 复用总分的时间序列来提供考试顺序（若没有总分，则回退为整体的考试标签顺序）
    total_rank_long = ts_long[ts_long["项目"] == "总分_校次"].copy()
    if not total_rank_long.empty:
        available_exams = list(dict.fromkeys(total_rank_long.sort_values("考试顺序")["考试标签"]))
    else:
        available_exams = list(dict.fromkeys(ts_long.sort_values("考试顺序")["考试标签"]))
    selected_exams_for_radar = st.multiselect("选择要比较的考试 (2~3 次更直观)", available_exams, default=available_exams[-2:] if len(available_exams) >= 2 else available_exams)

    if selected_exams_for_radar:
        # 复合科目（语数英/7选3）不纳入雷达图，避免与单科混合
        subjects_for_radar = [s for s in subjects if s not in COMPOSITE_SUBJECTS]
        radar_subject_ranks = ts_long[(ts_long["姓名"] == student_name) & (ts_long["考试标签"].isin(selected_exams_for_radar))]
        # 仅保留学科 rank 行
        subj_rank_mask = radar_subject_ranks["项目"].isin([f"{s}_校次" for s in subjects_for_radar])
        radar_subject_ranks = radar_subject_ranks[subj_rank_mask].copy()
        radar_subject_ranks["学科"] = radar_subject_ranks["项目"].str.replace("_校次", "", regex=False)
        transformed = transform_rank_for_radar(radar_subject_ranks)
        # 构造雷达
        fig_radar = go.Figure()
        categories = [s for s in subjects_for_radar if s in transformed["学科"].unique()]
        if not categories:
            st.info("所选考试缺少学科排名数据。")
        else:
            max_val = transformed["雷达值"].max() if "雷达值" in transformed.columns else None
            if pd.isna(max_val) or max_val is None:
                max_val = 1
            for exam in selected_exams_for_radar:
                sub = transformed[transformed["考试标签"] == exam]
                sub = sub.set_index("学科").reindex(categories)
                r_vals = sub["雷达值"].tolist() if "雷达值" in sub.columns else [None]*len(categories)
                fig_radar.add_trace(go.Scatterpolar(r=r_vals, theta=categories, fill='toself', name=exam))
            fig_radar.update_layout(title=f"{student_name} 各科排名雷达图 (数值已反转，面积越大表示排名越前)", polar=dict(radialaxis=dict(visible=True, range=[0, max_val])), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)
            export_figs[f"{student_name} 各科排名雷达图"] = fig_radar

   
    st.caption("提示：雷达图数值对排名做了反转，面积越大表示名次越前。若某科缺失排名则该科为空。")


    st.markdown("---")  

    # ================= 总分跨考试柱状对比 =================
    st.subheader("📊 总分跨考试对比")
    if "总分" in filtered_df.columns:
        total_df = filtered_df[filtered_df["姓名"] == student_name][["考试标签", "考试顺序", "总分"]].copy()
        total_df["考试标签"] = pd.Categorical(total_df["考试标签"], categories=exam_label_order, ordered=True)
        if total_df.empty:
            st.info("该学生无总分数据可对比。")
        else:
            total_df["显示总分"] = total_df["总分"].apply(_fmt_one_decimal)
            fig_total = px.bar(
                total_df.sort_values("考试顺序"),
                x="考试标签", y="总分", text="显示总分", color="考试标签",
                category_orders={"考试标签": exam_label_order},
                title=f"{student_name} 历次考试总分对比"
            )
            fig_total.update_traces(texttemplate="%{text}", textposition="outside")
            # 自适应Y轴起始值
            if auto_y_start and total_df["总分"].notna().any():
                vmin = float(total_df["总分"].min())
                vmax = float(total_df["总分"].max())
                y0 = max(0.0, vmin - float(offset_y))
                y1 = vmax * 1.05 if vmax > 0 else 1.0
                fig_total.update_yaxes(range=[y0, y1])
            
            fig_total.update_layout(yaxis_title="总分", xaxis_title="考试", height=420, showlegend=False)
            st.plotly_chart(fig_total, use_container_width=True)
            export_figs[f"{student_name} 历次考试总分对比"] = fig_total
    else:
        st.info("未找到总分列。")

    # ===== 在此默认增加：语数英 与 7选3 的跨考试对比（同一图：x=科目，颜色=考试标签，y=成绩） =====
    comp_candidates = ["语数英", "7选3"]
    present_comps = [c for c in comp_candidates if c in filtered_df.columns]
    comp_src = filtered_df[filtered_df["姓名"] == student_name]
    if present_comps and not comp_src.empty:
        comp_long = (
            comp_src[["考试标签", "考试顺序"] + present_comps]
            .melt(id_vars=["考试标签", "考试顺序"], value_vars=present_comps, var_name="科目", value_name="分数")
        )
        # 保持考试顺序与科目顺序
        comp_long["考试标签"] = pd.Categorical(comp_long["考试标签"], categories=exam_label_order, ordered=True)
        subj_order_comp = [s for s in comp_candidates if s in comp_long["科目"].unique()]
        comp_long["科目"] = pd.Categorical(comp_long["科目"], categories=subj_order_comp, ordered=True)

        if comp_long["分数"].notna().any():
            comp_long["显示分数"] = comp_long["分数"].apply(_fmt_one_decimal)
            fig_comp_mix = px.bar(
                comp_long.sort_values(["科目", "考试顺序"]),
                x="科目", y="分数", color="考试标签", text="显示分数",
                barmode="group",
                category_orders={"科目": subj_order_comp, "考试标签": exam_label_order},
                title=f"{student_name} 历次考试 语数英/7选3 对比"
            )
            fig_comp_mix.update_traces(texttemplate="%{text}", textposition="outside")
            if auto_y_start and comp_long["分数"].notna().any():
                vmin = float(comp_long["分数"].min())
                vmax = float(comp_long["分数"].max())
                y0 = max(0.0, vmin - float(offset_y))
                y1 = vmax * 1.05 if vmax > 0 else 1.0
                fig_comp_mix.update_yaxes(range=[y0, y1])
            fig_comp_mix.update_layout(yaxis_title="分数", xaxis_title="科目", height=420)
            st.plotly_chart(fig_comp_mix, use_container_width=True)
            export_figs[f"{student_name} 历次考试 语数英_7选3 对比"] = fig_comp_mix
        else:
            st.info("该学生在复合科目（语数英/7选3）没有有效分数可对比。")

    st.markdown("---")
    # ================= 跨考试成绩对比（X=科目 颜色=考试，排除总分） =================
    st.subheader("📊 跨考试成绩对比（按科目分类，颜色区分考试，不含总分）")
    if score_cols_subjects_only:
        score_all_long = (
            filtered_df[filtered_df["姓名"] == student_name][["考试标签", "考试顺序"] + score_cols_subjects_only]
            .melt(id_vars=["考试标签", "考试顺序"], var_name="科目", value_name="分数")
        )
        score_all_long["考试标签"] = pd.Categorical(score_all_long["考试标签"], categories=exam_label_order, ordered=True)
        subject_order = [s for s in subjects if s in score_all_long["科目"].unique()]
        score_all_long["科目"] = pd.Categorical(score_all_long["科目"], categories=subject_order, ordered=True)
        if score_all_long.empty:
            st.info("该学生无单科成绩数据可对比。")
        else:
            score_all_long["显示分数"] = score_all_long["分数"].apply(_fmt_one_decimal)
            fig_scores_all = px.bar(
                score_all_long.sort_values(["科目", "考试顺序"]),
                x="科目", y="分数", color="考试标签", text="显示分数",
                barmode="group",
                category_orders={"科目": subject_order, "考试标签": exam_label_order},
                title=f"{student_name} 历次考试各科成绩对比（颜色=考试标签）"
            )
            fig_scores_all.update_traces(texttemplate="%{text}", textposition="outside")
            # 自适应Y轴起始值
            if auto_y_start and score_all_long["分数"].notna().any():
                vmin = float(score_all_long["分数"].min())
                vmax = float(score_all_long["分数"].max())
                y0 = max(0.0, vmin - float(offset_y))
                y1 = vmax * 1.05 if vmax > 0 else 1.0
                fig_scores_all.update_yaxes(range=[y0, y1])
            fig_scores_all.update_layout(yaxis_title="分数", xaxis_title="科目", height=480)
            st.plotly_chart(fig_scores_all, use_container_width=True)
            export_figs[f"{student_name} 各科成绩对比（颜色=考试标签）"] = fig_scores_all
    else:
        st.info("未找到单科成绩列，无法生成成绩对比图。")


    st.markdown("---")  
    # ================= 跨考试排名对比（X=科目 颜色=考试） =================
    st.subheader("📊 跨考试校次排名对比（按科目分类，颜色区分考试）")
    rank_cols_exist = (["总分_校次"] if "总分_校次" in filtered_df.columns else []) + \
                      [f"{s}_校次" for s in subjects if f"{s}_校次" in filtered_df.columns]
    if rank_cols_exist:
        rank_long = (
            filtered_df[filtered_df["姓名"] == student_name][["考试标签", "考试顺序"] + rank_cols_exist]
            .melt(id_vars=["考试标签", "考试顺序"], var_name="项目", value_name="校次排名")
        )
        rank_long["科目"] = rank_long["项目"].str.replace("_校次", "", regex=False)
        rank_long.loc[rank_long["项目"] == "总分_校次", "科目"] = "总分"
        rank_long["考试标签"] = pd.Categorical(rank_long["考试标签"], categories=exam_label_order, ordered=True)
        subject_order_rank = (["总分"] if "总分" in rank_long["科目"].unique() else []) + [s for s in subjects if s in rank_long["科目"].unique()]
        rank_long["科目"] = pd.Categorical(rank_long["科目"], categories=subject_order_rank, ordered=True)
        if rank_long.empty:
            st.info("该学生无校次排名数据可对比。")
        else:
            rank_long["显示排名"] = rank_long["校次排名"].apply(_fmt_one_decimal)
            fig_ranks_all = px.bar(
                rank_long.sort_values(["科目", "考试顺序"]),
                x="科目", y="校次排名", color="考试标签", text="显示排名",
                barmode="group",
                category_orders={"科目": subject_order_rank, "考试标签": exam_label_order},
                title=f"{student_name} 历次考试各科校次排名对比（颜色=考试标签）"
            )
            fig_ranks_all.update_traces(texttemplate="%{text}", textposition="outside")
            fig_ranks_all.update_yaxes(autorange="reversed")
            fig_ranks_all.update_layout(yaxis_title="名次(越小越好)", xaxis_title="科目", height=480)
            st.plotly_chart(fig_ranks_all, use_container_width=True)
            export_figs[f"{student_name} 各科校次排名对比（颜色=考试标签）"] = fig_ranks_all

    
