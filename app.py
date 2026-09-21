import streamlit as st
import pandas as pd
import joblib

# ---------- ตั้งค่าหน้าเว็บ ----------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.write("กรอกข้อมูลผู้โดยสารด้านล่าง แล้วกดปุ่มเพื่อทำนายว่าจะรอดชีวิตหรือไม่")

# ---------- โหลดโมเดล ----------
# ใช้ cache_resource เพื่อให้โหลดโมเดลแค่ครั้งเดียว ไม่ต้องโหลดใหม่ทุกครั้งที่ interact กับ widget
@st.cache_resource
def load_model():
    return joblib.load("titanic_tree.joblib")

model = load_model()

# ---------- ฟอร์มรับข้อมูล ----------
st.subheader("ข้อมูลผู้โดยสาร")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "ชั้นโดยสาร (Pclass)",
        options=[1, 2, 3],
        index=2,
        help="1 = ชั้นหนึ่ง, 2 = ชั้นสอง, 3 = ชั้นสาม"
    )
    sex = st.radio("เพศ (Sex)", options=["หญิง", "ชาย"])
    age = st.slider("อายุ (Age)", min_value=0, max_value=100, value=30)

with col2:
    fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0, value=32.0, step=1.0)
    sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสที่มาด้วย (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("จำนวนพ่อแม่/ลูกที่มาด้วย (Parch)", min_value=0, max_value=10, value=0)

# โมเดลนี้ใช้ FamilySize = SibSp + Parch + 1 (ตัวผู้โดยสารเอง)
family_size = sibsp + parch + 1
sex_female = 1 if sex == "หญิง" else 0

st.caption(f"คำนวณ FamilySize อัตโนมัติ = SibSp + Parch + 1 = **{family_size}**")

# ---------- ทำนายผล ----------
if st.button("🔮 ทำนายผล", type="primary"):
    # ต้องเรียงคอลัมน์ตามลำดับที่โมเดลถูกเทรนมา: Pclass, Sex_female, Age, Fare, FamilySize
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex_female": sex_female,
        "Age": age,
        "Fare": fare,
        "FamilySize": family_size
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("ผลการทำนาย")
    if prediction == 1:
        st.success(f"✅ รอดชีวิต (Survived) — ความน่าจะเป็น {proba[1]*100:.1f}%")
    else:
        st.error(f"❌ ไม่รอดชีวิต (Not Survived) — ความน่าจะเป็นไม่รอด {proba[0]*100:.1f}%")

    with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล"):
        st.dataframe(input_df)
        st.write(f"ความน่าจะเป็นไม่รอด: {proba[0]*100:.1f}% | ความน่าจะเป็นรอด: {proba[1]*100:.1f}%")
