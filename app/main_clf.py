import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container

# Set page config for better appearance
st.set_page_config(
    page_title="NutriGuide AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4b6cb7 0%, #182848 100%) !important;
        color: white !important;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s;
        margin-bottom: 20px;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    /* Custom header */
    .custom-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Custom subheader */
    .custom-subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4b6cb7;
        margin-bottom: 1rem;
    }
    
    /* Custom metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background: white;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4b6cb7 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)



# Load datasets
@st.cache_data
def load_data():
    foods_df = pd.read_csv("datasets/Final_Data_Set.csv")
    meals_df = pd.read_csv("datasets/Meal_Suggestions.csv")
    nutrients_df = pd.read_csv("datasets/Micro_and_Macro_Nutrients.csv")
    return foods_df, meals_df, nutrients_df

foods_df, meals_df, nutrients_df = load_data()

# Load trained model
model = joblib.load("model/meal_classifier_model.pkl")


def calculate_bmr(weight, height, age, gender):
    if gender == "male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def calculate_tdee(bmr, activity_level):
    factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9
    }
    return bmr * factors.get(activity_level, 1.2)

def calculate_macronutrients(tdee, goal):
    if goal == "lose":
        protein = 2.2  # g per kg of body weight
        fat = 0.25 * tdee / 9  # 25% of calories from fat
        carbs = (tdee - (protein + fat)) / 4
    elif goal == "gain":
        protein = 1.6
        fat = 0.3 * tdee / 9
        carbs = (tdee - (protein + fat)) / 4
    else:  # maintain
        protein = 1.8
        fat = 0.25 * tdee / 9
        carbs = (tdee - (protein + fat)) / 4
    return protein, fat, carbs



# Hero Section
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<h1 class="custom-header">NutriGuide AI</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 1.2rem; color: #555;">
    Your personalized nutrition assistant that helps you discover the perfect meals 
    tailored to your body, goals, and preferences. Get science-backed recommendations 
    to optimize your health and performance.
    </p>
    """, unsafe_allow_html=True)
    
with col2:
    st.image("https://images.unsplash.com/photo-1490645935967-10de6ba17061?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=500&q=80", 
             caption="Healthy Eating Made Simple")

# Sidebar: User Profile Input
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: white; margin-bottom: 5px;">👤 User Profile</h2>
        <p style="color: #ddd;">Personalize your recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Basic Information", expanded=True):
        age = st.number_input("Age", 10, 100, 25)
        gender = st.selectbox("Gender", ["male", "female", "other"])
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
    
    with st.expander("Activity & Goals"):
        activity = st.selectbox("Activity Level", 
                              ["sedentary", "light", "moderate", "active", "very active"],
                              help="Sedentary = little or no exercise\nLight = light exercise 1-3 days/week\nModerate = moderate exercise 3-5 days/week\nActive = hard exercise 6-7 days/week\nVery active = very hard exercise & physical job")
        goal = st.selectbox("Goal", ["lose", "gain", "maintain"])
        diet = st.selectbox("Diet Preference", ["veg", "non-veg", "vegan", "pescatarian"])
    
    with st.expander("Health Considerations"):
        weekly_activity_days = st.number_input("Weekly Activity Days", 0, 7, 3)
        disease = st.text_input("Disease (if any)")
        food_allergies = st.text_input("Food Allergies (if any)")
    
    if st.button("Get Personalized Recommendations", key="recommend_button"):
        st.session_state['recommendations_generated'] = True

# Main Content Area
if 'recommendations_generated' in st.session_state and st.session_state['recommendations_generated']:
    # Calculate BMR and TDEE
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity)
    protein_need, fat_need, carbs_need = calculate_macronutrients(tdee, goal)
    
    # Display Metrics in a Card Layout
    st.markdown('<h2 class="custom-subheader">Your Daily Nutrition Needs</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with stylable_container(
            key="metric_card_bmr",
            css_styles="""
                {
                    background: white;
                    border-radius: 12px;
                    padding: 15px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
                h3 {
                    color: #4b6cb7;
                    font-size: 1rem;
                    margin-bottom: 5px;
                }
                p {
                    color: #333;
                    font-size: 1.5rem;
                    font-weight: 700;
                    margin: 0;
                }
            """
        ):
            st.markdown("<h3>Basal Metabolic Rate</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{bmr:.0f} kcal</p>", unsafe_allow_html=True)
            st.caption("Calories your body needs at rest")
    
    with col2:
        with stylable_container(
            key="metric_card_tdee",
            css_styles="""
                {
                    background: white;
                    border-radius: 12px;
                    padding: 15px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
                h3 {
                    color: #4b6cb7;
                    font-size: 1rem;
                    margin-bottom: 5px;
                }
                p {
                    color: #333;
                    font-size: 1.5rem;
                    font-weight: 700;
                    margin: 0;
                }
            """
        ):
            st.markdown("<h3>Total Daily Energy Expenditure</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{tdee:.0f} kcal</p>", unsafe_allow_html=True)
            st.caption("Calories needed based on activity level")
    
    with col3:
        with stylable_container(
            key="metric_card_protein",
            css_styles="""
                {
                    background: white;
                    border-radius: 12px;
                    padding: 15px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
                h3 {
                    color: #4b6cb7;
                    font-size: 1rem;
                    margin-bottom: 5px;
                }
                p {
                    color: #333;
                    font-size: 1.5rem;
                    font-weight: 700;
                    margin: 0;
                }
            """
        ):
            st.markdown("<h3>Protein Target</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{protein_need:.1f} g/kg</p>", unsafe_allow_html=True)
            st.caption(f"{(protein_need * weight):.0f}g per day")
    
    with col4:
        with stylable_container(
            key="metric_card_macro",
            css_styles="""
                {
                    background: white;
                    border-radius: 12px;
                    padding: 15px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
                h3 {
                    color: #4b6cb7;
                    font-size: 1rem;
                    margin-bottom: 5px;
                }
                p {
                    color: #333;
                    font-size: 1.5rem;
                    font-weight: 700;
                    margin: 0;
                }
            """
        ):
            st.markdown("<h3>Macro Split</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{carbs_need:.0f}g C / {protein_need*weight:.0f}g P / {fat_need:.0f}g F</p>", unsafe_allow_html=True)
            st.caption("Carbs / Protein / Fat")
    
    # In the meal recommendation section (replace the existing code):

    # Meal Plan Section
    st.markdown("---")
    st.markdown('<h2 class="custom-subheader">Your Personalized Meal Plan</h2>', unsafe_allow_html=True)

    # Filter meals based on dietary preference
    if diet == "veg":
        meals_df = meals_df[~meals_df['Breakfast'].str.contains('egg|meat|chicken|fish', case=False, regex=True)]
        meals_df = meals_df[~meals_df['Lunch'].str.contains('egg|meat|chicken|fish', case=False, regex=True)]
        meals_df = meals_df[~meals_df['Dinner'].str.contains('egg|meat|chicken|fish', case=False, regex=True)]
        meals_df = meals_df[~meals_df['Snacks'].str.contains('egg|meat|chicken|fish', case=False, regex=True)]
    elif diet == "vegan":
        meals_df = meals_df[~meals_df['Breakfast'].str.contains('egg|meat|chicken|fish|dairy|milk|cheese|yogurt', case=False, regex=True)]
        meals_df = meals_df[~meals_df['Lunch'].str.contains('egg|meat|chicken|fish|dairy|milk|cheese|yogurt', case=False, regex=True)]
        meals_df = meals_df[~meals_df['Dinner'].str.contains('egg|meat|chicken|fish|dairy|milk|cheese|yogurt', case=False, regex=True)]
        meals_df = meals_df[~meals_df['Snacks'].str.contains('egg|meat|chicken|fish|dairy|milk|cheese|yogurt', case=False, regex=True)]
    elif diet == "pescatarian":
        meals_df = meals_df[~meals_df['Breakfast'].str.contains('meat|chicken', case=False, regex=True)]
        meals_df = meals_df[~meals_df['Lunch'].str.contains('meat|chicken', case=False, regex=True)]
        meals_df = meals_df[~meals_df['Dinner'].str.contains('meat|chicken', case=False, regex=True)]
        meals_df = meals_df[~meals_df['Snacks'].str.contains('meat|chicken', case=False, regex=True)]

    # Select the best matching meal plan (based on TDEE)
    meal_plan = meals_df.merge(foods_df, left_on="Daily_Calories", right_on="Daily_Calories", how="left")
    meal_plan['diff'] = abs(meal_plan['Daily_Calories'] - tdee)
    best_meal_plan = meal_plan.sort_values(by='diff').head(1)

# Display the meal plan tabs...
    
    # Select the best matching meal plan (based on TDEE)
    meal_plan = meals_df.merge(foods_df, left_on="Daily_Calories", right_on="Daily_Calories", how="left")
    meal_plan['diff'] = abs(meal_plan['Daily_Calories'] - tdee)
    best_meal_plan = meal_plan.sort_values(by='diff').head(3)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Breakfast", "Lunch", "Dinner", "Snacks"])
    
    with tab1:
        if not best_meal_plan.empty:
            st.markdown(f"""
            <div class="card">
                <h3>Breakfast</h3>
                <p>{best_meal_plan['Breakfast'].values[0]}</p>
                <p class="small" style="color: #666;">~400 kcal</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        if not best_meal_plan.empty:
            st.markdown(f"""
            <div class="card">
                <h3> Lunch</h3>
                <p>{best_meal_plan['Lunch'].values[0]}</p>
                <p class="small" style="color: #666;">~600 kcal</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        if not best_meal_plan.empty:
            st.markdown(f"""
            <div class="card">
                <h3>Dinner</h3>
                <p>{best_meal_plan['Dinner'].values[0]}</p>
                <p class="small" style="color: #666;">~500 kcal</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        if not best_meal_plan.empty:
            st.markdown(f"""
            <div class="card">
                <h3> Snacks</h3>
                <p>{best_meal_plan['Snacks'].values[0]}</p>
                <p class="small" style="color: #666;">~200 kcal</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Water Intake Recommendation
    st.markdown(f"""
    <div class="card">
        <h3> Hydration</h3>
        <p>Recommended water intake: {best_meal_plan['Water_Intake_L'].values[0]} liters per day</p>
        <div style="height: 10px; background: #e0f2fe; border-radius: 5px; margin-top: 10px;">
            <div style="width: 75%; height: 100%; background: #4b6cb7; border-radius: 5px;"></div>
        </div>
        <p class="small" style="color: #666;">Track your water intake throughout the day</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Nutrient Comparison Section
    st.markdown("---")
    st.markdown('<h2 class="custom-subheader">Nutrient Analysis</h2>', unsafe_allow_html=True)
    
    # Actual needs based on the user input
    actual = {
        "Calories": tdee,
        "Protein": weight * protein_need,
        "Fat": fat_need,
        "Carbs": carbs_need
    }
    
    # Find the row closest to the user's TDEE from the nutrients_df
    nutrient_row = nutrients_df.iloc[(nutrients_df["Daily_Calories"] - tdee).abs().argsort()[:1]].iloc[0]
    
    # Required nutrients based on the closest match
    required = {
        "Calories": nutrient_row["Daily_Calories"],
        "Protein": nutrient_row["Protein_g"],
        "Fat": nutrient_row["Fat_g"],
        "Carbs": (nutrient_row["Daily_Calories"] - (nutrient_row["Protein_g"]*4 + nutrient_row["Fat_g"]*9)) / 4
    }
    
    # Create a DataFrame for visualization
    df_nutrients = pd.DataFrame({
        "Nutrient": list(actual.keys()),
        "Your Needs": list(actual.values()),
        "Recommended": list(required.values())
    })
    
    # Melt the DataFrame for seaborn
    df_melted = df_nutrients.melt(id_vars="Nutrient", var_name="Type", value_name="Amount")
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_palette("pastel")
    
    barplot = sns.barplot(
        data=df_melted,
        x="Nutrient",
        y="Amount",
        hue="Type",
        ax=ax
    )
    
    # Customize the plot
    ax.set_title("Your Nutrient Needs vs. Recommended Daily Allowance", fontsize=16, pad=20)
    ax.set_xlabel("Nutrients", fontsize=12)
    ax.set_ylabel("Amount (g or kcal)", fontsize=12)
    ax.legend(title='', fontsize=10)
    
    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 10),
            textcoords='offset points',
            fontsize=10
        )
    
    # Display the plot
    st.pyplot(fig)
    
    # Additional Tips Section
    st.markdown("---")
    st.markdown('<h2 class="custom-subheader">Nutrition Tips for You</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3> Protein Sources</h3>
            <ul>
                <li>Chicken breast</li>
                <li>Greek yogurt</li>
                <li>Lentils</li>
                <li>Tofu</li>
                <li>Eggs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Meal Timing</h3>
            <ul>
                <li>Eat within 1 hour of waking</li>
                <li>Space meals 3-4 hours apart</li>
                <li>Include protein in every meal</li>
                <li>Post-workout meal within 45 min</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3>Progress Tracking</h3>
            <ul>
                <li>Weigh yourself weekly</li>
                <li>Take progress photos monthly</li>
                <li>Track energy levels</li>
                <li>Monitor workout performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Final CTA
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border-radius: 12px; color: white;">
        <h2>Ready to Transform Your Nutrition?</h2>
        <p style="font-size: 1.1rem;">Save your meal plan and track your progress with our premium features</p>
        <button style="background: white; color: #4b6cb7; border: none; border-radius: 8px; padding: 12px 30px; font-weight: 600; font-size: 1rem; margin-top: 15px; cursor: pointer;">
            Upgrade to Premium
        </button>
    </div>
    """, unsafe_allow_html=True)

else:
    # Default state before recommendations are generated
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-top: 50px;">
        <h2 style="color: #4b6cb7;">Get Started with Personalized Nutrition</h2>
        <p style="font-size: 1.1rem; color: #555; max-width: 600px; margin: 0 auto 30px;">
        Fill out your profile information in the sidebar and click "Get Personalized Recommendations" 
        to receive your custom meal plan and nutrition analysis.
        </p>
        <img src="https://images.unsplash.com/photo-1546069901-ba9590a7edd63?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=500&q=80" 
             style="width: 100%; max-width: 400px; border-radius: 8px; margin: 0 auto;">
    </div>
    """, unsafe_allow_html=True)