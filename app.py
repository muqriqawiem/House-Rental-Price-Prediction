import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

file_path = "IDSmodel.pkl"
model1 = joblib.load(file_path)
mean1 = np.load("mean.npy")
std1 = np.load("std.npy")

with st.sidebar:
    selected = option_menu('Rental Price Prediction and Recommendation Website',['Rental Price Prediction','Rental Market Analysis'],default_index=0)

if(selected == 'Rental Price Prediction'):
    # Define the mappings for categorical variables
    furnished_mapping = {'Not furnished': 0, 'Partially furnished': 1, 'Fully furnished': 2}
    nearby_railways_mapping = {'Yes': 1, 'No': 0}
    minimart_availability_mapping = {'Yes': 1, 'No': 0}
    security_availability_mapping = {'Yes': 1, 'No': 0}
    property_type_mapping = {
        'Condominium': {'Condominium': 1, 'Apartment': 0, 'Service Residence': 0, 'Others': 0},
        'Apartment': {'Condominium': 0, 'Apartment': 1, 'Service Residence': 0, 'Others': 0},
        'Service Residence': {'Condominium': 0, 'Apartment': 0, 'Service Residence': 1, 'Others': 0},
        'Others': {'Condominium': 0, 'Apartment': 0, 'Service Residence': 0, 'Others': 1}
    }


    st.title('Property Renting Price Prediction')

    # Input fields
    rooms = st.text_input('Number of rooms')
    parking = st.text_input('Number of parking')
    bathroom = st.text_input('Number of bathroom')
    size = st.text_input('Size in square feet')
    furnished = st.selectbox('Furnishing', ['Not furnished', 'Partially furnished', 'Fully furnished'])
    nearKTMLRT = st.selectbox('Nearby KTM or LRT', ['Yes', 'No'])
    minimart_availability = st.selectbox('Minimart availability', ['Yes', 'No'])
    security_availability = st.selectbox('Security availability', ['Yes', 'No'])
    location_name = st.text_input('Desired location')
    property_type = st.selectbox('Property type', ['Apartment', 'Condominium', 'Service Residence', 'Others'])
    region = st.selectbox('Region', ['Selangor', 'Kuala Lumpur'])
    price_range = st.text_input('Accepted range of price difference with the predicted house price')

    # Convert categorical variables to numerical format
    furnished_val = furnished_mapping[furnished]
    nearby_railways_val = nearby_railways_mapping[nearKTMLRT]
    minimart_availability_val = minimart_availability_mapping[minimart_availability]
    security_availability_val = security_availability_mapping[security_availability]
    property_type_val = property_type_mapping[property_type]

    # Set all location columns to 0 initially
    location_columns = {'Cheras', 'Setapak', 'Sentul', 'Kepong', 'Bukit Jalil', 'Ampang', 'Wangsa Maju', 'Old Klang Road', 'Taman Desa', 'Mont Kiara','Keramat','Jalan Ipoh','Sungai Besi','Kuchai Lama','KL City','Jalan Kuching','Segambut','Desa Pandan','KLCC','Bangsar South','Cyberjaya', 'Kajang', 'Puchong', 'Seri Kembangan', 'Shah Alam', 'Petaling Jaya', 'Semenyih', 'Subang Jaya', 'Setia Alam', 'Bangi','Damansara Perdana','Batu Caves','Damansara Damai','Sepang','Kota Damansara','Klang','Selayang','Gombak','Dengkil','Ara Damansara','Other'}

    # Check if the entered location matches any location in the location list
    if location_name in location_columns:
        # Set the selected location to 1 and all other locations to 0
        location_columns = {location: 1 if location == location_name else 0 for location in location_columns}
    else:
        # If no match, set 'Others' to 1 and all locations to 0
        location_columns = {location: 0 for location in location_columns}
        location_columns['Other'] = 1

    size_float = float(size) if size else 0
    rooms_int = int(rooms) if rooms else 0
    parking_int = int(parking) if parking else 0
    bathroom_int = int(bathroom) if bathroom else 0
    selangor_int = 1 if region == 'Selangor' else 0
    kuala_lumpur_int = 1 if region == 'Kuala Lumpur' else 0
    price_range_float = float(price_range) if price_range else 0

    # Create the user inputs DataFrame
    user_inputs = {
        'rooms': [rooms_int],
        'parking': [parking_int],
        'bathroom': [bathroom_int],
        'size(sq.ft.)': [size_float],
        'furnished': [furnished_val],
        'near_KTM-LRT': [nearby_railways_val],
        'minimart_availability': [minimart_availability_val],
        'security_availability': [security_availability_val],
        **location_columns,  # Include all location columns in the dictionary
        **property_type_val,
        'Selangor': [selangor_int],
        'Kuala Lumpur': [kuala_lumpur_int]
    }

    user_inputs_df = pd.DataFrame(user_inputs)

    recommendation_df = pd.read_csv("df_final.csv")
    filtered_recommendation_df = recommendation_df[
        (recommendation_df['rooms'] == rooms_int) &
        (recommendation_df['parking'] == parking_int) &
        (recommendation_df['bathroom'] == bathroom_int) &
        (recommendation_df['size(sq.ft.)']<= size_float +100) &
        (recommendation_df['size(sq.ft.)']>= size_float -100) &
        (recommendation_df['furnished'] == furnished_val) &
        (recommendation_df['near_KTM-LRT'] == nearby_railways_val) &
        (recommendation_df['minimart availability'] == minimart_availability_val) &
        (recommendation_df['security availability'] == security_availability_val) &
        (recommendation_df['Selangor'] == selangor_int) &
        (recommendation_df['Kuala Lumpur'] == kuala_lumpur_int)
    ]

    # Filter based on location columns
    for location, value in location_columns.items():
        if value == 1:
            filtered_recommendation_df = filtered_recommendation_df[filtered_recommendation_df[location] == value]

    # Filter based on property type columns
    for property_type, value in property_type_val.items():
        if value == 1:
            filtered_recommendation_df = filtered_recommendation_df[filtered_recommendation_df[property_type] == value]

    if st.button('House Price Test Result'):
        # Scale the input data using the mean and standard deviation
        # Scale the user inputs
        # Scale the user inputs

        user_inputs_scaled = (user_inputs_df - mean1) / std1

        # Ensure the input data has the correct number of features
        if user_inputs_scaled.shape[1] != len(mean1):
            st.error(f'Invalid number of features. Expected {len(mean1)} features, but received {user_inputs_scaled.shape[1]}')
        else:
            # Make the prediction using the loaded model
            price_pred = model1.predict(user_inputs_scaled.values.reshape(1, -1))
            st.write(f"Predicted Price: RM {price_pred[0]:,.2f}")
            
            filtered_recommendation_df = filtered_recommendation_df[
            (filtered_recommendation_df['monthly_rent'] >= price_pred[0] - price_range_float) &
            (filtered_recommendation_df['monthly_rent'] <= price_pred[0] + price_range_float)
            ]

            st.title("Property Recommendation")
            for index, row in filtered_recommendation_df.iterrows():
                st.write(f"Property Name: {row['prop_name']}")
                st.write(f"Price: RM {row['monthly_rent']:,.2f}")

if selected == 'Rental Market Analysis':

    import matplotlib.pyplot as plt
    df = pd.read_csv("cleaned-mudah-apartment-kl-selangor.csv")

# Displaying the text with formatting
    st.markdown("## Top 5 Features Affecting the Rental Price:")
    st.write("1. Size")
    st.write("2. Furnishing level")
    st.write("3. Number of bathrooms")
    st.write("4. Number of rooms")
    st.write("5. Number of parking")

    # Plot average property price by region
    filtered_df = df[df['region'].isin(['Kuala Lumpur', 'Selangor'])]
    region_avg_price = filtered_df.groupby('region')['monthly_rent'].mean()
    fig1, ax1 = plt.subplots()
    region_avg_price.plot(kind='bar', ax=ax1)
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Average Price')
    ax1.set_title('Average Property Price: Kuala Lumpur vs Selangor')
    st.pyplot(fig1)

    # Plot property size by region
    kl_df = df[df['region'] == 'Kuala Lumpur']
    selangor_df = df[df['region'] == 'Selangor']
    # Filter the data for Kuala Lumpur and Selangor
    kl_sizes = kl_df['size(sq.ft.)']
    selangor_sizes = selangor_df['size(sq.ft.)']
    # Set the number of bins for the histogram
    num_bins = 20
    # Plotting the histograms
    fig, ax = plt.subplots()
    ax.hist(kl_sizes, bins=num_bins, alpha=0.5, label='Kuala Lumpur')
    ax.hist(selangor_sizes, bins=num_bins, alpha=0.5, label='Selangor')
    ax.set_xlabel('Property Size')
    ax.set_ylabel('Count of Properties')
    ax.set_title('Property Size Distribution by Region')
    ax.legend()
    st.pyplot(fig)

    # Plot proportion of furnished properties
    no_furniture_count = df[df['furnished'] == 0].shape[0]
    partial_furniture_count = df[df['furnished'] == 1].shape[0]
    fully_furnished_count = df[df['furnished'] == 2].shape[0]
    labels = ['No Furniture', 'Partially Furnished', 'Fully Furnished']
    sizes = [no_furniture_count, partial_furniture_count, fully_furnished_count]
    fig4, ax4 = plt.subplots()
    ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax4.axis('equal')
    ax4.set_title('Proportion of Furnished Properties')
    st.pyplot(fig4)

    # Plot property bathrooms by region
    kl_df = df[df['region'] == 'Kuala Lumpur']
    selangor_df = df[df['region'] == 'Selangor']
    bathroom_categories = sorted(list(set(kl_df['bathroom'].unique()) | set(selangor_df['bathroom'].unique())))
    kl_bathroom_counts = kl_df['bathroom'].value_counts().sort_index().reindex(bathroom_categories, fill_value=0)
    selangor_bathroom_counts = selangor_df['bathroom'].value_counts().sort_index().reindex(bathroom_categories, fill_value=0)
    fig3, ax3 = plt.subplots()
    bar_width = 0.35
    x = range(len(bathroom_categories))
    ax3.bar(x, kl_bathroom_counts, width=bar_width, label='Kuala Lumpur')
    ax3.bar(x, selangor_bathroom_counts, width=bar_width, label='Selangor', bottom=kl_bathroom_counts)
    ax3.set_xlabel('Number of bathrooms')
    ax3.set_ylabel('Count of Properties')
    ax3.set_title('Property Bathrooms by Region')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bathroom_categories)
    ax3.legend()
    st.pyplot(fig3)

    # Plot property rooms by region
    kl_df = df[df['region'] == 'Kuala Lumpur']
    selangor_df = df[df['region'] == 'Selangor']
    room_categories = sorted(list(set(kl_df['rooms'].unique()) | set(selangor_df['rooms'].unique())))
    kl_room_counts = kl_df['rooms'].value_counts().sort_index().reindex(room_categories, fill_value=0)
    selangor_room_counts = selangor_df['rooms'].value_counts().sort_index().reindex(room_categories, fill_value=0)
    fig3, ax3 = plt.subplots()
    bar_width = 0.35
    x = range(len(room_categories))
    ax3.bar(x, kl_room_counts, width=bar_width, label='Kuala Lumpur')
    ax3.bar(x, selangor_room_counts, width=bar_width, label='Selangor', bottom=kl_room_counts)
    ax3.set_xlabel('Number of Rooms')
    ax3.set_ylabel('Count of Properties')
    ax3.set_title('Property Rooms by Region')
    ax3.set_xticks(x)
    ax3.set_xticklabels(room_categories)
    ax3.legend()
    st.pyplot(fig3)

    # Plot property parking by region
    kl_df = df[df['region'] == 'Kuala Lumpur']
    selangor_df = df[df['region'] == 'Selangor']
    parking_categories = sorted(list(set(kl_df['parking'].unique()) | set(selangor_df['parking'].unique())))
    kl_parking_counts = kl_df['parking'].value_counts().sort_index().reindex(parking_categories, fill_value=0)
    selangor_parking_counts = selangor_df['parking'].value_counts().sort_index().reindex(parking_categories, fill_value=0)
    fig3, ax3 = plt.subplots()
    bar_width = 0.35
    x = range(len(parking_categories))
    ax3.bar(x, kl_parking_counts, width=bar_width, label='Kuala Lumpur')
    ax3.bar(x, selangor_parking_counts, width=bar_width, label='Selangor', bottom=kl_parking_counts)
    ax3.set_xlabel('Number of Parking')
    ax3.set_ylabel('Count of Properties')
    ax3.set_title('Property Parking by Region')
    ax3.set_xticks(x)
    ax3.set_xticklabels(parking_categories)
    ax3.legend()
    st.pyplot(fig3)

    # Plot property type by region
    property_counts = df.groupby(['region', 'property_type']).size().unstack()
    fig2, ax2 = plt.subplots()
    property_counts.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_xlabel('Region')
    ax2.set_ylabel('Count of Properties')
    ax2.set_title('Property Type by Region')
    ax2.legend(title='Property Type')
    st.pyplot(fig2)
