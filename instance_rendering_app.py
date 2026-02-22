import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="DDS Instance Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DDSInstanceGenerator:
    def __init__(self, num_customers, grid_size, max_cargo_capacity, set_type="corner", random_seed=None, clients_distribution=np.random.uniform):
        self.num_customers = num_customers
        self.max_cargo_capacity = max_cargo_capacity
        self.grid_size = grid_size
        self.set_type = set_type
        self.clients_distribution = clients_distribution
        
        if random_seed is not None:
            np.random.seed(random_seed)
  
    def generate_delivery_and_depot_coordinates(self):
        if self.set_type == "corner":
            self.depot_coordinates = np.array([0.0, 0.0])
        else:
            self.depot_coordinates = np.array([self.grid_size/2.0, self.grid_size/2.0])

        self.customers_coordinates = self.clients_distribution(
            0, self.grid_size, size=(self.num_customers, 2)
        )

    def generate_customer_demands(self):
        self.demands = np.zeros(self.num_customers)
        split = int(0.4 * self.num_customers)

        self.demands[:split] = np.random.uniform(0.1, 0.7, split)
        self.demands[split:] = np.random.uniform(0.1, 1.5, self.num_customers - split)

    def compute_required_drones(self):
        self.total_demand = np.sum(self.demands)
        self.allocated_drones = math.ceil(self.total_demand / (self.max_cargo_capacity / 3.0))

    def compute_travel_times(self):
        self.travel_times = np.linalg.norm(self.customers_coordinates - self.depot_coordinates, axis=1)

    def compute_time_horizon(self):
        sorted_times = np.sort(self.travel_times)[::-1]
        customers_drone_ratio = math.ceil(self.num_customers / self.allocated_drones)
        total_time_spent = np.sum(sorted_times[:customers_drone_ratio])
        
        self.initial_operating_time = 0
        self.final_operating_time = math.ceil(2 * total_time_spent)

    def generate_time_windows(self):
        self.lower_delivery_limits = np.zeros(self.num_customers)
        self.upper_delivery_limits = np.zeros(self.num_customers)

        for customer in range(self.num_customers):
            travel_time_to_client = self.travel_times[customer]
            latest_departure = self.final_operating_time - travel_time_to_client

            center_delivery_window = np.random.uniform(travel_time_to_client, latest_departure)

            mean_window_size = 0.25 * (latest_departure - travel_time_to_client)
            std_dev_window_size = 0.05 * (latest_departure - travel_time_to_client)
            client_window = np.random.normal(mean_window_size, std_dev_window_size)
            client_window = max(1, client_window)

            client_lower_limit = max(math.ceil(travel_time_to_client), math.floor(center_delivery_window - 0.5 * client_window))
            client_upper_limit = min(math.floor(latest_departure), math.floor(center_delivery_window + 0.5 * client_window))
            
            self.lower_delivery_limits[customer] = client_lower_limit
            self.upper_delivery_limits[customer] = client_upper_limit
  
    def generate_instances(self):
        self.generate_delivery_and_depot_coordinates()
        self.generate_customer_demands()
        self.compute_required_drones()
        self.compute_travel_times()
        self.compute_time_horizon()
        self.generate_time_windows()

    def load_from_dataframe(self, df):
        self.num_customers = len(df)
        self.customers_coordinates = df[['X Coord', 'Y Coord']].values
        self.demands = df['Demand'].values
        self.travel_times = df['Travel Time'].values
        self.lower_delivery_limits = df['Window Start'].values
        self.upper_delivery_limits = df['Window End'].values

        if self.set_type == "corner":
            self.depot_coordinates = np.array([0.0, 0.0])
        else:
            self.depot_coordinates = np.array([self.grid_size/2.0, self.grid_size/2.0])

        self.compute_required_drones()
        self.compute_time_horizon()
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        node_sizes = self.demands * 150 
        
        ax.scatter(self.customers_coordinates[:, 0], self.customers_coordinates[:, 1], 
                    s=node_sizes, c='dodgerblue', edgecolors='black', alpha=0.7, label='Clients')
        
        ax.scatter(self.depot_coordinates[0], self.depot_coordinates[1], 
                    c='red', marker='s', s=200, edgecolors='black', label='Main Depot')
        
        ax.set_title(f"DDS Instance Visualization ({self.num_customers} Clients)", fontsize=14, fontweight='bold')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        return fig


st.title("Drone Delivery System Generator")

st.sidebar.header("Configuration")
with st.sidebar.form("config_form"):
    num_customers = st.number_input("Number of Customers", min_value=1, max_value=500, value=30, step=1)
    grid_size = st.number_input("Grid Size", min_value=10, max_value=1000, value=100, step=10)
    max_cargo_capacity = st.number_input("Max Cargo Capacity", min_value=1.0, max_value=100.0, value=5.0, step=0.5)
    set_type = st.selectbox("Depot Location Strategy", options=["center", "corner"])
    
    use_seed = st.checkbox("Use Random Seed", value=True)
    random_seed = st.number_input("Seed Value", min_value=0, max_value=99999, value=42)
    
    submit_button = st.form_submit_button(label="Generate New Instance", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Instance CSV", type=["csv"])

if submit_button or uploaded_file is not None:
    seed_to_use = random_seed if use_seed else None
    
    generator = DDSInstanceGenerator(
        num_customers=num_customers, 
        grid_size=grid_size, 
        max_cargo_capacity=max_cargo_capacity, 
        set_type=set_type, 
        random_seed=seed_to_use
    )
    
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        generator.load_from_dataframe(df_uploaded)
        st.success("Instance successfully loaded from CSV.")
    else:
        with st.spinner("Generating instance data..."):
            generator.generate_instances()
        st.success("Instance successfully generated.")

    st.markdown("### Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Demand", f"{generator.total_demand:.2f}")
    col2.metric("Allocated Drones", f"{generator.allocated_drones}")
    col3.metric("Max Time Horizon", f"{generator.final_operating_time}")
    col4.metric("Avg Travel Time", f"{np.mean(generator.travel_times):.2f}")

    st.markdown("---")

    col_plot, col_data = st.columns([3, 2])
    
    with col_plot:
        st.markdown("### Network Map")
        fig = generator.plot()
        st.pyplot(fig)
        
    with col_data:
        st.markdown("### Data")
        
        df_customers = pd.DataFrame({
            "Client ID": range(1, generator.num_customers + 1),
            "X Coord": generator.customers_coordinates[:, 0].round(2),
            "Y Coord": generator.customers_coordinates[:, 1].round(2),
            "Demand": generator.demands.round(2),
            "Travel Time": generator.travel_times.round(2),
            "Window Start": generator.lower_delivery_limits.astype(int),
            "Window End": generator.upper_delivery_limits.astype(int)
        })
        
        st.dataframe(df_customers, use_container_width=True, height=450)
        
        csv_data = df_customers.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name='dds_instance_data.csv',
            mime='text/csv',
            use_container_width=True
        )

else:
    st.info("Adjust parameters and click Generate, or upload a previously saved CSV file.")