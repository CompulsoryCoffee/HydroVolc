"""
D-CASCADE: Distributed sediment cascade model for simulating ash transport in river networks.
This implementation combines both the simplicity of the minimal model with the more
comprehensive features of the complete implementation.
"""

import math
import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import nnls
import geopandas as gpd
from typing import Dict, List, Optional, Union, Tuple

class Reach:
    """
    Represents a single river reach in the D-Cascade network.
    
    Attributes:
        id (int): Unique reach identifier
        downstream_id (int): ID of the downstream reach (None for outlet)
        length (float): Reach length [m]
        slope (float): Reach slope (dimensionless)
        manning_n (float): Manning's roughness coefficient
        width (float): Channel width [m]
        storage (dict): Stored sediment mass by class, e.g. {'ash': 0.0}
        discharge (float): Current discharge [m³/s]
        metadata (dict): User-defined extras (e.g. A_sub, w)
    """
    def __init__(self, 
                id: int, 
                downstream_id: Optional[int], 
                length: float, 
                slope: float, 
                manning_n: float, 
                width: float, 
                storage: float = 0.0):
        """Initialize a river reach with physical and hydraulic parameters."""
        # Validate inputs
        if length <= 0:
            raise ValueError(f"Reach length must be positive, got {length}")
        if slope <= 0:
            raise ValueError(f"Reach slope must be positive, got {slope}")
        if manning_n <= 0:
            raise ValueError(f"Manning's n must be positive, got {manning_n}")
        if width <= 0:
            raise ValueError(f"Channel width must be positive, got {width}")
            
        self.id = id
        self.downstream_id = downstream_id
        self.length = length
        self.slope = slope
        self.manning_n = manning_n
        self.width = width
        self.storage = {'ash': storage}
        self.discharge = 0.0  # m³/s
        self.metadata = {}
        
        # For grain size distribution modeling
        self.deposit_gsd = None
        self.outflow_gsd = None

    def add_sediment_mass(self, sediment_type: str, mass_kg: float) -> None:
        """
        Add mass (kg) of a sediment type (e.g. 'ash') to this reach.
        
        Args:
            sediment_type: Type of sediment (e.g., 'ash', 'fine', 'medium', 'coarse')
            mass_kg: Mass to add in kg
        """
        if sediment_type not in self.storage:
            self.storage[sediment_type] = 0.0
        self.storage[sediment_type] += mass_kg

    def compute_hydraulic_radius(self) -> float:
        """
        Compute the hydraulic radius based on discharge and width.
        
        Returns:
            Estimated hydraulic radius [m]
        """
        if self.discharge <= 0:
            return 0.01  # Minimum value to avoid division by zero
            
        # Approximate hydraulic radius R from discharge and width
        # Using power-law relationship derived from Manning's equation
        return (self.discharge / self.width)**0.375

    def compute_velocity(self) -> float:
        """
        Compute flow velocity using Manning's equation.
        
        Returns:
            Flow velocity [m/s]
        """
        R = self.compute_hydraulic_radius()
        
        # Manning formula for velocity
        velocity = (1.0 / self.manning_n) * (R**(2.0/3.0)) * math.sqrt(self.slope)
        return velocity

    def compute_transport_capacity(self) -> Union[float, np.ndarray]:
        """
        Compute the reach's transport capacity based on current discharge,
        slope, width, Manning n, and reach length using a shear-based formula.
        
        Returns:
            Transport capacity [kg] or array of capacities for different grain sizes
        """
        # Physical constants
        rho_w = 1000.0  # water density [kg/m³]
        g = 9.81  # gravity [m/s²]

        # Get hydraulic radius
        R = self.compute_hydraulic_radius()

        # Bed shear stress
        shear = rho_w * g * R * self.slope

        # Simple capacity: capacity ∝ shear^(3/2) × width × length
        capacity = (shear**1.5) * self.width * self.length * 1e-3  # scale factor
        
        # If we have grain size distribution, compute capacity per class
        if self.deposit_gsd is not None:
            # Simplified Ackers-White proxy for each sediment class
            # Adjust capacity based on grain size
            k = 0.05  # simplified coefficient
            class_factors = np.linspace(0.8, 1.2, len(self.deposit_gsd))  # Fine to coarse
            return capacity * class_factors
            
        return capacity

    def step(self, incoming_flux_kg: float, dt: float = 1.0) -> float:
        """
        Advance one time step:
        
        Args:
            incoming_flux_kg: Sediment mass entering this reach from upstream [kg]
            dt: Time step duration [s]
            
        Returns:
            The mass exported downstream after deposition/entrainment [kg]
        """
        # 1) Add incoming sediment
        self.add_sediment_mass('ash', incoming_flux_kg)

        # 2) Compute how much can move: capacity vs. available
        available = self.storage.get('ash', 0.0)
        capacity = self.compute_transport_capacity()
        
        # Handle grain size distributions if present
        if isinstance(capacity, np.ndarray) and self.deposit_gsd is not None:
            # For multiclass sediment routing
            self.outflow_gsd = np.minimum(self.deposit_gsd, capacity)
            exported = np.sum(self.outflow_gsd)
            self.deposit_gsd = self.deposit_gsd - self.outflow_gsd
            return exported

        # Single sediment class routing
        if available <= capacity:
            # If capacity exceeds available: everything entrains
            exported = available
            self.storage['ash'] = 0.0
        else:
            # Otherwise deposit the excess
            exported = capacity
            self.storage['ash'] = available - capacity

        return exported


class DCascade:
    """
    D-Cascade model: holds all reaches and performs network-wide routing.
    
    Attributes:
        reaches (dict): Mapping reach_id → Reach object
        graph (nx.DiGraph): Directed graph representation of the river network
    """
    def __init__(self, reaches: Dict[int, Reach]):
        """
        Initialize with a dictionary of reaches.
        
        Args:
            reaches: Dictionary mapping reach IDs to Reach objects
        """
        self.reaches = reaches
        self.graph = self._build_graph()
        
        # Verify network has no cycles
        try:
            self.topo_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("River network contains cycles, which is physically impossible")

    def _build_graph(self) -> nx.DiGraph:
        """
        Build a directed graph from reaches based on downstream_id connections.
        
        Returns:
            NetworkX DiGraph representation of the river network
        """
        G = nx.DiGraph()
        
        # Add all reaches as nodes
        for rid in self.reaches:
            G.add_node(rid)
        
        # Add edges based on downstream connections
        for rid, reach in self.reaches.items():
            if reach.downstream_id is not None:
                if reach.downstream_id in self.reaches:
                    G.add_edge(rid, reach.downstream_id)
                else:
                    print(f"Warning: Reach {rid} points to non-existent downstream reach {reach.downstream_id}")
        
        return G

    def _upstream_ids(self, rid: int) -> List[int]:
        """
        Return list of reach_ids that flow directly into reach `rid`.
        
        Args:
            rid: Reach ID to find upstream reaches for
            
        Returns:
            List of upstream reach IDs
        """
        return list(self.graph.predecessors(rid))

    def step(self, dt: float = 1.0) -> Dict[int, float]:
        """
        Advance the entire network one time step of duration dt [s].
        It pulls exports from upstream, steps each reach, and routes to downstream.
        
        Args:
            dt: Time step duration [s]
            
        Returns:
            Dictionary of exports from each reach
        """
        # 1) Compute exports for each reach in topological order
        exports = {}
        
        for rid in self.topo_order:
            reach = self.reaches[rid]
            # Sum all exports from immediately upstream reaches
            incoming = sum(exports.get(up, 0.0) for up in self._upstream_ids(rid))
            exported = reach.step(incoming, dt)
            exports[rid] = exported

        return exports


def load_edges(path: str) -> gpd.GeoDataFrame:
    """
    Load river network edges from a GeoJSON file.
    
    Args:
        path: Path to GeoJSON file
        
    Returns:
        GeoDataFrame with river reach data
    """
    edges = gpd.read_file(path)
    return edges


def add_topo_and_manning(edges: gpd.GeoDataFrame, 
                         n_downstream: float = 0.035, 
                         n_head: float = 0.05) -> gpd.GeoDataFrame:
    """
    Compute topological order and Manning's n values for each reach.
    
    Args:
        edges: GeoDataFrame with river reach data
        n_downstream: Manning's n value for downstream reaches
        n_head: Manning's n value for headwater reaches
        
    Returns:
        GeoDataFrame with added topo_order and manning_n columns
    """
    # Build DiGraph
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        rid = int(r['reach_id'])
        G.add_node(rid)
        if pd.notna(r['downstream_id']):
            G.add_edge(rid, int(r['downstream_id']))
            
    # Compute level (shortest-path from any source)
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    level = {}
    for src in sources:
        for nid, d in nx.single_source_shortest_path_length(G, src).items():
            level[nid] = min(level.get(nid, np.inf), d)
    
    edges['topo_order'] = edges['reach_id'].map(level)
    max_order = edges['topo_order'].max()
    
    # Linear interpolation of Manning's n
    edges['manning_n'] = (
        n_head
        + (n_downstream - n_head) * edges['topo_order'] / max_order
    )
    
    return edges


def load_timeseries(rain_fp: str, q_fp: str, ash_fp: Optional[str] = None) -> pd.DataFrame:
    """
    Load rainfall, discharge, and optionally ashfall time series data.
    
    Args:
        rain_fp: Path to rainfall CSV file
        q_fp: Path to discharge CSV file
        ash_fp: Path to ashfall CSV file (optional)
        
    Returns:
        DataFrame with combined time series data
    """
    rain = pd.read_csv(rain_fp, parse_dates=['date']).set_index('date')
    qday = pd.read_csv(q_fp, parse_dates=['date']).set_index('date')
    df = rain.join(qday, how='inner').dropna()
    df['P'] = df['precip_mm'] / 1000  # m
    df['Q_obs'] = df['discharge_m3s']  # m3/s
    df['D'] = 0.0
    
    if ash_fp:
        ash = pd.read_csv(ash_fp, parse_dates=['date']).set_index('date')
        df['D'] = ash['ash_mm'] / 1000  # m
        
    return df


def fit_weights(edges: gpd.GeoDataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit catchment weights using non-negative least squares.
    
    Args:
        edges: GeoDataFrame with river reach data
        df: DataFrame with time series data
        
    Returns:
        DataFrame with catchment weights
    """
    catch = (
        edges[['catchment_id', 'catchment_area_m2']]
        .drop_duplicates('catchment_id')
        .set_index('catchment_id')
    )
    areas = catch['catchment_area_m2'].values
    X = np.outer(df['P'].values, areas)
    w, _ = nnls(X, df['Q_obs'].values)
    catch['w'] = w
    
    return catch


def runoff_coeff(H: float, C0: float, Hstar: float, Cmax: float, k2f: float) -> float:
    """
    Calculate runoff coefficient for ash mobilization.
    
    Args:
        H: Ash depth [m]
        C0: Baseline runoff coefficient
        Hstar: Threshold ash depth [m]
        Cmax: Maximum runoff coefficient
        k2f: Decay factor for ash depths > Hstar
        
    Returns:
        Runoff coefficient value
    """
    k1 = (Cmax - C0) / Hstar
    k2 = k1 * k2f
    
    if H <= Hstar:
        return C0 + k1 * H
    else:
        return max(0.0, C0 + k1 * Hstar - k2 * (H - Hstar))


def build_network(edges: gpd.GeoDataFrame) -> DCascade:
    """
    Build D-Cascade network from GeoDataFrame of river reaches.
    
    Args:
        edges: GeoDataFrame with river reach data
        
    Returns:
        DCascade model instance
    """
    reaches = {}
    for _, r in edges.iterrows():
        try:
            reach = Reach(
                id=int(r['reach_id']),
                downstream_id=int(r['downstream_id']) if pd.notna(r['downstream_id']) else None,
                length=float(r['length']),
                slope=float(r['slope']),
                manning_n=float(r['manning_n']),
                width=float(r['width']),
                storage=float(r.get('initial_storage_ash_kg', 0.0))
            )
            # attach hydrology metadata
            reach.metadata['A_sub'] = r['A_sub']
            reach.metadata['w'] = r['w']
            reaches[reach.id] = reach
        except (ValueError, KeyError) as e:
            print(f"Error creating reach {r['reach_id']}: {e}")
            
    return DCascade(reaches)


def run_model(edges: gpd.GeoDataFrame, 
              catch: pd.DataFrame, 
              df: pd.DataFrame, 
              rho_ash: float = 1000.0,
              validate_catchments: bool = True) -> Dict[str, float]:
    """
    Run the D-Cascade model for the entire time series.
    
    Args:
        edges: GeoDataFrame with river reach data
        catch: DataFrame with catchment parameters
        df: DataFrame with time series data
        rho_ash: Ash density [kg/m³]
        validate_catchments: Whether to validate catchment area distributions
        
    Returns:
        Dictionary with model results
    """
    catch = catch.copy()
    catch['H'] = 0.0
    total_washed = 0.0
    peak_Q = 0.0
    ash_deposits = []  # Track ash deposits over time

    dc = build_network(edges)
    
    # Calculate the total area per catchment and the portion contributing to each reach
    area_sum = edges.groupby('catchment_id')['A_sub'].sum().rename('A_sum')
    
    # Validate that A_sub sums properly add up to total catchment areas
    if validate_catchments:
        print("Validating catchment area distributions...")
        for catchment_id in edges['catchment_id'].unique():
            subset = edges[edges['catchment_id'] == catchment_id]
            total_a_sub = subset['A_sub'].sum()
            a_sum = area_sum.loc[catchment_id]
            if not np.isclose(total_a_sub, a_sum, rtol=1e-3):
                print(f"  Warning: A_sub sum ({total_a_sub:.2f}) does not match A_sum ({a_sum:.2f}) for catchment {catchment_id}")
                print(f"  This catchment is shared by {len(subset)} reaches")
        
        # Count and report shared catchments
        catchment_counts = edges['catchment_id'].value_counts()
        shared_catchments = catchment_counts[catchment_counts > 1]
        if len(shared_catchments) > 0:
            print(f"Found {len(shared_catchments)} catchments shared by multiple reaches:")
            for catchment_id, count in shared_catchments.items():
                print(f"  Catchment {catchment_id}: shared by {count} reaches")
    
    # Begin time series simulation
    for t, row in df.iterrows():
        P_t = row['P']
        D_t = row['D']
        catch['H'] += D_t
        catch['C_i'] = catch.apply(
            lambda ct: runoff_coeff(ct['H'], ct['C0'], ct['Hstar'], ct['Cmax'], ct['k2f']),
            axis=1
        )
        catch['M_avail'] = rho_ash * catch['catchment_area_m2'] * catch['H']
        catch['M_wash'] = catch['C_i'] * catch['M_avail']
        catch['H'] = (1 - catch['C_i']) * catch['H']

        # discharge metrics
        Qt = (catch['w'] * catch['catchment_area_m2'] * P_t).sum()
        peak_Q = max(peak_Q, Qt)
        total_washed += catch['M_wash'].sum()

        # Route ash to reaches, handling shared catchments
        # We merge the wash load with the edges and distribute proportionally
        # based on the subcatchment area contribution
        m = edges.merge(catch[['M_wash']], left_on='catchment_id', right_index=True)
        m = m.merge(area_sum, left_on='catchment_id', right_index=True)
        
        # Distribute sediment load proportionally to each reach
        # This ensures shared catchments correctly distribute their sediment
        m['M_edge'] = m['M_wash'] * (m['A_sub'] / m['A_sum'])
        
        # Check for any discrepancies in mass conservation
        if validate_catchments and t == df.index[0]:  # Only check first timestep
            mass_check = m.groupby('catchment_id').agg(
                total_edge_mass=('M_edge', 'sum'),
                catchment_mass=('M_wash', 'first')
            )
            mass_check['diff_percent'] = 100 * (mass_check['total_edge_mass'] - mass_check['catchment_mass']) / mass_check['catchment_mass']
            if not np.allclose(mass_check['diff_percent'], 0, atol=1e-3):
                problematic = mass_check[abs(mass_check['diff_percent']) > 1e-3]
                if len(problematic) > 0:
                    print("Warning: Mass conservation issue in these catchments:")
                    print(problematic)

        # Set discharge and inject sediment for each reach
        for _, er in m.iterrows():
            rid = int(er['reach_id'])
            rc = dc.reaches[rid]
            # set discharge forcing - proportionally based on subcatchment area
            rc.discharge = er['w'] * er['A_sub'] * P_t
            # inject ash - already proportionally distributed
            rc.add_sediment_mass('ash', er['M_edge'])
            
        dc.step(dt=1)
        
        # Store ash deposits after this timestep
        current_deposits = {rid: r.storage.get('ash', 0.0) for rid, r in dc.reaches.items()}
        ash_deposits.append({'date': t, 'deposits': current_deposits})

    return {
        'peak_discharge': peak_Q,
        'total_ash_washed': total_washed,
        'final_mean_H': catch['H'].mean(),
        'ash_deposits': ash_deposits,
        'shared_catchment_info': {
            'num_shared_catchments': len(shared_catchments) if validate_catchments else None,
            'max_reaches_per_catchment': catchment_counts.max() if validate_catchments else None
        }
    }


def sensitivity_sweep(edges: gpd.GeoDataFrame, 
                     catch: pd.DataFrame, 
                     df: pd.DataFrame, 
                     C0_vals: List[float], 
                     Hstar_vals: List[float], 
                     Cmax_vals: List[float], 
                     k2_vals: List[float]) -> pd.DataFrame:
    """
    Perform sensitivity analysis on key model parameters.
    
    Args:
        edges: GeoDataFrame with river reach data
        catch: DataFrame with catchment parameters
        df: DataFrame with time series data
        C0_vals: List of baseline runoff coefficient values
        Hstar_vals: List of threshold ash depth values
        Cmax_vals: List of maximum runoff coefficient values
        k2_vals: List of decay factor values
        
    Returns:
        DataFrame with sensitivity analysis results
    """
    records = []
    total_combinations = len(C0_vals) * len(Hstar_vals) * len(Cmax_vals) * len(k2_vals)
    completed = 0
    
    for C0 in C0_vals:
        for Hstar in Hstar_vals:
            for Cmax in Cmax_vals:
                for k2f in k2_vals:
                    completed += 1
                    print(f"Running scenario {completed}/{total_combinations}: C0={C0}, Hstar={Hstar}, Cmax={Cmax}, k2f={k2f}")
                    
                    # Set parameters
                    for col, val in zip(['C0', 'Hstar', 'Cmax', 'k2f'], [C0, Hstar, Cmax, k2f]):
                        catch[col] = val
                        
                    # Run model
                    try:
                        out = run_model(edges, catch, df)
                        out.update({'C0': C0, 'Hstar': Hstar, 'Cmax': Cmax, 'k2f': k2f})
                        records.append(out)
                    except Exception as e:
                        print(f"Error in scenario: {e}")
                        records.append({
                            'C0': C0, 'Hstar': Hstar, 'Cmax': Cmax, 'k2f': k2f,
                            'error': str(e)
                        })
                    
    return pd.DataFrame(records)


def synthetic_ashfall():


if __name__ == "__main__":
    edges_fp = "../data/joined_rivers_catchments.geojson"
    rain_fp = "../data/Rain_D.csv"
    q_fp = "../data/Q_Day.csv"
    ash_fp = "../data/Ashfall_mm.csv"

    print("=" * 80)
    print("D-CASCADE: Ash Transport in River Networks")
    print("=" * 80)
    print(f"Loading river network from: {edges_fp}")
    
    # load & preprocess
    edges = load_edges(edges_fp)
    
    # Analyze network topology
    print(f"Network loaded: {len(edges)} reaches")
    
    # Check for shared catchments
    catchment_counts = edges['catchment_id'].value_counts()
    print(f"Found {len(catchment_counts)} unique catchments")
    shared_catchments = catchment_counts[catchment_counts > 1]
    if len(shared_catchments) > 0:
        print(f"Note: {len(shared_catchments)} catchments are shared by multiple reaches")
        print("Top 5 most shared catchments:")
        print(shared_catchments.sort_values(ascending=False).head(5))
    
    # Add topological order and Manning's n values
    print("\nCalculating topological order and Manning's n values...")
    edges = add_topo_and_manning(edges,
                              n_downstream=0.035,  # known at outlet
                              n_head=0.05)         # assumed for headwaters
    
    print(f"Topological levels: {edges['topo_order'].min()} to {edges['topo_order'].max()}")
    print(f"Manning's n range: {edges['manning_n'].min():.3f} to {edges['manning_n'].max():.3f}")

    print("\nLoading time series data...")
    df = load_timeseries(rain_fp, q_fp, ash_fp)
    print(f"Time series loaded: {len(df)} days from {df.index.min().date()} to {df.index.max().date()}")
    
    print("\nFitting catchment weights...")
    catch = fit_weights(edges, df)
    print(f"Catchment weight range: {catch['w'].min():.4f} to {catch['w'].max():.4f}")
    
    # Add required parameters for runoff model
    catch['C0'] = 0.2     # Baseline runoff coefficient
    catch['Hstar'] = 0.005  # Threshold ash depth [m]
    catch['Cmax'] = 0.8    # Maximum runoff coefficient
    catch['k2f'] = 0.5     # Decay factor
    
    print("\nMerging catchment weights with river network...")
    edges = edges.merge(catch[['w']], left_on='catchment_id', right_index=True)

    print("\nRunning model with default parameters...")
    results = run_model(edges, catch, df, validate_catchments=True)
    
    print("\nBasic model results:")
    print(f"  Peak discharge: {results['peak_discharge']:.2f} m³/s")
    print(f"  Total ash washed: {results['total_ash_washed']/1000:.2f} metric tons")
    print(f"  Final mean ash depth: {results['final_mean_H']*1000:.2f} mm")
    
    if results['shared_catchment_info']['num_shared_catchments'] is not None:
        print(f"  Shared catchments: {results['shared_catchment_info']['num_shared_catchments']}")
        print(f"  Maximum reaches per catchment: {results['shared_catchment_info']['max_reaches_per_catchment']}")


    # Create a visualization function for catchment sharing
    def create_catchment_sharing_viz(edges_gdf):
        """Create visualization of catchment sharing."""
        # This would be implemented with matplotlib or other visualization tools
        # For now, just print a summary
        catchment_sharing = edges_gdf.groupby('catchment_id').size().value_counts()
        print("\nCatchment sharing statistics:")
        print(f"  Number of catchments shared by N reaches:")
        for num_reaches, count in catchment_sharing.items():
            print(f"    {num_reaches} reach(es): {count} catchment(s)")
    
    create_catchment_sharing_viz(edges)


    print("\nDefining parameter grids for sensitivity analysis...")
    # define parameter grids for sensitivity analysis
    C0_vals = np.linspace(0.1, 0.3, 5)
    Hstar_vals = np.linspace(0.001, 0.01, 5)
    Cmax_vals = np.linspace(0.6, 0.9, 4)
    k2_vals = np.linspace(0.2, 1.0, 5)
    
    print(f"Total parameter combinations: {len(C0_vals) * len(Hstar_vals) * len(Cmax_vals) * len(k2_vals)}")
    
    run_sensitivity = input("\nRun sensitivity analysis? (y/n): ").lower().strip() == 'y'
    
    if run_sensitivity:
        print("\nRunning sensitivity analysis...")
        df_res = sensitivity_sweep(edges, catch, df,
                                  C0_vals, Hstar_vals, Cmax_vals, k2_vals)
        output_file = "sensitivity_results.csv"
        df_res.to_csv(output_file, index=False)
        print(f"Done → {output_file}")
        
        # Show summary of sensitivity results
        print("\nSensitivity analysis summary:")
        print(f"  Parameter combinations tested: {len(df_res)}")
        print(f"  Peak discharge range: {df_res['peak_discharge'].min():.2f} to {df_res['peak_discharge'].max():.2f} m³/s")
        print(f"  Total ash washed range: {df_res['total_ash_washed'].min()/1000:.2f} to {df_res['total_ash_washed'].max()/1000:.2f} metric tons")
        
        # Find most sensitive parameters
        for metric in ['peak_discharge', 'total_ash_washed', 'final_mean_H']:
            print(f"\nParameters with highest impact on {metric}:")
            
            # Simple correlation analysis
            corr = df_res[[metric, 'C0', 'Hstar', 'Cmax', 'k2f']].corr()[metric].abs().sort_values(ascending=False)
            print(corr[1:])  # Skip self-correlation
    else:
        print("\nSkipping sensitivity analysis.")
        
    print("\nModel execution complete.")
