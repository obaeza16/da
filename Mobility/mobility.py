import os
import requests
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from keplergl import KeplerGl
from shapely import geometry
from shapely.geometry import Point
from functools import partial
import pyproj
from shapely.ops import transform
from shapely.geometry import mapping
import mercantile
from datetime import datetime, timedelta

# Taken from https://www.mapbox.com/blog/how-to-utilize-mapbox-movement-data-for-mobility-insights-a-guide-for-analysts-data-scientists-and-developers

def download_csv_to_df(url, parse_dates=False, sep=","):
    """
    Given a URL, downloads and reads the data to a Pandas DataFrame.

    Args:
        url (string): Includes a protocol (http), a hostname (www.example.com),
        and a file name (e.g., index.html).
        parse_dates (boolean or list): Boolean or list of column names (str), default False.
        sep (string): The separator used in the CSV, default ",".

    Raises:
        requests.exceptions.RequestException: An exception raised while
        handling your request.

    Returns:
        A Pandas DataFrame.
    """
    try:
        res = requests.get(url)
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    return pd.read_csv(StringIO(res.text), parse_dates=parse_dates, sep=sep)



# Download the sample CSV file consisting of January 1st, 2020 Movement data covering the San Francisco Bay Area.
movement_url = 'https://mapbox-movement-public-sample-dataset.s3.amazonaws.com/v0.2/daily-24h/v2.0/US/quadkey/total/2020/01/01/data/0230102.csv'
movement_df = download_csv_to_df(movement_url, sep="|")

# Rename a few columns for clarity:
movement_df = movement_df.rename(
columns={"xlon": "lon", "xlat": "lat"}
)[["lon", "lat", "activity_index_total"]]

# Load an empty keplergl map
sample_z7_map = KeplerGl()

# Add movement sample csv. Kepler accepts CSV, GeoJSON or DataFrame!
sample_z7_map.add_data(data=movement_df, name='data_2')

# Optional: Save map to an html file
sample_z7_map.save_to_html(file_name='sample_z7_map.html')

# Load kepler.gl widget below a cell
sample_z7_map

# Download airport dataset
airport_url = "https://ourairports.com/data/airports.csv"
airports_df = download_csv_to_df(airport_url)

# Drop unnecessary columns
airports_df = airports_df.drop(columns=['id', 'ident', 'elevation_ft','continent', 'scheduled_service', 'gps_code', 'local_code', 'home_link', 'wikipedia_link', 'keywords'])

# Filter to US only airports
us_airports_df = airports_df[airports_df["iso_country"] == 'US']
es_airports_df = airports_df[airports_df['iso_country'] == 'ES']

# Filter dataframe to include only large airports
us_airports_df_large = us_airports_df[us_airports_df["type"] == "large_airport" ]
us_airports_df_large = us_airports_df_large.sort_values(by=["iso_region"])

es_airports_df_large = es_airports_df[es_airports_df["type"] == "large_airport" ]
es_airports_df_large = es_airports_df_large.sort_values(by=["iso_region"])

# Inspect the first few rows of the DataFrame
us_airports_df_large.head()
es_airports_df_large.head()

# Integrate longitude and latitude into a Shapely point
def coords2points(df_row):
    """
    Converts "longitude_deg" and "latitude_deg" to Shapely Point object.

    Args:
    df_row (Pandas DataFrame row): a particular airport (row) or a pandas.core.series.Series
    from the original airports DataFrame.

    Returns:
    A shapely.geometry.point object.
    """
    return Point(df_row['longitude_deg'], df_row['latitude_deg'])

# Apply the coords2points function to each row and store results in a new column
us_airports_df_large['airport_center'] = us_airports_df_large.apply(coords2points, axis=1)

# Next, we will create a circle with a radius 1,000 meters around each airport’s Shapely Point
# object via Shapely’s buffer() method. This circle will serve as our “approximate boundary” 
# for every airport. It is stored as a Shapely Polygon object. Because we will be calculating
# circles on the surface of the Earth, this process involves a bit of projection magic. For 
# more information about why we need to transform the coordinates back and forth between AEQD 
# projection and WGS84, please see section I in the Appendix.

def aeqd_reproj_buffer(center, radius=1000):
    """
    Converts center coordinates to AEQD projection,
    draws a circle of given radius around the center coordinates,
    converts both polygons back to WGS84.

    Args:
    center (shapely.geometry Point): center coordinates of a circle.
    radius (integer): circle's radius in meters.

    Returns:
    A shapely.geometry Polygon object for cirlce of given radius.
    """
    # Get the latitude, longitude of the center coordinates
    lat = center.y
    long = center.x

    # Define the projections
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(
    lat, long
    )
    wgs84_to_aeqd = partial(
    pyproj.transform,
    pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
    pyproj.Proj(local_azimuthal_projection),
    )
    aeqd_to_wgs84 = partial(
    pyproj.transform,
    pyproj.Proj(local_azimuthal_projection),
    pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
    )

    # Transform the center coordinates from WGS84 to AEQD
    point_transformed = transform(wgs84_to_aeqd, center)
    buffer = point_transformed.buffer(radius)

    # Get the polygon with lat lon coordinates
    circle_poly = transform(aeqd_to_wgs84, buffer)

    return circle_poly

# Map the aeqd_reproj_buffer function to all airport coordinates.
us_airports_df_large["aeqd_reproj_circle"] = us_airports_df_large["airport_center"].apply(aeqd_reproj_buffer)


def plot_circles_kepler(airport, to_html=False):
    """
    Plots the center and 1 km circle polygons using kepler.

    Args:
    airport (pandas.core.series.Series): an specific airport (row) from the
    main airports DataFrame.
    to_html (boolean): if True, saves map to an html file named
    {airport_iata_code}_circle.html.

    Returns:
    a kepler interactive map object.
    """
    # Get the center (Shapely Point), circle (Shapely Polygon), and iata code of the airport
    circle_center = airport["airport_center"]
    circle_poly = airport["aeqd_reproj_circle"]
    iata = airport["iata_code"]

    # Define the GeoJSON object for the 1 km circle
    circle_feature = {
    "type": "Feature",
    "properties": {"name": "Circle"},
    "geometry": mapping(circle_poly)
    }

    # Define the GeoJSON object for the center of the circle
    center_feature = {
    "type": "Feature",
    "properties": {"name": "Center"},
    "geometry": mapping(circle_center)
    }

    # Define the Kepler map configurations
    config = {
    'version': 'v1',
    'config': {
    'mapState': {
    'latitude': airport["latitude_deg"],
    'longitude': airport["longitude_deg"],
    'zoom': 12
    }
    }
    }

    # Load the keplergl map of the circle and add center
    circle_map = KeplerGl(data={'Circle': circle_feature})
    circle_map.add_data(data=center_feature, name='Center')
    circle_map.config = config

    # Optional: Save map to an html file
    if to_html:
        circle_map.save_to_html(file_name=f'{iata}_circle.html')

    return circle_map



# Plot 1 km circle polygon around San Francisco International Airport
sfo = us_airports_df_large[us_airports_df_large["iata_code"] == "SFO"].iloc[0]
plot_circles_kepler(sfo, to_html=True)



def generate_quadkeys(circle_poly, zoom):
    """
    Generate a list of quadkeys that overlap with an airport.

    Args:
    circle_poly (shapely.geometry Polygon): circle polygon object drawn
    around an airport.
    zoom (integer): zoom level.

    Return:
    List of quadkeys as string
    """

    return [mercantile.quadkey(x) for x in mercantile.tiles(*circle_poly.bounds, zoom)]


# Create a list of overlapping z18 quadkeys for each airport and add to a new column
us_airports_df_large['z18_quadkeys'] = us_airports_df_large.apply(lambda x: generate_quadkeys(x['aeqd_reproj_circle'], 18),axis=1)

# Create a list of overlapping z7 quadkeys for each airport and add to a new column
us_airports_df_large['z7_quadkeys'] = us_airports_df_large.apply(lambda x: generate_quadkeys(x['aeqd_reproj_circle'], 7),axis=1)

def download_sample_data_to_df(start_date, end_date, z7_quadkey_list, local_dir, verbose=True):
    """
    Downloads Movement z7 quadkey CSV files to a local dir and reads
    Read the CSV file into a Pandas DataFrame. This DataFrame contains
    **ALL** Z18 quadkey data in this Z7 quadkey.

    Args:
    start_date (string): start date as YYYY-MM-DD string.
    end_date (string): end date as YYYY-MM-DD string.
    z7_quadkey_list (list): list of zoom 7 quadkeys as string.
    local_dir (string): local directory to store downloaded sample data.
    verbose (boolean): print download status.

    Raises:
    requests.exceptions.RequestException: An exception raised while
    handling your request.

    Returns:
    a Pandas DataFrame consists of Z7 quadkey data. DataFrame contains
    **ALL** Z18 quadkey data in this Z7 quadkey.

    """

    bucket = "mapbox-movement-public-sample-dataset"

    # Generate range of dates between start and end date in %Y-%m-%d string format
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    num_days = int((end - start).days)
    days_range = num_days + 1
    date_range = [(start + timedelta(n)).strftime('%Y-%m-%d') for n in range(days_range)]

    sample_data = []
    for z7_quadkey in z7_quadkey_list:
        for i in range(len(date_range)):
            yr, month, date = date_range[i].split('-')
            url = f"https://{bucket}.s3.amazonaws.com/v0.2/daily-24h/v2.0/US/quadkey/total/2020/{month}/{date}/data/{z7_quadkey}.csv"

    if not os.path.isdir(os.path.join(local_dir, month, date)):
        os.makedirs(os.path.join(local_dir, month, date))

    local_path = os.path.join(local_dir, month, date, f'{z7_quadkey}.csv')

    if verbose:
        print (z7_quadkey, month, date)
    print (f'local_path : {local_path}')

    try:
        res = requests.get(url)
        df = pd.read_csv(StringIO(res.text), sep='|')
        convert_dict = {'agg_day_period': 'datetime64[ns]', 'activity_index_total': float, 'geography': str}
        df = df.astype(convert_dict)
        # Keep leading zeros and save as string
        df['z18_quadkey'] = df.apply(lambda x: x['geography'].zfill(18), axis=1).astype('str')
        df['z7_quadkey'] = df.apply(lambda x: x['geography'][:6].zfill(7), axis=1).astype('str')

        sample_data.append(df)
        df.to_csv(local_path, index=False)

    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

    if verbose:
        print (f'Download completed for {z7_quadkey} over date range {start_dt_str} to {end_dt_str}')

    return pd.concat(sample_data)


# Tweak the following set of parameters to include broader time frame or more airports
start_dt_str = "2020-01-01"
end_dt_str = "2020-01-02"

# Find SFO and DEN from the us_airports_df_large DataFrame
sfo = us_airports_df_large[us_airports_df_large["iata_code"] == "SFO"].iloc[0]
den = us_airports_df_large[us_airports_df_large['name'].str.contains("Denver")].iloc[0]

# Add these two airports
airports = [sfo, den]

# Creates a list of z7 quadkeys to download
z7_quadkeys_to_download = []
for airport in airports:
    for z7_quadkey in airport["z7_quadkeys"]:
        z7_quadkeys_to_download.append(z7_quadkey)

# Define a list to append all newly created DataFrames
sample_data_airports = []
for z7 in z7_quadkeys_to_download:
    local_directory = os.path.join(os.getcwd(), f'sample_data_{z7}')
print ([z7])
print (local_directory)

# Run the download script
sample_data_airports.append(download_sample_data_to_df(start_dt_str, end_dt_str, [z7], local_directory))

# Create a DataFrame of all z7 quadkey activity data
sample_data_airports_df = pd.concat(sample_data_airports).sort_values(by=['agg_day_period', 'z18_quadkey'])

# Filter DataFrame to include activity data from SFO z7 quadkeys only. Excludes DEN z7 quadkey data.
sample_data_sfo = []
for z7_quadkey in sfo["z7_quadkeys"]:
    sample_data_sfo.append(sample_data_airports_df[sample_data_airports_df["z7_quadkey"] == z7_quadkey])

# Create a DataFrame for all SFO activity data
sample_data_df_sfo = pd.concat(sample_data_sfo)

# Filter the DataFrame to include only entries of Z18 quadkey that overlap with the 1 km circle
z18_sample_data_sfo_df = sample_data_df_sfo[sample_data_df_sfo["z18_quadkey"].isin([z18_qk for z18_qk in sfo["z18_quadkeys"]])]


# Find the San Francisco International Airport row in the DataFrame
sfo = us_airports_df_large[us_airports_df_large["iata_code"] == "SFO"].iloc[0]

# Define the url of SFO's pre-processed CSV file
z18_sample_data_sfo_url = 'https://mapbox-movement-public-sample-dataset.s3.amazonaws.com/airports_dataset/data/z18_sample_data_sfo.csv'

# Download filtered sample data for sfo and read into a DataFrame. Parse agg_day_period as datetime object
z18_sample_data_sfo_df = download_csv_to_df(z18_sample_data_sfo_url, parse_dates=['agg_day_period'])

# Add airport iata code to help identify each airport when we're plotting statistics later on
z18_sample_data_sfo_df["iata"] = sfo["iata_code"]

# Sort the DataFrame by date and inspect the DataFrame
z18_sample_data_sfo_df = z18_sample_data_sfo_df.sort_values(by=['agg_day_period', 'z18_quadkey'])
z18_sample_data_sfo_df

# Find the Denver International Airport row in the DataFrame
den = us_airports_df_large[us_airports_df_large['name'].str.contains("Denver")].iloc[0]

# Define the url of DEN's pre-processed CSV file
z18_sample_data_den_url = 'https://mapbox-movement-public-sample-dataset.s3.amazonaws.com/airports_dataset/data/z18_sample_data_den.csv'

# Download sample data for den and read into a DataFrame. Parse agg_day_period as datetime object
z18_sample_data_den_df = download_csv_to_df(z18_sample_data_den_url, parse_dates=['agg_day_period'])

# Add airport iata code to help identify each airport when we're plotting statistics later on
z18_sample_data_den_df["iata"] = den["iata_code"]

# Inspect the DataFrame
z18_sample_data_den_df


# Find the Dallas/Fort Worth International Airport row in the DataFrame
dfw = us_airports_df_large[us_airports_df_large['name'].str.contains("Dallas Fort Worth International Airport")].iloc[0]

# Define the url of DFW's pre-processed CSV file
z18_sample_data_dfw_url = 'https://mapbox-movement-public-sample-dataset.s3.amazonaws.com/airports_dataset/data/z18_sample_data_dfw.csv'

# Download sample data for dfw and read into a DataFrame. Parse agg_day_period as datetime object
z18_sample_data_dfw_df = download_csv_to_df(z18_sample_data_dfw_url, parse_dates=['agg_day_period'])

# Add airport iata code to help identify each airport when we're plotting statistics later on
z18_sample_data_dfw_df["iata"] = dfw["iata_code"]

# Inspect the DataFrame
z18_sample_data_dfw_df

# Find the Dulles International Aiport row in the DataFrame
iad = us_airports_df_large[us_airports_df_large['name'].str.contains("Washington Dulles International Airport")].iloc[0]

# Define the url of IAD's pre-processed CSV file
z18_sample_data_iad_url = 'https://mapbox-movement-public-sample-dataset.s3.amazonaws.com/airports_dataset/data/z18_sample_data_iad.csv'

# Download sample data for iad and read into a DataFrame. Parse agg_day_period as datetime object
z18_sample_data_iad_df = download_csv_to_df(z18_sample_data_iad_url, parse_dates=['agg_day_period'])

# Add airport iata code to help identify each airport when we're plotting statistics later on
z18_sample_data_iad_df["iata"] = iad["iata_code"]

# Inspect the DataFrame
z18_sample_data_iad_df

# Calculate the aggregated activity across all z18 quadkeys per day for SFO and construct a DataFrame with date as the index
z18_sample_df = z18_sample_data_sfo_df.copy()
sum_data = z18_sample_df.groupby(['agg_day_period']).sum()['activity_index_total']

sfo_ai_stats_df = pd.DataFrame(sum_data).reset_index().rename(columns={'activity_index_total':'sum_ai_daily'})
sfo_ai_stats_df = sfo_ai_stats_df.set_index('agg_day_period')

# Inspect the DataFrame
sfo_ai_stats_df

# Find the timestamp for Jan 4 to Jan 31, 2020 and generate a list of dates
jan_start_date = datetime.strptime('2020-01-04', "%Y-%m-%d")
jan_end_date = datetime.strptime('2020-01-31', "%Y-%m-%d")
jan_dates = pd.date_range(jan_start_date, jan_end_date, freq='D')

# Compute the aggregated activity from Jan 4 to Jan 31, 2020 for SFO
sfo_sum_ai_Jan_2020 = sfo_ai_stats_df[sfo_ai_stats_df.index.isin([x for x in jan_dates])].sum()['sum_ai_daily']
sfo_ai_stats_df["sum_ai_Jan_2020"] = sfo_sum_ai_Jan_2020

# Normalize the daily sum of activity by the total sum of activity from Jan 4 to Jan 31, 2020, add to DataFrame.
sfo_ai_stats_df["normalized_sum"] = 28 * sfo_ai_stats_df["sum_ai_daily"] / sfo_sum_ai_Jan_2020

# Inspect the DataFrame
sfo_ai_stats_df

# Compute the 7-day rolling average of the daily aggregated activity add to the DataFrame
sfo_ai_stats_df["sum_ai_daily_rolling_avg"] = sfo_ai_stats_df["sum_ai_daily"].rolling(window=7).mean()

# Inspect the DataFrame
sfo_ai_stats_df

# Compute the 7-day rolling average of the normalized daily aggregated activity
sfo_ai_stats_df["norm_sum_rolling_avg"] = sfo_ai_stats_df["sum_ai_daily"].rolling(window=7).mean()

# Inspect the DataFrame
sfo_ai_stats_df

def generate_airport_stats(airport_z18_sample_df,
                           label='iata',
                           normalizer_start_date='2020-01-04',
                           normalizer_end_date='2020-01-31'):
    """
    Given a Pandas DataFrame consists of an airport's Movement sample data,
    generate a new Pandas DataFrame consists of the aggregated activity index,
    7-day rolling average of aggregated activity, normalized activity index,
    7-day rolling average of normalized activity.
    
    
    Args:
        airport_z18_sample_df (Pandas DataFrame): DataFrame consists of data from 
            z18 quadkey that overlaps with 1 km circle. Should include "iata" of the airport as a column.
        label (string): 
            
    Raises:
        ValueError: If 'iata' column is missing from the airport_z18_sample_df DataFrame.
        TypeError: If 'agg_day_period' column is datetime.datetime.
        
    Returns:
        airport_stats_ai_df(Pandas DataFrame): DataFrame consists of airport statistics.
    """
    
    if label == 'iata' and 'iata' not in airport_z18_sample_df.columns:
        raise ValueError('Missing iata column in the Pandas DataFrame')
        
    if not isinstance(airport_z18_sample_df['agg_day_period'].iloc[0], datetime):
        errstr = "agg_day_period is of type " + type(airport_z18_sample_df['agg_day_period'].iloc[0]).__name__ + \
                ", should be of datetime.datetime object"
        raise TypeError(errstr)
        
    if label == 'iata':
        col_label = airport_z18_sample_df['iata'].iloc[0]
    else:
        col_label = label

    airport_stats_ai_df = airport_z18_sample_df.copy()

    # Calculate the aggregated AI per day
    airport_stats_ai_df = airport_stats_ai_df.groupby(['agg_day_period']).sum()["activity_index_total"].reset_index()
    airport_stats_ai_df = airport_stats_ai_df.rename(columns={'activity_index_total': f'sum_ai_daily_{col_label}'})
    airport_stats_ai_df = airport_stats_ai_df.set_index('agg_day_period')

    # Create a list of dates between start_date and end_date
    jan_start = datetime.strptime(normalizer_start_date, "%Y-%m-%d")
    jan_end = datetime.strptime(normalizer_end_date, "%Y-%m-%d")
    jan_dates = pd.date_range(jan_start, jan_end, freq='D')


    sum_ai_Jan_2020 = airport_stats_ai_df[airport_stats_ai_df.index.isin([x for x in jan_dates])].sum()[f'sum_ai_daily_{col_label}']
    airport_stats_ai_df[f'normalized_sum_{col_label}'] = 28 * airport_stats_ai_df[f'sum_ai_daily_{col_label}'] / sum_ai_Jan_2020
    airport_stats_ai_df[f'sum_ai_daily_rolling_avg_{col_label}'] = airport_stats_ai_df[f'sum_ai_daily_{col_label}'].rolling(window=7).mean()
    airport_stats_ai_df[f'norm_sum_rolling_avg_{col_label}'] = airport_stats_ai_df[f'normalized_sum_{col_label}'].rolling(window=7).mean()

    return airport_stats_ai_df


# Generate adjusted daily stats for all four airport, append them to a list and create a large DataFrame
all_airports_stats_ai = []
airport_z18_sample_dfList = [z18_sample_data_sfo_df, z18_sample_data_den_df, z18_sample_data_dfw_df, z18_sample_data_iad_df]

for airport_z18_sample_df in airport_z18_sample_dfList:
    all_airports_stats_ai.append(generate_airport_stats(airport_z18_sample_df))
all_airports_stats_ai_df = pd.concat(all_airports_stats_ai, axis=1)

# Append airport iata code to column names, so that one airport's statistic is distinguishable from another.
all_airports_stats_ai_df

## Plot daily aggregated AI for all four airports
all_airports_sum = all_airports_stats_ai_df[[sum_col for sum_col in all_airports_stats_ai_df.columns if 'sum_ai_daily' in sum_col and 'rolling' not in sum_col]]

# Use the default colors from color cycle. You can also pass in a list of colors to the color parameter.
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
all_airports_sum.plot(color=colors, figsize=(20, 8))
plt.legend(all_airports_sum.columns, loc='upper right')
plt.title('Aggregated Daily Activity Index: SFO, DEN, DFW, IAD')
plt.show()

all_airports_sum_rolling = all_airports_stats_ai_df[[sum_col for sum_col in all_airports_stats_ai_df.columns if 'sum_ai_daily_rolling' in sum_col]]
all_airports_sum_rolling.plot(figsize=(20, 8))
plt.legend(all_airports_sum_rolling.columns, loc='upper right')
plt.title('7-day Rolling Average of Aggregated Daily Activity Index: SFO, DEN, DFW, IAD')
plt.show()

## Plot normalized daily aggregated AI for all four airports
all_airports_norm = all_airports_stats_ai_df[[sum_col for sum_col in all_airports_stats_ai_df.columns if 'normalized_sum' in sum_col and 'rolling' not in sum_col]]
all_airports_norm.plot(figsize=(20, 8))
plt.legend(all_airports_norm.columns, loc='upper right')
plt.title('Normalized Aggregated Daily Activity Index: SFO, DEN, DFW, IAD')
plt.show()

all_airports_norm_rolling = all_airports_stats_ai_df[[sum_col for sum_col in all_airports_stats_ai_df.columns if 'norm_sum_rolling' in sum_col]]
all_airports_norm_rolling.plot(figsize=(20, 8))
plt.legend(all_airports_norm_rolling.columns, loc='upper right')
plt.title('7-day Rolling Average of Normalized Aggregated Daily Activity Index: SFO, DEN, DFW, IAD')
plt.show()

