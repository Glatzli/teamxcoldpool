"""Script to read in hatrpo file"""
import pandas as pd
import xarray as xr
from metpy.units import units

from main_code.confg import hatpro_vertical_levels, hatpro_folder


def __read_hatpro_intern(filepath):
    """internally used function to read in hatpro Temperature or Humidity depending on the filepath (height in meter)"""
    height_int = [int(height) for height in hatpro_vertical_levels["height"]]

    # Read in the DataFrame from the CSV file
    df = pd.read_csv(filepath,
                     sep=";")

    # Convert the 'rawdate' column to datetime if it's not already
    df['rawdate'] = pd.to_datetime(df['rawdate'])

    # Set the 'rawdate' column as the index
    df.set_index('rawdate', inplace=True)

    # Rename the columns to v01, v02, ..., v39
    df.columns = [f"v{i:02d}" for i in range(1, 40)]

    # Create a new index that includes 'rawdate' and 'v1' to 'v39'
    new_index = pd.MultiIndex.from_product([df.index, df.columns], names=['rawdate', 'height_level'])

    # Create a new DataFrame with the new index
    if "temp" in filepath:
        df_new = pd.DataFrame(index=new_index, data=df.values.flatten(), columns=['T'])
    elif "humidity" in filepath:
        df_new = pd.DataFrame(index=new_index, data=df.values.flatten(), columns=['humidity'])

    # Convert the DataFrame to an xarray dataset
    dataset = xr.Dataset.from_dataframe(df_new)

    # Assign the 'height_level' coordinate
    dataset["height_level"] = height_int
    if "T" in list(dataset.keys()):
        # Set the units attribute for temperature variable 'T'
        dataset["T"].attrs['units'] = "K"

        dataset["T"].values = dataset["T"].values * units.kelvin
        dataset["T"] = dataset["T"].metpy.convert_units("degC")
    elif "humidity" in list(dataset.keys()):
        dataset["humidity"].attrs['units'] = "g/m^3"  # absolute humidity
        dataset['humidity'] = dataset['humidity'].metpy.convert_units("g/m^3")

        print(dataset["humidity"])
    return dataset


def read_hatpro_extern():
    """read hatpro function can be called from externally"""
    dataset1 = __read_hatpro_intern(
        filepath=f"{hatpro_folder}/data_HATPRO_temp.csv")
    dataset2 = __read_hatpro_intern(
        filepath=f"{hatpro_folder}/data_HATPRO_humidity.csv")
    merged_dataset = xr.merge([dataset1, dataset2])
    return merged_dataset
