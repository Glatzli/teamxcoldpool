"""Create a gif where the AROME temperature on 2D and the wind are visible"""

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from main_code.AROME.read_in_arome import read_2D_variables_AROME, read_3D_variables_AROME
from main_code.confg import cities, dir_PLOTS


def animate(i):
    if i >= 0:
        ax.clear()  # Clear the axis to draw a new frame
        temperature = ds['tsk'].isel(time=i)  # Select the temperature data for the current time step
        geopotential_height = df["z"].isel(time=i)

        ax.pcolormesh(temperature.longitude, temperature.latitude, temperature,
                      shading='auto', cmap='coolwarm')

        geopotential_height_contours = ax.contour(geopotential_height.longitude, geopotential_height.latitude,
                                                  geopotential_height,
                                                  colors='black')  # You can adjust the colors and number of levels

        # Optionally, add labels to the contours
        ax.clabel(geopotential_height_contours, inline=True, fontsize=8)

        # Plot wind vectors
        subsample = 4
        u_sub = df['u'].isel(time=i)[::subsample, ::subsample]
        v_sub = df['v'].isel(time=i)[::subsample, ::subsample]

        # Plot wind vectors
        ax.quiver(df.longitude[::subsample], df.latitude[::subsample], u_sub, v_sub,
                  color='black', scale=50)

        for city, coord in cities.items():
            if city == "Muenchen":
                continue

            ax.plot(coord['lon'], coord['lat'], 'o', color="grey")  # 'ro' for red circles, change as needed

            if city == "Innsbruck Airport":
                ax.text(coord['lon'], coord['lat'] - 0.02, city)  # innsbruck text collides

            else:
                ax.text(coord['lon'], coord['lat'], city)  # Add city name text

        # Set titles and labels
        ax.set_title(f'Surface Temperature on {temperature.time.values} in Celsius')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')


# GIF Beschreibung
if __name__ == '__main__':
    # slice over lat and long
    latitude = slice(47.1, 48.2)
    longitude = slice(10.5, 12.3)
    time = slice(4, None)

    ds = read_2D_variables_AROME(variableList=["tsk"], lat=latitude, lon=longitude, slice_lat_lon=True)
    ds["tsk"] = ds["tsk"].metpy.convert_units("degC")
    # ds["ps"] = ds["ps"].metpy.convert_units("hPa")

    df = read_3D_variables_AROME(variables=["u", "v", "z"], method="sel", level=90, lat=latitude, lon=longitude,
                                 slice_lat_lon=True)

    # Plot a GIF of the temperature of 2D
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # Dummy plot for colorbar creation
    temp_plot = ax.pcolormesh(ds['longitude'], ds['latitude'], ds['tsk'].isel(time=0), shading='auto', cmap='coolwarm')

    cbar = fig.colorbar(temp_plot, ax=ax, label='Temperature (°C)')  # Add color bar only once

    # Create the animation, iterating over all time steps
    anim = FuncAnimation(fig, animate, frames=len(ds.time), interval=1500)

    # Save the animation as a GIF
    anim.save(f'{dir_PLOTS}/temperature/temperature_animation_zoomed_Innvalley_with_wind.gif', writer='pillow', fps=1)
    print("Animation has finished!")
