#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def main():
    cols = list(range(2, 26))
    data2013 = pd.read_csv("wheat-2013-supervised.csv", usecols=cols)
    data2014 = pd.read_csv("wheat-2014-supervised.csv", usecols=cols)
    data = pd.concat([data2013, data2014])
    unique_locations = data[['Latitude', 'Longitude']].drop_duplicates()
    print("# of unique locations: {}".format(len(unique_locations)))

    data['Date'] = pd.to_datetime(data['Date'].str.rstrip('0: '),
                                  format="%m/%d/%Y")

    min_lat = unique_locations['Latitude'].min()
    max_lat = unique_locations['Latitude'].max()
    min_lon = unique_locations['Longitude'].min()
    max_lon = unique_locations['Longitude'].max()
    avg_lat = .5 * (max_lat - min_lat) + min_lat
    avg_lon = .5 * (max_lon - min_lon) + min_lon

    for i, data_by_date in enumerate(data.groupby('Date')):
        lons = data_by_date[1]['Longitude'].values
        lats = data_by_date[1]['Latitude'].values

        fig = plt.figure()
        fig.add_subplot(111)
        fig.suptitle(data_by_date[0])

        lat_u_margin = 0
        lat_l_margin = 2
        lon_l_margin = 0
        lon_r_margin = 10
        m = Basemap(projection='stere',
                    llcrnrlat=min_lat-lat_l_margin,
                    urcrnrlat=max_lat+lat_u_margin,
                    llcrnrlon=min_lon-lon_l_margin,
                    urcrnrlon=max_lon+lon_r_margin,
                    lon_0=avg_lon, lat_0=avg_lat)
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()
        m.drawcounties()
        sc = m.scatter(lons, lats, c=data_by_date[1]['Yield'], latlon=True)
        cbar = m.colorbar(sc, location='right', pad="5%")
        cbar.set_label('Yield')

        if i > 2:
            break
    plt.show()

    for col in data:
        if col in ('Date', 'Yield'):
            continue
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(col)
        ax.set_ylabel('Yield')
        ax.scatter(data[col], data['Yield'])
        plt.show()


if __name__ == "__main__":
    main()
