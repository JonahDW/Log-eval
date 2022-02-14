import sys
import re
import numpy as np

from astropy.time import Time
from astropy.time import TimeFITS

from astropy.coordinates import Angle
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii

def parse_file(filename):
    '''
    Read a file and return a list of its lines
    '''
    with open(filename) as f:
        file = f.read()
        lines = file.split('\n')

    return lines

def match_format_string(format_str, s):
    '''
    Match s against the given format string, return dict of matches.

    We assume all of the arguments in format string are named keyword arguments (i.e. no {} or
    {:0.2f}). We also assume that all chars are allowed in each keyword argument, so separators
    need to be present which aren't present in the keyword arguments (i.e. '{one}{two}' won't work
    reliably as a format string but '{one}-{two}' will if the hyphen isn't used in {one} or {two}).

    We raise if the format string does not match s.
    Source: https://stackoverflow.com/questions/10663093/use-python-format-string-in-reverse-for-parsing

    Example:
    fs = '{test}-{flight}-{go}'
    s = fs.format('first', 'second', 'third')
    match_format_string(fs, s) -> {'test': 'first', 'flight': 'second', 'go': 'third'}
    '''

    # First split on any keyword arguments, note that the names of keyword arguments will be in the
    # 1st, 3rd, ... positions in this list
    tokens = re.split(r'\{(.*?)\}', format_str)
    keywords = tokens[1::2]

    # Now replace keyword arguments with named groups matching them. We also escape between keyword
    # arguments so we support meta-characters there. Re-join tokens to form our regexp pattern
    tokens[1::2] = map(u'(?P<{}>.*)'.format, keywords)
    tokens[0::2] = map(re.escape, tokens[0::2])
    pattern = ''.join(tokens)

    # Use our pattern to match the given string, raise if it doesn't match
    matches = re.match(pattern, s)
    if not matches:
        raise Exception("Format string did not match")

    # Return a dict with all of our keywords and their values
    return {x: matches.group(x) for x in keywords}


def observatory_coords(obs='MeerKAT'):
    # MeerKAT Coordinates from WIKI
    #
    from astropy.coordinates import Angle
    from astropy import units as u

    if obs == 'MeerKAT':
        latitude  = '-30d43m16s'
        longitude =  '21d24m40s'
        height    = 1053.0  #meter

    return (Angle(latitude),Angle(longitude),height * u.m)

def get_LST_range(coords1,coords2,obsdatetime,timer,elevationlimit,frame='fk5'):

    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy import units as u
    from astropy.time import Time
    import numpy as np

    timerange         = np.linspace(timer[0],timer[1],100)*u.hour

    latitude, longitude, height = observatory_coords()

    obs_location      = EarthLocation(lat=latitude.deg, lon=longitude.deg, height=height)
    observing_time    = Time(Time(obsdatetime).utc + timerange, scale='utc', location=obs_location)
    LST               = observing_time.sidereal_time('mean')

    if frame == 'galactic':
        pointing_coords   = SkyCoord(coords1,coords2,frame=frame,unit='deg')
        pointing_radec    = pointing_coords.transform_to('fk5')
    else:
        pointing_radec    = SkyCoord(coords1,coords2,frame=frame,unit='deg')

    pointing_altaz    = pointing_radec.transform_to(AltAz(obstime=observing_time,location=obs_location))
    sel_elevation     = pointing_altaz.alt.deg  > elevationlimit

    idxmax = list(pointing_altaz.alt.deg).index(max(list(pointing_altaz.alt.deg)))

    # get the index of the highest_culmination
    maxidx = list(pointing_altaz.alt.deg)[idxmax]

    return LST[sel_elevation],observing_time[sel_elevation],list(LST)[idxmax],list(observing_time)[idxmax]


def get_parallactic_angle(pointing_radec,obsdatetime,timer,elevationlimit):
    '''
    Get parallactic angle for a set of coordinates

    Keyword arguments:
    coords1, coords2  (float)-- Coordinates of target in specified frame
    obsdatetime (string)     -- Observing date and time
    timer (tuple)            -- Timerange for observing as [start,end]
    elevationlimit (float)   -- Lower elevation limit of source
    '''

    from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
    from astropy import units as u
    from astropy.time import Time
    import numpy as np

    timerange         = np.linspace(timer[0],timer[1], 100)*u.hour

    latitude, longitude, height = observatory_coords()

    obs_location      = EarthLocation(lat=latitude.deg,lon=longitude.deg,height=height)
    observing_time    = Time(Time(obsdatetime).utc + timerange, scale='utc', location=obs_location)
    LST               = observing_time.sidereal_time('mean')

    pointing_altaz    = pointing_radec.transform_to(AltAz(obstime=observing_time,location=obs_location))
    sel_elevation     = pointing_altaz.alt.deg  > elevationlimit

    H = (LST[sel_elevation].radian - pointing_radec.ra.radian)

    #
    # formulae from 
    # https://github.com/astropy/astroplan/blob/master/astroplan/observer.py
    # # Eqn (14.1) of Meeus' Astronomical Algorithms
    #
    PA = np.arctan2(np.sin(H),
                        (np.tan(latitude.radian) *
                        np.cos(pointing_radec.dec.radian) -
                        np.sin(pointing_radec.dec.radian)*np.cos(H)))*u.radian

    return list(Angle(PA).deg), LST[sel_elevation], observing_time[sel_elevation], pointing_altaz[sel_elevation]

def parse_listobs(listobs_file, output='scans'):
    '''
    Parse a listobs file and get output table depending on required data
    '''
    listobs_lines = parse_file(listobs_file)

    if output == 'scans':
        for i, line in enumerate(listobs_lines):
            if 'Date' and 'Timerange' in line:
                start_i = i+1
            if 'nRows = Total number of rows per scan' in line:
                end_i = i
                break

        listobs_table = ascii.read(listobs_lines[start_i:end_i],
                                   format='fixed_width_no_header',
                                   names=['Date','Start_Time','End_Time','Scan',
                                          'FieldID','FieldName','nRows','SpwIDs',
                                          'Interval','ScanIntent'],
                                   col_starts=[2,14,27,40,49,51,74,83,88,92],
                                   col_ends=[12,24,37,43,50,61,81,86,91,-1],
                                   guess=False)

        return listobs_table

    if output == 'fields':
        for i, line in enumerate(listobs_lines):
            if 'Fields:' in line:
                start_i = i+2
            if 'Spectral Windows:' in line:
                end_i = i
                break

        listobs_table = ascii.read(listobs_lines[start_i:end_i],
                                   format='fixed_width_no_header',
                                   names=['ID','Code','Name','RA',
                                          'DEC','Epoch','SrcID','nRows'],
                                   col_starts=[2,7,12,32,48,64,72,82],
                                   col_ends=[5,9,22,47,63,69,74,88],
                                   guess=False)

        return listobs_table

    if output == 'antennas':
        for i, line in enumerate(listobs_lines):
            if 'Antennas:' in line:
                start_i = i+3
                break

        listobs_table = ascii.read(listobs_lines[start_i:-1],
                                   format='fixed_width_no_header',
                                   names=['ID','Name','Station','Diameter',
                                          'Long','Lat','East_offset','North_offset',
                                          'Elevation','x','y','z'],
                                   col_starts=[2,7,13,23,32,46,63,77,90,103,119,134],
                                   col_ends=[4,11,17,29,44,57,73,87,101,117,133,149],
                                   guess=False)

        return listobs_table

    if output == 'all':
        return listobs_lines