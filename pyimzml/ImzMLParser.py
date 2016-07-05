# -*- coding: utf-8 -*-

# Copyright 2015 Dominik Fay
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#J: allow compressed read
# allow fast read and larger-than-memory read
# use obo codes instead of names

import time

from bisect import bisect_left
import sys
import xml.etree.ElementTree as ET
try:
    from lxml.etree import iterparse
except ImportError:
    try:
        from xml.etree.cElementTree import iterparse
    except ImportError:
        from xml.etree.ElementTree import iterparse
import struct
from warnings import warn
import numpy as np
from collections import namedtuple
from StringIO import StringIO

from pyimzml.compression import NoCompression, ZlibCompression
from encoding import Float64, Float32, Integer64, Integer32

param_group_elname = "referenceableParamGroup"
data_processing_elname = "dataProcessing"
instrument_confid_elname = "instrumentConfiguration"

SpectrumBase = namedtuple('Spectrum', 'index coords mz_len mz_offset mz_enc_len int_len int_offset int_enc_len')
Coords = namedtuple('Coords', 'xmin xmax ymin ymax')

class Spectrum(SpectrumBase):
    def get_spectrum(self):
        if self.ibd is None:
            raise ValueError("No ibd object defined")
        return self.ibd.get_spectrum(self)

    def get_physical_coordinates(self):
        """
        For a pixel index i, return the real-world coordinates in nanometers.

        This is equivalent to multiplying the image coordinates of the given pixel with the pixel size.

        :param i: the pixel index
        :return: a tuple of x and y coordinates.
        :rtype: Tuple[float]
        :raises KeyError: if the .imzML file does not specify the attributes "pixel size x" and "pixel size y"
        """
        if self.meta is None:
            raise ValueError("No meta data defined")
        try:
            pixel_size_x = self.meta["pixel size x"]
            pixel_size_y = self.meta["pixel size y"]
        except KeyError:
            raise KeyError("Could not find all pixel size attributes in imzML file")
        image_x, image_y = self.coords[:2]
        return image_x * pixel_size_x, image_y * pixel_size_y

class IBDParser(object):
    def open(self, master):
            #~ mz_compression,
            #~ mz_encoding,
            #~ int_compresssion,
            #~ int_encoding,
            #~ cont_mode)
        self.master = master
        meta = master.meta
        self.meta = meta

        if meta.get("mzArray/no compression") is not None:
            self.mz_compression = NoCompression()
        elif meta.get("mzArray/zlib") is not None:
            self.mz_compression = ZlibCompression()
        else:
            raise TypeError("Unknown or unspecified compression mode for mzArray")

        if meta.get("intensityArray/no compression") is not None:
            self.intensity_compression = NoCompression()
        elif meta.get("intensityArray/zlib") is not None:
            self.intensity_compression = ZlibCompression()
        else:
            raise TypeError("Unknown or unspecified compression mode for intensityArray")

        if meta.get("mzArray/64-bit float") is not None:
            self.mz_encoding = Float64()
        elif meta.get("mzArray/32-bit float") is not None:
            self.mz_encoding = Float32()
        elif meta.get("mzArray/64-bit integer") is not None:
            self.mz_encoding = Integer64()
        elif meta.get("mzArray/32-bit integer") is not None:
            self.mz_encoding = Integer32()
        else:
            raise TypeError("Unknown or unspecified data type for mzArray")

        if meta.get("intensityArray/64-bit float") is not None:
            self.intensity_encoding = Float64()
        elif meta.get("intensityArray/32-bit float") is not None:
            self.intensity_encoding = Float32()
        elif meta.get("intensityArray/64-bit integer") is not None:
            self.intensity_encoding = Integer64()
        elif meta.get("intensityArray/32-bit integer") is not None:
            self.intensity_encoding = Integer32()
        else:
            raise TypeError("Unknown or unspecified data type for intensityArray")

    def get_intensity(self, spectrum):
        self.ibd.seek(spectrum.int_offset)
        int_data = self.ibd.read(spectrum.int_enc_len)
        int_data = self.intensity_compression.decompress(int_data)
        int_data = self.intensity_encoding.decode(int_data)
        return int_data

    def get_mz(self, spectrum):
        self.ibd.seek(spectrum.mz_offset)
        mz_data = self.ibd.read(spectrum.mz_enc_len)
        mz_data = self.mz_compression.decompress(mz_data)
        mz_data = self.mz_encoding.decode(mz_data)
        return mz_data

    def get_spectrum(self, spectrum):
        return self.get_mz(spectrum), self.get_intensity(spectrum)

    def get_eic(self, spectra, mz_value, tol=0.1, reduce_func=max):
        '''
        :param spectra:
          a list of Spectrum objects
        :param mz_value:
          the center of the requested m/z range
        :param tol: (tolerance)
          the plus / minus from the mz_value. Range will be 2x tol
        '''
        eic = []
        cont_mode = self.master.meta.get('continuous') is not None
        if cont_mode:
            #bisection is expensive. Bisect only once in continuous mode.
            mz_data = self.get_mz(spectra[0])
            start, end = _bisect_spectrum(mz_data, mz_value, tol)
        for spec in spectra:
        #~ for i, spec in enumerate(spectra):
            #~ print i
            if cont_mode:
                int_data = self.get_intensity(spec)
            else:
                mz_data, int_data = self.get_spectrum(spec)
                start, end = _bisect_spectrum(mz_data, mz_value, tol)
            value = reduce_func(int_data[start:end])
            eic.append(value)
        return eic

    def close(self):
        self.ibd.close()

class IBDFileParser(IBDParser):
    def open(self, master):
        meta = master.meta
        IBDParser.open(self, master)

        filename = meta.filename[:-5]+"ibd"
        self.ibd = open(filename, 'rb')

class IBDMemoryParser(IBDParser):
    '''load the entire .ibd file into memory for fast access'''
    def open(self, master):
        IBDParser.open(self, master)

        filename = master.filename[:-5]+"ibd"
        with open(filename, 'rb') as f:
            self.ibd = StringIO(f.read())

class IBDSpeedParser(IBDFileParser):
    '''if the ibd file is in continuous mode and the intensity data is not compressed speed mode can be used'''
    def open(self, master):
        self.master = master
        meta = master.meta
        if meta.get('continuous') is None:
            raise TypeError("IBDSpeedParser can only be used for continuous mode")
        if meta.get('intensityArray/no compression') is None:
            raise TypeError("IBDSpeedParser can only be used for uncompressed data")
        #~ if not isinstance(self.intensity_compression, NoCompression):
            #~ raise TypeError("IBDSpeedParser can only be used for uncompressed data")

        IBDFileParser.open(self, master)
        self.mz_data = None

    def get_eic(self, spectra, mz_value, tol=0.1, reduce_func=max):
        if self.mz_data is None:
            self.mz_data, _ = self.master[0].get_spectrum()
        eic = []
        start, end = _bisect_spectrum(self.mz_data, mz_value, tol)
        offset = self.intensity_encoding.width * start
        length = self.intensity_encoding.width * (end - start)
        for s in spectra:
            self.ibd.seek(s.int_offset + offset)
            data = self.intensity_encoding.decode(self.ibd.read(length))
            eic.append(reduce_func(data))
        return eic

class MetaParser(dict):
    '''parse a .imzML file and extract the file-wide variables

    :param filename:
            name of the XML file. Must end with .imzML.
    '''
    #ImzML files can be parsed by name or by obo code. Well-formed files will have
    #identical results in either mode.
    use_obo_codes = False # use names if False

    def __init__(self, filename):
        """This method reads only a subset of the available meta information and may be extended in the future. The keys
        are named similarly to the imzML names. Currently supported keys: "max dimension x", "max dimension y",
        "pixel size x", "pixel size y", "matrix solution concentration", "wavelength", "focus diameter x",
        "focus diameter y", "pulse energy", "pulse duration", "attenuation".

        If a key is not found in the XML tree, it will not be in the dict either.

        :return:
            self (a list)
        :rtype:
            list
        :raises Warning:
            if an xml attribute has a number format different from the imzML specification
        """


        self.filename = filename
        self.root = ET.parse(filename).getroot()
        #get schema location and add it to namespace
        self.ns = {'imzml':self.root.tag.split('}')[0].strip('{')}
        self.imzml = None

        self._parse_meta()

    def _find(self, name, root=None):
        '''find a tag in the .ImzML file'''
        if root is None:
            root = self.root
        return root.find(".//imzml:"+name, self.ns)


    def _add_group(self, name, root=None, data=None):
        if data is None:
            data = self
        for group in self._find(name, root):
            group_id = group.attrib['id']
            for element in group:
                data["%s/%s"%(group_id, element.attrib['name'])] = element.attrib['value']
                data["%s/%s"%(group_id, element.attrib['accession'])] = element.attrib['value']

    def _parse_meta(self):
        '''add both the name and obo code to self. We trust the author of the
        imzML file matched the names and obo codes...'''

        for element in self._find("fileContent"):
            self[element.attrib['name']] = element.attrib['value']
            self[element.attrib['accession']] = element.attrib['value']

        self._add_group("referenceableParamGroupList")

        self._add_group("softwareList")

        self._add_group("scanSettingsList")

        self._add_group("instrumentConfigurationList")

        for group in self._find("dataProcessingList"):
            group_id = group.attrib['id']
            for process in group:
                process_id = process.attrib['softwareRef']
                for element in process:
                    self["%s/%s/%s"%(group_id, process_id, element.attrib['name'])] = element.attrib['value']
                    self["%s/%s/%s"%(group_id, process_id, element.attrib['accession'])] = element.attrib['value']

        if self.use_obo_codes:
            pass

    def get_spectra(self, ibd=None):
        for spectrum in self._find("spectrumList"):
            data = {}

            index = spectrum.attrib['index']

            for element in self._find("scan", spectrum):
                data[element.attrib['accession']] = element.attrib['value']

            for array in self._find("binaryDataArrayList", spectrum):
                array_type = self._find('referenceableParamGroupRef', array).attrib['ref']
                for element in array:
                    if 'accession' in element.attrib:
                        data[array_type+element.attrib['accession']] = element.attrib['value']
                        data[array_type+element.attrib['name']] = element.attrib['value']

            if 'position z' in data:
                coords = (int(data["IMS:1000050"]), int(data["IMS:1000051"]), int(data["position z"]))
            else:
                coords = (int(data["IMS:1000050"]), int(data["IMS:1000051"]), None)

            s = Spectrum(
                int(index), # index as reported by the file
                coords, # coords
                int(data["mzArrayexternal array length"]), # mz_len
                int(data["mzArrayexternal offset"]), # mz_offset
                int(data["mzArrayexternal encoded length"]), # mz_enc_len
                int(data["intensityArrayIMS:1000103"]), # int_len
                int(data["intensityArrayIMS:1000102"]), # int_offset
                int(data["intensityArrayIMS:1000104"])) # int_enc_len

            s.imzml = self.imzml
            s.ibd = ibd
            yield s



class ImzMLParser(list):
    """
    Parser for imzML 1.1.0 files (see specification here:
    http://imzml.org/download/imzml/specifications_imzML1.1.0_RC1.pdf).

    Iteractively reads the .imzML file into memory while pruning the per-spectrum metadata (everything in
    <spectrumList> elements) during initialization. Returns a spectrum upon calling getspectrum(i). The binary file
    is read in every call of getspectrum(i). Use enumerate(parser.coordinates) to get all coordinates with their
    respective index. Relevant meta data is stored in parser.imzmldict
    """

    def __init__(self, filename, ibd_parser=None):
        """
        Opens the two files corresponding to the file name, reads the entire .imzML
        file and extracts required attributes. Does not read any binary data, yet.

        :param filename:
            name of the XML file. Must end with .imzML. Binary data file must be named equally but ending with .ibd
        """
        #dictionary of file-wide settings
        self.filename = filename
        self.meta = MetaParser(filename) # really no reason to be instance variable
        #~ self._parse_meta()

        if ibd_parser is None:
            self.ibd_parser = IBDFileParser()
        else:
            self.ibd_parser = ibd_parser
            #~ #auto-choose
            #~ if self.meta.get("IMS:1000030") is not None and self.meta.get("intensityArray/MS:1000576") is not None:
                #~ #continuous mode and uncompressed
                #~ self.ibd_parser = IBDSpeedParser()
            #~ else:

                #~ self.ibd_parser = IBDFileParser()
        #~ else:
            #~ self.ibd_parser = ibd_parser
        #~
        #~ self.ibd_parser = IBDSpeedParser()
        self.ibd_parser.open(self)


        for spectra in self.meta.get_spectra(self.ibd_parser):
            self.append(spectra)


        self.by_coords = {scan.coords:scan for scan in self}
        self.by_index = {scan.index:scan for scan in self}

        Xs, Ys, Zs = zip(*self.by_coords)
        self.coords = Coords(min(Xs), max(Xs), min(Ys), max(Ys))

        self.by_z = {z:[] for z in set(Zs)}
        for s in self:
            self.by_z[s.coords[2]].append(s)

    def get_eic(self, spectra, mz_value, tol=0.1, reduce_func=max):
        return self.ibd_parser.get_eic(spectra, mz_value, tol, reduce_func)

    # system method for use of 'with ... as'
    def __enter__(self):
        return self

    # system method for use of 'with ... as'
    def __exit__(self, exc_t, exc_v, trace):
        self.ibd_parser.close()

def getionimage(p, mz_value, tol=0.1, z=None, reduce_func=sum):
    """
    Get an image representation of the intensity distribution
    of the ion with specified m/z value.

    By default, the intensity values within the tolerance region are summed.

    :param p:
        the ImzMLParser (or anything else with similar attributes) for the desired dataset
    :param mz_value:
        m/z value for which the ion image shall be returned
    :param tol:
        Absolute tolerance for the m/z value, such that all ions with values
        mz_value-|tol| <= x <= mz_value+|tol| are included. Defaults to 0.1
    :param z:
        z Value if spectrogram is 3-dimensional.
    :param reduce_func:
        the bahaviour for reducing the intensities between mz_value-|tol| and mz_value+|tol| to a single value. Must
        be a function that takes a sequence as input and outputs a number. By default, the values are summed.

    :return:
        numpy matrix with each element representing the ion intensity in this
        pixel. Can be easily plotted with matplotlib
    """
    tol = abs(tol)
    if z:
        im = np.zeros((p.imzmldict["max count of pixels y"], p.imzmldict["max count of pixels x"], z))
        for i, (x, y, z) in enumerate(p.coordinates):
            mzs, ints = p.getspectrum(i)
            min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
            im[y - 1, x - 1, z - 1] = reduce_func(ints[min_i:max_i])
    else:
        im = np.zeros((p.imzmldict["max count of pixels y"], p.imzmldict["max count of pixels x"]))
        for i, (x, y,) in enumerate(p.coordinates):
            mzs, ints = p.getspectrum(i)
            min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
            im[y - 1, x - 1] = reduce_func(ints[min_i:max_i])
    return im

def _bisect_spectrum(mzs, mz_value, tol):
    return bisect_left(mzs, mz_value - tol), bisect_left(mzs, mz_value + tol) + 1


if __name__ == '__main__':
    fn = r'C:\Egnyte\Private\jbrauer\Data\2016-05-11_mixed ps\2016-05-11_03\test.imzML'
    #~ hm = MetaParser(fn)
    hm = ImzMLParser(fn, ibd_parser=IBDSpeedParser())
    #~ hm = ImzMLParser(fn, ibd_parser=IBDFileParser())
    #~ hm = ImzMLParser(fn, ibd_parser=IBDMemoryParser())
    start = time.time()
    data = hm.get_eic(hm, 106, tol=2)
    elapsed = time.time() - start
    print data
    print elapsed, 'seconds'
    print 'data len:', len(data)

    #~ for k,v in hm.by_coords.items():
        #~ print k,v
