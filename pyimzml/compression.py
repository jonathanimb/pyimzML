
import zlib

class Compression(object):
    def __repr__(self):
        return type(self).__name__ + "()"

class NoCompression(Compression):
    name = "no compression"

    def rounding(self, data):
        return data

    def compress(self, bytes):
        return bytes

    def decompress(self, bytes):
        return bytes

class ZlibCompression(Compression):
    name = "zlib compression"

    def __init__(self, round_amt=None):
        self.round_amt = round_amt

    def rounding(self, data):
        if self.round_amt is not None:
            return [round(x,self.round_amt) for x in data] #rounding helps the compression, but is lossy
        return data

    def compress(self, bytes):
        return zlib.compress(bytes)

    def decompress(self, bytes):
        return zlib.decompress(bytes)
