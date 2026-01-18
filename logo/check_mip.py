import struct, sys

def mipcount(path):
    b=open(path,'rb').read()
    if b[:4] != b"DDS ":
        return None
    hdr=b[4:4+124]
    mip=struct.unpack_from("<I", hdr, 24)[0]
    return mip if mip else 1

for p in sys.argv[1:]:
    print(p, "mip=", mipcount(p))
