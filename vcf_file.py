import vobject
import pandas as pd
import numpy as np

vcfile = '/home/oscar/Descargas/Contactos.vcf'

with open(vcfile) as source_file:
    for vcard in vobject.readComponents(source_file):
        print(vcard.fn.value)
        # print(vcard.title.value)
        print(vcard.tel.value)
        # print(vcard.email.value)

with open(vcfile) as file:
    for vcard in vobject.readComponents(file):
        print(vcard.prettyPrint())  # Prints the pretty card

with open(vcfile) as inf:
    indata = inf.read()
    vc = vobject.readComponents(indata)
    vo = next(vc, None)
    while vo is not None:
        vo.prettyPrint()
        vo = next(vc, None)


file2 = open(vcfile)
file2

file3 = (vcfile)

for vcard in vobject.readComponents(file2):
    newvcardtext = vcard.replace( '=\n=', '=')

print(vobject.readOne(el, allowQP=True))

for vcard in vobject.readComponents(file2):
    print(vcard.fn.value)  # Prints the full name
    
