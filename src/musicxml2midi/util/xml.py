from __future__ import annotations
from xml.etree import ElementTree as ET

def get_ns(root):
    return {"m": root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

def F(elem, tag, ns):
    return elem.find(f"m:{tag}", ns) if ns else elem.find(tag)

def FA(elem, tag, ns):
    return elem.findall(f"m:{tag}", ns) if ns else elem.findall(tag)
