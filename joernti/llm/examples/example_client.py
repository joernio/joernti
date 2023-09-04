from telnetlib import Telnet
import json

with open('./waynecz_reworm.json', 'r') as f:
    raw_slice = f.read()
    # use default port
    with Telnet('localhost', 1337) as tn:
        tn.write(raw_slice.encode() + b'\r\r')
        response = tn.read_all().decode()

    json_dct = json.loads(response)
    print(response)
