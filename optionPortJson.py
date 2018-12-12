
import json

filter = "JSON file (*.json) | *.json | All Files (*.*) | *.*||"
filename = rs.OpenFileName("Open JSON File", filter)

if filenae:
    with open(filename, 'r') as f:
        datastore = json.load(f)



print datastore