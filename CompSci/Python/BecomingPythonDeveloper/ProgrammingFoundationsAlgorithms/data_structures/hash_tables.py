items = dict({'key1': 1, 'key2': 2, 'key3': 'three', 'key4': 4})
print(items)

items2 = dict()
items2['key1'] = 1
items2['key2'] = 2
items2['key3'] = 'three'
print(items2)

for key, value in items.items():
    print(f'Key: {key}, Value: {value}')
