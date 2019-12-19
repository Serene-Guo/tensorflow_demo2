
keyword='你好'
vec_key_byte = bytes(keyword, encoding = "utf-8")

print (type(vec_key_byte))
print (type(vec_key_byte.decode('utf-8')))
print (vec_key_byte.decode('utf-8'))
