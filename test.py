def sum(a,b, c=None):
    if c:
        return a+b+c
    return a+b

res = sum(1,2,4)
print(res)