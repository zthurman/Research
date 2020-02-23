def greatest_common_denominator(a, b):
    while b != 0:
        temp = a
        a = b
        b = temp % b

    return a


print(greatest_common_denominator(20, 8))
print(greatest_common_denominator(60, 96))
