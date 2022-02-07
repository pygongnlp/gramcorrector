test_dict = {
    "a" : 1,
    "b" : 2
}
print([key for key in test_dict.items()])
print(f"test dict  {' '.join([f'{key}:{value}' for key, value in test_dict.items()])}")