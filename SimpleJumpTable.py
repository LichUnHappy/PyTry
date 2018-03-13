#　以字典的形式跳转　switch 当然也可以 if elif else
def f(x):
    return {
        '0': "You are o.\n",
        '1': "You are 1.\n",
        '2': "You are 2.\n",
    }.get(x)

if __name__ == "__main__":
    code = input("Only single-digit numbers are welcome.\n")
    print(f(code))

